"""I-RAVEN benchmark for CubeMind.

Evaluates CubeMind on the RAVEN visual reasoning dataset (Raven's Progressive
Matrices). Uses HuggingFaceM4/RAVEN from Hugging Face, which provides 7
configurations with 2000 test problems each.

Dataset: https://huggingface.co/datasets/HuggingFaceM4/RAVEN
Paper:   https://arxiv.org/abs/1903.02741

Each problem has 8 context panels (a 3x3 grid with the bottom-right missing)
and 8 answer choices. The model must pick the correct answer. Visual attributes
(shape, size, color, angle) follow relational rules (Progression, Arithmetic,
Constant, Distribute Three) that the model must detect.

CubeMind approach:
  1. Encode each panel image as a block-code vector via the perception encoder
  2. Feed context panel block-codes through the HMM ensemble to predict the
     missing panel's block-code
  3. Compare prediction to each choice's block-code via similarity
  4. Select the choice with highest similarity

Configurations map to CubeMind paper naming:
  - center_single                              -> Center
  - distribute_four                            -> 2x2 Grid
  - distribute_nine                            -> 3x3 Grid
  - left_center_single_right_center_single     -> Left-Right (L-R)
  - up_center_single_down_center_single        -> Up-Down (U-D)
  - in_center_single_out_center_single         -> Out-In (O-IC)
  - in_distribute_four_out_center_single       -> In-Out (O-IG)

Usage:
    from benchmarks.iraven import run_iraven_benchmark
    results = run_iraven_benchmark(model)

    # Or run a specific configuration
    results = run_iraven_benchmark(model, configs=["center_single"])

    # CLI
    python -m benchmarks.iraven
    python -m benchmarks.iraven --configs center_single distribute_four
    python -m benchmarks.iraven --max-problems 100 --no-train
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes
from cubemind.telemetry import metrics

logger = logging.getLogger(__name__)

# ── Dataset Configuration ─────────────────────────────────────────────────────

DATASET_ID = "HuggingFaceM4/RAVEN"

# All 7 RAVEN configurations
ALL_CONFIGS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
]

# Mapping from HF config names to short display names (paper-style)
CONFIG_DISPLAY_NAMES = {
    "center_single": "Center",
    "distribute_four": "2x2 Grid",
    "distribute_nine": "3x3 Grid",
    "left_center_single_right_center_single": "L-R",
    "up_center_single_down_center_single": "U-D",
    "in_center_single_out_center_single": "O-IC",
    "in_distribute_four_out_center_single": "O-IG",
}


# ── Dataset Loading ───────────────────────────────────────────────────────────


def _check_datasets_available() -> bool:
    """Check if the HuggingFace datasets library is installed."""
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def _check_pillow_available() -> bool:
    """Check if Pillow is installed for image processing."""
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def load_raven_split(
    config: str = "center_single",
    split: str = "test",
    cache_dir: str | Path | None = None,
) -> list[dict]:
    """Load a split of the RAVEN dataset from Hugging Face.

    Each returned dict has:
        - panels: list of 8 PIL images (context panels)
        - choices: list of 8 PIL images (answer choices)
        - target: int (correct answer index, 0-7)
        - metadata: str (XML with rules and attributes)

    Args:
        config: Dataset configuration name (e.g. "center_single").
        split: Dataset split ("train", "validation", or "test").
        cache_dir: Optional cache directory for downloaded data.

    Returns:
        List of problem dicts.

    Raises:
        ImportError: If datasets or Pillow are not installed.
        ValueError: If config is not a valid RAVEN configuration.
    """
    if not _check_datasets_available():
        raise ImportError(
            "The 'datasets' library is required for the RAVEN benchmark. "
            "Install it with: pip install datasets"
        )
    if not _check_pillow_available():
        raise ImportError(
            "Pillow is required for the RAVEN benchmark. "
            "Install it with: pip install Pillow"
        )

    if config not in ALL_CONFIGS:
        raise ValueError(
            f"Unknown RAVEN config '{config}'. "
            f"Valid configs: {', '.join(ALL_CONFIGS)}"
        )

    import datasets

    logger.info("Loading RAVEN dataset: config=%s, split=%s", config, split)

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    # Load only metadata + target columns to avoid image shape issues
    # in multi-component configs (L-R, U-D, in-out have variable panel sizes)
    try:
        ds = datasets.load_dataset(DATASET_ID, name=config, split=split, **kwargs)
        problems = []
        for row in ds:
            problems.append({
                "panels": row.get("panels", []),
                "choices": row.get("choices", []),
                "target": row.get("target", None),
                "metadata": row.get("metadata", ""),
            })
    except (ValueError, Exception) as e:
        # Multi-component configs may fail on image deserialization —
        # fall back to loading just metadata via select_columns
        logger.info("Standard load failed (%s), trying metadata-only load", e)
        ds = datasets.load_dataset(
            DATASET_ID, name=config, split=split, **kwargs
        ).select_columns(["metadata", "target"])
        problems = []
        for row in ds:
            problems.append({
                "panels": [],
                "choices": [],
                "target": row.get("target", None),
                "metadata": row.get("metadata", ""),
            })

    logger.info("Loaded %d problems from %s/%s", len(problems), config, split)
    return problems


# ── NVSA Panel Encoding (role-filler binding) ─────────────────────────────────
#
# Ported from the proven eval_iraven_learned.py. Encodes panels via symbolic
# attributes (Type, Size, Color, Angle) using role-filler binding, matching
# the NVSA approach (Hersche et al. 2023). Uses deterministic seeds so the
# same attribute value always maps to the same block-code.

ATTRS = ["Type", "Size", "Color", "Angle"]
N_ATTR_VALUES = 10  # I-RAVEN uses 0-9 for each attribute

# Multi-component configs: panels have 2+ independent components
MULTI_COMPONENT = {
    "left_center_single_right_center_single": [(0, "Left"), (1, "Right")],
    "up_center_single_down_center_single": [(0, "Up"), (1, "Down")],
    "in_center_single_out_center_single": [(0, "Out"), (1, "In")],
    "in_distribute_four_out_center_single": [(0, "Out"), (1, "In")],
}


def _stable_seed(name: str) -> int:
    """Deterministic seed from a string name."""
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return h


def _build_role_vectors(bc: BlockCodes) -> dict[str, np.ndarray]:
    """Build deterministic role vectors for attribute slots AND spatial positions."""
    roles = {
        attr: bc.random_discrete(seed=_stable_seed(f"role_{attr}"))
        for attr in ATTRS
    }
    # Spatial position roles for grid configs (up to 9 for 3x3)
    for i in range(9):
        roles[f"pos_{i}"] = bc.random_discrete(seed=_stable_seed(f"role_pos_{i}"))
    return roles


def _build_attr_codebooks(bc: BlockCodes) -> dict[str, np.ndarray]:
    """Build deterministic codebooks for each attribute (10 values each)."""
    return {
        attr: bc.codebook_discrete(N_ATTR_VALUES, seed=_stable_seed(f"cb_{attr}"))
        for attr in ATTRS
    }


# Module-level caches (rebuilt per bc instance via _get_encoding_tables)
_encoding_cache: dict[int, tuple[dict, dict]] = {}


def _get_encoding_tables(bc: BlockCodes):
    """Get or build role vectors and attribute codebooks for a BlockCodes instance."""
    key = id(bc)
    if key not in _encoding_cache:
        _encoding_cache[key] = (_build_role_vectors(bc), _build_attr_codebooks(bc))
    return _encoding_cache[key]


def encode_entity_nvsa(entity_attrs: dict, bc: BlockCodes) -> np.ndarray:
    """Encode a single entity via role-filler binding.

    entity_vec = bind(role_type, cb_type[val]) ⊛ bind(role_size, cb_size[val]) ⊛ ...

    Args:
        entity_attrs: Dict with Type, Size, Color, Angle (int values 0-9).
        bc: BlockCodes instance.

    Returns:
        Block-code vector (k, l).
    """
    roles, codebooks = _get_encoding_tables(bc)

    parts = []
    for attr in ATTRS:
        val = int(entity_attrs.get(attr, 0))
        val = max(0, min(val, N_ATTR_VALUES - 1))
        role = roles[attr]
        filler = codebooks[attr][val]
        parts.append(bc.bind(role, filler))

    # Bind all attribute bindings together
    result = parts[0]
    for p in parts[1:]:
        result = bc.bind(result, p)
    return result


def encode_panel_nvsa(entities: list[dict], bc: BlockCodes) -> np.ndarray:
    """Encode a full panel: role-filler encode each entity, bind to position, then bundle.

    Each entity is bound to a spatial position role before bundling.
    This preserves which object is where in the grid — critical for
    distribute configs where spatial arrangement follows rules.

    Panel = (Obj0 x Pos0) + (Obj1 x Pos1) + ... + (ObjN x PosN)

    Args:
        entities: List of entity dicts, each with Type/Size/Color/Angle.
        bc: BlockCodes instance.

    Returns:
        Discrete block-code vector (k, l).
    """
    if not entities:
        return bc.random_discrete(seed=0)

    roles, _ = _get_encoding_tables(bc)

    positioned = []
    for i, e in enumerate(entities):
        ent_vec = encode_entity_nvsa(e, bc)
        # Bind entity to its spatial position
        pos_role = roles.get(f"pos_{i}", roles["pos_0"])
        positioned.append(bc.bind(pos_role, ent_vec))

    if len(positioned) == 1:
        return bc.discretize(positioned[0])
    return bc.discretize(bc.bundle(positioned))


def parse_panel_entities(metadata_xml: str, panel_index: int) -> list[dict]:
    """Parse entity attributes from RAVEN metadata XML for any panel index.

    Works for both context panels (0-7) and choice panels (8-15).

    Args:
        metadata_xml: XML metadata string from RAVEN dataset.
        panel_index: Panel index (0-7 context, 8-15 choices).

    Returns:
        List of entity dicts with Type/Size/Color/Angle keys.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(metadata_xml)
    except ET.ParseError:
        return [{"Type": 0, "Size": 0, "Color": 0, "Angle": 0}]

    panels = root.findall(".//Panel")
    if panel_index >= len(panels):
        return [{"Type": 0, "Size": 0, "Color": 0, "Angle": 0}]

    panel = panels[panel_index]
    entities = panel.findall(".//Entity")

    if not entities:
        return [{"Type": 0, "Size": 0, "Color": 0, "Angle": 0}]

    return [
        {
            "Type": int(e.get("Type", "0")),
            "Size": int(e.get("Size", "0")),
            "Color": int(e.get("Color", "0")),
            "Angle": int(e.get("Angle", "0")),
        }
        for e in entities
    ]


def encode_panel_from_metadata(
    metadata_xml: str, panel_index: int, bc: BlockCodes
) -> np.ndarray:
    """Encode a panel from metadata XML using NVSA role-filler binding.

    Works for both context panels (indices 0-7) and choice panels (indices 8-15).

    Args:
        metadata_xml: XML metadata string.
        panel_index: Panel index in the XML (0-7 context, 8-15 choices).
        bc: BlockCodes instance.

    Returns:
        Discrete block-code vector (k, l).
    """
    entities = parse_panel_entities(metadata_xml, panel_index)
    return encode_panel_nvsa(entities, bc)


def encode_panel_image(
    image,
    bc: BlockCodes,
    target_size: tuple[int, int] = (80, 80),
) -> np.ndarray:
    """Encode a panel image to a block-code (fallback when no metadata).

    Simple grayscale projection — much weaker than NVSA metadata encoding.
    Use encode_panel_from_metadata when XML metadata is available.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    img = image.convert("L").resize(target_size, Image.Resampling.BILINEAR)
    pixels = np.array(img, dtype=np.float32) / 255.0
    flat = pixels.ravel()
    d_vsa = bc.k * bc.l

    if flat.size >= d_vsa:
        indices = np.linspace(0, flat.size - 1, d_vsa, dtype=int)
        projected = flat[indices]
    else:
        repeats = (d_vsa // flat.size) + 1
        projected = np.tile(flat, repeats)[:d_vsa]

    return bc.discretize(projected.reshape(bc.k, bc.l))


# ── Multi-View Observation Sequences ──────────────────────────────────────────
#
# Ported from the proven eval_iraven_learned.py approach. Three views capture
# different structural patterns in the panel sequence:
#   absolute:   raw panel vectors (captures direct patterns)
#   delta:      unbind(p_{t+1}, p_t) (captures transformations)
#   row_bundle: bundle per row (captures cross-row structure)


def make_views(
    panel_vecs: list[np.ndarray],
    bc: BlockCodes,
) -> dict[str, list[np.ndarray]]:
    """Create three observation views from panel vectors.

    Args:
        panel_vecs: List of block-code panel vectors (8 context or 9 with answer).
        bc: BlockCodes instance for bind/unbind/bundle ops.

    Returns:
        Dict mapping view name to list of observation vectors.
    """
    views = {}

    # Absolute: raw sequence
    views["absolute"] = list(panel_vecs)

    # Delta: consecutive differences via unbinding
    deltas = []
    for t in range(len(panel_vecs) - 1):
        deltas.append(bc.unbind(panel_vecs[t + 1], panel_vecs[t]))
    views["delta"] = deltas if deltas else list(panel_vecs)

    # Row bundle: group into rows of 3
    row_bundles = []
    if len(panel_vecs) >= 3:
        row_bundles.append(bc.bundle([panel_vecs[0], panel_vecs[1], panel_vecs[2]]))
    if len(panel_vecs) >= 6:
        row_bundles.append(bc.bundle([panel_vecs[3], panel_vecs[4], panel_vecs[5]]))
    remaining = list(panel_vecs[6:])
    if remaining:
        row_bundles.append(
            bc.bundle(remaining) if len(remaining) >= 2 else remaining[0]
        )
    views["row_bundle"] = row_bundles if row_bundles else list(panel_vecs)

    return views


def train_multiview_hmms(
    train_problems: list[dict],
    bc: BlockCodes,
    use_metadata: bool = True,
    n_states: int = 16,
    em_epochs: int = 20,
    batch_size: int = 30,
    temperature: float = 40.0,
    seed: int = 42,
) -> dict:
    """Train one HMM per view using Baum-Welch EM on training problems.

    Args:
        train_problems: List of RAVEN problem dicts (with panels, choices, target).
        bc: BlockCodes instance.
        use_metadata: Use XML metadata encoding.
        n_states: Number of HMM codebook states.
        em_epochs: Number of Baum-Welch EM epochs.
        batch_size: Batch size for EM updates.
        temperature: Emission softmax temperature.
        seed: Random seed.

    Returns:
        Dict mapping view name to trained HMMRule.
    """
    from cubemind.reasoning.hmm_rule import HMMRule

    # Build codebook for HMM states
    codebook = bc.codebook_discrete(n_states, seed=seed)

    # Encode all training problems and build multi-view sequences
    view_sequences: dict[str, list[list[np.ndarray]]] = {
        "absolute": [], "delta": [], "row_bundle": [],
    }

    for prob in train_problems:
        # Encode context panels (NVSA metadata for consistent encoding)
        if use_metadata and prob.get("metadata"):
            ctx_vecs = [
                encode_panel_from_metadata(prob["metadata"], i, bc)
                for i in range(len(prob["panels"]))
            ]
        else:
            ctx_vecs = [encode_panel_image(p, bc) for p in prob["panels"]]

        # Append correct answer (same encoding space as context!)
        target_idx = prob.get("target")
        if target_idx is not None:
            if use_metadata and prob.get("metadata"):
                answer_vec = encode_panel_from_metadata(
                    prob["metadata"], 8 + target_idx, bc
                )
            else:
                answer_vec = encode_panel_image(prob["choices"][target_idx], bc)
            full_seq = ctx_vecs + [answer_vec]
        else:
            full_seq = ctx_vecs

        views = make_views(full_seq, bc)
        for vname in view_sequences:
            view_sequences[vname].append(views[vname])

    # Train one HMM per view
    view_seeds = {"absolute": seed + 100, "delta": seed + 200, "row_bundle": seed + 300}
    hmms = {}

    for vname in ["absolute", "delta", "row_bundle"]:
        seqs = view_sequences[vname]
        hmm = HMMRule(codebook, temperature=temperature, seed=view_seeds[vname])

        if not seqs:
            hmms[vname] = hmm
            continue

        n_seqs = len(seqs)
        for epoch in range(em_epochs):
            indices = list(range(n_seqs))
            np.random.default_rng(seed + epoch).shuffle(indices)

            epoch_ll = 0.0
            n_batches = 0
            for start in range(0, n_seqs, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = [seqs[i] for i in batch_idx]
                batch = [s for s in batch if len(s) >= 2]
                if not batch:
                    continue
                ll = hmm.train_step_em(batch)
                epoch_ll += ll
                n_batches += 1

            avg_ll = epoch_ll / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"    {vname:>12} epoch {epoch+1:2d}/{em_epochs}: avg_ll={avg_ll:.4f}",
                    flush=True,
                )

        hmms[vname] = hmm

    return hmms


def predict_multiview(
    ctx_vecs: list[np.ndarray],
    cand_vecs: list[np.ndarray],
    hmms: dict,
    bc: BlockCodes,
) -> int:
    """Predict the best answer using multi-view HMM routing.

    For each view:
      1. Run HMM forward on context to get detection confidence (normalized LL)
      2. Predict next panel vector via posterior-weighted transition
      3. Score all candidates against prediction via similarity

    Route via the view with highest confidence.

    Args:
        ctx_vecs: List of 8 context panel block-codes.
        cand_vecs: List of 8 candidate answer block-codes.
        hmms: Dict mapping view name to trained HMMRule.
        bc: BlockCodes instance.

    Returns:
        Predicted answer index (0-7).
    """
    from cubemind.reasoning.hmm_rule import _logsumexp

    best_view_score = -np.inf
    best_candidate_idx = 0

    for view_name, hmm in hmms.items():
        views = make_views(ctx_vecs, bc)
        obs_seq = views[view_name]

        if len(obs_seq) < 2:
            continue

        # Detection confidence: length-normalized log-likelihood
        try:
            log_ll, log_alpha = hmm.forward(obs_seq)
            norm_ll = log_ll / len(obs_seq)
        except Exception:
            norm_ll = -np.inf

        # Predict next panel vector
        try:
            predicted = hmm.predict(obs_seq)
        except Exception:
            continue

        # Score candidates via similarity
        predicted_disc = bc.discretize(predicted)
        cand_scores = np.array([
            bc.similarity(predicted_disc, cv) for cv in cand_vecs
        ])

        if norm_ll > best_view_score:
            best_view_score = norm_ll
            best_candidate_idx = int(np.argmax(cand_scores))

    return best_candidate_idx


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_problem_dataset(
    problem: dict,
    hmms: dict,
    bc: BlockCodes,
    use_metadata: bool = True,
) -> tuple[bool, int, float]:
    """Evaluate on a single RAVEN problem using integer detectors + HMM tiebreaker.

    Primary signal: deterministic integer-domain rule detectors on per-attribute
    3x3 grids (constant, progression, arithmetic, distribute-three).
    Secondary signal: multi-view HMM prediction for tiebreaking.

    Args:
        problem: Dict with panels, choices, target, metadata.
        hmms: Dict mapping view name to trained HMMRule.
        bc: BlockCodes instance.
        use_metadata: If True, encode from XML metadata (NVSA-style).

    Returns:
        (correct, predicted_idx, latency_ms)
    """
    from cubemind.reasoning.rule_detectors import score_candidates as detector_score

    t0 = time.perf_counter()

    # Parse problem into per-component context + candidates
    config = problem.get("_config", "center_single")
    if problem.get("metadata"):
        components = parse_problem_components(problem["metadata"], config)
    else:
        # No metadata — fall back to multi-view HMM only
        context_codes = [encode_panel_image(p, bc) for p in problem["panels"]]
        choice_codes = [encode_panel_image(c, bc) for c in problem["choices"]]
        pred_idx = predict_multiview(context_codes, choice_codes, hmms, bc)
        latency = (time.perf_counter() - t0) * 1000
        target = problem.get("target")
        return pred_idx == target if target is not None else False, pred_idx, latency

    if not components:
        pred_idx = 0
        latency = (time.perf_counter() - t0) * 1000
        target = problem.get("target")
        return pred_idx == target if target is not None else False, pred_idx, latency

    # Grid configs have multiple entities per panel — use Sinkhorn alignment
    is_grid = config in (
        "distribute_four", "distribute_nine",
        "in_distribute_four_out_center_single",
    )

    # Multi-component scoring: accumulate detector scores across all components
    # n_choices = 8 for all RAVEN configs (even when images aren't loaded)
    n_choices = len(problem["choices"]) if problem["choices"] else 8
    combined_scores = [0.0] * n_choices
    attrs_to_check = ["Type", "Size", "Color", "Number"]

    for comp in components:
        # ── Grid configs: use position-aware scoring ────────────────
        if is_grid and use_metadata and problem.get("metadata"):
            # Primary: aggregated detector scoring
            comp_scores = detector_score(
                comp["context"], comp["candidates"], attrs=attrs_to_check
            )

            # Position tiebreaker: extract bbox-based position fingerprints
            # and use position distribution rules to disambiguate tied candidates
            pos_scores = _score_position_rules(
                problem["metadata"], config, n_choices,
            )
            for i in range(n_choices):
                comp_scores[i] += pos_scores[i]

        else:
            # ── Standard aggregated path for single/compound configs ─
            comp_scores = detector_score(
                comp["context"], comp["candidates"], attrs=attrs_to_check
            )

        # VSA set-completion scoring for distribution rules.
        # When integer detectors tie, this breaks ties using bundled
        # row signatures — the candidate that makes Row 2 match the
        # Master Set (average of Row 0 + Row 1 signatures) wins.
        if use_metadata and problem.get("metadata"):
            try:
                ctx_codes = [
                    encode_panel_from_metadata(problem["metadata"], i, bc)
                    for i in range(8)
                ]
                cand_codes = [
                    encode_panel_from_metadata(problem["metadata"], 8 + i, bc)
                    for i in range(n_choices)
                ]
                dist_scores = _score_distribution_rule(ctx_codes, cand_codes, bc)
                # VSA distribution signal as tiebreaker
                for i in range(n_choices):
                    comp_scores[i] += dist_scores[i] * 0.5
            except Exception:
                pass
        for i in range(n_choices):
            combined_scores[i] += comp_scores[i]

    # Check for clear winner or tie (epsilon for float precision)
    max_score = max(combined_scores)
    eps = 1e-5
    n_tied = sum(1 for s in combined_scores if abs(s - max_score) < eps)

    if n_tied == 1:
        pred_idx = int(np.argmax(combined_scores))
    else:
        # Tie — use multi-view HMM as tiebreaker
        if use_metadata and problem.get("metadata"):
            context_codes = [
                encode_panel_from_metadata(problem["metadata"], i, bc)
                for i in range(8)
            ]
            choice_codes = [
                encode_panel_from_metadata(problem["metadata"], 8 + i, bc)
                for i in range(n_choices)
            ]
        else:
            context_codes = [encode_panel_image(p, bc) for p in problem["panels"]]
            choice_codes = [encode_panel_image(c, bc) for c in problem["choices"]]

        hmm_idx = predict_multiview(context_codes, choice_codes, hmms, bc)

        if combined_scores[hmm_idx] == max_score:
            pred_idx = hmm_idx
        else:
            pred_idx = int(np.argmax(combined_scores))

    latency = (time.perf_counter() - t0) * 1000

    target = problem.get("target")
    correct = pred_idx == target if target is not None else False

    return correct, pred_idx, latency


def _aggregate_entities(entities: list[dict], layout_number: int = 0) -> dict:
    """Aggregate multiple entities into a single attribute dict (mode per attr).

    Includes 'Number' from the Layout element — critical for distribute configs.
    """
    from collections import Counter

    if not entities:
        return {"Type": 0, "Size": 0, "Color": 0, "Number": layout_number}

    result = {"Number": layout_number}
    for attr in ["Type", "Size", "Color"]:
        vals = [e.get(attr, 0) for e in entities]
        counter = Counter(vals)
        result[attr] = counter.most_common(1)[0][0]
    return result


def parse_panel_entities_component(
    metadata_xml: str, panel_index: int, component_idx: int | None = None
) -> tuple[list[dict], int]:
    """Parse entity attributes from a specific component of a panel.

    For multi-component configs (L-R, U-D, in-out), each panel has multiple
    Component elements. This extracts entities from a specific component.

    Args:
        metadata_xml: XML metadata string.
        panel_index: Panel index (0-7 context, 8-15 choices).
        component_idx: Component index within the panel (None = all entities).

    Returns:
        Tuple of (entities, layout_number) where entities is a list of dicts
        and layout_number is the Number attribute from the Layout element.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(metadata_xml)
    except ET.ParseError:
        return [{"Type": 0, "Size": 0, "Color": 0}], 0

    panels = root.findall(".//Panel")
    if panel_index >= len(panels):
        return [{"Type": 0, "Size": 0, "Color": 0}], 0

    panel = panels[panel_index]
    layout_number = 0

    if component_idx is not None:
        components = panel.findall(".//Component")
        if component_idx < len(components):
            comp = components[component_idx]
            layout_el = comp.find(".//Layout")
            if layout_el is not None:
                layout_number = int(layout_el.get("Number", "0"))
            entities = comp.findall(".//Entity")
        else:
            entities = []
    else:
        layout_el = panel.find(".//Layout")
        if layout_el is not None:
            layout_number = int(layout_el.get("Number", "0"))
        entities = panel.findall(".//Entity")

    if not entities:
        return [{"Type": 0, "Size": 0, "Color": 0}], layout_number

    return [
        {
            "Type": int(e.get("Type", "0")),
            "Size": int(e.get("Size", "0")),
            "Color": int(e.get("Color", "0")),
        }
        for e in entities
    ], layout_number


def parse_problem_components(
    metadata_xml: str, config: str
) -> list[dict] | None:
    """Parse a RAVEN problem into per-component context + candidates.

    For single-component configs: returns 1 component.
    For multi-component configs: returns 2 components (e.g., Left + Right).

    Each component dict has:
        context: list of 8 aggregated attribute dicts
        candidates: list of 8 aggregated attribute dicts

    Returns None on parse error.
    """
    comp_configs = MULTI_COMPONENT.get(config, None)

    if comp_configs is None:
        # Single component — aggregate entities (mode per attribute)
        context = []
        context_entities = []
        for i in range(8):
            ents, num = parse_panel_entities_component(metadata_xml, i)
            context.append(_aggregate_entities(ents, num))
            context_entities.append(ents)
        candidates = []
        candidate_entities = []
        for i in range(8):
            ents, num = parse_panel_entities_component(metadata_xml, 8 + i)
            candidates.append(_aggregate_entities(ents, num))
            candidate_entities.append(ents)
        return [{"context": context, "candidates": candidates,
                 "context_entities": context_entities,
                 "candidate_entities": candidate_entities}]
    else:
        # Multi-component — extract per-component
        components = []
        for comp_idx, comp_name in comp_configs:
            context = []
            context_entities = []
            for i in range(8):
                ents, num = parse_panel_entities_component(
                    metadata_xml, i, component_idx=comp_idx
                )
                context.append(_aggregate_entities(ents, num))
                context_entities.append(ents)
            candidates = []
            candidate_entities = []
            for i in range(8):
                ents, num = parse_panel_entities_component(
                    metadata_xml, 8 + i, component_idx=comp_idx
                )
                candidates.append(_aggregate_entities(ents, num))
                candidate_entities.append(ents)
            components.append({"context": context, "candidates": candidates,
                              "context_entities": context_entities,
                              "candidate_entities": candidate_entities})
        return components


# ── Position-Aware Scoring (Grid Configs) ────────────────────────────────────


def _extract_position_signature(metadata_xml: str, panel_index: int) -> tuple:
    """Extract a position signature from entity bboxes in a panel.

    Returns a sorted tuple of discretized bbox centers, which uniquely
    identifies the spatial layout of entities in the panel.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(metadata_xml)
    except ET.ParseError:
        return ()

    panels = root.findall(".//Panel")
    if panel_index >= len(panels):
        return ()

    entities = panels[panel_index].findall(".//Entity")
    if not entities:
        return ()

    positions = []
    for e in entities:
        bbox_str = e.get("bbox", "")
        if bbox_str:
            try:
                bbox = eval(bbox_str) if isinstance(bbox_str, str) else bbox_str
                # Discretize to grid positions (round to nearest 0.25)
                cx = round(bbox[0] * 4) / 4
                cy = round(bbox[1] * 4) / 4
                positions.append((cx, cy))
            except Exception:
                pass

    return tuple(sorted(positions))


def _score_position_rules(
    metadata_xml: str,
    config: str,
    n_choices: int,
) -> list[float]:
    """Score candidates by position pattern consistency across rows.

    Extracts position signatures from all panels and checks if the
    candidate's position layout is consistent with the row/column
    patterns established by the context panels.

    Args:
        metadata_xml: XML metadata string.
        config: RAVEN configuration name.
        n_choices: Number of candidates (typically 8).

    Returns:
        List of position-based scores per candidate.
    """
    scores = [0.0] * n_choices

    # Extract position signatures for context panels
    ctx_sigs = [_extract_position_signature(metadata_xml, i) for i in range(8)]
    cand_sigs = [_extract_position_signature(metadata_xml, 8 + i) for i in range(n_choices)]

    # Row-wise position pattern: panels [0,1,2], [3,4,5], [6,7,?]
    # Check if position signatures follow a row-wise rule

    # Pattern: constant position within rows
    r0_const = ctx_sigs[0] == ctx_sigs[1] == ctx_sigs[2]
    r1_const = ctx_sigs[3] == ctx_sigs[4] == ctx_sigs[5]

    if r0_const and r1_const:
        # Row-constant: candidate should match row2 known panels
        expected = ctx_sigs[6]
        for i in range(n_choices):
            if cand_sigs[i] == expected:
                scores[i] += 3.0

    # Pattern: column-wise position consistency
    # Column 2 panels: [2, 5, ?]
    if ctx_sigs[2] == ctx_sigs[5]:
        for i in range(n_choices):
            if cand_sigs[i] == ctx_sigs[2]:
                scores[i] += 2.0

    # Pattern: distribute-three on positions
    # Each row contains a permutation of the same set of position signatures
    r0_set = set(ctx_sigs[0:3])
    r1_set = set(ctx_sigs[3:6])
    if r0_set == r1_set and len(r0_set) >= 2:
        r2_known = set(ctx_sigs[6:8])
        missing = r0_set - r2_known
        if len(missing) == 1:
            expected_sig = missing.pop()
            for i in range(n_choices):
                if cand_sigs[i] == expected_sig:
                    scores[i] += 4.0

    # Pattern: position count consistency (entity count = position count)
    r0_counts = [len(s) for s in ctx_sigs[0:3]]
    r1_counts = [len(s) for s in ctx_sigs[3:6]]
    r2_counts = [len(s) for s in ctx_sigs[6:8]]

    # Apply same number-detection logic to position counts
    # Constant: all same count per row
    if len(set(r0_counts)) == 1 and len(set(r1_counts)) == 1:
        if r2_counts and len(set(r2_counts)) == 1:
            expected_count = r2_counts[0]
            for i in range(n_choices):
                if len(cand_sigs[i]) == expected_count:
                    scores[i] += 1.5

    return scores


# ── Entity Set Consistency Scoring (Grid Configs) ────────────────────────────


def _entity_attr_multiset(entities: list[dict], attr: str) -> tuple:
    """Extract sorted multiset of an attribute from entities."""
    return tuple(sorted(e.get(attr, -1) for e in entities))


def _score_entity_set_consistency(
    context_entities: list[list[dict]],
    candidate_entities: list[list[dict]],
    attrs: tuple = ("Type", "Size", "Color"),
) -> list[float]:
    """Score candidates by entity attribute set consistency across rows.

    In distribute configs, each row's entity attribute multisets follow
    rules (constant set, progression of counts, etc.). This scores
    candidates by how well they complete the row 2 pattern relative
    to rows 0 and 1.

    Args:
        context_entities: 8 panels, each a list of entity dicts.
        candidate_entities: 8 candidates, each a list of entity dicts.
        attrs: Attributes to compare.

    Returns:
        List of 8 scores (higher = better).
    """
    n = len(candidate_entities)
    scores = [0.0] * n

    # Row panels: row0 = [0,1,2], row1 = [3,4,5], row2 = [6,7,?]
    for attr in attrs:
        # Build per-row multisets
        row0 = [_entity_attr_multiset(context_entities[i], attr) for i in range(3)]
        row1 = [_entity_attr_multiset(context_entities[i], attr) for i in range(3, 6)]
        row2_known = [_entity_attr_multiset(context_entities[i], attr) for i in range(6, 8)]

        # Check if rows 0 and 1 share a pattern
        # Pattern 1: same multiset across all panels in a row (constant)
        r0_const = len(set(row0)) == 1
        r1_const = len(set(row1)) == 1

        if r0_const and r1_const:
            # Constant rule: each row has identical entity sets
            # Row 2's candidate should match row2_known pattern
            expected = row2_known[0] if row2_known else row0[0]
            for i in range(n):
                cand_set = _entity_attr_multiset(candidate_entities[i], attr)
                if cand_set == expected:
                    scores[i] += 2.0

        # Pattern 2: same multiset across ALL rows (distribute)
        all_sets = set(row0 + row1)
        if len(all_sets) == 1:
            expected = row0[0]
            for i in range(n):
                cand_set = _entity_attr_multiset(candidate_entities[i], attr)
                if cand_set == expected:
                    scores[i] += 3.0

        # Pattern 3: column-wise consistency
        for col in range(3):
            col_sets = []
            if col < 3:
                col_sets.append(_entity_attr_multiset(context_entities[col], attr))
            if col + 3 < 6:
                col_sets.append(_entity_attr_multiset(context_entities[col + 3], attr))
            if len(col_sets) == 2 and col_sets[0] == col_sets[1] and col == 2:
                # Column 2 is constant — candidate should match
                for i in range(n):
                    cand_set = _entity_attr_multiset(candidate_entities[i], attr)
                    if cand_set == col_sets[0]:
                        scores[i] += 1.5

    # Entity count consistency
    row0_counts = [len(context_entities[i]) for i in range(3)]
    row1_counts = [len(context_entities[i]) for i in range(3, 6)]
    row2_known_counts = [len(context_entities[i]) for i in range(6, 8)]

    # Check count pattern across rows
    for i in range(n):
        cand_count = len(candidate_entities[i])
        # If count follows same pattern as rows 0 and 1
        if (row0_counts[2] == row1_counts[2]
                and len(row2_known_counts) >= 1
                and row2_known_counts[0] == row0_counts[0]):
            if cand_count == row0_counts[2]:
                scores[i] += 2.0

    return scores


# ── VSA Set-Completion Scoring (Distribution Rules) ──────────────────────────


def _score_distribution_rule(
    context_vecs: list[np.ndarray],
    cand_vecs: list[np.ndarray],
    bc: BlockCodes,
) -> list[float]:
    """Score candidates by VSA set distribution (bundling).

    For distribution rules, each row is a permutation of the same attribute set.
    The bundled signature of a complete row should match a "master set" derived
    from the known rows. The candidate that makes Row 2 most similar to the
    master set is the best answer.

    Also checks column-wise distribution for additional signal.

    Args:
        context_vecs: 8 block-code vectors for context panels.
        cand_vecs: 8 block-code vectors for candidates.
        bc: BlockCodes instance.

    Returns:
        List of 8 similarity scores (higher = better set completion).
    """
    # Row signatures
    row0_sig = bc.bundle([context_vecs[0], context_vecs[1], context_vecs[2]])
    row1_sig = bc.bundle([context_vecs[3], context_vecs[4], context_vecs[5]])
    master_row = bc.bundle([row0_sig, row1_sig])

    # Column signatures
    col0_sig = bc.bundle([context_vecs[0], context_vecs[3], context_vecs[6]])
    col1_sig = bc.bundle([context_vecs[1], context_vecs[4], context_vecs[7]])
    master_col = bc.bundle([col0_sig, col1_sig])

    scores = []
    for cand in cand_vecs:
        # Row completion: Row 2 + candidate should match master row
        row2_sig = bc.bundle([context_vecs[6], context_vecs[7], cand])
        row_sim = float(bc.similarity(
            bc.discretize(master_row), bc.discretize(row2_sig)
        ))

        # Column completion: Col 2 + candidate should match master col
        col2_sig = bc.bundle([context_vecs[2], context_vecs[5], cand])
        col_sim = float(bc.similarity(
            bc.discretize(master_col), bc.discretize(col2_sig)
        ))

        # Take the max of row and column signals
        scores.append(max(row_sim, col_sim))

    return scores


# ── Synthetic Fallback (original code, preserved for offline testing) ─────────


def generate_synthetic_problem(
    bc: BlockCodes,
    n_context: int = 8,
    n_choices: int = 8,
    seed: int = 0,
) -> dict:
    """Generate a synthetic RAVEN-style problem as block codes.

    Used as a fallback when the real dataset is not available. Creates
    block-code sequences with learnable patterns that CubeMind's HMM
    can detect.

    Args:
        bc: BlockCodes instance.
        n_context: Number of context panels.
        n_choices: Number of answer choices.
        seed: Random seed.

    Returns:
        Dict with: panels (list of block-codes), choices (list),
        target (int), rule_type (str).
    """
    rng = np.random.default_rng(seed)

    # Generate a "rule" as a binding key
    rule = bc.random_discrete(seed=int(rng.integers(0, 2**31)))

    # Context panels follow the rule: each panel = bind(prev, rule)
    panels = []
    current = bc.random_discrete(seed=int(rng.integers(0, 2**31)))
    for _ in range(n_context):
        panels.append(current)
        current = bc.bind(current, rule)

    # Correct answer is the next in sequence
    correct = current

    # Generate distractors
    target = int(rng.integers(0, n_choices))
    choices = []
    for i in range(n_choices):
        if i == target:
            choices.append(correct)
        else:
            choices.append(bc.random_discrete(seed=int(rng.integers(0, 2**31))))

    return {
        "panels": panels,
        "choices": choices,
        "target": target,
        "rule_type": "sequential_bind",
        "metadata": None,
    }


def evaluate_synthetic_problem(model, problem: dict) -> tuple[bool, int, float]:
    """Evaluate CubeMind on a synthetic problem (block-code inputs).

    Args:
        model: CubeMind instance.
        problem: Dict from generate_synthetic_problem.

    Returns:
        (correct, predicted_idx, latency_ms)
    """
    t0 = time.perf_counter()

    prediction, weights = model.hmm.predict(problem["panels"])

    bc = BlockCodes(K_BLOCKS, L_BLOCK)
    best_idx = -1
    best_sim = -1.0
    for i, choice in enumerate(problem["choices"]):
        sim = bc.similarity(bc.discretize(prediction), choice)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    latency = (time.perf_counter() - t0) * 1000
    correct = best_idx == problem["target"]

    return correct, best_idx, latency


# ── Main Benchmark Runner ─────────────────────────────────────────────────────


def run_iraven_benchmark(
    model,
    configs: list[str] | None = None,
    split: str = "test",
    max_problems: int | None = None,
    train_first: bool = True,
    train_split: str = "train",
    train_epochs: int = 3,
    train_problems: int = 50,
    train_lr: float = 0.05,
    use_metadata: bool = True,
    use_synthetic_fallback: bool = True,
    cache_dir: str | Path | None = None,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Run the full I-RAVEN benchmark on the real dataset.

    Downloads and evaluates CubeMind on the HuggingFaceM4/RAVEN dataset
    across all (or selected) configurations. Falls back to synthetic
    problems if the dataset library is not available.

    Args:
        model: CubeMind instance.
        configs: List of configuration names to evaluate. None = all 7.
        split: Evaluation split ("test" or "validation").
        max_problems: Maximum problems per configuration (None = all).
        train_first: Whether to train the HMM on training data first.
        train_split: Split to use for training ("train").
        train_epochs: Number of training epochs.
        train_problems: Number of training problems per config.
        train_lr: Training learning rate.
        use_metadata: Use XML metadata encoding (NVSA-style) vs raw images.
        use_synthetic_fallback: Fall back to synthetic data if dataset
            loading fails.
        cache_dir: Cache directory for dataset downloads.
        seed: Random seed.

    Returns:
        Dict with:
            overall_accuracy: float
            overall_latency_ms: float
            per_config: dict mapping config name to per-config results
            n_total: int
            n_correct: int
            dataset_source: str ("HuggingFaceM4/RAVEN" or "synthetic")
            wall_clock_s: float
    """
    if configs is None:
        configs = list(ALL_CONFIGS)

    # Use smaller block-code dimensions for I-RAVEN (proven in eval_iraven_learned.py)
    # K=8, L=64 (d_vsa=512) gives better EM convergence than K=16, L=128 (d_vsa=2048)
    iraven_k = kwargs.get("iraven_k", 8)
    iraven_l = kwargs.get("iraven_l", 64)
    bc = BlockCodes(iraven_k, iraven_l)
    logger.info("Using block-code dims: k=%d, l=%d (d_vsa=%d)", iraven_k, iraven_l, iraven_k * iraven_l)
    dataset_available = _check_datasets_available() and _check_pillow_available()

    benchmark_start = time.perf_counter()
    per_config = {}
    total_correct = 0
    total_problems = 0
    total_latency = 0.0
    dataset_source = "synthetic"

    for config in configs:
        display_name = CONFIG_DISPLAY_NAMES.get(config, config)
        logger.info("Evaluating config: %s (%s)", config, display_name)

        # ── Load dataset ──────────────────────────────────────────────────
        problems = None
        is_synthetic = True

        if dataset_available:
            try:
                problems = load_raven_split(config, split, cache_dir)
                if max_problems is not None:
                    problems = problems[:max_problems]
                is_synthetic = False
                dataset_source = DATASET_ID
            except Exception as e:
                logger.warning(
                    "Failed to load RAVEN/%s: %s. Falling back to synthetic.",
                    config, e,
                )

        if problems is None:
            if not use_synthetic_fallback:
                logger.error(
                    "Dataset not available and synthetic fallback disabled. "
                    "Install 'datasets' and 'Pillow': "
                    "pip install datasets Pillow"
                )
                continue
            n = max_problems if max_problems else 200
            problems = [
                generate_synthetic_problem(bc, seed=seed + i)
                for i in range(n)
            ]

        # ── Multi-view EM training ────────────────────────────────────────
        hmms = None
        if train_first and not is_synthetic:
            try:
                train_data = load_raven_split(config, train_split, cache_dir)
                train_data = train_data[:train_problems]
                logger.info(
                    "Training multi-view HMMs on %d %s problems (%d EM epochs)",
                    len(train_data), config, train_epochs,
                )
                hmms = train_multiview_hmms(
                    train_data, bc,
                    use_metadata=use_metadata,
                    n_states=model.codebook.shape[0],
                    em_epochs=train_epochs,
                    temperature=40.0,
                    seed=seed,
                )
            except Exception as e:
                logger.warning("Training failed for %s: %s", config, e)

        elif train_first and is_synthetic:
            # Train on synthetic data using multi-view EM
            hmms = train_multiview_hmms(
                problems[:min(50, len(problems))], bc,
                use_metadata=False,
                n_states=model.codebook.shape[0],
                em_epochs=train_epochs,
                temperature=40.0,
                seed=seed,
            )

        # Fall back to model's HMM ensemble if multi-view training failed
        if hmms is None:
            from cubemind.reasoning.hmm_rule import HMMRule
            hmms = {"absolute": model.hmm.rules[0]}

        # ── Evaluate ──────────────────────────────────────────────────────
        config_correct = 0
        config_latency = 0.0
        config_results = []

        for idx, prob in enumerate(problems):
            prob["_config"] = config  # pass config name for multi-component handling
            if is_synthetic:
                correct, pred_idx, latency = evaluate_synthetic_problem(
                    model, prob
                )
            else:
                correct, pred_idx, latency = evaluate_problem_dataset(
                    prob, hmms, bc, use_metadata=use_metadata
                )

            config_correct += int(correct)
            config_latency += latency
            config_results.append({
                "idx": idx,
                "correct": correct,
                "predicted": pred_idx,
                "target": prob.get("target"),
                "latency_ms": latency,
            })

        n_eval = len(problems)
        config_accuracy = config_correct / max(n_eval, 1)
        config_avg_latency = config_latency / max(n_eval, 1)

        per_config[config] = {
            "display_name": display_name,
            "accuracy": config_accuracy,
            "correct_count": config_correct,
            "n_problems": n_eval,
            "avg_latency_ms": config_avg_latency,
            "is_synthetic": is_synthetic,
            "per_problem": config_results,
        }

        total_correct += config_correct
        total_problems += n_eval
        total_latency += config_latency

        metrics.record(f"benchmark.iraven.{config}.accuracy", config_accuracy)
        metrics.record(
            f"benchmark.iraven.{config}.latency_ms", config_avg_latency
        )

        logger.info(
            "  %s: %.1f%% (%d/%d) avg=%.1fms",
            display_name,
            config_accuracy * 100,
            config_correct,
            n_eval,
            config_avg_latency,
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    overall_accuracy = total_correct / max(total_problems, 1)
    overall_avg_latency = total_latency / max(total_problems, 1)
    wall_clock = time.perf_counter() - benchmark_start

    metrics.record("benchmark.iraven.overall_accuracy", overall_accuracy)
    metrics.record("benchmark.iraven.overall_latency_ms", overall_avg_latency)

    return {
        "overall_accuracy": overall_accuracy,
        "overall_latency_ms": overall_avg_latency,
        "per_config": per_config,
        "n_total": total_problems,
        "n_correct": total_correct,
        "dataset_source": dataset_source,
        "wall_clock_s": wall_clock,
        "configs_evaluated": configs,
    }


# ── Pretty Printing ───────────────────────────────────────────────────────────


def print_results(results: dict) -> None:
    """Print benchmark results in a readable table format."""
    print()
    print("=" * 72)
    print("  I-RAVEN Benchmark Results")
    print(f"  Dataset: {results['dataset_source']}")
    print(f"  Wall clock: {results['wall_clock_s']:.1f}s")
    print("=" * 72)
    print()
    print(f"  {'Configuration':<35} {'Accuracy':>10} {'N':>6} {'Latency':>10}")
    print("  " + "-" * 65)

    for config_name, config_data in results["per_config"].items():
        display = config_data["display_name"]
        acc = config_data["accuracy"] * 100
        n = config_data["n_problems"]
        lat = config_data["avg_latency_ms"]
        syn = " (syn)" if config_data["is_synthetic"] else ""
        print(f"  {display + syn:<35} {acc:>9.1f}% {n:>6} {lat:>8.1f}ms")

    print("  " + "-" * 65)
    overall_acc = results["overall_accuracy"] * 100
    n_total = results["n_total"]
    overall_lat = results["overall_latency_ms"]
    print(f"  {'Overall':<35} {overall_acc:>9.1f}% {n_total:>6} {overall_lat:>8.1f}ms")
    print()

    # Context
    print(f"  Random baseline: 12.5% (1/8 choices)")
    print(f"  NVSA (Hersche 2023) target: ~87.7% overall")
    print(f"  CubeMind target: 97.5% on Center Single")
    print()


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run the I-RAVEN benchmark on CubeMind",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=ALL_CONFIGS,
        default=None,
        help="Configurations to evaluate (default: all 7)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "validation", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Max problems per config (default: all)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training phase",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--train-problems",
        type=int,
        default=50,
        help="Training problems per config (default: 50)",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Encode from images instead of metadata",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Fail instead of falling back to synthetic data",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for dataset downloads",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--k", type=int, default=8,
        help="Block code k (number of blocks, default: 8)",
    )
    parser.add_argument(
        "--l", type=int, default=64,
        help="Block code l (block length, default: 64)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from cubemind.model import CubeMind

    logger.info("Initializing CubeMind model...")
    model = CubeMind(
        n_codebook=16,
        n_hmm_rules=4,
        cache_size=1000,
        d_hidden=128,
        seed=args.seed,
    )

    results = run_iraven_benchmark(
        model,
        configs=args.configs,
        split=args.split,
        max_problems=args.max_problems,
        train_first=not args.no_train,
        train_epochs=args.train_epochs,
        train_problems=args.train_problems,
        use_metadata=not args.use_images,
        use_synthetic_fallback=not args.no_synthetic_fallback,
        cache_dir=args.cache_dir,
        seed=args.seed,
        iraven_k=args.k,
        iraven_l=args.l,
    )

    print_results(results)


if __name__ == "__main__":
    main()
