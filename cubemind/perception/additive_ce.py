"""Additive Cross-Entropy Training for VSA-CNN Perception.

Implements the bundle-predictive learning paradigm:
1. Frozen codebook W of atomic attribute block-codes
2. Target = bundled superposition of attributes present in panel
3. Loss = additive CE (cosine sim against codebook, temperature-scaled)
4. Post-hoc algorithmic binding in symbolic backend

The CNN learns to produce a query vector similar to the BUNDLED
attributes, preserving topological continuity. The binding hash is
never seen by the gradient — applied algorithmically after inference.

Reference: Hersche et al. (2023), "A Neuro-Vector-Symbolic Architecture
for Solving Raven's Progressive Matrices", Nature Machine Intelligence.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes

# ── Codebook ────────────────────────────────────────────────────────────────

ATTRS = ["Type", "Size", "Color", "Angle"]
N_VALUES = 10  # RAVEN uses 0-9 per attribute


class VSACodebook:
    """Frozen dictionary of atomic attribute block-codes.

    Each attribute value gets a unique, quasi-orthogonal block-code.
    Total codebook size = len(ATTRS) * N_VALUES = 40 vectors.

    Args:
        bc: BlockCodes instance (k, l).
        seed: Deterministic seed for reproducibility.
    """

    def __init__(self, bc: BlockCodes, seed: int = 42) -> None:
        self.bc = bc
        self.k = bc.k
        self.l = bc.l
        self.d = bc.k * bc.l

        # Generate unique codes for each (attr, value) pair
        self.codes = {}  # (attr, val) -> (k, l) block-code
        self.flat_codes = []  # list of (d,) vectors for similarity search
        self.labels = []  # list of (attr, val) tuples

        rng_seed = seed
        for attr in ATTRS:
            for val in range(N_VALUES):
                code = bc.random_discrete(seed=rng_seed)
                self.codes[(attr, val)] = code
                self.flat_codes.append(code.ravel())
                self.labels.append((attr, val))
                rng_seed += 1

        # Codebook matrix W: (n_codes, d) for batch similarity
        self.W = np.array(self.flat_codes, dtype=np.float32)  # (40, 512)
        self.n_codes = len(self.flat_codes)

    def bundle_target(self, entity_attrs: dict) -> np.ndarray:
        """Create bundled target from entity attributes.

        Target = sum of attribute codes (unbound, similarity-preserving).

        Args:
            entity_attrs: Dict with Type, Size, Color, Angle (int 0-9).

        Returns:
            Bundled block-code (k, l), L1-normalized per block.
        """
        vectors = []
        for attr in ATTRS:
            val = int(entity_attrs.get(attr, 0))
            val = max(0, min(val, N_VALUES - 1))
            vectors.append(self.codes[(attr, val)])

        return self.bc.bundle(vectors, normalize=True)

    def bundle_target_flat(self, entity_attrs: dict) -> np.ndarray:
        """Bundled target as flat vector (d,)."""
        return self.bundle_target(entity_attrs).ravel()

    def target_indices(self, entity_attrs: dict) -> list[int]:
        """Get codebook indices for the attributes present in entity.

        Args:
            entity_attrs: Dict with Type, Size, Color, Angle.

        Returns:
            List of codebook indices (one per attribute).
        """
        indices = []
        for attr in ATTRS:
            val = int(entity_attrs.get(attr, 0))
            val = max(0, min(val, N_VALUES - 1))
            idx = self.labels.index((attr, val))
            indices.append(idx)
        return indices


# ── Additive Cross-Entropy Loss ─────────────────────────────────────────────


def additive_cross_entropy(
    query: np.ndarray,
    codebook_W: np.ndarray,
    target_indices: list[int],
    temperature: float = 10.0,
) -> tuple[float, np.ndarray]:
    """Additive Cross-Entropy loss with temperature scaling.

    L = -log( exp(s * sum_j sim(q, w_yj)) / sum_i exp(s * sim(q, w_i)) )

    The numerator sums cosine similarities to ALL target attribute codes,
    preserving the bundling structure. Temperature s sharpens gradients.

    Args:
        query: CNN output vector (d,) — continuous, not block-coded.
        codebook_W: Frozen codebook (n_codes, d).
        target_indices: Indices of target attributes in codebook.
        temperature: Inverse temperature scalar (higher = sharper).

    Returns:
        (loss, grad) where grad is w.r.t. query vector (d,).
    """
    d = query.shape[0]
    n = codebook_W.shape[0]

    # Cosine similarities: sim(q, w_i) for all i
    q_norm = np.linalg.norm(query) + 1e-8
    w_norms = np.linalg.norm(codebook_W, axis=1, keepdims=True) + 1e-8
    q_hat = query / q_norm
    w_hat = codebook_W / w_norms

    sims = np.clip(w_hat @ q_hat, -1.0, 1.0)  # (n,) cosine similarities

    # Temperature-scaled logits (clamp to prevent overflow)
    logits = np.clip(temperature * sims, -50.0, 50.0)  # (n,)

    # Average target logits (not sum — prevents overflow with 4 targets)
    target_logit_avg = np.mean([logits[idx] for idx in target_indices])

    # Log-sum-exp denominator (numerically stable)
    max_logit = logits.max()
    log_denom = max_logit + np.log(np.exp(logits - max_logit).sum() + 1e-8)

    # Loss: negative log probability of target
    loss = -(target_logit_avg - log_denom)
    loss = np.clip(loss, 0.0, 50.0)  # prevent extreme values

    # Gradient w.r.t. query
    # dL/dq = temperature * (softmax(logits) @ W_hat - sum_j W_hat[yj]) / ||q||
    probs = np.exp(logits - max_logit)
    probs /= probs.sum()

    # Expected codebook vector under softmax
    expected = probs @ w_hat  # (d,)

    # Target codebook vectors (averaged, matching loss)
    target_avg = np.mean([w_hat[idx] for idx in target_indices], axis=0)

    # Gradient (clipped for stability)
    grad = temperature * (expected - target_avg) / q_norm
    grad = np.clip(grad, -1.0, 1.0)

    return float(loss), grad.astype(np.float32)


# ── Training Loop ───────────────────────────────────────────────────────────


def train_additive_ce(
    problems: list[dict],
    bc: BlockCodes,
    n_epochs: int = 30,
    lr: float = 0.01,
    temperature: float = 10.0,
    max_problems: int | None = None,
) -> tuple:
    """Train CNN with Additive Cross-Entropy on RAVEN panels.

    Args:
        problems: RAVEN problem dicts with panels + metadata.
        bc: BlockCodes instance.
        n_epochs: Training epochs.
        lr: Learning rate.
        temperature: Additive CE temperature.
        max_problems: Max problems.

    Returns:
        (model, codebook) tuple.
    """
    import xml.etree.ElementTree as ET
    from PIL import Image
    from cubemind.perception.cnn_encoder import CNNEncoder

    codebook = VSACodebook(bc)
    model = CNNEncoder(k=bc.k, l=bc.l, grid_size=(1, 1), temperature=1.0)

    if max_problems:
        problems = problems[:max_problems]

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_sim = 0.0
        n_panels = 0

        for prob in problems:
            panels = prob.get("panels", [])
            metadata = prob.get("metadata", "")
            if not panels or not metadata:
                continue

            try:
                root = ET.fromstring(metadata)
            except ET.ParseError:
                continue

            xml_panels = root.findall(".//Panel")

            for pi in range(min(len(panels), 8)):
                if pi >= len(xml_panels):
                    break

                entities = xml_panels[pi].findall(".//Entity")
                if not entities:
                    continue

                # Target: first entity's attributes
                e = entities[0]
                entity_attrs = {
                    "Type": int(e.get("Type", "0")),
                    "Size": int(e.get("Size", "0")),
                    "Color": int(e.get("Color", "0")),
                    "Angle": int(e.get("Angle", "0")),
                }

                target_indices = codebook.target_indices(entity_attrs)
                bundle_target = codebook.bundle_target(entity_attrs)

                # Prepare image
                img = panels[pi]
                if isinstance(img, Image.Image):
                    img = np.array(img.convert("L").resize((80, 80)),
                                   dtype=np.float32) / 255.0
                else:
                    img = np.asarray(img, dtype=np.float32)
                    if img.max() > 1.0:
                        img /= 255.0

                # Forward — use raw logits, not block-softmax output
                model.zero_grad()
                pred = model.forward(img)  # (k, l) after softmax

                # Get raw logits from cache (pre-softmax)
                logits_raw = model._cache.get('logits')
                if logits_raw is not None:
                    query = logits_raw.ravel().astype(np.float32)
                else:
                    query = pred.ravel()  # fallback

                # Additive CE loss
                loss, grad = additive_cross_entropy(
                    query, codebook.W, target_indices, temperature,
                )
                total_loss += loss

                # Similarity to bundled target
                sim = bc.similarity(bc.discretize(pred), bundle_target)
                total_sim += sim

                # Clip gradient
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm

                # Backward + step
                grad_2d = grad.reshape(bc.k, bc.l)
                model.backward(grad_2d)
                model.step(lr=lr)

                n_panels += 1

        avg_loss = total_loss / max(n_panels, 1)
        avg_sim = total_sim / max(n_panels, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f} "
                  f"sim={avg_sim:.4f} (bundled target)")

    return model, codebook


if __name__ == "__main__":
    import sys
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import io

    config = sys.argv[1] if len(sys.argv) > 1 else "center_single"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    print(f"Loading {config}/train (max {max_n})...")
    path = hf_hub_download(
        repo_id="HuggingFaceM4/RAVEN", repo_type="dataset",
        filename=f"{config}/train-00000-of-00001.parquet",
    )
    table = pq.read_table(path)
    problems = []
    for i in range(min(max_n, len(table))):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        panels = []
        if "panels" in row and row["panels"]:
            for p in row["panels"]:
                if isinstance(p, dict) and "bytes" in p:
                    panels.append(Image.open(io.BytesIO(p["bytes"])))
        problems.append({
            "panels": panels,
            "metadata": row.get("metadata", ""),
            "target": row.get("target"),
        })
    print(f"Loaded {len(problems)} problems")

    bc = BlockCodes(8, 64)
    print(f"\nTraining with Additive CE (temp=5.0, lr=0.01, {epochs} epochs)...")
    model, codebook = train_additive_ce(
        problems, bc, n_epochs=epochs, lr=0.01, temperature=5.0,
        max_problems=max_n,
    )
