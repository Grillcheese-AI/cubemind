"""VSA4VQA — Visual Question Answering via Vector Symbolic Architecture.

Inspired by Penzkofer et al. (CogSci 2024): encodes objects with spatial
semantic pointers (SSP) into a bundled scene memory, then answers questions
by unbinding, spatial querying, and attribute verification.

Pipeline:
  Image → object detection (bounding boxes) → SSP encoding (x,y,w,h)
  → bundle into scene memory M → parse question → execute VSA program
  → answer

SSP encoding: each object is bound with its 4D spatial location:
  SSP_i = SP_i ⊗ X^x ⊗ Y^y ⊗ W^w ⊗ H^h

Scene memory: M = Σ(SSP_i) — superposition of all object pointers.

Query by unbinding: select("lamp") = M ⊗ lamp^{-1} → spatial location.
Spatial relations: learned query masks ("to_the_right_of", "above", etc.)

Attribute verification: VisionEncoder (CLIP/SigLIP) for color, material, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 8
    L_BLOCK = 64


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """An object detected in an image with bounding box."""
    label: str
    x: float        # Center x, normalized [0, 1]
    y: float        # Center y, normalized [0, 1]
    w: float        # Width, normalized [0, 1]
    h: float        # Height, normalized [0, 1]
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)  # color, material, etc.


@dataclass
class VQAResult:
    """Result of a visual question answering query."""
    answer: str
    confidence: float
    program: list[str]
    intermediate: dict = field(default_factory=dict)


# ── Spatial Semantic Pointers ─────────────────────────────────────────────

class SpatialEncoder:
    """Encodes 4D spatial locations (x, y, w, h) as VSA block-code vectors.

    Uses fractional power binding: S(x,y,w,h) = X^x ⊗ Y^y ⊗ W^w ⊗ H^h
    where X, Y, W, H are random axis vectors and the power is achieved
    via repeated self-binding (discretized to grid resolution).

    Args:
        k:          Number of VSA blocks.
        l:          Block length.
        grid_res:   Spatial grid resolution (100 = 1% precision).
        seed:       Random seed.
    """

    def __init__(
        self, k: int = K_BLOCKS, l: int = L_BLOCK,
        grid_res: int = 100, seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self.grid_res = grid_res

        rng = np.random.default_rng(seed)

        # Four random axis vectors for spatial dimensions
        self.X_axis = self.bc.random_discrete(seed=seed)
        self.Y_axis = self.bc.random_discrete(seed=seed + 1)
        self.W_axis = self.bc.random_discrete(seed=seed + 2)
        self.H_axis = self.bc.random_discrete(seed=seed + 3)

        # Precompute power table for each axis: axis^i for i in [0, grid_res]
        # Power via repeated binding: axis^n = bind(axis, axis, ..., n times)
        self._X_powers = self._precompute_powers(self.X_axis, grid_res)
        self._Y_powers = self._precompute_powers(self.Y_axis, grid_res)
        self._W_powers = self._precompute_powers(self.W_axis, grid_res)
        self._H_powers = self._precompute_powers(self.H_axis, grid_res)

    def _precompute_powers(self, axis: np.ndarray, n: int) -> list[np.ndarray]:
        """Precompute axis^0, axis^1, ..., axis^n via repeated binding."""
        # Identity for block-code binding: one-hot at index 0 in each block
        identity = np.zeros((self.k, self.l), dtype=np.float32)
        identity[:, 0] = 1.0
        powers = [identity]  # axis^0 = identity
        current = axis.copy()
        for _ in range(n):
            powers.append(current.copy())
            current = self.bc.bind(current, axis)
        return powers

    def encode_location(self, x: float, y: float, w: float, h: float) -> np.ndarray:
        """Encode a 4D spatial location as a block-code vector.

        Args:
            x, y: Center position, normalized [0, 1].
            w, h: Width/height, normalized [0, 1].

        Returns:
            (k, l) block-code vector.
        """
        xi = int(np.clip(x * self.grid_res, 0, self.grid_res))
        yi = int(np.clip(y * self.grid_res, 0, self.grid_res))
        wi = int(np.clip(w * self.grid_res, 0, self.grid_res))
        hi = int(np.clip(h * self.grid_res, 0, self.grid_res))

        # S(x,y,w,h) = X^x ⊗ Y^y ⊗ W^w ⊗ H^h
        spatial = self._X_powers[xi]
        spatial = self.bc.bind(spatial, self._Y_powers[yi])
        spatial = self.bc.bind(spatial, self._W_powers[wi])
        spatial = self.bc.bind(spatial, self._H_powers[hi])
        return spatial


# ── Spatial Query Masks ───────────────────────────────────────────────────

# 12 core spatial relations with their query regions
# Each is a function: (ref_x, ref_y, ref_w, ref_h) → mask region (x_min, y_min, x_max, y_max)
SPATIAL_RELATIONS = {
    "to_the_right_of": lambda x, y, w, h: (x + w / 2, 0, 1.0, 1.0),
    "to_the_left_of":  lambda x, y, w, h: (0, 0, x - w / 2, 1.0),
    "above":           lambda x, y, w, h: (0, 0, 1.0, y - h / 2),
    "below":           lambda x, y, w, h: (0, y + h / 2, 1.0, 1.0),
    "on":              lambda x, y, w, h: (x - w, y - h * 1.5, x + w, y - h / 2),
    "under":           lambda x, y, w, h: (x - w, y + h / 2, x + w, y + h * 1.5),
    "near":            lambda x, y, w, h: (x - w * 2, y - h * 2, x + w * 2, y + h * 2),
    "in_front_of":     lambda x, y, w, h: (x - w, y + h / 4, x + w, y + h),
    "behind":          lambda x, y, w, h: (x - w, y - h, x + w, y - h / 4),
    "inside":          lambda x, y, w, h: (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
    "surrounding":     lambda x, y, w, h: (x - w * 1.5, y - h * 1.5, x + w * 1.5, y + h * 1.5),
    "next_to":         lambda x, y, w, h: (x - w * 1.5, y - h / 2, x + w * 1.5, y + h / 2),
}


# ── Scene Memory ──────────────────────────────────────────────────────────

class SceneMemory:
    """VSA scene memory — bundled spatial semantic pointers.

    Encodes all detected objects with their spatial locations into a
    single superposed memory vector M. Supports:
      - select(obj): unbind to retrieve spatial location
      - relate(obj, relation): find objects in spatial relation to obj
      - query attributes via VisionEncoder

    Args:
        k: Number of VSA blocks.
        l: Block length.
        seed: Random seed.
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK, seed: int = 42) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self.spatial = SpatialEncoder(k=k, l=l, seed=seed)

        # Scene state
        self.objects: list[DetectedObject] = []
        self.object_vectors: dict[str, np.ndarray] = {}  # label → SP vector
        self.memory: np.ndarray | None = None  # Bundled SSP memory M

        # Object identity codebook: random vectors per label
        self._identity_cache: dict[str, np.ndarray] = {}
        self._seed = seed

    def _get_identity(self, label: str) -> np.ndarray:
        """Get or create a deterministic identity vector for an object label."""
        if label not in self._identity_cache:
            seed = hash(label) % (2**31)
            self._identity_cache[label] = self.bc.random_discrete(seed=seed)
        return self._identity_cache[label]

    def encode_scene(self, objects: list[DetectedObject]) -> np.ndarray:
        """Encode all objects into SSP memory.

        M = Σ(SP_i ⊗ S(x_i, y_i, w_i, h_i))

        Args:
            objects: List of detected objects with bounding boxes.

        Returns:
            (k, l) bundled SSP memory vector.
        """
        self.objects = objects
        self.object_vectors = {}

        # Initialize memory as zeros
        M = np.zeros((self.k, self.l), dtype=np.float32)

        for obj in objects:
            # Object identity vector (deterministic from label)
            sp = self._get_identity(obj.label)

            # Spatial encoding
            spatial = self.spatial.encode_location(obj.x, obj.y, obj.w, obj.h)

            # SSP = SP ⊗ S(x,y,w,h)
            ssp = self.bc.bind(sp, spatial)
            self.object_vectors[obj.label] = sp

            # Bundle into memory: M += SSP
            M = M + ssp.astype(np.float32)

        self.memory = M
        return M

    def select(self, label: str) -> np.ndarray | None:
        """Retrieve spatial location of an object by unbinding from memory.

        select(label) = M ⊗ SP_label^{-1} → approximate S(x,y,w,h)

        Returns:
            (k, l) spatial vector, or None if label unknown.
        """
        if self.memory is None:
            return None
        sp = self._identity_cache.get(label)
        if sp is None:
            return None
        # Unbind: M ⊗ SP^{-1} (unbind = inverse binding)
        return self.bc.unbind(self.memory, sp)

    def relate(self, ref_label: str, relation: str) -> list[tuple[str, float]]:
        """Find objects in a spatial relation to the reference object.

        Args:
            ref_label: The reference object label (e.g., "lamp").
            relation:  Spatial relation (e.g., "to_the_right_of").

        Returns:
            List of (object_label, similarity_score) sorted by relevance.
        """
        if relation not in SPATIAL_RELATIONS:
            return []

        # Find reference object
        ref_obj = None
        for obj in self.objects:
            if obj.label == ref_label:
                ref_obj = obj
                break
        if ref_obj is None:
            return []

        # Get spatial query region
        region_fn = SPATIAL_RELATIONS[relation]
        x_min, y_min, x_max, y_max = region_fn(ref_obj.x, ref_obj.y, ref_obj.w, ref_obj.h)

        # Find objects within the query region
        results = []
        for obj in self.objects:
            if obj.label == ref_label:
                continue
            # Check if object center is within the query region
            in_region = (x_min <= obj.x <= x_max and y_min <= obj.y <= y_max)
            if in_region:
                # Compute VSA similarity for ranking
                obj_spatial = self.spatial.encode_location(obj.x, obj.y, obj.w, obj.h)
                ref_spatial = self.spatial.encode_location(ref_obj.x, ref_obj.y, ref_obj.w, ref_obj.h)
                sim = float(self.bc.similarity(obj_spatial, ref_spatial))
                results.append((obj.label, max(0.0, 1.0 - abs(sim))))

        # Also check via VSA unbinding similarity
        if self.memory is not None:
            ref_sp = self._identity_cache.get(ref_label)
            if ref_sp is not None:
                unbound = self.bc.unbind(self.memory, ref_sp)
                for obj in self.objects:
                    if obj.label == ref_label:
                        continue
                    obj_sp = self._identity_cache.get(obj.label)
                    if obj_sp is not None:
                        sim = float(self.bc.similarity(unbound, obj_sp))
                        # Add VSA similarity as tiebreaker
                        for i, (lbl, score) in enumerate(results):
                            if lbl == obj.label:
                                results[i] = (lbl, score + abs(sim) * 0.5)
                                break

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def filter_by_attribute(
        self, candidates: list[str], attr_name: str, attr_value: str,
    ) -> list[str]:
        """Filter objects by an attribute (e.g., color=red).

        Checks stored attributes first, then returns matches.
        """
        matches = []
        for label in candidates:
            for obj in self.objects:
                if obj.label == label:
                    if obj.attributes.get(attr_name) == attr_value:
                        matches.append(label)
                    break
        return matches if matches else candidates  # Fallback: return all if no match

    def list_objects(self) -> list[str]:
        """List all object labels in the scene."""
        return [obj.label for obj in self.objects]


# ── Question Parser ───────────────────────────────────────────────────────

class QuestionParser:
    """Parse natural language questions into VSA program steps.

    Maps questions to a sequence of functions following the
    Neural Module Network paradigm (Andreas et al., 2016).

    Supported function types (from VSA4VQA):
      select(obj)                     → position (SSP unbinding)
      relate(obj, relation)           → proposals (SSP query mask)
      filter(position, attr, value)   → true/false
      verify(obj, attr, value)        → true/false
      query(obj, attr)                → answer
      choose(obj, attr, value1, val2) → answer
    """

    # Relation keywords
    RELATION_MAP = {
        "right of": "to_the_right_of",
        "to the right of": "to_the_right_of",
        "left of": "to_the_left_of",
        "to the left of": "to_the_left_of",
        "above": "above",
        "over": "above",
        "on top of": "above",
        "below": "below",
        "under": "under",
        "beneath": "under",
        "on": "on",
        "near": "near",
        "next to": "next_to",
        "beside": "next_to",
        "in front of": "in_front_of",
        "behind": "behind",
        "inside": "inside",
        "in": "inside",
        "surrounding": "surrounding",
        "around": "surrounding",
    }

    # Question type patterns
    ATTRIBUTE_WORDS = {"color", "colour", "material", "shape", "size", "type"}

    def parse(self, question: str, known_objects: list[str]) -> list[dict]:
        """Parse a question into program steps.

        Args:
            question: Natural language question.
            known_objects: Labels of objects in the scene.

        Returns:
            List of program step dicts with 'function' and 'args'.
        """
        q = question.lower().strip().rstrip("?")
        steps = []

        # Detect spatial relation queries: "What is to the right of the X?"
        for phrase, relation in sorted(self.RELATION_MAP.items(), key=lambda x: -len(x[0])):
            if phrase in q:
                # Find the reference object
                ref_obj = self._find_object_in_text(q, known_objects, after=phrase)
                if ref_obj:
                    steps.append({"function": "select", "args": [ref_obj]})
                    steps.append({"function": "relate", "args": [ref_obj, relation]})

                    # Check if filtering is needed
                    for attr in self.ATTRIBUTE_WORDS:
                        if attr in q:
                            steps.append({"function": "filter_attr", "args": [attr]})
                            break

                    steps.append({"function": "query_result", "args": []})
                    return steps

        # "What color is the X?" / "What is the X made of?"
        for attr in self.ATTRIBUTE_WORDS:
            if attr in q:
                obj = self._find_object_in_text(q, known_objects)
                if obj:
                    steps.append({"function": "select", "args": [obj]})
                    steps.append({"function": "query_attr", "args": [obj, attr]})
                    return steps

        # "Is there a X?" / "How many X?"
        if q.startswith("is there") or q.startswith("are there"):
            obj = self._find_object_in_text(q, known_objects)
            if obj:
                steps.append({"function": "verify_exists", "args": [obj]})
                return steps

        if q.startswith("how many"):
            obj = self._find_object_in_text(q, known_objects)
            if obj:
                steps.append({"function": "count", "args": [obj]})
                return steps

        # "What is in the image?" / general query
        if "what" in q:
            obj = self._find_object_in_text(q, known_objects)
            if obj:
                steps.append({"function": "select", "args": [obj]})
                steps.append({"function": "describe", "args": [obj]})
                return steps

        # Fallback: list all objects
        steps.append({"function": "list_all", "args": []})
        return steps

    def _find_object_in_text(
        self, text: str, known_objects: list[str], after: str = "",
    ) -> str | None:
        """Find a known object label mentioned in the text."""
        search_text = text
        if after and after in text:
            search_text = text[text.index(after) + len(after):]

        # Try longest match first
        for obj in sorted(known_objects, key=len, reverse=True):
            if obj.lower() in search_text:
                return obj
        return known_objects[0] if known_objects else None


# ── VQA Engine ────────────────────────────────────────────────────────────

class VQAEngine:
    """Visual Question Answering engine using VSA scene memory.

    Args:
        k: Number of VSA blocks.
        l: Block length.
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK) -> None:
        self.scene = SceneMemory(k=k, l=l)
        self.parser = QuestionParser()

    def set_scene(self, objects: list[DetectedObject]) -> None:
        """Load a scene with detected objects."""
        self.scene.encode_scene(objects)

    def answer(self, question: str) -> VQAResult:
        """Answer a natural language question about the current scene.

        Args:
            question: Natural language question.

        Returns:
            VQAResult with answer, confidence, and program trace.
        """
        known = self.scene.list_objects()
        if not known:
            return VQAResult(answer="No objects in scene.", confidence=0.0, program=[])

        # Parse question into program
        program = self.parser.parse(question, known)
        program_trace = [f"{s['function']}({', '.join(str(a) for a in s['args'])})" for s in program]

        intermediate = {}
        answer = ""
        confidence = 0.0

        # Execute program steps
        for step in program:
            fn = step["function"]
            args = step["args"]

            if fn == "select":
                result = self.scene.select(args[0])
                intermediate["selected"] = args[0]

            elif fn == "relate":
                ref_label, relation = args[0], args[1]
                candidates = self.scene.relate(ref_label, relation)
                intermediate["relation"] = relation
                intermediate["candidates"] = candidates
                if candidates:
                    answer = candidates[0][0]
                    confidence = candidates[0][1]

            elif fn == "filter_attr":
                attr = args[0]
                if "candidates" in intermediate:
                    labels = [c[0] for c in intermediate["candidates"]]
                    # Can't filter without VisionEncoder, return top candidate
                    if labels:
                        answer = labels[0]

            elif fn == "query_result":
                # Return the best candidate from relation query
                if "candidates" in intermediate and intermediate["candidates"]:
                    answer = intermediate["candidates"][0][0]
                    confidence = intermediate["candidates"][0][1]

            elif fn == "query_attr":
                obj_label, attr = args[0], args[1]
                for obj in self.scene.objects:
                    if obj.label == obj_label and attr in obj.attributes:
                        answer = str(obj.attributes[attr])
                        confidence = 1.0
                        break
                if not answer:
                    answer = f"unknown {attr}"
                    confidence = 0.3

            elif fn == "verify_exists":
                exists = any(o.label == args[0] for o in self.scene.objects)
                answer = "yes" if exists else "no"
                confidence = 1.0

            elif fn == "count":
                count = sum(1 for o in self.scene.objects if args[0].lower() in o.label.lower())
                answer = str(count)
                confidence = 1.0

            elif fn == "describe":
                obj_label = args[0]
                for obj in self.scene.objects:
                    if obj.label == obj_label:
                        attrs = ", ".join(f"{k}={v}" for k, v in obj.attributes.items())
                        answer = f"{obj.label} at ({obj.x:.2f}, {obj.y:.2f})"
                        if attrs:
                            answer += f" [{attrs}]"
                        confidence = 0.8
                        break

            elif fn == "list_all":
                answer = ", ".join(known)
                confidence = 1.0

        return VQAResult(
            answer=answer or "I don't know.",
            confidence=confidence,
            program=program_trace,
            intermediate=intermediate,
        )
