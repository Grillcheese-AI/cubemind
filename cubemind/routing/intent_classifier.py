# cubemind/routing/intent_classifier.py
"""Multi-head intent classifier via VSA codebook bundling.

CubeMind-native approach: no PyTorch, no backpropagation.
Uses BGE-M3 GGUF embeddings projected to block-code VSA space,
bundled into per-class centroids, classified by cosine similarity.

Three classification heads (matching agentic-intent-classifier taxonomy):
  1. intent_type:     10 classes (informational, commercial, transactional, ...)
  2. decision_phase:   7 classes (awareness, research, consideration, ...)
  3. subtype:         18 classes (education, comparison, evaluation, ...)

Each head is a VSA codebook where each class = bundled centroid of
training examples. Classification = encode query → cosine sim → top-1.
Centroids adapt via Oja's rule on each new query (plastic routing).

Dataset: cloned/agentic-intent-classifier/data/*.jsonl
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128

# Label taxonomies (from agentic-intent-classifier config.py)
INTENT_TYPE_LABELS = (
    "informational", "exploratory", "commercial", "transactional",
    "support", "personal_reflection", "creative_generation",
    "chit_chat", "ambiguous", "prohibited",
)

DECISION_PHASE_LABELS = (
    "awareness", "research", "consideration", "decision",
    "action", "post_purchase", "support",
)

SUBTYPE_LABELS = (
    "education", "product_discovery", "comparison", "evaluation",
    "deal_seeking", "provider_selection", "signup", "purchase",
    "booking", "download", "contact_sales", "task_execution",
    "onboarding_setup", "troubleshooting", "account_help",
    "billing_help", "follow_up", "emotional_reflection",
)


def _oja_update(w, x, eta=0.01):
    """Single-step Oja update: w <- w + eta * y * (x - y * w)."""
    y = float(np.dot(w.ravel(), x.ravel()))
    dw = eta * y * (x.ravel() - y * w.ravel())
    return (w.ravel() + dw).astype(np.float32)


@dataclass
class IntentResult:
    """Result from the multi-head intent classifier."""
    intent_type: str = "ambiguous"
    intent_type_confidence: float = 0.0
    decision_phase: str = "awareness"
    decision_phase_confidence: float = 0.0
    subtype: str = "education"
    subtype_confidence: float = 0.0
    all_scores: dict[str, dict[str, float]] = field(default_factory=dict)


class VSACodebookHead:
    """A single classification head: per-class VSA centroid codebook.

    Each class is represented by a bundled centroid of training examples
    projected into (k, l) block-code space. Classification is cosine
    similarity against all centroids.
    """

    def __init__(
        self, labels: tuple[str, ...], k: int, l: int,  # noqa: E741
        oja_eta: float = 0.005,
    ) -> None:
        self.labels = labels
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self.oja_eta = oja_eta

        # Per-class centroids: {label: (k, l) float32}
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}

        for label in labels:
            self.centroids[label] = np.zeros((k, l), dtype=np.float32)
            self.counts[label] = 0

    def add_example(self, label: str, vec: np.ndarray) -> None:
        """Add a training example to a class centroid via running average."""
        if label not in self.centroids:
            return
        n = self.counts[label]
        if n == 0:
            self.centroids[label] = vec.copy().astype(np.float32)
        else:
            # Running average: centroid = (n * old + new) / (n + 1)
            self.centroids[label] = (
                (n * self.centroids[label] + vec) / (n + 1)
            ).astype(np.float32)
        self.counts[label] = n + 1

    def classify(
        self, query_vec: np.ndarray, adapt: bool = False,
    ) -> tuple[str, float, dict[str, float]]:
        """Classify a query vector against all class centroids.

        Args:
            query_vec: (k, l) block-code vector.
            adapt: If True, apply Oja update to winning centroid.

        Returns:
            (best_label, confidence, all_scores)
        """
        scores: dict[str, float] = {}
        for label, centroid in self.centroids.items():
            if self.counts[label] == 0:
                scores[label] = 0.0
                continue
            scores[label] = float(self.bc.similarity(query_vec, centroid))

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        # Normalize confidence: (best - mean) / std
        vals = np.array(list(scores.values()))
        if vals.std() > 1e-8:
            confidence = float((best_score - vals.mean()) / vals.std())
            confidence = max(0.0, min(1.0, confidence / 3.0))  # scale to [0,1]
        else:
            confidence = 0.0

        # Plastic adaptation: sharpen winning centroid toward query
        if adapt and self.counts[best_label] > 0:
            old = self.centroids[best_label]
            self.centroids[best_label] = _oja_update(
                old, query_vec, self.oja_eta,
            ).reshape(self.k, self.l)

        return best_label, confidence, scores

    @property
    def trained(self) -> bool:
        """Whether any examples have been added."""
        return any(n > 0 for n in self.counts.values())


class IntentClassifier:
    """Multi-head intent classifier using VSA codebook bundling.

    Three heads classify simultaneously:
      - intent_type (10 classes)
      - decision_phase (7 classes)
      - subtype (18 classes)

    Args:
        k: Number of VSA blocks.
        l: Block length.
        encoder: Object with .encode_action(text) -> (k, l) ndarray.
            Can be SemanticEncoder (BGE-M3) or WorldEncoder (hash fallback).
        oja_eta: Learning rate for plastic centroid adaptation.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        encoder=None,
        oja_eta: float = 0.005,
    ) -> None:
        self.k = k
        self.l = l
        self.encoder = encoder

        self.intent_type = VSACodebookHead(
            INTENT_TYPE_LABELS, k, l, oja_eta,
        )
        self.decision_phase = VSACodebookHead(
            DECISION_PHASE_LABELS, k, l, oja_eta,
        )
        self.subtype = VSACodebookHead(
            SUBTYPE_LABELS, k, l, oja_eta,
        )

    def train_from_jsonl(self, path: str | Path) -> int:
        """Train from a JSONL file with text + label fields.

        Supports fields: text, intent_type, decision_phase, subtype.
        Missing fields are skipped for that head.

        Returns number of examples processed.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder set — pass encoder= to __init__")

        path = Path(path)
        if not path.exists():
            return 0

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = rec.get("text", "")
                if not text:
                    continue

                vec = self.encoder.encode_action(text)

                if "intent_type" in rec:
                    self.intent_type.add_example(rec["intent_type"], vec)
                if "decision_phase" in rec:
                    self.decision_phase.add_example(rec["decision_phase"], vec)
                if "subtype" in rec:
                    self.subtype.add_example(rec["subtype"], vec)

                count += 1
        return count

    def train_from_directory(self, data_dir: str | Path) -> int:
        """Train from all JSONL files in a directory."""
        data_dir = Path(data_dir)
        total = 0
        for f in sorted(data_dir.glob("**/*.jsonl")):
            n = self.train_from_jsonl(f)
            if n > 0:
                total += n
        return total

    def classify(
        self, text: str, adapt: bool = False,
    ) -> IntentResult:
        """Classify a query text across all three heads.

        Args:
            text: Input query string.
            adapt: If True, plastically update winning centroids.

        Returns:
            IntentResult with predictions from all three heads.
        """
        if self.encoder is None:
            return IntentResult()

        vec = self.encoder.encode_action(text)

        it_label, it_conf, it_scores = self.intent_type.classify(
            vec, adapt=adapt,
        )
        dp_label, dp_conf, dp_scores = self.decision_phase.classify(
            vec, adapt=adapt,
        )
        st_label, st_conf, st_scores = self.subtype.classify(
            vec, adapt=adapt,
        )

        return IntentResult(
            intent_type=it_label,
            intent_type_confidence=it_conf,
            decision_phase=dp_label,
            decision_phase_confidence=dp_conf,
            subtype=st_label,
            subtype_confidence=st_conf,
            all_scores={
                "intent_type": it_scores,
                "decision_phase": dp_scores,
                "subtype": st_scores,
            },
        )

    def save(self, path: str | Path) -> None:
        """Save trained centroids to .npz file."""
        arrays = {}
        for head_name, head in [
            ("intent_type", self.intent_type),
            ("decision_phase", self.decision_phase),
            ("subtype", self.subtype),
        ]:
            for label, centroid in head.centroids.items():
                arrays[f"{head_name}__{label}"] = centroid
            arrays[f"{head_name}__counts"] = np.array(
                [head.counts[l] for l in head.labels], dtype=np.int32,
            )
        np.savez_compressed(str(path), **arrays)

    def load(self, path: str | Path) -> None:
        """Load trained centroids from .npz file."""
        data = np.load(str(path))
        for head_name, head in [
            ("intent_type", self.intent_type),
            ("decision_phase", self.decision_phase),
            ("subtype", self.subtype),
        ]:
            counts_key = f"{head_name}__counts"
            if counts_key in data:
                counts = data[counts_key]
                for i, label in enumerate(head.labels):
                    if i < len(counts):
                        head.counts[label] = int(counts[i])

            for label in head.labels:
                key = f"{head_name}__{label}"
                if key in data:
                    head.centroids[label] = data[key].astype(np.float32)
