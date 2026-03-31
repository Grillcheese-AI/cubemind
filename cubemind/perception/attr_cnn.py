"""Attribute-Head CNN — predicts discrete attributes, encodes to block-codes.

Instead of learning to produce block-codes directly (hard — targets are
hash-like one-hot patterns from role-filler binding), this CNN predicts
discrete attributes (Type, Size, Color, Angle) as separate classification
heads, then uses the deterministic NVSA encoder to produce block-codes.

Architecture:
    Image (1, 80, 80)
    -> Conv(1,32,3)+ReLU+MaxPool(2)    -> (32, 40, 40)
    -> Conv(32,64,3)+ReLU+MaxPool(2)   -> (64, 20, 20)
    -> Conv(64,128,3)+ReLU+MaxPool(2)  -> (128, 10, 10)
    -> GlobalAvgPool                    -> (128,)
    -> 4 classification heads:
       Type:  Linear(128, 10) -> softmax
       Size:  Linear(128, 10) -> softmax
       Color: Linear(128, 10) -> softmax
       Angle: Linear(128, 10) -> softmax

Training: cross-entropy per attribute head against XML metadata labels.
Inference: argmax each head -> encode_entity_nvsa -> block-code.

This decouples visual recognition from VSA encoding, following the
VSA4VQA insight (Penzkofer et al., 2024).
"""

from __future__ import annotations

import numpy as np

N_ATTR_VALUES = 10  # RAVEN uses 0-9 for each attribute
ATTRS = ["Type", "Size", "Color", "Angle"]


# ── Simple numpy CNN ────────────────────────────────────────────────────────


def _relu(x):
    return np.maximum(x, 0).astype(np.float32)


def _softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def _conv2d(x, w, b, padding=1):
    """Simple conv2d, NCHW."""
    n, ci, hi, wi = x.shape
    co, _, kh, kw = w.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, hp, wp = x.shape
    ho, wo = hp - kh + 1, wp - kw + 1
    out = np.zeros((n, co, ho, wo), dtype=np.float32)
    for oc in range(co):
        for khi in range(kh):
            for kwi in range(kw):
                out[:, oc] += (
                    x[:, :, khi:khi + ho, kwi:kwi + wo]
                    * w[oc, :, khi, kwi].reshape(1, ci, 1, 1)
                ).sum(axis=1)
        out[:, oc] += b[oc]
    return out


def _maxpool2d(x, k=2, s=2):
    n, c, h, w = x.shape
    oh, ow = h // s, w // s
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i*s:i*s+k, j*s:j*s+k].max(axis=(2, 3))
    return out


def _he_init(shape, fan_in, rng):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


class AttrCNN:
    """CNN with per-attribute classification heads.

    Args:
        n_classes: Number of classes per attribute (default 10).
        channels: Conv channel widths.
        seed: Random seed.
    """

    def __init__(
        self,
        n_classes: int = N_ATTR_VALUES,
        channels: tuple[int, ...] = (32, 64, 128),
        seed: int = 42,
    ) -> None:
        self.n_classes = n_classes
        self.channels = channels
        rng = np.random.default_rng(seed)

        # Conv layers
        ch_in = [1] + list(channels)
        self.conv_w = []
        self.conv_b = []
        for i in range(len(channels)):
            self.conv_w.append(_he_init((channels[i], ch_in[i], 3, 3), ch_in[i] * 9, rng))
            self.conv_b.append(np.zeros(channels[i], dtype=np.float32))

        # Per-attribute classification heads
        feat_dim = channels[-1]
        self.heads_w = {}
        self.heads_b = {}
        for attr in ATTRS:
            self.heads_w[attr] = _he_init((n_classes, feat_dim), feat_dim, rng)
            self.heads_b[attr] = np.zeros(n_classes, dtype=np.float32)

        # Gradient accumulators
        self.conv_w_grad = [np.zeros_like(w) for w in self.conv_w]
        self.conv_b_grad = [np.zeros_like(b) for b in self.conv_b]
        self.heads_w_grad = {a: np.zeros_like(self.heads_w[a]) for a in ATTRS}
        self.heads_b_grad = {a: np.zeros_like(self.heads_b[a]) for a in ATTRS}

        # Cache
        self._features = None

    def forward(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Forward: image -> per-attribute class probabilities.

        Args:
            image: (80, 80) float32 [0, 1].

        Returns:
            Dict mapping attr name to probability vector (n_classes,).
        """
        if image.ndim == 2:
            x = image[np.newaxis, np.newaxis, :, :].astype(np.float32)
        else:
            x = image.reshape(1, 1, image.shape[-2], image.shape[-1]).astype(np.float32)

        # Conv backbone
        self._conv_cache = []
        for i in range(len(self.channels)):
            self._conv_cache.append(x)  # pre-conv input
            x = _relu(_conv2d(x, self.conv_w[i], self.conv_b[i]))
            x = _maxpool2d(x)

        # Global average pool -> feature vector
        features = x.mean(axis=(2, 3)).ravel()  # (feat_dim,)
        self._features = features

        # Per-attribute heads
        probs = {}
        self._logits = {}
        for attr in ATTRS:
            logits = self.heads_w[attr] @ features + self.heads_b[attr]
            self._logits[attr] = logits
            probs[attr] = _softmax(logits)

        return probs

    def predict(self, image: np.ndarray) -> dict[str, int]:
        """Predict discrete attribute values.

        Args:
            image: (80, 80) float32 [0, 1].

        Returns:
            Dict mapping attr name to predicted class index.
        """
        probs = self.forward(image)
        return {attr: int(np.argmax(p)) for attr, p in probs.items()}

    def loss_and_grad(
        self,
        probs: dict[str, np.ndarray],
        targets: dict[str, int],
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Cross-entropy loss and gradients for all attribute heads.

        Args:
            probs: Dict of probability vectors from forward().
            targets: Dict of target class indices.

        Returns:
            (total_loss, grad_dict) where grad_dict maps attr to gradient.
        """
        total_loss = 0.0
        grads = {}

        for attr in ATTRS:
            p = probs[attr]
            t = targets.get(attr, 0)
            # Cross-entropy: -log(p[t])
            total_loss -= np.log(max(p[t], 1e-8))
            # Gradient of CE w.r.t. logits: p - one_hot(t)
            g = p.copy()
            g[t] -= 1.0
            grads[attr] = g

        return total_loss / len(ATTRS), grads

    def backward(self, grads: dict[str, np.ndarray]) -> None:
        """Backward through heads + conv stack.

        Args:
            grads: Dict of gradients from loss_and_grad().
        """
        features = self._features
        if features is None:
            return

        # Head gradients
        g_features = np.zeros_like(features)
        for attr in ATTRS:
            g = grads[attr] / len(ATTRS)  # average across heads
            # dL/dW = outer(g, features), dL/db = g
            self.heads_w_grad[attr] += np.outer(g, features)
            self.heads_b_grad[attr] += g
            # dL/d(features) += W^T @ g
            g_features += self.heads_w[attr].T @ g

        # Backward through conv stack (simplified — update heads only for now,
        # conv backward needs im2col which is slow in numpy)
        # TODO: wire grilly conv backward when running on GPU

    def zero_grad(self):
        for g in self.conv_w_grad:
            g.fill(0)
        for g in self.conv_b_grad:
            g.fill(0)
        for attr in ATTRS:
            self.heads_w_grad[attr].fill(0)
            self.heads_b_grad[attr].fill(0)

    def step(self, lr: float = 0.01):
        """SGD update."""
        for attr in ATTRS:
            self.heads_w[attr] -= lr * self.heads_w_grad[attr]
            self.heads_b[attr] -= lr * self.heads_b_grad[attr]
        # Conv weights updated when backward wired
        for i in range(len(self.conv_w)):
            if np.any(self.conv_w_grad[i] != 0):
                self.conv_w[i] -= lr * self.conv_w_grad[i]
                self.conv_b[i] -= lr * self.conv_b_grad[i]


# ── Training + Evaluation ──────────────────────────────────────────────────


def train_attr_cnn(
    problems: list[dict],
    n_epochs: int = 30,
    lr: float = 0.01,
    max_problems: int | None = None,
) -> AttrCNN:
    """Train attribute CNN on RAVEN problems.

    Args:
        problems: RAVEN problem dicts with panels and metadata.
        n_epochs: Training epochs.
        lr: Learning rate.
        max_problems: Max problems to use.

    Returns:
        Trained AttrCNN.
    """
    import xml.etree.ElementTree as ET
    from PIL import Image

    model = AttrCNN()

    if max_problems:
        problems = problems[:max_problems]

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_correct = {a: 0 for a in ATTRS}
        n_total = 0

        for prob in problems:
            panels = prob.get("panels", [])
            metadata = prob.get("metadata", "")
            if not panels or not metadata:
                continue

            # Parse targets from XML
            try:
                root = ET.fromstring(metadata)
            except ET.ParseError:
                continue

            xml_panels = root.findall(".//Panel")

            for pi in range(min(len(panels), 8)):
                if pi >= len(xml_panels):
                    break

                # Get target attributes from first entity
                entities = xml_panels[pi].findall(".//Entity")
                if not entities:
                    continue
                e = entities[0]
                targets = {
                    "Type": int(e.get("Type", "0")),
                    "Size": int(e.get("Size", "0")),
                    "Color": int(e.get("Color", "0")),
                    "Angle": int(e.get("Angle", "0")),
                }
                # Clamp to valid range
                for a in ATTRS:
                    targets[a] = max(0, min(targets[a], N_ATTR_VALUES - 1))

                # Prepare image
                img = panels[pi]
                if isinstance(img, Image.Image):
                    img = img.convert("L").resize((80, 80))
                    img = np.array(img, dtype=np.float32) / 255.0
                else:
                    img = np.asarray(img, dtype=np.float32)
                    if img.max() > 1.0:
                        img /= 255.0

                # Forward
                model.zero_grad()
                probs = model.forward(img)

                # Loss
                loss, grads = model.loss_and_grad(probs, targets)
                total_loss += loss

                # Accuracy
                for a in ATTRS:
                    if int(np.argmax(probs[a])) == targets[a]:
                        n_correct[a] += 1
                n_total += 1

                # Backward + step
                model.backward(grads)
                model.step(lr=lr)

        avg_loss = total_loss / max(n_total, 1)
        acc = {a: n_correct[a] / max(n_total, 1) for a in ATTRS}
        avg_acc = sum(acc.values()) / len(ATTRS)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f} "
                  f"acc={avg_acc:.1%} "
                  f"[T={acc['Type']:.1%} S={acc['Size']:.1%} "
                  f"C={acc['Color']:.1%} A={acc['Angle']:.1%}]")

    return model


if __name__ == "__main__":
    import sys

    # Load data via parquet
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

    print(f"\nTraining AttrCNN ({epochs} epochs, lr=0.01)...")
    model = train_attr_cnn(problems, n_epochs=epochs, lr=0.01)

    # Quick eval
    print("\nSample predictions:")
    for i in range(3):
        img = problems[i]["panels"][0]
        if isinstance(img, Image.Image):
            img = np.array(img.convert("L").resize((80, 80)), dtype=np.float32) / 255.0
        preds = model.predict(img)
        print(f"  Problem {i}: {preds}")
