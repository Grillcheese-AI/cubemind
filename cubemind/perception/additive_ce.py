"""Additive Cross-Entropy Training — bundle-predictive learning.

CNN outputs a raw 512-dim query vector (no softmax). Additive CE measures
cosine similarity against a frozen codebook of atomic attribute block-codes.
Temperature scaling sharpens gradients. Post-hoc algorithmic binding.

The CNN never sees the hash-bound targets. It learns to produce a vector
similar to the BUNDLED superposition of attributes in the image.
"""

from __future__ import annotations

import numpy as np
import time

from cubemind.ops.block_codes import BlockCodes

ATTRS = ["Type", "Size", "Color", "Angle"]
N_VALUES = 10


class VSACodebook:
    """Frozen dictionary of atomic attribute block-codes."""

    def __init__(self, bc: BlockCodes, seed: int = 42) -> None:
        self.bc = bc
        self.d = bc.k * bc.l
        self.codes = {}
        self.flat_codes = []
        self.labels = []

        s = seed
        for attr in ATTRS:
            for val in range(N_VALUES):
                code = bc.random_discrete(seed=s)
                self.codes[(attr, val)] = code
                self.flat_codes.append(code.ravel())
                self.labels.append((attr, val))
                s += 1

        self.W = np.array(self.flat_codes, dtype=np.float32)
        self.n_codes = len(self.flat_codes)

        # Pre-compute normalized codebook
        norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-8
        self.W_hat = (self.W / norms).astype(np.float32)

    def target_indices(self, entity_attrs: dict) -> list[int]:
        indices = []
        for attr in ATTRS:
            val = int(entity_attrs.get(attr, 0))
            val = max(0, min(val, N_VALUES - 1))
            indices.append(self.labels.index((attr, val)))
        return indices

    def bundle_target(self, entity_attrs: dict) -> np.ndarray:
        vecs = []
        for attr in ATTRS:
            val = int(entity_attrs.get(attr, 0))
            val = max(0, min(val, N_VALUES - 1))
            vecs.append(self.codes[(attr, val)])
        return self.bc.bundle(vecs, normalize=True)


def additive_ce_loss(query, W_hat, target_indices, temperature=5.0):
    """Additive CE: cosine sim against codebook, temperature-scaled.

    Args:
        query: Raw CNN output (d,) — NOT softmax'd.
        W_hat: Normalized codebook (n, d).
        target_indices: Indices of target attribute codes.
        temperature: Inverse temperature.

    Returns:
        (loss, grad_query)
    """
    q_norm = np.linalg.norm(query) + 1e-8
    q_hat = query / q_norm

    # Cosine similarities
    sims = W_hat @ q_hat  # (n,)
    sims = np.clip(sims, -0.999, 0.999)

    # Scaled logits
    logits = temperature * sims

    # Stable softmax
    logits_shifted = logits - logits.max()
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / (exp_logits.sum() + 1e-8)

    # Loss: average negative log-prob of target indices
    target_probs = [max(probs[idx], 1e-8) for idx in target_indices]
    loss = -np.mean([np.log(p) for p in target_probs])

    # Gradient w.r.t. query
    # dL/dq = (temperature / ||q||) * (E[w] - E_target[w])
    expected_w = probs @ W_hat  # (d,)
    target_w = np.mean([W_hat[idx] for idx in target_indices], axis=0)

    grad = temperature * (expected_w - target_w) / q_norm
    grad = np.clip(grad, -1.0, 1.0).astype(np.float32)

    return float(loss), grad


def train(problems, bc, n_epochs=30, lr=0.01, temperature=5.0, max_n=None):
    """Train with Additive CE using grilly Conv2d (GEMM backward)."""
    import xml.etree.ElementTree as ET
    from PIL import Image
    from grilly.nn.conv import Conv2d
    from grilly.nn.pooling import MaxPool2d
    from grilly.nn.linear import Linear

    if max_n:
        problems = problems[:max_n]

    codebook = VSACodebook(bc)

    # Build model: 3-layer conv + linear projection to d_vsa
    convs = [
        Conv2d(1, 32, kernel_size=3, padding=1),
        Conv2d(32, 64, kernel_size=3, padding=1),
        Conv2d(64, 128, kernel_size=3, padding=1),
    ]
    pools = [MaxPool2d(2, 2), MaxPool2d(2, 2), MaxPool2d(2, 2)]
    proj = Linear(128, bc.k * bc.l)

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_sim = 0.0
        n_panels = 0
        t0 = time.perf_counter()

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

                e = entities[0]
                entity_attrs = {
                    "Type": int(e.get("Type", "0")),
                    "Size": int(e.get("Size", "0")),
                    "Color": int(e.get("Color", "0")),
                    "Angle": int(e.get("Angle", "0")),
                }
                target_indices = codebook.target_indices(entity_attrs)

                # Prepare image
                img = panels[pi]
                if isinstance(img, Image.Image):
                    img = np.array(img.convert("L").resize((80, 80)),
                                   dtype=np.float32) / 255.0
                x = img.reshape(1, 1, 80, 80).astype(np.float32)

                # ── Forward ──
                # Zero grads
                for c in convs:
                    c.zero_grad()
                proj.zero_grad()

                # Conv stack: conv -> ReLU -> pool
                activations = [x]
                h = x
                for c, p in zip(convs, pools):
                    h = c.forward(h)
                    h = np.maximum(h, 0)  # ReLU
                    activations.append(h)
                    h = p.forward(h)

                # Global avg pool -> flatten
                feat = h.mean(axis=(2, 3)).reshape(1, -1)  # (1, 128)

                # Linear projection to d_vsa (raw logits, NO softmax)
                query = proj.forward(feat).ravel()  # (512,)

                # ── Loss ──
                loss, grad_query = additive_ce_loss(
                    query, codebook.W_hat, target_indices, temperature,
                )
                total_loss += loss

                # Similarity to bundled target
                bundle_target = codebook.bundle_target(entity_attrs)
                q_disc = bc.discretize(bc.from_flat(query, bc.k))
                sim = bc.similarity(q_disc, bundle_target)
                total_sim += sim

                # ── Backward ──
                # Through linear projection
                g_feat = proj.backward(
                    grad_query.reshape(1, -1), x=feat
                )  # (1, 128)

                # Through global avg pool
                _, c_dim, h_dim, w_dim = h.shape
                g_pool = (g_feat.reshape(1, c_dim, 1, 1)
                          / (h_dim * w_dim)).astype(np.float32)
                g_pool = np.broadcast_to(g_pool, h.shape).copy()

                # Through conv stack (reverse)
                g = g_pool
                for i in range(len(convs) - 1, -1, -1):
                    g = pools[i].backward(g)
                    # ReLU backward
                    g = g * (activations[i + 1] > 0).astype(np.float32)
                    g = convs[i].backward(g)

                # ── SGD step (in-place) ──
                for c in convs:
                    w = np.asarray(c.weight)
                    wg = getattr(c, '_grad_w', None)
                    if wg is None:
                        wg = getattr(c.weight, 'grad', None)
                    if wg is not None:
                        w -= lr * np.asarray(wg)
                    b = np.asarray(c.bias) if c.bias is not None else None
                    bg = getattr(c, '_grad_b', None)
                    if bg is None and c.bias is not None:
                        bg = getattr(c.bias, 'grad', None)
                    if b is not None and bg is not None:
                        b -= lr * np.asarray(bg)

                pw = np.asarray(proj.weight)
                pwg = getattr(proj, '_grad_w', None)
                if pwg is None:
                    pwg = getattr(proj.weight, 'grad', None)
                if pwg is not None:
                    pw -= lr * np.asarray(pwg)
                pb = np.asarray(proj.bias) if proj.bias is not None else None
                pbg = getattr(proj, '_grad_b', None)
                if pbg is None and proj.bias is not None:
                    pbg = getattr(proj.bias, 'grad', None)
                if pb is not None and pbg is not None:
                    pb -= lr * np.asarray(pbg)

                n_panels += 1

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_panels, 1)
        avg_sim = total_sim / max(n_panels, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f} "
                  f"sim={avg_sim:.4f} ({elapsed:.1f}s)")

    return convs, proj, codebook


if __name__ == "__main__":
    import sys
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import io

    config = sys.argv[1] if len(sys.argv) > 1 else "center_single"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
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
        })
    print(f"Loaded {len(problems)} problems")

    bc = BlockCodes(8, 64)
    print(f"\nAdditive CE training (temp=5, lr=0.01, {epochs} epochs)...")
    convs, proj, codebook = train(problems, bc, n_epochs=epochs, lr=0.01,
                                   temperature=5.0, max_n=max_n)
