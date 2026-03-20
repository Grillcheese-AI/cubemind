"""ResNet-18 → VSA Block-Code via Additive Cross-Entropy.

Uses PyTorch pretrained ResNet-18 as frozen feature extractor (512-dim),
then trains a lightweight projection head with additive CE to map
features into the VSA block-code space.

Architecture:
    Image (1, 80, 80) → repeat to (3, 224, 224)
    → ResNet-18 (frozen, pretrained) → (512,) features
    → Linear(512, d_vsa) projection head (trainable)
    → Additive CE against frozen VSA codebook

Only the projection head is trained. ResNet-18 features are excellent
for geometric shape recognition out of the box.
"""

from __future__ import annotations

import numpy as np
import time

from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.additive_ce import VSACodebook, additive_ce_loss

ATTRS = ["Type", "Size", "Color", "Angle"]

# ── PyTorch ResNet-18 feature extractor ─────────────────────────────────────

_TORCH = False
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _TORCH = True
except ImportError:
    pass


class ResNetVSA:
    """ResNet-18 frozen backbone + trainable VSA projection head.

    Args:
        bc: BlockCodes instance.
        temperature: Additive CE temperature.
        lr: Learning rate for projection head.
        seed: Random seed.
    """

    def __init__(
        self,
        bc: BlockCodes,
        temperature: float = 5.0,
        lr: float = 0.01,
        seed: int = 42,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch required for ResNet-18 backbone")

        self.bc = bc
        self.d_vsa = bc.k * bc.l
        self.temperature = temperature
        self.lr = lr
        self.codebook = VSACodebook(bc, seed=seed)

        # Frozen ResNet-18 feature extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.eval()
        # Remove classification head — use avgpool output (512-dim)
        self.feature_dim = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity()

        # Freeze all ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Trainable projection head: 512 → d_vsa
        torch.manual_seed(seed)
        self.proj = nn.Linear(self.feature_dim, self.d_vsa)
        self.optimizer = torch.optim.Adam(self.proj.parameters(), lr=lr)

    @torch.no_grad()
    def _extract_features(self, pil_image) -> np.ndarray:
        """Extract ResNet-18 features from a PIL image."""
        from PIL import Image
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(
                (np.asarray(pil_image) * 255).astype(np.uint8))

        # Convert grayscale to RGB by repeating channels
        img = pil_image.convert("RGB")
        x = self.transform(img).unsqueeze(0)  # (1, 3, 224, 224)
        features = self.resnet(x)  # (1, 512)
        return features.numpy().ravel()

    def train_step(self, pil_image, entity_attrs: dict) -> tuple[float, float]:
        """One training step: image → features → proj → additive CE.

        Args:
            pil_image: PIL Image of the panel.
            entity_attrs: Dict with Type, Size, Color, Angle.

        Returns:
            (loss, similarity) tuple.
        """
        # Extract frozen features
        features_np = self._extract_features(pil_image)
        features = torch.from_numpy(features_np).unsqueeze(0)  # (1, 512)

        # Forward through projection head
        self.optimizer.zero_grad()
        query = self.proj(features).squeeze(0)  # (d_vsa,)
        query_np = query.detach().numpy()

        # Additive CE loss (numpy)
        target_indices = self.codebook.target_indices(entity_attrs)
        loss, grad_np = additive_ce_loss(
            query_np, self.codebook.W_hat, target_indices, self.temperature,
        )

        # Backward through projection head only
        query.backward(torch.from_numpy(grad_np).float())
        self.optimizer.step()

        # Similarity to bundled target
        bundle_target = self.codebook.bundle_target(entity_attrs)
        q_disc = self.bc.discretize(self.bc.from_flat(query_np, self.bc.k))
        sim = self.bc.similarity(q_disc, bundle_target)

        return loss, sim

    def encode_panel(self, pil_image) -> np.ndarray:
        """Encode a panel image to a block-code.

        Args:
            pil_image: PIL Image.

        Returns:
            Discrete block-code (k, l).
        """
        features_np = self._extract_features(pil_image)
        features = torch.from_numpy(features_np).unsqueeze(0)
        with torch.no_grad():
            query = self.proj(features).squeeze(0).numpy()
        return self.bc.discretize(self.bc.from_flat(query, self.bc.k))

    def decode_attributes(self, block_code: np.ndarray) -> dict[str, int]:
        """Decode predicted attributes from block-code via codebook lookup.

        Args:
            block_code: (k, l) block-code.

        Returns:
            Dict with predicted Type, Size, Color, Angle.
        """
        code_flat = block_code.ravel()
        # Find most similar codebook entry per attribute
        result = {}
        for attr in ATTRS:
            best_val = 0
            best_sim = -1.0
            for val in range(10):
                idx = self.codebook.labels.index((attr, val))
                cb_flat = self.codebook.W[idx]
                sim = float(np.dot(code_flat, cb_flat))
                if sim > best_sim:
                    best_sim = sim
                    best_val = val
            result[attr] = best_val
        return result


def train_resnet_vsa(problems, bc, n_epochs=30, lr=0.01, temperature=5.0,
                      max_n=None):
    """Train ResNet-18 → VSA projection on RAVEN panels."""
    import xml.etree.ElementTree as ET
    from PIL import Image

    model = ResNetVSA(bc, temperature=temperature, lr=lr)

    if max_n:
        problems = problems[:max_n]

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_sim = 0.0
        n_correct = {a: 0 for a in ATTRS}
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
                gt = {
                    "Type": int(e.get("Type", "0")),
                    "Size": int(e.get("Size", "0")),
                    "Color": int(e.get("Color", "0")),
                    "Angle": int(e.get("Angle", "0")),
                }

                loss, sim = model.train_step(panels[pi], gt)
                total_loss += loss
                total_sim += sim

                # Accuracy check
                bc_pred = model.encode_panel(panels[pi])
                pred_attrs = model.decode_attributes(bc_pred)
                for a in ATTRS:
                    if pred_attrs[a] == gt[a]:
                        n_correct[a] += 1

                n_panels += 1

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_panels, 1)
        avg_sim = total_sim / max(n_panels, 1)
        acc = {a: n_correct[a] / max(n_panels, 1) for a in ATTRS}
        avg_acc = sum(acc.values()) / len(ATTRS)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f} "
                  f"sim={avg_sim:.4f} acc={avg_acc:.1%} "
                  f"[T={acc['Type']:.0%} S={acc['Size']:.0%} "
                  f"C={acc['Color']:.0%} A={acc['Angle']:.0%}] "
                  f"({elapsed:.1f}s)")

    return model


if __name__ == "__main__":
    import sys
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import io

    config = sys.argv[1] if len(sys.argv) > 1 else "center_single"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
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
    print(f"\nResNet-18 -> VSA (Additive CE, temp=5, lr=0.01, {epochs} epochs)...")
    model = train_resnet_vsa(problems, bc, n_epochs=epochs, lr=0.01,
                              temperature=5.0, max_n=max_n)
