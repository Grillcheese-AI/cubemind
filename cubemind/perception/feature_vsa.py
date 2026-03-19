"""Feature-VSA Encoder — classical CV features → block-codes.

Extracts interpretable visual features that map directly to RAVEN
attributes, then encodes each as a VSA block-code via role-filler binding:

- Edge map (Sobel) → Shape/Type information
- Mean intensity → Color/Fill pattern
- Object area (thresholded pixel count) → Size
- Gradient orientation histogram → Angle

No training. No CNN. No NaN. Deterministic and fast.
Each feature is quantized, looked up in a frozen codebook, and
role-filler bound — same algebra as the NVSA metadata encoder.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes

N_LEVELS = 10  # Match RAVEN's 0-9 attribute range


def _sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel edge detection (3x3)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    from scipy.ndimage import convolve
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    return gx, gy


def _sobel_numpy(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel without scipy."""
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx[y, x] = (-img[y-1, x-1] + img[y-1, x+1]
                         - 2*img[y, x-1] + 2*img[y, x+1]
                         - img[y+1, x-1] + img[y+1, x+1])
            gy[y, x] = (-img[y-1, x-1] - 2*img[y-1, x] - img[y-1, x+1]
                         + img[y+1, x-1] + 2*img[y+1, x] + img[y+1, x+1])
    return gx, gy


def extract_features(img: np.ndarray) -> dict[str, int]:
    """Extract quantized visual features from a grayscale panel.

    Args:
        img: (H, W) float32 [0, 1].

    Returns:
        Dict with Type, Size, Color, Angle as integers 0-9.
    """
    h, w = img.shape

    # ── Color: mean intensity of the object region ──
    # Threshold to find object pixels (RAVEN: shapes are darker than background)
    threshold = 0.5
    obj_mask = img < threshold
    if obj_mask.sum() < 10:
        # Try inverted (light shapes on dark background)
        obj_mask = img > threshold

    if obj_mask.sum() > 0:
        mean_intensity = img[obj_mask].mean()
    else:
        mean_intensity = img.mean()
    color = int(np.clip(mean_intensity * N_LEVELS, 0, N_LEVELS - 1))

    # ── Size: fraction of object pixels ──
    obj_fraction = obj_mask.sum() / (h * w)
    size = int(np.clip(obj_fraction * N_LEVELS * 2, 0, N_LEVELS - 1))

    # ── Type (Shape): edge density pattern ──
    try:
        gx, gy = _sobel(img)
    except ImportError:
        gx, gy = _sobel_numpy(img)

    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_density = (edge_mag > edge_mag.mean()).sum() / (h * w)

    # Shape encoding via edge compactness:
    # circles have smooth edges (low variance), polygons have sharp corners
    if obj_mask.sum() > 0:
        edge_in_obj = edge_mag[obj_mask]
        edge_variance = edge_in_obj.var() if len(edge_in_obj) > 1 else 0
        # Combine density + variance for shape discrimination
        shape_score = edge_density * 0.5 + min(edge_variance * 10, 0.5)
    else:
        shape_score = edge_density

    shape_type = int(np.clip(shape_score * N_LEVELS, 0, N_LEVELS - 1))

    # ── Angle: dominant gradient orientation ──
    if obj_mask.sum() > 0:
        gx_obj = gx[obj_mask]
        gy_obj = gy[obj_mask]
        # Weighted average angle
        angles = np.arctan2(gy_obj, gx_obj + 1e-8)
        # Quantize to 10 bins (0 to pi, since shapes are symmetric)
        angle_bins = np.clip(((angles + np.pi) / (2 * np.pi) * N_LEVELS).astype(int),
                             0, N_LEVELS - 1)
        # Mode angle
        counts = np.bincount(angle_bins, minlength=N_LEVELS)
        angle = int(np.argmax(counts))
    else:
        angle = 0

    return {"Type": shape_type, "Size": size, "Color": color, "Angle": angle}


class FeatureVSAEncoder:
    """Encodes images to block-codes via classical CV feature extraction.

    Extracts Type/Size/Color/Angle features from pixels, then uses
    the same NVSA role-filler encoding as the metadata pipeline.
    Drop-in replacement for XML metadata encoding.

    Args:
        k: Number of blocks.
        l: Block length.
        seed: Random seed for codebook.
    """

    def __init__(self, k: int = 8, l: int = 64, seed: int = 42) -> None:
        self.bc = BlockCodes(k, l)
        self.k = k
        self.l = l

        # Build the same role + codebook vectors as benchmarks/iraven.py
        self.roles = {}
        self.codebooks = {}
        attrs = ["Type", "Size", "Color", "Angle"]

        for attr in attrs:
            self.roles[attr] = self.bc.random_discrete(
                seed=self._stable_seed(f"role_{attr}"))
            self.codebooks[attr] = self.bc.codebook_discrete(
                N_LEVELS, seed=self._stable_seed(f"cb_{attr}"))

    @staticmethod
    def _stable_seed(name: str) -> int:
        h = 0
        for ch in name:
            h = (h * 31 + ord(ch)) & 0x7FFFFFFF
        return h

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to block-code via feature extraction + NVSA binding.

        Args:
            image: (H, W) float32 [0, 1].

        Returns:
            Discrete block-code (k, l).
        """
        features = extract_features(image)
        return self.encode_attrs(features)

    def encode_attrs(self, attrs: dict[str, int]) -> np.ndarray:
        """Encode attribute dict to block-code (same as NVSA metadata encoder)."""
        parts = []
        for attr in ["Type", "Size", "Color", "Angle"]:
            val = max(0, min(int(attrs.get(attr, 0)), N_LEVELS - 1))
            role = self.roles[attr]
            filler = self.codebooks[attr][val]
            parts.append(self.bc.bind(role, filler))

        result = parts[0]
        for p in parts[1:]:
            result = self.bc.bind(result, p)
        return self.bc.discretize(result)

    def encode_panel(self, panel_image, target_size: int = 80) -> np.ndarray:
        """Encode a PIL Image to block-code."""
        from PIL import Image
        if isinstance(panel_image, Image.Image):
            img = panel_image.convert("L").resize((target_size, target_size))
            pixels = np.array(img, dtype=np.float32) / 255.0
        else:
            pixels = np.asarray(panel_image, dtype=np.float32)
            if pixels.max() > 1.0:
                pixels /= 255.0
        return self.encode(pixels)


if __name__ == "__main__":
    import sys
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import io
    import xml.etree.ElementTree as ET
    import time

    config = sys.argv[1] if len(sys.argv) > 1 else "center_single"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print(f"Loading {config}/test (max {max_n})...")
    path = hf_hub_download(
        repo_id="HuggingFaceM4/RAVEN", repo_type="dataset",
        filename=f"{config}/test-00000-of-00001.parquet",
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

    enc = FeatureVSAEncoder(k=8, l=64)

    # Compare extracted features vs XML ground truth
    print("\nFeature extraction accuracy (vs XML metadata):")
    attr_correct = {"Type": 0, "Size": 0, "Color": 0, "Angle": 0}
    n_total = 0

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

            img = panels[pi]
            if isinstance(img, Image.Image):
                img = np.array(img.convert("L").resize((80, 80)),
                               dtype=np.float32) / 255.0

            pred = extract_features(img)

            for attr in attr_correct:
                if pred[attr] == gt[attr]:
                    attr_correct[attr] += 1
            n_total += 1

    elapsed = time.perf_counter() - t0
    print(f"  Panels evaluated: {n_total}")
    for attr in attr_correct:
        acc = attr_correct[attr] / max(n_total, 1)
        print(f"  {attr}: {acc:.1%}")
    avg = sum(attr_correct.values()) / (len(attr_correct) * max(n_total, 1))
    print(f"  Average: {avg:.1%}")
    print(f"  Latency: {elapsed/max(n_total,1)*1000:.1f}ms/panel")
