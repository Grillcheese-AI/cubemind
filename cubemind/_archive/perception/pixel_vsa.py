"""Pixel-VSA Encoder — zero-training image to block-code via sign+bind.

Encodes images directly to VSA block-codes without any CNN or training:
1. Threshold pixels to bipolar {0, 1} per block position
2. Each pixel (x,y) has a pre-computed position block-code
3. Bind pixel intensity code to position code
4. Bundle all pixel bindings → panel block-code

This is a deterministic, algebraic perception frontend that maps
raw pixel data into the same VSA space as the NVSA role-filler
encoding, enabling direct comparison and rule detection.

No training required. No gradients. No NaN.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes


class PixelVSAEncoder:
    """Encodes images to block-codes via pixel-level VSA binding.

    Each spatial position gets a unique position code. Pixel intensities
    are quantized and bound to positions. The full image is the bundled
    superposition of all pixel-position bindings.

    Args:
        k: Number of blocks.
        l: Block length.
        image_size: Target image size (square).
        n_intensity_levels: Quantization levels for pixel intensity.
        downsample: Spatial downsampling factor (e.g. 4 = 80->20).
        seed: Random seed for position codes.
    """

    def __init__(
        self,
        k: int = 8,
        l: int = 64,
        image_size: int = 80,
        n_intensity_levels: int = 8,
        downsample: int = 4,
        seed: int = 42,
    ) -> None:
        self.bc = BlockCodes(k, l)
        self.k = k
        self.l = l
        self.image_size = image_size
        self.n_levels = n_intensity_levels
        self.downsample = downsample

        ds = image_size // downsample
        self.ds_size = ds

        rng_seed = seed

        # Pre-compute position codes: one per (x, y) in downsampled grid
        self.pos_codes = np.zeros((ds, ds, k, l), dtype=np.float32)
        for y in range(ds):
            for x in range(ds):
                self.pos_codes[y, x] = self.bc.random_discrete(seed=rng_seed)
                rng_seed += 1

        # Pre-compute intensity codes: one per quantized level
        self.intensity_codes = np.zeros((n_intensity_levels, k, l), dtype=np.float32)
        for lev in range(n_intensity_levels):
            self.intensity_codes[lev] = self.bc.random_discrete(seed=rng_seed)
            rng_seed += 1

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode a grayscale image to a block-code.

        Args:
            image: (H, W) float32 [0, 1] or (H, W) uint8.

        Returns:
            Block-code (k, l).
        """
        img = np.asarray(image, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # Resize to target
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            from PIL import Image as PILImage
            pil = PILImage.fromarray((img * 255).astype(np.uint8), mode='L')
            pil = pil.resize((self.image_size, self.image_size))
            img = np.array(pil, dtype=np.float32) / 255.0

        # Downsample via block averaging
        ds = self.ds_size
        block_h = self.image_size // ds
        block_w = self.image_size // ds
        small = np.zeros((ds, ds), dtype=np.float32)
        for y in range(ds):
            for x in range(ds):
                small[y, x] = img[y*block_h:(y+1)*block_h,
                                  x*block_w:(x+1)*block_w].mean()

        # Quantize intensities
        quantized = np.clip((small * self.n_levels).astype(int), 0, self.n_levels - 1)

        # Bind each pixel's intensity code to its position code, then bundle
        bindings = []
        for y in range(ds):
            for x in range(ds):
                lev = quantized[y, x]
                bound = self.bc.bind(self.pos_codes[y, x], self.intensity_codes[lev])
                bindings.append(bound)

        # Bundle all bindings
        bundled = self.bc.bundle(bindings, normalize=True)
        return self.bc.discretize(bundled)

    def encode_panel(self, panel_image, target_size: int = 80) -> np.ndarray:
        """Encode a PIL Image to block-code."""
        try:
            from PIL import Image
            if isinstance(panel_image, Image.Image):
                img = panel_image.convert("L").resize((target_size, target_size))
                pixels = np.array(img, dtype=np.float32) / 255.0
            else:
                pixels = np.asarray(panel_image, dtype=np.float32)
                if pixels.max() > 1.0:
                    pixels /= 255.0
        except ImportError:
            pixels = np.asarray(panel_image, dtype=np.float32)
            if pixels.max() > 1.0:
                pixels /= 255.0
        return self.encode(pixels)

    def encode_panels_batch(self, panels: list) -> list[np.ndarray]:
        """Encode a batch of panel images."""
        return [self.encode_panel(p) for p in panels]

    def similarity_matrix(self, panels: list) -> np.ndarray:
        """Compute pairwise similarity between panel encodings.

        Useful for detecting which panels are similar (constant rule)
        or which differ systematically (progression/arithmetic).

        Args:
            panels: List of panel images.

        Returns:
            (n, n) similarity matrix.
        """
        codes = self.encode_panels_batch(panels)
        n = len(codes)
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                sim[i, j] = self.bc.similarity(codes[i], codes[j])
        return sim


if __name__ == "__main__":
    import sys
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import io
    import time

    config = sys.argv[1] if len(sys.argv) > 1 else "center_single"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

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
            "choices": [],
            "metadata": row.get("metadata", ""),
            "target": row.get("target"),
        })
    print(f"Loaded {len(problems)} problems")

    # Encode and test
    enc = PixelVSAEncoder(k=8, l=64, downsample=4, n_intensity_levels=8)
    print(f"\nPixelVSA: ds={enc.ds_size}x{enc.ds_size}, {enc.n_levels} intensity levels")

    t0 = time.perf_counter()
    correct = 0
    for pi, prob in enumerate(problems):
        if len(prob["panels"]) < 8:
            continue

        # Encode context panels
        ctx_codes = [enc.encode_panel(p) for p in prob["panels"][:8]]

        # Load choice panels from parquet
        row = {col: table.column(col)[pi].as_py() for col in table.column_names}
        choice_panels = []
        if "choices" in row and row["choices"]:
            for p in row["choices"]:
                if isinstance(p, dict) and "bytes" in p:
                    choice_panels.append(Image.open(io.BytesIO(p["bytes"])))

        if len(choice_panels) < 8:
            continue

        choice_codes = [enc.encode_panel(c) for c in choice_panels]

        # Simple prediction: find choice most similar to predicted pattern
        # Row completion: choice should make row2 similar to row0/row1
        row0 = enc.bc.bundle([ctx_codes[0], ctx_codes[1], ctx_codes[2]])
        row1 = enc.bc.bundle([ctx_codes[3], ctx_codes[4], ctx_codes[5]])
        master = enc.bc.bundle([row0, row1])

        best_idx = 0
        best_sim = -1.0
        for ci, cc in enumerate(choice_codes):
            row2 = enc.bc.bundle([ctx_codes[6], ctx_codes[7], cc])
            sim = enc.bc.similarity(enc.bc.discretize(master), enc.bc.discretize(row2))
            if sim > best_sim:
                best_sim = sim
                best_idx = ci

        target = prob.get("target")
        if best_idx == target:
            correct += 1

        if pi < 3:
            print(f"  Problem {pi}: pred={best_idx}, target={target}, sim={best_sim:.4f}")

    elapsed = time.perf_counter() - t0
    acc = correct / max(len(problems), 1)
    print(f"\nAccuracy: {acc:.1%} ({correct}/{len(problems)})")
    print(f"Latency: {elapsed/len(problems)*1000:.1f}ms/problem")
