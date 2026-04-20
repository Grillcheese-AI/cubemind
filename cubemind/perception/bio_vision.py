"""Biologically-grounded vision neurons for CubeMind.

Implements three neuroscience-derived visual processing channels:

1. **Opponent-Color Neurons** (V1, ~30% of cells):
   C = w_L * L - w_M * M  (Derrington et al., J Physiology 1984)
   Red-green, blue-yellow opponent channels. Excited by one wavelength,
   inhibited by another.

2. **Motion-Tuned Neurons** (V1/MT):
   r(theta) = r_b + r_max * exp(-(theta - theta_0)^2 / (2*sigma^2))
   Gaussian tuning curves for direction-selective cells. Computed via
   temporal frame differencing.

3. **Luminance Neurons** (Weber-Fechner law):
   r = r_0 + k * log(1 + I/I_0)
   Log-compressed brightness response. Transient response to sudden
   changes (lights-on/off).

Plus developmental maturity:
   r_effective = p_mature * r_raw
   Starts with baby-brain broad tuning (wide sigma, p_mature ~0.2).
   STDP pruning narrows sigma over time → adult sharp tuning.

References:
  - Hodgkin & Huxley (1952) — spike generation
  - Derrington et al. (1984) — opponent color in V1
  - Roy et al. (2021) — EEG color complexity
  - Bonini et al. (2022) — mirror mechanism
"""

from __future__ import annotations

import numpy as np

from cubemind.core.registry import register


class OpponentColorNeurons:
    """V1 opponent-color processing channels.

    Three channels matching biological opponent pairs:
      - L-M (red vs green): long vs medium cone contrast
      - S-(L+M) (blue vs yellow): short vs long+medium
      - L+M (luminance): achromatic brightness

    Each channel produces a signed activation: positive = preferred color,
    negative = opponent color. Magnitude = strength.
    """

    def __init__(self, grid_h: int = 8, grid_w: int = 13) -> None:
        self.grid_h = grid_h
        self.grid_w = grid_w

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Extract opponent-color features from a BGR frame.

        Args:
            frame: (H, W, 3) uint8 BGR image.

        Returns:
            (grid_h * grid_w * 3,) float32 — 3 channels per grid cell:
              [L-M (red-green), S-(L+M) (blue-yellow), L+M (luminance)]
        """
        if frame.ndim == 2:
            # Greyscale — only luminance channel
            small = _resize(frame, self.grid_w, self.grid_h)
            lum = np.log1p(small.astype(np.float32) / 50.0)
            zeros = np.zeros_like(lum)
            return np.stack([zeros, zeros, lum], axis=-1).ravel().astype(np.float32)

        small = _resize(frame, self.grid_w, self.grid_h).astype(np.float32) / 255.0
        b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]

        # Approximate cone responses from RGB
        # L (long/red) ≈ 0.7*R + 0.3*G
        # M (medium/green) ≈ 0.2*R + 0.7*G + 0.1*B
        # S (short/blue) ≈ 0.02*R + 0.1*G + 0.9*B
        L_cone = 0.7 * r + 0.3 * g
        M_cone = 0.2 * r + 0.7 * g + 0.1 * b
        S_cone = 0.02 * r + 0.1 * g + 0.9 * b

        # Opponent channels
        rg = L_cone - M_cone                          # Red vs Green
        by = S_cone - 0.5 * (L_cone + M_cone)         # Blue vs Yellow
        lum = np.log1p((L_cone + M_cone) * 2.0)       # Weber-Fechner luminance

        return np.stack([rg, by, lum], axis=-1).ravel().astype(np.float32)


class MotionNeurons:
    """V1/MT direction-selective motion detection via temporal differencing.

    Computes optical flow magnitude and dominant direction per grid cell.
    Direction tuning follows Gaussian: r(theta) = r_b + r_max * exp(...)

    Maintains previous frame for temporal differencing.
    """

    def __init__(self, grid_h: int = 8, grid_w: int = 13,
                 n_directions: int = 8) -> None:
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_directions = n_directions
        self.prev_frame: np.ndarray | None = None

        # Direction tuning: 8 preferred directions (0, 45, 90, ..., 315)
        self.preferred_dirs = np.linspace(0, 2 * np.pi, n_directions,
                                           endpoint=False)
        self.sigma = np.pi / 4  # Tuning width (starts broad, narrows with maturity)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Extract motion features from current frame.

        Args:
            frame: (H, W, ...) image.

        Returns:
            (grid_h * grid_w * n_directions,) float32 — direction-tuned responses.
        """
        grey = _to_grey(frame)
        small = _resize(grey, self.grid_w, self.grid_h).astype(np.float32)

        if self.prev_frame is None:
            self.prev_frame = small.copy()
            return np.zeros(self.grid_h * self.grid_w * self.n_directions,
                            dtype=np.float32)

        # Temporal difference — proxy for motion energy
        diff = small - self.prev_frame
        self.prev_frame = small.copy()

        # Compute local gradients for direction
        # Horizontal gradient ≈ dx, vertical gradient ≈ dy
        dy = np.zeros_like(diff)
        dx = np.zeros_like(diff)
        dy[1:, :] = diff[1:, :] - diff[:-1, :]
        dx[:, 1:] = diff[:, 1:] - diff[:, :-1]

        # Direction angle per cell
        angle = np.arctan2(dy, dx + 1e-8)  # [-pi, pi]
        magnitude = np.sqrt(dx**2 + dy**2)

        # Apply Gaussian tuning curves for each preferred direction
        responses = np.zeros((self.grid_h, self.grid_w, self.n_directions),
                             dtype=np.float32)
        for i, theta_0 in enumerate(self.preferred_dirs):
            # Circular distance
            d = angle - theta_0
            d = np.arctan2(np.sin(d), np.cos(d))  # Wrap to [-pi, pi]
            tuning = np.exp(-d**2 / (2 * self.sigma**2))
            responses[:, :, i] = magnitude * tuning

        return responses.ravel().astype(np.float32)

    def set_maturity(self, p: float) -> None:
        """Adjust tuning width based on developmental maturity.

        p=0.2 (baby): sigma=pi/2 (very broad, responds to everything)
        p=1.0 (adult): sigma=pi/8 (sharp, direction-selective)
        """
        self.sigma = np.pi / (2 + 6 * p)  # pi/2 at p=0, pi/8 at p=1


class LuminanceNeurons:
    """Weber-Fechner luminance processing with transient detection.

    Steady-state: r = r_0 + k * log(1 + I/I_0)
    Transient: spikes on sudden brightness changes (lights-on/off)
    """

    def __init__(self, grid_h: int = 8, grid_w: int = 13,
                 k: float = 2.0, I_0: float = 50.0) -> None:
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.k = k
        self.I_0 = I_0
        self.prev_luminance: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Extract luminance features with transient detection.

        Returns:
            (grid_h * grid_w * 2,) float32 — [steady_state, transient] per cell.
        """
        grey = _to_grey(frame)
        small = _resize(grey, self.grid_w, self.grid_h).astype(np.float32)

        # Weber-Fechner log compression
        steady = self.k * np.log1p(small / self.I_0)

        # Transient: absolute change from previous frame
        if self.prev_luminance is not None:
            transient = np.abs(steady - self.prev_luminance)
        else:
            transient = np.zeros_like(steady)
        self.prev_luminance = steady.copy()

        return np.stack([steady, transient], axis=-1).ravel().astype(np.float32)


@register("encoder", "bio_vision")
class BioVisionEncoder:
    """Complete biologically-grounded visual feature extractor.

    Combines opponent-color + motion + luminance channels into a single
    feature vector suitable for SNN input.

    Developmental maturity (p_mature) controls:
      - Motion tuning width (sigma narrowing)
      - Feature scaling (r_effective = p_mature * r_raw)
      - Number of "pruned" (zeroed) weak features

    Args:
        grid_h, grid_w: Spatial grid resolution.
        n_directions: Motion direction bins.
        maturity: Initial developmental maturity [0, 1].
    """

    def __init__(self, grid_h: int = 8, grid_w: int = 13,
                 n_directions: int = 8, maturity: float = 0.3) -> None:
        self.color = OpponentColorNeurons(grid_h, grid_w)
        self.motion = MotionNeurons(grid_h, grid_w, n_directions)
        self.luminance = LuminanceNeurons(grid_h, grid_w)
        self.maturity = maturity
        self.motion.set_maturity(maturity)

        # Feature dimensions
        self.n_color = grid_h * grid_w * 3      # RG, BY, Lum per cell
        self.n_motion = grid_h * grid_w * n_directions
        self.n_luminance = grid_h * grid_w * 2  # steady + transient
        self.d_features = self.n_color + self.n_motion + self.n_luminance

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Extract all visual features from a frame.

        Returns:
            (d_features,) float32 feature vector.
        """
        color_feat = self.color.process(frame)
        motion_feat = self.motion.process(frame)
        lum_feat = self.luminance.process(frame)

        raw = np.concatenate([color_feat, motion_feat, lum_feat])

        # Developmental maturity scaling: r_effective = p_mature * r_raw
        # Baby brain: broad, noisy, weak. Adult brain: sharp, strong.
        effective = self.maturity * raw

        return effective.astype(np.float32)

    def grow(self, delta: float = 0.001) -> None:
        """Increase developmental maturity (called after each STDP update).

        Simulates synaptic pruning: tuning sharpens, responses strengthen.
        """
        self.maturity = min(1.0, self.maturity + delta)
        self.motion.set_maturity(self.maturity)

    @property
    def feature_names(self) -> list[str]:
        """Names for each feature dimension (for debugging/plotting)."""
        names = []
        for ch in ["RG", "BY", "Lum"]:
            for r in range(self.color.grid_h):
                for c in range(self.color.grid_w):
                    names.append(f"color_{ch}_{r}_{c}")
        for d in range(self.motion.n_directions):
            deg = int(np.degrees(self.motion.preferred_dirs[d]))
            for r in range(self.motion.grid_h):
                for c in range(self.motion.grid_w):
                    names.append(f"motion_{deg}deg_{r}_{c}")
        for ch in ["steady", "transient"]:
            for r in range(self.luminance.grid_h):
                for c in range(self.luminance.grid_w):
                    names.append(f"lum_{ch}_{r}_{c}")
        return names


# ── Helpers ──────────────────────────────────────────────────────────

def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize image to (h, w) using area averaging (no OpenCV dependency)."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    try:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except ImportError:
        # Numpy fallback: simple block averaging
        src_h, src_w = img.shape[:2]
        bh, bw = max(1, src_h // h), max(1, src_w // w)
        cropped = img[:bh * h, :bw * w]
        if img.ndim == 3:
            return cropped.reshape(h, bh, w, bw, -1).mean(axis=(1, 3)).astype(img.dtype)
        return cropped.reshape(h, bh, w, bw).mean(axis=(1, 3)).astype(img.dtype)


def _to_grey(frame: np.ndarray) -> np.ndarray:
    """Convert to greyscale using proper luminance weights."""
    if frame.ndim == 2:
        return frame
    # BGR → grey: 0.114*B + 0.587*G + 0.299*R
    return (frame[:, :, 0] * 0.114 + frame[:, :, 1] * 0.587
            + frame[:, :, 2] * 0.299).astype(np.uint8)
