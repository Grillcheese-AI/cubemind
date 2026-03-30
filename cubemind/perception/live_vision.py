"""Live Vision — webcam capture → SNN temporal encoding → VSA memory.

Captures frames from a webcam (or video file), processes them through
the visual perception pipeline, and produces temporal VSA memory vectors.

Pipeline:
  webcam → frame → resize/normalize → VisionEncoder or CNN features
  → SNN temporal binding (cyclic shift + XOR) → packed binary VSA

Scene understanding: accumulate frames into a temporal memory, then
retrieve nearest concept from ContinuousItemMemory.

Dependencies: opencv-python (optional, for webcam capture)
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

# Optional OpenCV
_CV2 = None
try:
    import cv2 as _cv2_module
    _CV2 = _cv2_module
except ImportError:
    pass

from cubemind.perception.snn import SNNEncoder


class FramePreprocessor:
    """Normalize and resize frames for the perception pipeline.

    Args:
        target_size: (height, width) output size.
        normalize:   If True, scale pixels to [0, 1].
        grayscale:   If True, convert to single channel.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (80, 80),
        normalize: bool = True,
        grayscale: bool = True,
    ) -> None:
        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a raw frame.

        Args:
            frame: (H, W, C) uint8 BGR frame from OpenCV, or (H, W) grayscale.

        Returns:
            (target_h, target_w) or (target_h, target_w, C) float32 array.
        """
        img = frame.copy()

        # Convert BGR → grayscale if requested
        if self.grayscale and img.ndim == 3 and img.shape[2] == 3:
            if _CV2 is not None:
                img = _CV2.cvtColor(img, _CV2.COLOR_BGR2GRAY)
            else:
                img = np.mean(img, axis=2).astype(np.uint8)

        # Resize
        h, w = self.target_size
        if _CV2 is not None and (img.shape[0] != h or img.shape[1] != w):
            img = _CV2.resize(img, (w, h), interpolation=_CV2.INTER_AREA)
        elif img.shape[0] != h or img.shape[1] != w:
            # Numpy fallback: nearest-neighbor resize
            src_h, src_w = img.shape[:2]
            row_idx = (np.arange(h) * src_h / h).astype(int)
            col_idx = (np.arange(w) * src_w / w).astype(int)
            img = img[np.ix_(row_idx, col_idx)]

        # Normalize to [0, 1]
        if self.normalize:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

        return img.astype(np.float32)


class WebcamCapture:
    """Webcam frame capture via OpenCV.

    Args:
        device:     Camera device index (0 = default webcam).
        fps_limit:  Max frames per second to capture.
    """

    def __init__(self, device: int = 0, fps_limit: float = 30.0) -> None:
        if _CV2 is None:
            raise ImportError("opencv-python required: pip install opencv-python")
        self.device = device
        self.fps_limit = fps_limit
        self._cap = None
        self._min_interval = 1.0 / fps_limit
        self._last_time = 0.0

    def open(self) -> bool:
        """Open the webcam. Returns True if successful."""
        self._cap = _CV2.VideoCapture(self.device)
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        """Read a single frame, respecting FPS limit.

        Returns:
            (H, W, 3) uint8 BGR frame, or None if unavailable.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        now = time.monotonic()
        if now - self._last_time < self._min_interval:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._last_time = now
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class LiveVisionEncoder:
    """Live webcam → SNN temporal VSA encoding.

    Captures frames, preprocesses them, extracts features via a
    configurable feature extractor, and feeds into the SNN encoder
    for temporal binding.

    Args:
        feature_dim:  Dimension of per-frame feature vector.
        snn_neurons:  Number of SNN neurons.
        d_vsa:        Binary VSA dimension.
        frame_size:   (H, W) frame size for preprocessing.
        buffer_size:  Number of recent frames to keep for scene analysis.
        device:       Webcam device index.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        snn_neurons: int = 512,
        d_vsa: int = 10240,
        frame_size: tuple[int, int] = (80, 80),
        buffer_size: int = 30,
        device: int = 0,
    ) -> None:
        self.feature_dim = feature_dim
        self.frame_size = frame_size

        self.preprocessor = FramePreprocessor(
            target_size=frame_size, normalize=True, grayscale=True,
        )
        self.snn = SNNEncoder(
            d_input=feature_dim, n_neurons=snn_neurons,
            d_vsa=d_vsa, neuron_type="lif",
        )

        # Feature extraction: simple spatial statistics by default.
        # Replace with VisionEncoder or CNN for richer features.
        self._feature_extractor = None

        # Frame buffer for scene analysis
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._feature_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

        self.device = device

    def set_feature_extractor(self, extractor) -> None:
        """Set a custom feature extractor (VisionEncoder, CNN, etc.).

        Must have a method: extract(image) → (feature_dim,) float32
        """
        self._feature_extractor = extractor

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract feature vector from a preprocessed frame.

        Args:
            frame: (H, W) float32 normalized frame.

        Returns:
            (feature_dim,) float32 feature vector.
        """
        # Custom extractor (VisionEncoder, CNN, etc.)
        if self._feature_extractor is not None:
            try:
                if hasattr(self._feature_extractor, 'encode_image'):
                    # VisionEncoder path — returns (k, l) block-code
                    bc = self._feature_extractor.encode_image(frame)
                    feat = bc.ravel()[:self.feature_dim]
                elif hasattr(self._feature_extractor, 'forward'):
                    # CNN path
                    bc = self._feature_extractor.forward(frame)
                    feat = bc.ravel()[:self.feature_dim]
                else:
                    feat = self._feature_extractor(frame)
                if len(feat) < self.feature_dim:
                    feat = np.pad(feat, (0, self.feature_dim - len(feat)))
                return feat.astype(np.float32)
            except Exception:
                pass

        # Default: spatial statistics feature extraction
        return self._spatial_features(frame)

    def _spatial_features(self, frame: np.ndarray) -> np.ndarray:
        """Simple spatial statistics as features (no model needed).

        Divides frame into a grid and computes per-cell statistics.
        """
        h, w = frame.shape[:2]
        img = frame if frame.ndim == 2 else frame[:, :, 0]

        # Grid: divide into sqrt(feature_dim/4) × sqrt(feature_dim/4) cells
        grid_n = max(2, int(np.sqrt(self.feature_dim / 4)))
        cell_h, cell_w = h // grid_n, w // grid_n

        features = []
        for i in range(grid_n):
            for j in range(grid_n):
                cell = img[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                features.extend([
                    float(np.mean(cell)),
                    float(np.std(cell)),
                    float(np.max(cell)),
                    float(np.min(cell)),
                ])

        feat = np.array(features[:self.feature_dim], dtype=np.float32)
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        return feat

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single raw frame through the full pipeline.

        Args:
            frame: Raw frame (H, W, C) uint8 or (H, W) float32.

        Returns:
            Dict with 'features', 'spikes', 'spike_rate', 'emotion'.
        """
        processed = self.preprocessor.process(frame)
        features = self.extract_features(processed)

        self._frame_buffer.append(processed)
        self._feature_buffer.append(features)

        spikes = self.snn.step(features)
        spike_rate = float(np.mean(spikes))

        return {
            "features": features,
            "spikes": spikes,
            "spike_rate": spike_rate,
            "emotion": self.snn.neurochemistry.dominant_emotion,
            "arousal": self.snn.neurochemistry.arousal,
        }

    def encode_buffer(self) -> np.ndarray:
        """Encode the current frame buffer as a temporal VSA vector.

        Returns:
            (words_per_vec,) uint32 packed binary temporal memory.
        """
        if not self._feature_buffer:
            return np.zeros(self.snn.words_per_vec, dtype=np.uint32)

        stream = np.stack(list(self._feature_buffer))
        return self.snn.encode_stream(stream)

    def capture_and_encode(self, duration_s: float = 1.0, fps: float = 10.0) -> dict:
        """Capture from webcam for a duration and encode to VSA.

        Args:
            duration_s: Capture duration in seconds.
            fps:        Frames per second to capture.

        Returns:
            Dict with 'temporal_vector', 'n_frames', 'avg_spike_rate', 'emotion'.
        """
        if _CV2 is None:
            raise ImportError("opencv-python required for webcam capture")

        cap = _CV2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {self.device}")

        interval = 1.0 / fps
        self.snn.reset()
        self._frame_buffer.clear()
        self._feature_buffer.clear()

        spike_rates = []
        t_start = time.monotonic()

        try:
            while time.monotonic() - t_start < duration_s:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.process_frame(frame)
                spike_rates.append(result["spike_rate"])
                time.sleep(max(0, interval - (time.monotonic() - t_start) % interval))
        finally:
            cap.release()

        temporal_vec = self.encode_buffer()

        return {
            "temporal_vector": temporal_vec,
            "n_frames": len(spike_rates),
            "avg_spike_rate": float(np.mean(spike_rates)) if spike_rates else 0.0,
            "emotion": self.snn.neurochemistry.dominant_emotion,
            "neurochemistry": self.snn.neurochemistry.to_dict(),
        }
