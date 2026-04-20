"""Face Perception — MediaPipe Face Landmarker → blendshapes → SNN.

Extracts 52 facial blendshapes (Action Units) + 468 3D landmarks from
webcam frames in <5ms using MediaPipe's BlazeFace model.

Blendshapes map to FACS Action Units:
  browInnerUp, eyeSquintLeft, mouthSmileLeft, jawOpen, cheekPuff, etc.

These feed directly into the SNN as a 52-dim feature vector. The SNN's
STDP learns which AU combinations correspond to which emotional states.

Micro-expression detection: frame-to-frame blendshape deltas reveal
subtle facial movements (onset/offset of expressions in 40-200ms).

Pipeline:
  frame → MediaPipe Face Landmarker → 52 blendshapes + 468 landmarks
  → SNN temporal encoding → neurochemistry modulation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# MediaPipe (lazy import to avoid overhead if not used)
_MP = None
_MP_TASKS = None

# Default model path
_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker_v2_with_blendshapes.task"

# 52 blendshape names (ARKit-compatible, loosely maps to FACS AUs)
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

# Emotion mapping: which blendshapes indicate which emotions
# Values are (blendshape_index, weight) pairs
EMOTION_INDICATORS = {
    "happy": [(44, 1.0), (45, 1.0), (7, 0.5), (8, 0.5)],  # smile + cheek squint
    "sad": [(30, 1.0), (31, 1.0), (3, 0.5)],  # mouth frown + brow inner up
    "surprised": [(21, 1.0), (22, 1.0), (25, 0.8), (3, 0.7)],  # eyes wide + jaw open + brow up
    "angry": [(1, 1.0), (2, 1.0), (36, 0.5), (37, 0.5)],  # brow down + mouth press
    "disgusted": [(50, 1.0), (51, 1.0), (40, 0.5)],  # nose sneer + mouth roll
    "fearful": [(21, 0.8), (22, 0.8), (3, 1.0), (46, 0.5), (47, 0.5)],  # eyes wide + brow up + mouth stretch
}


def _ensure_mediapipe():
    """Lazy-load mediapipe."""
    global _MP, _MP_TASKS
    if _MP is None:
        import mediapipe as mp
        _MP = mp
        _MP_TASKS = mp.tasks
    return _MP, _MP_TASKS


class FacePerception:
    """MediaPipe-based face perception with blendshapes and micro-expression detection.

    Args:
        model_path:         Path to face_landmarker.task model file.
        max_faces:          Maximum number of faces to detect.
        min_detection_conf: Minimum detection confidence.
        min_tracking_conf:  Minimum tracking confidence.
        micro_expr_window:  Frames to buffer for micro-expression detection.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        max_faces: int = 1,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        micro_expr_window: int = 15,
    ) -> None:
        mp, mp_tasks = _ensure_mediapipe()

        model_path = Path(model_path) if model_path else _MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        options = mp_tasks.vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_tasks.vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_tracking_conf,
        )
        self._landmarker = mp_tasks.vision.FaceLandmarker.create_from_options(options)

        self._micro_window = micro_expr_window
        self._blendshape_history: list[np.ndarray] = []
        self._prev_blendshapes: np.ndarray | None = None

    def process_frame(self, frame: np.ndarray) -> dict | None:
        """Process a single BGR frame and extract face features.

        Args:
            frame: (H, W, 3) uint8 BGR frame from OpenCV.

        Returns:
            Dict with 'blendshapes', 'landmarks', 'deltas', 'emotion',
            'micro_expressions', or None if no face detected.
        """
        mp, _ = _ensure_mediapipe()

        # Convert BGR → RGB for MediaPipe
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb = frame[:, :, ::-1].copy()
        else:
            rgb = frame

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_blendshapes or len(result.face_blendshapes) == 0:
            return None

        # Extract blendshapes as numpy array (52,)
        bs = result.face_blendshapes[0]
        blendshapes = np.array([c.score for c in bs], dtype=np.float32)

        # Extract landmarks as (468, 3) array
        landmarks = None
        if result.face_landmarks and len(result.face_landmarks) > 0:
            lm = result.face_landmarks[0]
            landmarks = np.array(
                [[p.x, p.y, p.z] for p in lm], dtype=np.float32,
            )

        # Frame-to-frame deltas (micro-expression signal)
        if self._prev_blendshapes is not None:
            deltas = blendshapes - self._prev_blendshapes
        else:
            deltas = np.zeros_like(blendshapes)
        self._prev_blendshapes = blendshapes.copy()

        # Buffer for micro-expression detection
        self._blendshape_history.append(blendshapes.copy())
        if len(self._blendshape_history) > self._micro_window:
            self._blendshape_history.pop(0)

        # Detect emotion from blendshapes
        emotion, emotion_scores = self._detect_emotion(blendshapes)

        # Detect micro-expressions from temporal deltas
        micro_exprs = self._detect_micro_expressions()

        return {
            "blendshapes": blendshapes,
            "blendshape_names": BLENDSHAPE_NAMES[:len(blendshapes)],
            "landmarks": landmarks,
            "deltas": deltas,
            "delta_magnitude": float(np.linalg.norm(deltas)),
            "emotion": emotion,
            "emotion_scores": emotion_scores,
            "micro_expressions": micro_exprs,
            "n_landmarks": len(landmarks) if landmarks is not None else 0,
        }

    def get_snn_features(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract a feature vector suitable for SNN input.

        Returns blendshapes (52D) + top deltas, or None if no face.
        """
        result = self.process_frame(frame)
        if result is None:
            return None
        # Concatenate: blendshapes (52) + deltas (52) = 104D
        return np.concatenate([
            result["blendshapes"],
            result["deltas"] * 5.0,  # Amplify deltas for SNN sensitivity
        ]).astype(np.float32)

    def _detect_emotion(self, blendshapes: np.ndarray) -> tuple[str, dict]:
        """Detect dominant emotion from blendshape activations."""
        scores = {}
        for emotion, indicators in EMOTION_INDICATORS.items():
            score = 0.0
            for idx, weight in indicators:
                if idx < len(blendshapes):
                    score += blendshapes[idx] * weight
            scores[emotion] = float(score / len(indicators))

        best = max(scores, key=scores.get)
        # Threshold: if best score is too low, it's neutral
        if scores[best] < 0.15:
            best = "neutral"
        return best, scores

    def _detect_micro_expressions(self) -> list[dict]:
        """Detect micro-expressions from blendshape history.

        Micro-expressions are rapid onset (< 200ms) + offset of AU activations
        that the person may not be aware of. At 30fps, that's ~6 frames.
        """
        if len(self._blendshape_history) < 5:
            return []

        micro_exprs = []
        history = np.stack(self._blendshape_history)  # (window, 52)

        # For each blendshape, check for rapid spike-then-return pattern
        for i in range(min(52, history.shape[1])):
            trace = history[:, i]
            if len(trace) < 5:
                continue

            # Spike detection: value rises > 0.2 above baseline then returns
            baseline = float(np.median(trace[:3]))
            peak = float(np.max(trace[2:-1]))
            end = float(np.mean(trace[-2:]))

            spike_height = peak - baseline
            returned = abs(end - baseline) < spike_height * 0.5

            if spike_height > 0.2 and returned:
                micro_exprs.append({
                    "blendshape": BLENDSHAPE_NAMES[i] if i < len(BLENDSHAPE_NAMES) else f"bs_{i}",
                    "spike_height": spike_height,
                    "baseline": baseline,
                    "peak": peak,
                })

        return micro_exprs

    def get_identity_features(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract face identity features from landmarks.

        Uses inter-landmark distances that are stable across expressions
        but unique to each person: eye width, nose length, jaw shape,
        face proportions, etc.

        Returns:
            (128,) float32 identity feature vector, or None if no face.
        """
        result = self.process_frame(frame)
        if result is None or result["landmarks"] is None:
            return None

        lm = result["landmarks"]  # (468, 3)

        # Key landmark indices for face geometry
        # Pairs of landmarks → distances that define face structure
        identity_pairs = [
            # Eye width (left/right)
            (33, 133), (362, 263),
            # Eye height
            (159, 145), (386, 374),
            # Inter-eye distance
            (33, 362),
            # Nose length
            (6, 4), (4, 1),
            # Nose width
            (48, 278),
            # Mouth width
            (61, 291),
            # Mouth height
            (0, 17),
            # Jaw width
            (234, 454),
            # Face height (forehead to chin)
            (10, 152),
            # Face width (cheek to cheek)
            (234, 454),
            # Eyebrow to eye
            (70, 33), (300, 362),
            # Chin shape
            (152, 377), (152, 148),
            # Cheekbone
            (116, 345),
            # Forehead width
            (54, 284),
        ]

        # Compute distances
        distances = []
        for i, j in identity_pairs:
            if i < len(lm) and j < len(lm):
                d = float(np.linalg.norm(lm[i] - lm[j]))
                distances.append(d)

        # Normalize by inter-eye distance for scale invariance
        inter_eye = distances[4] if len(distances) > 4 and distances[4] > 1e-6 else 1.0
        distances = [d / inter_eye for d in distances]

        # Also add key landmark positions (normalized by face bounding box)
        face_center = np.mean(lm, axis=0)
        face_scale = np.std(lm, axis=0).mean() + 1e-6

        key_indices = [1, 4, 6, 10, 33, 61, 133, 152, 159, 234, 263, 278, 284, 291, 362, 386, 454]
        for idx in key_indices:
            if idx < len(lm):
                normalized = (lm[idx] - face_center) / face_scale
                distances.extend(normalized.tolist())

        # Pad/truncate to 128D
        feat = np.array(distances[:128], dtype=np.float32)
        if len(feat) < 128:
            feat = np.pad(feat, (0, 128 - len(feat)))

        return feat

    def close(self) -> None:
        """Release the landmarker."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
