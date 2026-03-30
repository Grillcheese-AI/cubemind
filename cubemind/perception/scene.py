"""Scene Understanding + Taste — video analysis with emergent preferences.

Watches a video, segments it into scenes, builds a temporal scene graph,
and generates a critique informed by the system's accumulated taste.

Pipeline:
  Video → subsample frames → per-frame features (face + vision)
  → SNN temporal encoding per segment → scene VSA vectors
  → retrieve past experiences → compare with emotional memory
  → generate critique with taste-informed opinion

Taste = scene_vector XOR emotion_vector stored in memory.
Over repeated exposures, the system develops genuine preferences
that emerge from neurochemistry, not hardcoded rules.

Key concepts:
  - Scene segmentation: Hamming distance spikes between consecutive frame VSAs
  - Emotional binding: scene content XOR'd with neurochemical state
  - Taste formation: Hebbian accumulation of valence per scene type
  - Critique: structured analysis from scene graph + taste
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cubemind.ops.vsa_bridge import (
    LSHProjector,
    ContinuousItemMemory,
    binarize_and_pack,
    hamming_similarity,
)
from cubemind.perception.snn import SNNEncoder, NeurochemicalState

# Optional imports
_CV2 = None
try:
    import cv2 as _cv2
    _CV2 = _cv2
except ImportError:
    pass

_FACE = None
try:
    from cubemind.perception.face import FacePerception
    _FACE = FacePerception
except Exception:
    pass


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class SceneSegment:
    """A detected scene segment from a video."""
    start_frame: int
    end_frame: int
    duration_s: float
    scene_vector: np.ndarray           # Packed binary VSA
    emotion_vector: np.ndarray         # Scene XOR emotion = taste vector
    dominant_emotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    face_emotion: str | None = None
    micro_expressions: list = field(default_factory=list)
    description: str = ""
    taste_score: float = 0.0          # [-1, 1] how much the system liked it


@dataclass
class TasteProfile:
    """Accumulated preferences from past experiences."""
    positive_scenes: int = 0
    negative_scenes: int = 0
    total_scenes: int = 0
    avg_valence: float = 0.0
    preferred_emotions: dict = field(default_factory=dict)
    style_preferences: dict = field(default_factory=dict)


# ── Emotion → VSA encoding ────────────────────────────────────────────────

def _emotion_to_vector(neurochemistry: NeurochemicalState, d_vsa: int = 2048) -> np.ndarray:
    """Encode neurochemical state as a packed binary VSA vector.

    This creates a unique binary signature for each emotional state,
    so scene XOR emotion = taste-colored memory.
    """
    # Use hormone levels as seeds for deterministic binary vectors
    emotion_features = np.array([
        neurochemistry.cortisol,
        neurochemistry.dopamine,
        neurochemistry.serotonin,
        neurochemistry.oxytocin,
        neurochemistry.valence,
        neurochemistry.arousal,
        neurochemistry.stress,
        1.0 if neurochemistry.dominant_emotion == "joy" else 0.0,
        1.0 if neurochemistry.dominant_emotion == "curious" else 0.0,
        1.0 if neurochemistry.dominant_emotion == "anxious" else 0.0,
        1.0 if neurochemistry.dominant_emotion == "sad" else 0.0,
        1.0 if neurochemistry.dominant_emotion == "warm" else 0.0,
    ], dtype=np.float32)

    # Project to VSA space via random projection seeded by emotion hash
    rng = np.random.default_rng(int(np.sum(emotion_features * 1000) % (2**31)))
    projection = rng.choice([-1.0, 1.0], size=(len(emotion_features), d_vsa)).astype(np.float32)
    projected = emotion_features @ projection
    return binarize_and_pack(projected)


# ── Scene Analyzer ────────────────────────────────────────────────────────

class SceneAnalyzer:
    """Analyzes video content and builds scene understanding with taste.

    Args:
        feature_dim:    Per-frame feature dimension.
        snn_neurons:    Number of SNN neurons.
        d_vsa:          Binary VSA dimension.
        segment_thresh: Hamming distance threshold for scene change detection.
        taste_memory_capacity: Max scenes to remember for taste formation.
        seed:           Random seed.
    """

    def __init__(
        self,
        feature_dim: int = 104,
        snn_neurons: int = 256,
        d_vsa: int = 2048,
        segment_thresh: float = 0.35,
        taste_memory_capacity: int = 10000,
        seed: int = 42,
    ) -> None:
        self.feature_dim = feature_dim
        self.d_vsa = d_vsa
        self.segment_thresh = segment_thresh
        self.words_per_vec = int(np.ceil(d_vsa / 32))

        # SNN for temporal encoding
        self.snn = SNNEncoder(
            d_input=feature_dim,
            n_neurons=snn_neurons,
            d_vsa=d_vsa,
            neuron_type="lif",
            tau=15.0,
            v_threshold=0.1,
            seed=seed,
        )
        self.snn.stdp_lr_potentiate = 0.0002
        self.snn.stdp_lr_depress = 0.0001
        self.snn.stdp_weight_clip = 0.3

        # Face perception (optional)
        self.face: FacePerception | None = None
        if _FACE is not None:
            try:
                self.face = _FACE(max_faces=1)
            except Exception:
                pass

        # LSH for scene features when no face available
        self.scene_lsh = LSHProjector(d_input=feature_dim, d_output=d_vsa, seed=seed + 1)

        # Taste memory: stores scene XOR emotion vectors
        self.taste_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=taste_memory_capacity)

        # Taste profile: accumulated preferences
        self.taste = TasteProfile()

    # ── Video processing ──────────────────────────────────────────────────

    def analyze_video(
        self,
        video_path: str,
        subsample: int = 3,
        max_frames: int = 600,
    ) -> dict:
        """Analyze a video file and produce a scene-by-scene critique.

        Args:
            video_path:  Path to video file.
            subsample:   Process every Nth frame.
            max_frames:  Maximum frames to process.

        Returns:
            Dict with 'segments', 'critique', 'taste_score', 'scene_graph'.
        """
        if _CV2 is None:
            raise ImportError("opencv-python required: pip install opencv-python")

        cap = _CV2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(_CV2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"Analyzing: {video_path}")
        print(f"  {total_frames} frames, {fps:.0f} fps, {duration:.1f}s")

        # Process frames
        self.snn.reset()
        self._frame_neurochemistry = []
        frame_features = []
        frame_vectors = []
        frame_emotions = []
        frame_idx = 0
        processed = 0

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % subsample != 0:
                continue

            # Extract features
            feat, face_emo = self._extract_frame_features(frame)
            frame_features.append(feat)
            frame_emotions.append(face_emo)

            # SNN step
            feat_norm = feat / (np.std(feat) + 1e-6) * 0.3
            feat_norm += np.random.default_rng().standard_normal(len(feat_norm)).astype(np.float32) * 0.03
            spikes = self.snn.step(feat_norm)
            spike_rate = float(np.mean(spikes))

            # Compute visual novelty: how different is this frame from the last?
            if processed > 0 and len(frame_features) >= 2:
                frame_delta = float(np.mean(np.abs(frame_features[-1] - frame_features[-2])))
            else:
                frame_delta = 0.0

            # Visual qualities that drive neurochemistry even without faces:
            # - High brightness variance → interesting/dynamic → dopamine
            # - Smooth/uniform → calm → serotonin
            # - Rapid change → alerting → cortisol
            brightness = float(np.mean(feat[:self.feature_dim // 2]))
            variance = float(np.std(feat))

            visual_valence = float(np.clip(variance * 2.0 - 0.3, -0.5, 0.5))  # variety = pleasant
            visual_novelty = float(np.clip(frame_delta * 10.0 + spike_rate, 0, 1))
            visual_threat = float(np.clip(frame_delta * 5.0 - 0.3, 0, 0.5))  # sudden change = startle

            # Face emotion overrides if available
            face_valence = self._face_emotion_to_valence(face_emo)
            if face_emo and face_emo != "neutral":
                visual_valence = face_valence

            self.snn.neurochemistry.update(
                novelty=visual_novelty,
                valence=visual_valence,
                threat=visual_threat,
                focus=0.4 if face_emo else 0.2,
            )

            # Encode current frame state as binary VSA
            projected = self.scene_lsh.project(feat)
            frame_vec = binarize_and_pack(projected)
            frame_vectors.append(frame_vec)

            # Snapshot neurochemistry per frame for per-segment analysis
            if not hasattr(self, '_frame_neurochemistry'):
                self._frame_neurochemistry = []
            self._frame_neurochemistry.append({
                "valence": self.snn.neurochemistry.valence,
                "arousal": self.snn.neurochemistry.arousal,
                "emotion": self.snn.neurochemistry.dominant_emotion,
                "cortisol": self.snn.neurochemistry.cortisol,
                "dopamine": self.snn.neurochemistry.dopamine,
                "serotonin": self.snn.neurochemistry.serotonin,
            })

            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed} frames...")

        cap.release()

        if not frame_vectors:
            return {"segments": [], "critique": "No frames could be processed.", "taste_score": 0.0}

        # Segment the video by detecting scene changes
        segments = self._segment_scenes(frame_vectors, frame_emotions, fps, subsample)

        # Score each segment with taste
        for seg in segments:
            seg.taste_score = self._compute_taste_score(seg)

        # Build scene graph
        scene_graph = self._build_scene_graph(segments)

        # Generate critique with taste
        critique = self._generate_critique(segments, scene_graph, duration)

        # Store experiences for future taste development
        self._store_experiences(segments)

        overall_taste = float(np.mean([s.taste_score for s in segments])) if segments else 0.0

        return {
            "segments": segments,
            "scene_graph": scene_graph,
            "critique": critique,
            "taste_score": overall_taste,
            "n_segments": len(segments),
            "duration_s": duration,
            "frames_processed": processed,
            "neurochemistry": self.snn.neurochemistry.to_dict(),
        }

    def analyze_frames(self, frames: list[np.ndarray], fps: float = 30.0) -> dict:
        """Analyze a list of frames (from webcam capture, etc.)."""
        self.snn.reset()
        frame_vectors = []
        frame_emotions = []

        for frame in frames:
            feat, face_emo = self._extract_frame_features(frame)
            frame_emotions.append(face_emo)

            feat_norm = feat / (np.std(feat) + 1e-6) * 0.3
            self.snn.step(feat_norm)

            projected = self.scene_lsh.project(feat)
            frame_vectors.append(binarize_and_pack(projected))

        segments = self._segment_scenes(frame_vectors, frame_emotions, fps, 1)
        for seg in segments:
            seg.taste_score = self._compute_taste_score(seg)

        scene_graph = self._build_scene_graph(segments)
        critique = self._generate_critique(segments, scene_graph, len(frames) / fps)
        self._store_experiences(segments)

        return {
            "segments": segments,
            "scene_graph": scene_graph,
            "critique": critique,
            "taste_score": float(np.mean([s.taste_score for s in segments])) if segments else 0.0,
        }

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_frame_features(self, frame: np.ndarray) -> tuple[np.ndarray, str | None]:
        """Extract features from a single frame.

        Returns (features, face_emotion).
        """
        face_emo = None

        # Try face perception first (52 blendshapes + 52 deltas = 104D)
        if self.face is not None:
            snn_feat = self.face.get_snn_features(frame)
            if snn_feat is not None:
                result = self.face.process_frame(frame)
                face_emo = result["emotion"] if result else None
                if len(snn_feat) >= self.feature_dim:
                    return snn_feat[:self.feature_dim], face_emo
                return np.pad(snn_feat, (0, self.feature_dim - len(snn_feat))), face_emo

        # Fallback: grayscale spatial statistics
        if frame.ndim == 3:
            gray = np.mean(frame.astype(np.float32), axis=2) / 255.0
        else:
            gray = frame.astype(np.float32)
            if gray.max() > 1.0:
                gray /= 255.0

        # Grid features
        h, w = gray.shape[:2]
        grid_n = max(2, int(np.sqrt(self.feature_dim / 4)))
        cell_h, cell_w = max(1, h // grid_n), max(1, w // grid_n)
        features = []
        for i in range(grid_n):
            for j in range(grid_n):
                cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                if cell.size > 0:
                    features.extend([float(np.mean(cell)), float(np.std(cell)),
                                     float(np.max(cell)), float(np.min(cell))])
        feat = np.array(features[:self.feature_dim], dtype=np.float32)
        if len(feat) < self.feature_dim:
            feat = np.pad(feat, (0, self.feature_dim - len(feat)))
        return feat, face_emo

    # ── Scene segmentation ────────────────────────────────────────────────

    def _segment_scenes(
        self, frame_vectors: list[np.ndarray], frame_emotions: list,
        fps: float, subsample: int,
    ) -> list[SceneSegment]:
        """Detect scene boundaries via Hamming distance + time-based splitting.

        Two segmentation strategies combined:
          1. Hard cuts: Hamming similarity drops below threshold
          2. Time windows: force a segment every max_segment_s seconds
             so even gradual changes get captured
        """
        max_segment_frames = int(5.0 * fps / max(subsample, 1))  # ~5s per segment max

        if len(frame_vectors) < 2:
            if frame_vectors:
                emo_vec = _emotion_to_vector(self.snn.neurochemistry, self.d_vsa)
                return [SceneSegment(
                    start_frame=0, end_frame=1, duration_s=1.0 / fps,
                    scene_vector=frame_vectors[0],
                    emotion_vector=np.bitwise_xor(frame_vectors[0], emo_vec),
                    dominant_emotion=self.snn.neurochemistry.dominant_emotion,
                    valence=self.snn.neurochemistry.valence,
                    arousal=self.snn.neurochemistry.arousal,
                )]
            return []

        # Compute frame-to-frame Hamming similarity
        similarities = []
        for i in range(1, len(frame_vectors)):
            sim = hamming_similarity(frame_vectors[i - 1], frame_vectors[i], self.d_vsa)
            similarities.append(sim)

        # Find scene boundaries: hard cuts OR time-based splits
        boundaries = [0]
        frames_since_boundary = 0
        for i, sim in enumerate(similarities):
            frames_since_boundary += 1
            is_hard_cut = sim < self.segment_thresh
            is_time_split = frames_since_boundary >= max_segment_frames
            if is_hard_cut or is_time_split:
                boundaries.append(i + 1)
                frames_since_boundary = 0
        boundaries.append(len(frame_vectors))

        # Build segments
        segments = []
        for seg_idx in range(len(boundaries) - 1):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            if end <= start:
                continue

            # Average the frame vectors in this segment (majority vote via bundling)
            seg_vectors = frame_vectors[start:end]
            # Simple approach: use the middle frame's vector as representative
            mid = len(seg_vectors) // 2
            scene_vec = seg_vectors[mid]

            # Emotion vector: bind scene with current neurochemical state
            emo_vec = _emotion_to_vector(self.snn.neurochemistry, self.d_vsa)
            taste_vec = np.bitwise_xor(scene_vec, emo_vec)

            # Dominant face emotion in this segment
            seg_emotions = frame_emotions[start:end]
            face_emos = [e for e in seg_emotions if e and e != "neutral"]
            face_emo = max(set(face_emos), key=face_emos.count) if face_emos else None

            duration = (end - start) * subsample / fps

            # Use per-frame neurochemistry snapshots for this segment's stats
            seg_nc = getattr(self, '_frame_neurochemistry', [])
            if seg_nc and start < len(seg_nc):
                seg_slice = seg_nc[start:min(end, len(seg_nc))]
                if seg_slice:
                    seg_valence = float(np.mean([s["valence"] for s in seg_slice]))
                    seg_arousal = float(np.mean([s["arousal"] for s in seg_slice]))
                    # Most common emotion in this segment
                    seg_emotions_list = [s["emotion"] for s in seg_slice]
                    seg_emotion = max(set(seg_emotions_list), key=seg_emotions_list.count)
                else:
                    seg_valence = self.snn.neurochemistry.valence
                    seg_arousal = self.snn.neurochemistry.arousal
                    seg_emotion = self.snn.neurochemistry.dominant_emotion
            else:
                seg_valence = self.snn.neurochemistry.valence
                seg_arousal = self.snn.neurochemistry.arousal
                seg_emotion = self.snn.neurochemistry.dominant_emotion

            segments.append(SceneSegment(
                start_frame=start * subsample,
                end_frame=end * subsample,
                duration_s=duration,
                scene_vector=scene_vec,
                emotion_vector=taste_vec,
                dominant_emotion=seg_emotion,
                valence=seg_valence,
                arousal=seg_arousal,
                face_emotion=face_emo,
            ))

        return segments

    # ── Taste computation ─────────────────────────────────────────────────

    def _compute_taste_score(self, segment: SceneSegment) -> float:
        """Score a scene segment based on accumulated taste.

        Returns [-1, 1]: negative = dislike, 0 = neutral, positive = like.
        """
        if self.taste_memory.size == 0:
            # No prior experience — taste is purely from neurochemistry
            return float(np.clip(segment.valence * 0.5 + segment.arousal * 0.3, -1, 1))

        # Query taste memory: how similar is this scene+emotion to past experiences?
        results = self.taste_memory.retrieve(segment.emotion_vector, k=5)

        if not results:
            return float(np.clip(segment.valence, -1, 1))

        # Weighted average of past taste scores stored in labels
        weighted_score = 0.0
        total_weight = 0.0
        for _, sim, label in results:
            try:
                past_score = float(label)
            except (ValueError, TypeError):
                past_score = 0.0
            weighted_score += sim * past_score
            total_weight += sim

        if total_weight > 0:
            past_taste = weighted_score / total_weight
        else:
            past_taste = 0.0

        # Blend: 40% current neurochemistry reaction + 60% accumulated taste
        current = float(np.clip(segment.valence * 0.5 + segment.arousal * 0.3, -1, 1))
        return float(np.clip(0.4 * current + 0.6 * past_taste, -1, 1))

    def _store_experiences(self, segments: list[SceneSegment]) -> None:
        """Store scene experiences in taste memory for future preference formation."""
        for seg in segments:
            # Store the taste vector with the taste score as label
            label = f"{seg.taste_score:.3f}"
            self.taste_memory.learn(seg.emotion_vector, label=label)

        # Update taste profile
        for seg in segments:
            self.taste.total_scenes += 1
            if seg.taste_score > 0.2:
                self.taste.positive_scenes += 1
            elif seg.taste_score < -0.2:
                self.taste.negative_scenes += 1

            # Track preferred emotions
            emo = seg.face_emotion or seg.dominant_emotion
            if emo not in self.taste.preferred_emotions:
                self.taste.preferred_emotions[emo] = {"count": 0, "avg_taste": 0.0}
            entry = self.taste.preferred_emotions[emo]
            entry["count"] += 1
            entry["avg_taste"] = (entry["avg_taste"] * (entry["count"] - 1) + seg.taste_score) / entry["count"]

        if self.taste.total_scenes > 0:
            self.taste.avg_valence = float(np.mean([s.valence for s in segments]))

    # ── Scene graph ───────────────────────────────────────────────────────

    def _build_scene_graph(self, segments: list[SceneSegment]) -> list[dict]:
        """Build a temporal scene graph from segments."""
        graph = []
        for i, seg in enumerate(segments):
            node = {
                "index": i,
                "time_range": f"{seg.start_frame}-{seg.end_frame}",
                "duration_s": round(seg.duration_s, 1),
                "emotion": seg.face_emotion or seg.dominant_emotion,
                "valence": round(seg.valence, 2),
                "arousal": round(seg.arousal, 2),
                "taste_score": round(seg.taste_score, 2),
                "micro_expressions": len(seg.micro_expressions),
            }
            # Transition from previous segment
            if i > 0:
                prev = segments[i - 1]
                node["transition_from"] = prev.face_emotion or prev.dominant_emotion
                node["scene_similarity"] = round(
                    hamming_similarity(prev.scene_vector, seg.scene_vector, self.d_vsa), 2,
                )
            graph.append(node)
        return graph

    # ── Critique generation ───────────────────────────────────────────────

    def _generate_critique(
        self, segments: list[SceneSegment], scene_graph: list[dict], duration: float,
    ) -> str:
        """Generate a critique informed by taste and emotional experience.

        This is template-based. With MoQE LLM, it would be generated
        from the scene graph + taste profile.
        """
        if not segments:
            return "Nothing to critique — no scenes detected."

        n = len(segments)
        avg_taste = float(np.mean([s.taste_score for s in segments]))
        avg_valence = float(np.mean([s.valence for s in segments]))
        avg_arousal = float(np.mean([s.arousal for s in segments]))

        # Taste-informed opening — first person, opinionated
        if avg_taste > 0.4:
            opening = "I really liked this. Something about it resonated with me."
        elif avg_taste > 0.2:
            opening = "I enjoyed watching this — it held my attention throughout."
        elif avg_taste > 0.05:
            opening = "Pleasant enough. It has a quiet quality I can appreciate."
        elif avg_taste > -0.05:
            opening = "I'm neutral on this one. It didn't move me much either way."
        elif avg_taste > -0.2:
            opening = "Not quite my thing. I found my attention drifting."
        elif avg_taste > -0.4:
            opening = "I struggled with this. It felt flat to me."
        else:
            opening = "I'll be honest — this didn't work for me at all."

        # Add neurochemistry color to the opening
        if avg_arousal > 0.4:
            opening += " My arousal stayed high — this kept me engaged."
        elif avg_arousal < 0.15:
            opening += " Very low energy throughout — almost meditative."

        # Scene analysis
        lines = [opening, ""]

        # Pacing
        avg_duration = duration / max(n, 1)
        if avg_duration > 5.0:
            lines.append(f"Pacing: Slow and deliberate — {n} scenes across {duration:.0f}s.")
        elif avg_duration > 2.0:
            lines.append(f"Pacing: Measured — {n} scenes across {duration:.0f}s, good rhythm.")
        else:
            lines.append(f"Pacing: Quick — {n} scene changes in {duration:.0f}s, fast cuts.")

        # Per-segment observations with visual descriptors
        lines.append("")
        for i, seg in enumerate(segments):
            emo = seg.face_emotion or seg.dominant_emotion
            taste_icon = "+" if seg.taste_score > 0.1 else ("-" if seg.taste_score < -0.1 else "~")
            time_start = seg.start_frame / 30.0

            # Describe the segment's character from neurochemistry
            if seg.arousal > 0.4:
                energy = "dynamic"
            elif seg.arousal > 0.2:
                energy = "moderate energy"
            else:
                energy = "calm"

            if seg.valence > 0.2:
                mood = "uplifting"
            elif seg.valence > 0:
                mood = "pleasant"
            elif seg.valence > -0.2:
                mood = "neutral tone"
            else:
                mood = "somber"

            # Scene continuity from graph
            continuity = ""
            if i < len(scene_graph) and "scene_similarity" in scene_graph[i]:
                sim = scene_graph[i]["scene_similarity"]
                if sim > 0.7:
                    continuity = ", continuous"
                elif sim > 0.5:
                    continuity = ", gradual shift"
                else:
                    continuity = ", distinct change"

            lines.append(f"  [{taste_icon}] Scene {i+1} ({time_start:.0f}-{time_start + seg.duration_s:.0f}s): "
                         f"{energy}, {mood}{continuity} "
                         f"(v={seg.valence:+.2f} a={seg.arousal:.2f})")
        lines.append("")

        # Emotional arc
        emotions_seen = [s.face_emotion or s.dominant_emotion for s in segments]
        unique_emotions = list(dict.fromkeys(e for e in emotions_seen if e != "neutral"))
        if unique_emotions:
            arc = " -> ".join(unique_emotions[:8])
            lines.append(f"Emotional arc: {arc}")
        else:
            # Check if emotion was flat
            arousal_range = max(s.arousal for s in segments) - min(s.arousal for s in segments)
            if arousal_range < 0.1:
                lines.append("Emotional arc: Flat — low dynamic range. Needs more contrast.")
            else:
                lines.append("Emotional arc: Subtle shifts detected.")

        # Standout moments
        best_seg = max(segments, key=lambda s: s.taste_score)
        worst_seg = min(segments, key=lambda s: s.taste_score)
        if best_seg.taste_score > 0.1:
            lines.append(f"Best moment: scene {segments.index(best_seg) + 1} "
                         f"(taste: {best_seg.taste_score:+.2f}, "
                         f"emotion: {best_seg.face_emotion or best_seg.dominant_emotion})")
        if worst_seg.taste_score < -0.05 and worst_seg is not best_seg:
            lines.append(f"Weakest moment: scene {segments.index(worst_seg) + 1} "
                         f"(taste: {worst_seg.taste_score:+.2f})")

        # Taste development note
        if self.taste.total_scenes > 20:
            fav_emotions = sorted(
                self.taste.preferred_emotions.items(),
                key=lambda x: x[1]["avg_taste"], reverse=True,
            )
            if fav_emotions and fav_emotions[0][1]["avg_taste"] > 0.1:
                lines.append(f"\nMy emerging preference: I tend to respond well to "
                             f"'{fav_emotions[0][0]}' content "
                             f"(avg taste: {fav_emotions[0][1]['avg_taste']:+.2f} "
                             f"across {fav_emotions[0][1]['count']} scenes).")

        # Cortisol/serotonin commentary (what the system physically felt)
        nc = self.snn.neurochemistry
        if nc.cortisol > 0.4:
            lines.append(f"My cortisol was elevated ({nc.cortisol:.2f}) — I found parts of this stressful.")
        if nc.serotonin > 0.7:
            lines.append(f"High serotonin ({nc.serotonin:.2f}) — this had a stabilizing, soothing quality.")
        if nc.dopamine > 0.3:
            lines.append(f"Dopamine spike ({nc.dopamine:.2f}) — moments of genuine novelty or reward.")
        if nc.oxytocin > 0.5:
            lines.append(f"Oxytocin elevated ({nc.oxytocin:.2f}) — I felt a sense of warmth or connection.")

        # Overall rating
        stars = int(np.clip((avg_taste + 1) * 2.5, 0, 5))
        descriptors = {0: "Hard pass", 1: "Not for me", 2: "It's okay", 3: "Solid", 4: "Really good", 5: "Outstanding"}
        lines.append(f"\nOverall: {'*' * stars}{'.' * (5 - stars)} — {descriptors.get(stars, '')} ({avg_taste:+.2f})")

        return "\n".join(lines)

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _face_emotion_to_valence(emotion: str | None) -> float:
        return {
            "happy": 0.6, "surprised": 0.2, "warm": 0.4,
            "sad": -0.4, "angry": -0.5, "fearful": -0.3,
            "disgusted": -0.4,
        }.get(emotion or "", 0.0)

    def save(self, path: str) -> None:
        """Save taste memory and SNN brain."""
        self.taste_memory.save(f"{path}_taste")
        self.snn.save(f"{path}_snn.npz")
        # Save taste profile
        np.savez_compressed(
            f"{path}_profile.npz",
            total_scenes=np.int32(self.taste.total_scenes),
            positive_scenes=np.int32(self.taste.positive_scenes),
            negative_scenes=np.int32(self.taste.negative_scenes),
            avg_valence=np.float32(self.taste.avg_valence),
        )

    def load(self, path: str) -> None:
        """Load taste memory and SNN brain."""
        import os
        if os.path.exists(f"{path}_taste.npz"):
            self.taste_memory.load(f"{path}_taste")
        if os.path.exists(f"{path}_snn.npz"):
            self.snn.load(f"{path}_snn.npz")
        if os.path.exists(f"{path}_profile.npz"):
            data = np.load(f"{path}_profile.npz")
            self.taste.total_scenes = int(data["total_scenes"])
            self.taste.positive_scenes = int(data["positive_scenes"])
            self.taste.negative_scenes = int(data["negative_scenes"])
            self.taste.avg_valence = float(data["avg_valence"])
