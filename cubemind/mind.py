"""CubeMind — Unified Cognitive Architecture Orchestrator.

Single entry point that wires together all subsystems:
  Perception → SNN → Neurochemistry → Thalamus → Brain → Memory → Output

Accepts any input modality (text, image, video, audio, webcam) and routes
it through the appropriate perception pipeline, SNN temporal encoding,
brain cortex routing, and response generation.

Usage:
    mind = CubeMind()
    mind.see(frame)                    # Process a single image/frame
    mind.watch(video_path)             # Analyze a video with critique
    mind.hear(audio)                   # Process audio stream (future)
    mind.read(text)                    # Process text input
    mind.ask(question)                 # VQA about current scene
    mind.think()                       # Internal reflection/consolidation
    mind.respond(user_input)           # Full pipeline: perceive → reason → respond

    mind.save("cubemind_state")        # Persist everything
    mind.load("cubemind_state")        # Restore from disk
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from cubemind.ops.block_codes import BlockCodes
from cubemind.ops.vsa_bridge import (
    LSHProjector,
    ContinuousItemMemory,
    binarize_and_pack,
    hamming_similarity,
)
from cubemind.perception.snn import SNNEncoder, NeurochemicalState
from cubemind.brain.cortex import (
    CircadianCells,
    Thalamus,
    BasalGanglia,
    CentralNervousSystem,
    PersonalityLayer,
)

# Optional heavy imports — lazy loaded
_FacePerception = None
_SceneAnalyzer = None
_VQAEngine = None
_DetectedObject = None
_SemanticEncoder = None
_VisionEncoder = None
_AudioEncoder = None


def _lazy_imports():
    """Load heavy modules on first use."""
    global _FacePerception, _SceneAnalyzer, _VQAEngine, _DetectedObject
    global _SemanticEncoder, _VisionEncoder, _AudioEncoder
    if _FacePerception is None:
        try:
            from cubemind.perception.face import FacePerception as _FP
            _FacePerception = _FP
        except Exception:
            pass
    if _SceneAnalyzer is None:
        try:
            from cubemind.perception.scene import SceneAnalyzer as _SA
            _SceneAnalyzer = _SA
        except Exception:
            pass
    if _VQAEngine is None:
        try:
            from cubemind.reasoning.vqa import VQAEngine as _VE, DetectedObject as _DO
            _VQAEngine = _VE
            _DetectedObject = _DO
        except Exception:
            pass
    if _SemanticEncoder is None:
        try:
            from cubemind.perception.semantic_encoder import SemanticEncoder as _SE
            _SemanticEncoder = _SE
        except Exception:
            pass
    if _VisionEncoder is None:
        try:
            from cubemind.perception.vision_encoder import VisionEncoder as _VE2
            _VisionEncoder = _VE2
        except Exception:
            pass
    if _AudioEncoder is None:
        try:
            from cubemind.perception.audio import AudioEncoder as _AE
            _AudioEncoder = _AE
        except Exception:
            pass


try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 8
    L_BLOCK = 64


# ── State ─────────────────────────────────────────────────────────────────

@dataclass
class MindState:
    """Current state of the CubeMind system."""
    # What it's currently perceiving
    current_scene: dict = field(default_factory=dict)
    current_face: dict | None = None
    current_emotion: str = "neutral"

    # Brain routing
    thalamus_route: str = "reasoning"
    strategy: str = "default"
    consciousness: str = "ALERT"

    # Accumulated context
    interaction_count: int = 0
    session_start: float = 0.0
    last_input_type: str = ""

    # Taste
    total_experiences: int = 0
    taste_valence: float = 0.0


# ── Main Orchestrator ─────────────────────────────────────────────────────

class Mind:
    """CubeMind unified cognitive architecture.

    Wires together all subsystems into a single coherent pipeline.
    Each input is perceived, emotionally processed, routed through
    the brain cortex, and either stored in memory or used to generate
    a response.

    Args:
        k:          VSA block count.
        l:          VSA block length.
        d_vsa:      Binary VSA dimension for packed vectors.
        snn_neurons: Number of SNN neurons.
        d_model:    Personality/brain model dimension.
        enable_face: Enable MediaPipe face perception.
        seed:       Random seed.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        d_vsa: int = 2048,
        snn_neurons: int = 256,
        d_model: int = 256,
        enable_face: bool = True,
        seed: int = 42,
    ) -> None:
        _lazy_imports()

        self.k = k
        self.l = l
        self.d_vsa = d_vsa
        self.bc = BlockCodes(k=k, l=l)

        # ── Perception ────────────────────────────────────────────────
        self.snn = SNNEncoder(
            d_input=104,  # 52 blendshapes + 52 deltas (face) or grid features
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

        self.face = None
        if enable_face and _FacePerception is not None:
            try:
                self.face = _FacePerception(max_faces=1)
            except Exception:
                pass

        self.text_encoder = None
        if _SemanticEncoder is not None:
            try:
                self.text_encoder = _SemanticEncoder(k=k, l=l)
            except Exception:
                pass

        # VisionEncoder: SigLIP/CLIP → semantic image features → VSA
        self.vision_encoder = None
        if _VisionEncoder is not None:
            try:
                self.vision_encoder = _VisionEncoder(k=k, l=l)
                print(f"VisionEncoder: {self.vision_encoder.mode} ({self.vision_encoder.embed_dim}D)")
            except Exception:
                pass

        # AudioEncoder: mel spectrogram → SNN → VSA
        self.audio_encoder = None
        if _AudioEncoder is not None:
            try:
                self.audio_encoder = _AudioEncoder(
                    snn_neurons=snn_neurons, d_vsa=d_vsa, seed=seed + 5,
                )
            except Exception:
                pass

        # ── Brain Cortex ──────────────────────────────────────────────
        self.circadian = CircadianCells()
        self.thalamus = Thalamus(embedding_dim=d_model)
        self.basal_ganglia = BasalGanglia()
        self.cns = CentralNervousSystem()
        self.personality = PersonalityLayer(d_model=d_model)

        # ── Memory ────────────────────────────────────────────────────
        self.episodic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=50000)
        self.semantic_memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=50000)
        self.lsh = LSHProjector(d_input=d_model, d_output=d_vsa, seed=seed + 10)

        # ── Scene Understanding ───────────────────────────────────────
        self.scene_analyzer = None
        if _SceneAnalyzer is not None:
            self.scene_analyzer = _SceneAnalyzer(
                feature_dim=104, snn_neurons=snn_neurons, d_vsa=d_vsa,
            )

        # ── VQA ───────────────────────────────────────────────────────
        self.vqa = None
        if _VQAEngine is not None:
            self.vqa = _VQAEngine(k=k, l=l)

        # ── State ─────────────────────────────────────────────────────
        self.state = MindState(session_start=time.time())

    # ── Public API ────────────────────────────────────────────────────────

    def see(self, frame: np.ndarray) -> dict:
        """Process a single image/frame through the full perception pipeline.

        frame → face perception → SNN encoding → neurochemistry → thalamus
        → brain routing → memory storage

        Returns dict with perception results, emotion, and routing decision.
        """
        self.state.last_input_type = "vision"
        self.state.interaction_count += 1

        result = {
            "modality": "vision",
            "face": None,
            "spikes": None,
            "neurochemistry": {},
            "route": {},
            "strategy": {},
            "emotion": "neutral",
        }

        # 1. Face perception
        face_feat = None
        face_emotion = None
        if self.face is not None:
            snn_feat = self.face.get_snn_features(frame)
            if snn_feat is not None:
                face_feat = snn_feat
                face_result = self.face.process_frame(frame)
                if face_result:
                    face_emotion = face_result["emotion"]
                    result["face"] = {
                        "emotion": face_emotion,
                        "blendshapes_active": int(np.sum(face_result["blendshapes"] > 0.1)),
                        "micro_expressions": face_result.get("micro_expressions", []),
                    }
                    self.state.current_face = result["face"]

        # 2. SNN encoding
        if face_feat is not None:
            feat = face_feat[:104]
            if len(feat) < 104:
                feat = np.pad(feat, (0, 104 - len(feat)))
        else:
            # Fallback: spatial grid features from frame
            feat = self._grid_features(frame, 104)

        feat_norm = feat / (np.std(feat) + 1e-6) * 0.3
        spikes = self.snn.step(feat_norm)
        spike_rate = float(np.mean(spikes))
        result["spikes"] = {"rate": spike_rate, "active": int(np.sum(spikes > 0))}

        # 3. Neurochemistry update
        face_valence = self._emotion_to_valence(face_emotion)
        self.snn.neurochemistry.update(
            novelty=spike_rate,
            valence=face_valence,
            threat=0.0,
            focus=0.4 if face_emotion else 0.1,
        )
        nc = self.snn.neurochemistry
        result["neurochemistry"] = nc.to_dict()
        result["emotion"] = nc.dominant_emotion
        self.state.current_emotion = nc.dominant_emotion

        # 4. Thalamus routing
        embedding = feat[:min(len(feat), self.thalamus.embedding_dim)]
        if len(embedding) < self.thalamus.embedding_dim:
            embedding = np.pad(embedding, (0, self.thalamus.embedding_dim - len(embedding)))
        route = self.thalamus.route(embedding, arousal=nc.arousal, valence=nc.valence)
        result["route"] = route
        self.state.thalamus_route = route["primary_route"]

        # 5. Strategy selection
        strategy = self.basal_ganglia.select_strategy(
            route["routes"], valence=nc.valence, arousal=nc.arousal, stress=nc.stress,
        )
        result["strategy"] = strategy
        self.state.strategy = strategy["strategy"]

        # 6. CNS update
        cns_state = self.cns.update(arousal=nc.arousal, stress=nc.stress)
        self.state.consciousness = cns_state.consciousness.name

        # 7. Store in episodic memory
        projected = self.lsh.project(embedding[:self.lsh.d_input])
        packed = binarize_and_pack(projected)
        self.episodic_memory.learn(packed, label=f"see:{nc.dominant_emotion}")

        return result

    def hear(self, audio: np.ndarray) -> dict:
        """Process audio through mel spectrogram → SNN → neurochemistry.

        Args:
            audio: (N,) float32 audio samples (16kHz mono).

        Returns dict with temporal VSA vector, spike info, neurochemistry.
        """
        self.state.last_input_type = "audio"
        self.state.interaction_count += 1

        if self.audio_encoder is None:
            return {"error": "AudioEncoder not available"}

        # Encode full audio clip to temporal VSA
        temporal_vec = self.audio_encoder.encode_audio(audio)
        nc = self.audio_encoder.snn.neurochemistry

        # Merge audio neurochemistry into main SNN
        # Audio emotions influence the global emotional state
        self.snn.neurochemistry.update(
            novelty=0.3,
            valence=nc.valence,
            threat=nc.stress,
            focus=0.2,
        )

        # Store in episodic memory
        self.episodic_memory.learn(temporal_vec, label=f"hear:{nc.dominant_emotion}")

        return {
            "modality": "audio",
            "temporal_vector": temporal_vec,
            "duration_s": len(audio) / 16000.0,
            "neurochemistry": nc.to_dict(),
            "emotion": nc.dominant_emotion,
            "main_emotion": self.snn.neurochemistry.dominant_emotion,
        }

    def see_semantic(self, frame: np.ndarray) -> dict:
        """Process an image through VisionEncoder for semantic features.

        Uses SigLIP/CLIP when available for rich visual understanding.
        Falls back to standard see() if no VL model loaded.

        Returns dict with block-code vector, similarity to known concepts.
        """
        if self.vision_encoder is None:
            return self.see(frame)

        self.state.last_input_type = "vision_semantic"
        self.state.interaction_count += 1

        # VisionEncoder: image → dense embedding → (k, l) block-code
        block_code = self.vision_encoder.encode_image(frame)

        # Store in episodic memory via LSH (aligns to d_vsa)
        feat = block_code.ravel()[:self.lsh.d_input]
        if len(feat) < self.lsh.d_input:
            feat = np.pad(feat, (0, self.lsh.d_input - len(feat)))
        packed = binarize_and_pack(self.lsh.project(feat))
        self.episodic_memory.learn(packed, label=f"see_semantic:{self.vision_encoder.mode}")

        return {
            "modality": "vision_semantic",
            "backend": self.vision_encoder.mode,
            "embed_dim": self.vision_encoder.embed_dim,
            "block_code_shape": block_code.shape,
            "emotion": self.snn.neurochemistry.dominant_emotion,
        }

    def watch(self, video_path: str, **kwargs) -> dict:
        """Analyze a video and produce a taste-informed critique.

        Returns scene analysis with segments, emotional arc, and critique.
        """
        self.state.last_input_type = "video"
        self.state.interaction_count += 1

        if self.scene_analyzer is None:
            return {"error": "SceneAnalyzer not available"}

        result = self.scene_analyzer.analyze_video(video_path, **kwargs)
        self.state.total_experiences = self.scene_analyzer.taste.total_scenes
        return result

    def read(self, text: str) -> dict:
        """Process text input through semantic encoding + brain routing.

        Returns encoded representation, routing decision, and memory storage.
        """
        self.state.last_input_type = "text"
        self.state.interaction_count += 1

        result = {"modality": "text", "text": text[:100]}

        # Encode text to VSA
        if self.text_encoder is not None:
            vec = self.text_encoder.encode_action(text)
            embedding = vec.ravel()[:self.thalamus.embedding_dim]
        else:
            # Hash fallback
            embedding = np.zeros(self.thalamus.embedding_dim, dtype=np.float32)
            for i, ch in enumerate(text.encode()[:self.thalamus.embedding_dim]):
                embedding[i] = float(ch) / 255.0

        if len(embedding) < self.thalamus.embedding_dim:
            embedding = np.pad(embedding, (0, self.thalamus.embedding_dim - len(embedding)))

        # Route through brain
        nc = self.snn.neurochemistry
        route = self.thalamus.route(embedding, arousal=nc.arousal, valence=nc.valence)
        strategy = self.basal_ganglia.select_strategy(
            route["routes"], valence=nc.valence, arousal=nc.arousal, stress=nc.stress,
        )

        # Personality styling
        styled, style_name = self.personality.forward(
            embedding, valence=nc.valence, arousal=nc.arousal, stress=nc.stress,
        )

        result["route"] = route
        result["strategy"] = strategy
        result["personality_style"] = style_name
        result["emotion"] = nc.dominant_emotion

        # Store in semantic memory
        projected = self.lsh.project(embedding[:self.lsh.d_input])
        packed = binarize_and_pack(projected)
        self.semantic_memory.learn(packed, label=f"read:{text[:50]}")

        return result

    def ask(self, question: str, objects: list | None = None) -> dict:
        """Answer a question about the current scene via VQA.

        Args:
            question: Natural language question.
            objects:  Optional list of DetectedObject. Uses current scene if None.

        Returns:
            VQA result with answer, confidence, and program trace.
        """
        if self.vqa is None:
            return {"answer": "VQA not available", "confidence": 0.0}

        if objects is not None and _DetectedObject is not None:
            self.vqa.set_scene(objects)

        result = self.vqa.answer(question)
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "program": result.program,
        }

    def respond(self, user_input: str) -> dict:
        """Full pipeline: understand input → route → generate response.

        This is the main interaction point. Determines input type,
        routes through the appropriate pipeline, and produces a
        response shaped by personality and emotional state.
        """
        self.state.interaction_count += 1
        nc = self.snn.neurochemistry

        # Determine input type
        if "?" in user_input:
            # Question — try VQA first, then text reasoning
            if self.vqa and self.vqa.scene.objects:
                vqa_result = self.ask(user_input)
                if vqa_result["confidence"] > 0.3:
                    return {
                        "type": "vqa",
                        "response": vqa_result["answer"],
                        "confidence": vqa_result["confidence"],
                        "emotion": nc.dominant_emotion,
                        "strategy": self.state.strategy,
                    }

        # Text processing
        text_result = self.read(user_input)

        # Build response metadata
        temporal = self.circadian.get_state()

        return {
            "type": "text",
            "strategy": text_result["strategy"]["strategy"],
            "personality_style": text_result["personality_style"],
            "emotion": nc.dominant_emotion,
            "consciousness": self.state.consciousness,
            "route": text_result["route"]["primary_route"],
            "temporal": temporal["time_of_day"],
            "interaction": self.state.interaction_count,
            # The actual text response would come from MoQE LLM here
            "response": self._generate_response(user_input, text_result),
        }

    def learn(self, label: str, frame: np.ndarray | None = None,
              audio: np.ndarray | None = None) -> dict:
        """Multi-modal learning: bind see + hear + feel into one concept.

        When you show it a bear and say "this is a bear", it binds:
          visual_vec XOR audio_vec XOR emotion_vec = concept "bear"

        Later, any single modality retrieves the others:
          see bear → retrieves word "bear" + emotion
          hear "bear" → retrieves visual pattern + emotion

        This is how humans learn: simultaneous multi-sensory experience
        fused into a single memory trace.

        Args:
            label:  What this concept is called (e.g., "bear").
            frame:  Optional image/frame of the concept.
            audio:  Optional audio of someone saying the label.

        Returns:
            Dict with the concept vector and what was bound.
        """
        self.state.interaction_count += 1
        modalities_bound = []

        # Start with the label as text
        if self.text_encoder is not None:
            text_vec = self.text_encoder.encode_action(label).ravel()
            text_proj = binarize_and_pack(self.lsh.project(
                text_vec[:self.lsh.d_input] if len(text_vec) >= self.lsh.d_input
                else np.pad(text_vec, (0, self.lsh.d_input - len(text_vec)))
            ))
            modalities_bound.append("text")
        else:
            # Hash fallback
            seed = hash(label) % (2**31)
            rng = np.random.default_rng(seed)
            text_proj = rng.integers(0, 2**32, size=self.episodic_memory.words_per_vec, dtype=np.uint32)
            modalities_bound.append("text_hash")

        concept_vec = text_proj.copy()

        # Bind visual modality if provided
        if frame is not None:
            visual_feat = self._grid_features(frame, self.lsh.d_input)
            if self.vision_encoder is not None:
                bc = self.vision_encoder.encode_image(frame)
                visual_feat = bc.ravel()[:self.lsh.d_input]
                if len(visual_feat) < self.lsh.d_input:
                    visual_feat = np.pad(visual_feat, (0, self.lsh.d_input - len(visual_feat)))
            visual_proj = binarize_and_pack(self.lsh.project(visual_feat))
            concept_vec = np.bitwise_xor(concept_vec, visual_proj)
            modalities_bound.append("vision")

            # Also process through SNN for neurochemistry
            snn_feat = self._grid_features(frame, 104)
            self.snn.step(snn_feat / (np.std(snn_feat) + 1e-6) * 0.3)

        # Bind audio modality if provided
        if audio is not None and self.audio_encoder is not None:
            audio_vec = self.audio_encoder.encode_audio(audio)
            concept_vec = np.bitwise_xor(concept_vec, audio_vec)
            modalities_bound.append("audio")

        # Bind current emotional state (how it FEELS about this concept)
        from cubemind.perception.scene import _emotion_to_vector
        emotion_vec = _emotion_to_vector(self.snn.neurochemistry, self.d_vsa)
        concept_vec = np.bitwise_xor(concept_vec, emotion_vec)
        modalities_bound.append("emotion")

        # Store the multi-modal concept
        self.semantic_memory.learn(concept_vec, label=label)

        # Also store each modality separately for cross-modal retrieval
        # see bear → query semantic memory → retrieve "bear" label
        if "vision" in modalities_bound:
            self.episodic_memory.learn(visual_proj, label=f"visual:{label}")
        if "audio" in modalities_bound:
            self.episodic_memory.learn(audio_vec, label=f"audio:{label}")

        return {
            "label": label,
            "modalities": modalities_bound,
            "emotion": self.snn.neurochemistry.dominant_emotion,
            "semantic_memories": self.semantic_memory.size,
            "episodic_memories": self.episodic_memory.size,
        }

    def recall(self, query_frame: np.ndarray | None = None,
               query_audio: np.ndarray | None = None,
               query_text: str | None = None,
               k: int = 3) -> list[dict]:
        """Cross-modal recall: given one modality, retrieve the full concept.

        See a bear → recall returns "bear" + associated emotion.
        Hear "bear" → recall returns visual memory + emotion.

        Args:
            query_frame: Image to search for.
            query_audio: Audio to search for.
            query_text:  Text label to search for.
            k:           Number of results.

        Returns:
            List of {label, similarity, memory_type} dicts.
        """
        results = []

        if query_frame is not None:
            feat = self._grid_features(query_frame, self.lsh.d_input)
            if self.vision_encoder is not None:
                bc = self.vision_encoder.encode_image(query_frame)
                feat = bc.ravel()[:self.lsh.d_input]
                if len(feat) < self.lsh.d_input:
                    feat = np.pad(feat, (0, self.lsh.d_input - len(feat)))
            packed = binarize_and_pack(self.lsh.project(feat))
            # Search both memories
            for mem, mem_type in [(self.episodic_memory, "episodic"), (self.semantic_memory, "semantic")]:
                if mem.size > 0:
                    hits = mem.retrieve(packed, k=k)
                    for idx, sim, label in hits:
                        results.append({"label": label, "similarity": sim, "memory": mem_type})

        if query_audio is not None and self.audio_encoder is not None:
            packed = self.audio_encoder.encode_audio(query_audio)
            for mem, mem_type in [(self.episodic_memory, "episodic"), (self.semantic_memory, "semantic")]:
                if mem.size > 0:
                    hits = mem.retrieve(packed, k=k)
                    for idx, sim, label in hits:
                        results.append({"label": label, "similarity": sim, "memory": mem_type})

        if query_text is not None and self.text_encoder is not None:
            vec = self.text_encoder.encode_action(query_text).ravel()
            feat = vec[:self.lsh.d_input]
            if len(feat) < self.lsh.d_input:
                feat = np.pad(feat, (0, self.lsh.d_input - len(feat)))
            packed = binarize_and_pack(self.lsh.project(feat))
            for mem, mem_type in [(self.semantic_memory, "semantic"), (self.episodic_memory, "episodic")]:
                if mem.size > 0:
                    hits = mem.retrieve(packed, k=k)
                    for idx, sim, label in hits:
                        results.append({"label": label, "similarity": sim, "memory": mem_type})

        # Deduplicate and sort by similarity
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["similarity"], reverse=True):
            key = f"{r['label']}:{r['memory']}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:k]

    def think(self) -> dict:
        """Internal reflection — consolidate memories, update personality.

        Called periodically or between interactions. The system reflects
        on accumulated experience and adjusts its personality weights.
        """
        nc = self.snn.neurochemistry

        # Hebbian update: reinforce current personality style
        embedding = np.random.default_rng().standard_normal(
            self.personality.d_model
        ).astype(np.float32) * 0.1
        _, style = self.personality.forward(
            embedding, valence=nc.valence, arousal=nc.arousal, stress=nc.stress,
        )
        # Strengthen the dominant style based on recent experience
        style_idx = list(self.personality.STYLES).index(style) if style in self.personality.STYLES else 0
        self.personality.hebbian_update(embedding, style_idx)

        # CNS update
        self.cns.update(arousal=nc.arousal, stress=nc.stress)

        return {
            "personality_style": style,
            "emotion": nc.dominant_emotion,
            "consciousness": self.state.consciousness,
            "episodic_memories": self.episodic_memory.size,
            "semantic_memories": self.semantic_memory.size,
            "taste_experiences": self.state.total_experiences,
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save entire mind state to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        self.snn.save(f"{path}_snn.npz")
        if self.episodic_memory.size > 0:
            self.episodic_memory.save(f"{path}_episodic")
        if self.semantic_memory.size > 0:
            self.semantic_memory.save(f"{path}_semantic")
        if self.scene_analyzer:
            self.scene_analyzer.save(f"{path}_scene")

        # Save mind state
        np.savez_compressed(
            f"{path}_mind.npz",
            interaction_count=np.int32(self.state.interaction_count),
            total_experiences=np.int32(self.state.total_experiences),
        )
        print(f"Mind saved to {path}_*")

    def load(self, path: str) -> None:
        """Load mind state from disk."""
        if os.path.exists(f"{path}_snn.npz"):
            self.snn.load(f"{path}_snn.npz")
        if os.path.exists(f"{path}_episodic.npz"):
            self.episodic_memory.load(f"{path}_episodic")
        if os.path.exists(f"{path}_semantic.npz"):
            self.semantic_memory.load(f"{path}_semantic")
        if self.scene_analyzer and os.path.exists(f"{path}_scene_taste.npz"):
            self.scene_analyzer.load(f"{path}_scene")
        if os.path.exists(f"{path}_mind.npz"):
            data = np.load(f"{path}_mind.npz")
            self.state.interaction_count = int(data["interaction_count"])
            self.state.total_experiences = int(data["total_experiences"])

        print(f"Mind loaded: {self.episodic_memory.size} episodic, "
              f"{self.semantic_memory.size} semantic memories")

    # ── Private helpers ───────────────────────────────────────────────────

    def _grid_features(self, frame: np.ndarray, dim: int) -> np.ndarray:
        """Extract spatial grid features from a frame."""
        if frame.ndim == 3:
            gray = np.mean(frame.astype(np.float32), axis=2) / 255.0
        else:
            gray = frame.astype(np.float32)
            if gray.max() > 1.0:
                gray /= 255.0
        h, w = gray.shape[:2]
        grid_n = max(2, int(np.sqrt(dim / 4)))
        cell_h, cell_w = max(1, h // grid_n), max(1, w // grid_n)
        features = []
        for i in range(grid_n):
            for j in range(grid_n):
                cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                if cell.size > 0:
                    features.extend([float(np.mean(cell)), float(np.std(cell)),
                                     float(np.max(cell)), float(np.min(cell))])
        feat = np.array(features[:dim], dtype=np.float32)
        if len(feat) < dim:
            feat = np.pad(feat, (0, dim - len(feat)))
        return feat

    @staticmethod
    def _emotion_to_valence(emotion: str | None) -> float:
        return {"happy": 0.6, "surprised": 0.2, "warm": 0.4,
                "sad": -0.4, "angry": -0.5, "fearful": -0.3}.get(emotion or "", 0.0)

    def _generate_response(self, user_input: str, context: dict) -> str:
        """Generate a response based on brain state.

        Currently template-based. With MoQE LLM, this would be:
        VSA memory → adapter → LLM embedding → autoregressive generation.
        """
        strategy = context["strategy"]["strategy"]
        emotion = self.snn.neurochemistry.dominant_emotion
        style = context["personality_style"]
        temporal = self.circadian.get_state()

        # Template responses shaped by personality + strategy + emotion
        prefix = ""
        if strategy == "empathetic":
            prefix = "I understand. "
        elif strategy == "questioning":
            prefix = "Interesting — "
        elif strategy == "action":
            prefix = "Let me help. "

        # This is where MoQE LLM would generate the actual text
        return (f"[{style}/{strategy}] {prefix}"
                f"(emotion: {emotion}, route: {context['route']['primary_route']}) "
                f"Processing: '{user_input[:80]}'")

    @property
    def neurochemistry(self) -> NeurochemicalState:
        return self.snn.neurochemistry

    @property
    def emotion(self) -> str:
        return self.snn.neurochemistry.dominant_emotion

    def __repr__(self) -> str:
        nc = self.snn.neurochemistry
        return (f"CubeMind(emotion={nc.dominant_emotion}, "
                f"consciousness={self.state.consciousness}, "
                f"memories={self.episodic_memory.size + self.semantic_memory.size}, "
                f"interactions={self.state.interaction_count})")
