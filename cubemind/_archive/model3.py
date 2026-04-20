"""CubeMind v3 — Integrated Neuro-Vector-Symbolic Cognitive Architecture.

Wires together all brain modules into a unified perception → reasoning → memory loop:

  Perception (vision + hearing + text)
      ↓
  SNN Processing (GIFNeuron + Synapsis + STDP)
      ↓
  Neurochemistry (5-hormone ODE modulates routing)
      ↓
  Routing (entropy-gated MoQE or AdditionLinear)
      ↓
  HybridFFN (MLP + SNN blend via learnable gate)
      ↓
  HippocampalFormation (store/retrieve episodic memories)
      ↓
  Neurogenesis (grow/prune neurons based on residual error)
      ↓
  DecisionOracle (many-worlds Active Inference)
      ↓
  Output (VSA block-code answer)

All params configurable. All ops through grilly when available.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS, L_BLOCK = 80, 128


class CubeMindV3:
    """Integrated cognitive architecture.

    Args:
        k: VSA blocks.
        l: Block length.
        d_hidden: Hidden dimension for SNN/FFN layers.
        n_gif_levels: GIF multi-bit spike levels.
        gif_tau: GIF membrane time constant.
        gif_threshold: GIF firing threshold.
        gif_alpha: GIF threshold adaptation rate.
        snn_timesteps: Spike timesteps per position.
        snn_ratio: HybridFFN SNN blend ratio (0=MLP, 1=SNN).
        enable_stdp: Enable STDP on synapses.
        n_place_cells: HippocampalFormation place cells.
        n_time_cells: HippocampalFormation time cells.
        n_grid_cells: HippocampalFormation grid cells.
        max_memories: Memory bank capacity.
        initial_neurons: NeurogenesisController starting neurons.
        max_neurons: NeurogenesisController max neurons.
        growth_threshold: Residual EMA threshold for neurogenesis.
        enable_neurochemistry: Enable 5-hormone neurochemistry ODE.
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        d_hidden: int = 128,
        n_gif_levels: int = 16,
        gif_tau: float = 10.0,
        gif_threshold: float = 1.0,
        gif_alpha: float = 0.01,
        snn_timesteps: int = 4,
        snn_ratio: float = 0.5,
        enable_stdp: bool = True,
        n_place_cells: int = 2000,
        n_time_cells: int = 100,
        n_grid_cells: int = 200,
        max_memories: int = 100_000,
        initial_neurons: int = 64,
        max_neurons: int = 10000,
        growth_threshold: float = 0.35,
        enable_neurochemistry: bool = True,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.d_hidden = d_hidden

        self.bc = BlockCodes(k=k, l=l)

        # ── Perception ───────────────────────────────────────────────────
        from cubemind.perception.encoder import Encoder
        self.text_encoder = Encoder(k=k, l=l)

        # Harrier embedding (if available)
        self._harrier = None
        try:
            from cubemind.perception.harrier_encoder import HarrierEncoder
            self._harrier = HarrierEncoder(k=k, l=l)
        except Exception:
            pass

        # Vision (bio-inspired)
        self._vision = None
        try:
            from cubemind.perception.bio_vision import BioVisionEncoder
            self._vision = BioVisionEncoder(k=k, l=l)
        except Exception:
            pass

        # Audio
        self._audio = None
        try:
            from cubemind.perception.audio import AudioEncoder
            self._audio = AudioEncoder(
                sample_rate=16000, n_mels=40,
                snn_neurons=d_hidden, d_vsa=self.d_vsa, seed=seed,
            )
        except Exception:
            pass

        # ── SNN Processing ───────────────────────────────────────────────
        from cubemind.brain.snn_ffn import HybridFFN
        self.snn_ffn = HybridFFN(
            input_dim=d_hidden, hidden_dim=d_hidden * 2,
            snn_ratio=snn_ratio, num_timesteps=snn_timesteps,
            L=n_gif_levels, tau=gif_tau, threshold=gif_threshold,
            alpha=gif_alpha, enable_stdp=enable_stdp, seed=seed,
        )

        # Input projection: d_vsa → d_hidden
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (self.d_vsa + d_hidden))
        self._proj_in = rng.normal(0, std, (d_hidden, self.d_vsa)).astype(np.float32)

        # Output projection: d_hidden → d_vsa
        std_out = np.sqrt(2.0 / (d_hidden + self.d_vsa))
        self._proj_out = rng.normal(0, std_out, (self.d_vsa, d_hidden)).astype(np.float32)

        # ── Neurochemistry ───────────────────────────────────────────────
        self._neurochemistry = None
        if enable_neurochemistry:
            try:
                from cubemind.brain.neurochemistry import Neurochemistry
                self._neurochemistry = Neurochemistry()
            except Exception:
                pass

        # ── Hippocampal Memory ───────────────────────────────────────────
        from cubemind.memory.formation import HippocampalFormation
        self.hippocampus = HippocampalFormation(
            spatial_dimensions=2,
            n_place_cells=n_place_cells,
            n_time_cells=n_time_cells,
            n_grid_cells=n_grid_cells,
            max_memories=max_memories,
            feature_dim=d_hidden,
            seed=seed,
        )

        # ── Neurogenesis ─────────────────────────────────────────────────
        from cubemind.brain.neurogenesis import NeurogenesisController
        self.neurogenesis = NeurogenesisController(
            initial_neurons=initial_neurons,
            max_neurons=max_neurons,
            feature_dim=d_hidden,
            growth_threshold=growth_threshold,
            seed=seed,
        )

        # ── LLM Interface ────────────────────────────────────────────────
        self._llm = None

        # ── Spike↔VSA Bridge ────────────────────────────────────────────
        from cubemind.brain.spike_vsa_bridge import SpikeVSABridge
        self.spike_bridge = SpikeVSABridge(k=k, l=l, seed=seed)

        # ── State ────────────────────────────────────────────────────────
        self._step_count = 0
        self._last_output = None

    def forward(
        self,
        text: str | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
        phi: np.ndarray | None = None,
        location: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """Full forward pass through the cognitive architecture.

        Args:
            text: Input text (encoded via Harrier or hash encoder).
            image: Image frame (BGR, any size) for bio-vision.
            audio: Audio chunk (float32 PCM) for audio perception.
            phi: Pre-encoded VSA block-code (k, l). Overrides text/image.
            location: Spatial location (2D) for hippocampal context.

        Returns:
            Dict with: output_hv, confidence, memories, neurogenesis_info,
                       neurochemistry, spatial_context, temporal_context.
        """
        self._step_count += 1

        # ── 1. Perception: all modalities fire in parallel, bind together ─
        modality_hvs = []

        if phi is not None:
            modality_hvs.append(phi)

        if text is not None:
            if self._harrier is not None:
                modality_hvs.append(self._harrier.encode(text))
            else:
                modality_hvs.append(self.text_encoder.encode(text))

        if image is not None and self._vision is not None:
            modality_hvs.append(self._vision.encode(image))

        if audio is not None and self._audio is not None:
            modality_hvs.append(self._audio.encode_audio(audio))

        # Fuse modalities: bundle all, then discretize
        if not modality_hvs:
            input_hv = self.bc.random_discrete(seed=self._step_count)
        elif len(modality_hvs) == 1:
            input_hv = modality_hvs[0]
        else:
            # Bundle: superposition of all modalities in VSA space
            bundled = np.zeros((self.k, self.l), dtype=np.float32)
            for hv in modality_hvs:
                bundled += np.asarray(hv, dtype=np.float32)
            input_hv = self.bc.discretize(bundled)

        # Flatten to dense vector
        input_flat = self.bc.to_flat(input_hv).astype(np.float32)

        # ── 2. Project to hidden dim ─────────────────────────────────────
        h = (self._proj_in @ input_flat).astype(np.float32)

        # ── 3. Hippocampal spatial/temporal context ──────────────────────
        if location is not None:
            self.hippocampus.update_spatial_state(location)
        spatial_ctx = self.hippocampus.get_spatial_context()
        temporal_ctx = self.hippocampus.get_temporal_context()

        # ── 4. Memory retrieval (RAG) ────────────────────────────────────
        retrieved = self.hippocampus.retrieve_similar_memories(h, k=5)
        if retrieved:
            # Inject memory via additive gating
            for mem_id, score in retrieved[:3]:
                if mem_id in self.hippocampus.id_to_idx:
                    idx = self.hippocampus.id_to_idx[mem_id]
                    mem_feat = self.hippocampus.memory_features[idx]
                    h = h + score * 0.1 * mem_feat  # gated injection

        # ── 5. SNN processing (HybridFFN: MLP + spiking blend) ──────────
        h_seq = h.reshape(1, 1, -1)  # (1, 1, d_hidden)
        h_processed = self.snn_ffn.forward(h_seq).reshape(-1)

        # ── 6. Neurochemistry modulation ─────────────────────────────────
        neuro_state = {}
        if self._neurochemistry is not None:
            # Compute drives from processing
            novelty = float(1.0 - max(s for _, s in retrieved) if retrieved else 1.0)
            intensity = float(np.linalg.norm(h_processed))
            self._neurochemistry.step(
                novelty=novelty,
                threat=0.0,
                focus=min(intensity / 10.0, 1.0),
                valence=0.5,
                dt=0.05,
            )
            neuro_state = self._neurochemistry.get_state()

        # ── 7. Neurogenesis update ───────────────────────────────────────
        neuro_info = self.neurogenesis.step(h_processed)

        # ── 8. Store in hippocampal memory ───────────────────────────────
        self.hippocampus.create_episodic_memory(features=h_processed)

        # ── 9. Project back to VSA space ─────────────────────────────────
        output_flat = (self._proj_out @ h_processed).astype(np.float32)
        output_hv = self.bc.discretize(
            self.bc.from_flat(output_flat, self.k))

        # ── 10. Confidence from similarity to input ──────────────────────
        confidence = float(self.bc.similarity(input_hv, output_hv))

        self._last_output = output_hv

        return {
            "output_hv": output_hv,
            "confidence": confidence,
            "input_hv": input_hv,
            "hidden": h_processed,
            "memories_retrieved": len(retrieved),
            "neurogenesis": neuro_info,
            "neurochemistry": neuro_state,
            "spatial_context": spatial_ctx,
            "temporal_context": temporal_ctx,
            "step": self._step_count,
        }

    def recall(self, query: str | np.ndarray, k: int = 5) -> List:
        """Recall memories by text query or block-code."""
        if isinstance(query, str):
            if self._harrier:
                hv = self._harrier.encode(query)
            else:
                hv = self.text_encoder.encode(query)
            flat = self.bc.to_flat(hv)
            h = (self._proj_in @ flat).astype(np.float32)
        else:
            h = np.asarray(query, dtype=np.float32).ravel()
            if len(h) == self.d_vsa:
                h = (self._proj_in @ h).astype(np.float32)
        return self.hippocampus.retrieve_similar_memories(h, k=k)

    def attach_llm(
        self,
        model_path: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        inject_layers: bool = True,
        injection_strength: float = 0.1,
        use_mindforge: bool = False,
        **kwargs,
    ) -> None:
        """Attach an LLM for language generation with brain-state injection.

        Args:
            model_path: Path to GGUF model file.
            api_url: OpenAI-compatible API URL.
            api_key: API key.
            inject_layers: Enable brain-state injection into LLM layers.
            injection_strength: How much brain state influences LLM.
            use_mindforge: Use MindForge LoRA adapters for injection.
            **kwargs: Additional args for LLMInterface.
        """
        from cubemind.brain.llm_interface import LLMInterface
        self._llm = LLMInterface(
            model_path=model_path, api_url=api_url,
            api_key=api_key, **kwargs,
        )

        if inject_layers:
            from cubemind.brain.llm_injector import LLMInjector
            self._injector = LLMInjector(
                brain=self,
                n_layers=32,  # Llama3.3-8B = 32 layers
                d_model=4096,  # Llama3.3-8B = 4096 hidden
                d_brain=self.d_hidden,
                injection_strength=injection_strength,
                use_mindforge=use_mindforge,
                k=self.k, l=self.l,
            )
            self._llm.attach_injector(self._injector)
        else:
            self._injector = None

    def think(
        self,
        prompt: str,
        text: str | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
        location: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """Full cognitive cycle: perceive → reason → remember → speak.

        Runs forward() for perception/memory, then generates language via LLM.

        Args:
            prompt: What to think about / respond to.
            text: Additional text context.
            image: Visual input.
            audio: Audio input.
            location: Spatial location.

        Returns:
            Dict with: response (str), plus all forward() outputs.
        """
        # Perceive and process
        result = self.forward(
            text=text or prompt, image=image,
            audio=audio, location=location,
        )

        # Update injector with current brain state
        if hasattr(self, '_injector') and self._injector is not None:
            self._injector.update_brain_state(
                brain_hidden=result.get("hidden"),
                brain_hv=result.get("input_hv"),
                neurochemistry=result.get("neurochemistry"),
            )

        # Generate language response
        response = ""
        if self._llm is not None and self._llm.available:
            # Retrieve memory summaries
            memories = self.recall(prompt, k=3)
            mem_strs = [f"Memory {mid} (score={s:.2f})" for mid, s in memories]

            response = self._llm.generate(
                prompt=prompt,
                context=result,
                memories=mem_strs,
                neurochemistry=result.get("neurochemistry"),
            )

            # Store the response as a memory too
            if response:
                resp_hv = self.text_encoder.encode(response)
                resp_flat = self.bc.to_flat(resp_hv)
                h = (self._proj_in @ resp_flat).astype(np.float32)
                self.hippocampus.create_episodic_memory(features=h)

        result["response"] = response
        return result

    def train_step(
        self,
        text: str | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
        target_hv: np.ndarray | None = None,
        location: np.ndarray | None = None,
        extract_logits: bool = False,
    ) -> Dict[str, Any]:
        """One training step — forward + STDP + neurogenesis + memory + logits.

        The whole system trains together:
        - Perception encodes input to VSA (all modalities in parallel)
        - SNN processes with STDP (if enabled)
        - Neurogenesis grows/prunes based on residual
        - Hippocampus stores episode
        - If target_hv provided, compute VSA similarity loss
        - If extract_logits and LLM attached, extract teacher logits on the fly

        Args:
            text: Text input.
            image: Visual input.
            audio: Audio input.
            target_hv: Target VSA block-code for supervision.
            location: Spatial location.
            extract_logits: Extract LLM logits for MoQE distillation.

        Returns:
            Dict with: loss, similarity, neurogenesis info, teacher_logits, etc.
        """
        result = self.forward(
            text=text, image=image, audio=audio, location=location,
        )

        loss = 0.0
        similarity = 0.0
        if target_hv is not None:
            similarity = float(self.bc.similarity(result["output_hv"], target_hv))
            loss = 1.0 - similarity

        # Extract LLM teacher logits for live distillation
        teacher_logits = None
        if extract_logits and text and self._llm is not None:
            teacher_logits = self._llm.get_logits(text)

        result["loss"] = loss
        result["similarity"] = similarity
        result["teacher_logits"] = teacher_logits
        return result

    def stats(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "hippocampus": self.hippocampus.stats(),
            "neurogenesis": self.neurogenesis.stats(),
            "d_vsa": self.d_vsa,
            "d_hidden": self.d_hidden,
            "harrier": self._harrier is not None,
            "vision": self._vision is not None,
            "audio": self._audio is not None,
            "neurochemistry": self._neurochemistry is not None,
        }
