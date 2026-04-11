"""CubeMind — Integrated Neuro-Vector-Symbolic Cognitive Architecture.

Perception (vision + hearing + text)
    ↓
SNN Processing (GIFNeuron + Synapsis + STDP)
    ↓
Neurochemistry (5-hormone ODE modulates routing)
    ↓
HippocampalFormation (store/retrieve episodic memories)
    ↓
Neurogenesis (grow/prune neurons based on residual error)
    ↓
Output (VSA block-code answer)

All modules are independent and fault-isolated. If any module fails at
init or runtime, CubeMind degrades gracefully — the pipeline continues
without the failed module. This is enforced via try/except boundaries
around every module call.

All ops route through grilly when available.

Usage:
    # Via DI container (recommended):
    from cubemind.container import CubeMindContainer
    container = CubeMindContainer()
    container.config.from_dict({"k": 8, "l": 64, "d_hidden": 64})
    brain = container.cubemind()

    # Via factory (quick):
    from cubemind.model import create_cubemind
    brain = create_cubemind(k=8, l=64)

    # Direct (full control):
    brain = CubeMind(bc=my_bc, text_encoder=my_enc, ...)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK

# Re-export Oja-plastic attention engine (used by tests and reasoning pipeline)
from cubemind._archive.model_v2 import HyperAxialAttention, PlasticCodebook  # noqa: F401

logger = logging.getLogger(__name__)


# ── Oja Plasticity Kernels ──────────────────────────────────────────────────


def oja_update(m: np.ndarray, x: np.ndarray, eta: float = 0.01) -> np.ndarray:
    m_flat = m.ravel().astype(np.float32)
    x_flat = x.ravel().astype(np.float32)
    y = float(np.dot(m_flat, x_flat))
    m_flat = m_flat + (eta * y) * (x_flat - y * m_flat)
    return m_flat.reshape(m.shape)


def oja_update_batch(
    memories: np.ndarray, inputs: np.ndarray, eta: float = 0.01,
) -> np.ndarray:
    y = np.sum(memories * inputs, axis=-1, keepdims=True)
    updated = memories + (eta * y) * (inputs - y * memories)
    return updated.astype(np.float32)


# ── CubeMind Orchestrator ──────────────────────────────────────────────────


class CubeMind:
    """Integrated cognitive architecture with fault-isolated modules.

    Every module is optional and independently recoverable. If a module
    is None or raises at runtime, that pipeline stage is skipped and
    the system continues with degraded output.

    Args:
        bc: BlockCodes VSA ops (required).
        text_encoder: Text → VSA encoder (required).
        harrier_encoder: Harrier embedding encoder (optional, None = skip).
        vision_encoder: Bio-vision encoder (optional, None = skip).
        audio_encoder: Audio encoder (optional, None = skip).
        snn_ffn: Hybrid SNN/FFN processor (required).
        hippocampus: HippocampalFormation memory (required).
        neurochemistry: Neurochemistry ODE modulator (optional, None = skip).
        neurogenesis: NeurogenesisController (required).
        spike_bridge: Spike↔VSA bridge (required).
        k: VSA block count.
        l: VSA block length.
        d_hidden: Hidden dimension for SNN/FFN layers.
        seed: Random seed.
    """

    def __init__(
        self,
        bc: Any,
        text_encoder: Any,
        snn_ffn: Any,
        hippocampus: Any,
        neurogenesis: Any,
        spike_bridge: Any,
        harrier_encoder: Any | None = None,
        vision_encoder: Any | None = None,
        audio_encoder: Any | None = None,
        neurochemistry: Any | None = None,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        d_hidden: int = 128,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.d_hidden = d_hidden

        # Required modules
        self.bc = bc
        self.text_encoder = text_encoder
        self.snn_ffn = snn_ffn
        self.hippocampus = hippocampus
        self.neurogenesis = neurogenesis
        self.spike_bridge = spike_bridge

        # Optional modules (None = disabled)
        self._harrier = harrier_encoder
        self._vision = vision_encoder
        self._audio = audio_encoder
        self._neurochemistry = neurochemistry

        # Projections: d_vsa ↔ d_hidden
        rng = np.random.default_rng(seed)
        std_in = np.sqrt(2.0 / (self.d_vsa + d_hidden))
        std_out = np.sqrt(2.0 / (d_hidden + self.d_vsa))
        self._proj_in = rng.normal(0, std_in, (d_hidden, self.d_vsa)).astype(np.float32)
        self._proj_out = rng.normal(0, std_out, (self.d_vsa, d_hidden)).astype(np.float32)

        # LLM (attached later via attach_llm)
        self._llm = None
        self._injector = None

        # State
        self._step_count = 0
        self._last_output = None

    # ── Forward Pass ─────────────────────────────────────────────────────

    def forward(
        self,
        text: str | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
        phi: np.ndarray | None = None,
        location: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Full forward pass through the cognitive architecture.

        Each module is wrapped in its own error boundary. If any module
        fails, the pipeline continues — that stage is skipped and a
        warning is logged.

        Args:
            text: Input text.
            image: Image frame (BGR, any size).
            audio: Audio chunk (float32 PCM).
            phi: Pre-encoded VSA block-code (k, l). Overrides text/image.
            location: Spatial location (2D) for hippocampal context.

        Returns:
            Dict with: output_hv, confidence, input_hv, hidden,
            memories_retrieved, neurogenesis, neurochemistry,
            spatial_context, temporal_context, step.
        """
        self._step_count += 1

        # ── 1. Perception: each modality in its own error boundary ───────
        modality_hvs: list[np.ndarray] = []

        if phi is not None:
            modality_hvs.append(phi)

        if text is not None:
            hv = self._safe_call(
                "harrier_encoder",
                lambda: self._harrier.encode(text) if self._harrier else None,
            )
            if hv is None:
                hv = self._safe_call(
                    "text_encoder",
                    lambda: self.text_encoder.encode(text),
                )
            if hv is not None:
                modality_hvs.append(hv)

        if image is not None:
            hv = self._safe_call(
                "vision_encoder",
                lambda: self._vision.encode(image) if self._vision else None,
            )
            if hv is not None:
                modality_hvs.append(hv)

        if audio is not None:
            hv = self._safe_call(
                "audio_encoder",
                lambda: self._audio.encode_audio(audio) if self._audio else None,
            )
            if hv is not None:
                modality_hvs.append(hv)

        # Fuse modalities
        if not modality_hvs:
            input_hv = self.bc.random_discrete(seed=self._step_count)
        elif len(modality_hvs) == 1:
            input_hv = modality_hvs[0]
        else:
            bundled = np.zeros((self.k, self.l), dtype=np.float32)
            for hv in modality_hvs:
                bundled += np.asarray(hv, dtype=np.float32)
            input_hv = self.bc.discretize(bundled)

        input_flat = self.bc.to_flat(input_hv).astype(np.float32)

        # ── 2. Project to hidden dim ─────────────────────────────────────
        h = (self._proj_in @ input_flat).astype(np.float32)

        # ── 3. Hippocampal spatial/temporal context ──────────────────────
        spatial_ctx = {}
        temporal_ctx = {}
        if location is not None:
            self._safe_call(
                "hippocampus.spatial",
                lambda: self.hippocampus.update_spatial_state(location),
            )
        spatial_ctx = self._safe_call(
            "hippocampus.spatial_ctx",
            lambda: self.hippocampus.get_spatial_context(),
        ) or {}
        temporal_ctx = self._safe_call(
            "hippocampus.temporal_ctx",
            lambda: self.hippocampus.get_temporal_context(),
        ) or {}

        # ── 4. Memory retrieval ──────────────────────────────────────────
        retrieved = self._safe_call(
            "hippocampus.retrieve",
            lambda: self.hippocampus.retrieve_similar_memories(h, k=5),
        ) or []

        if retrieved:
            for mem_id, score in retrieved[:3]:
                try:
                    if mem_id in self.hippocampus.id_to_idx:
                        idx = self.hippocampus.id_to_idx[mem_id]
                        mem_feat = self.hippocampus.memory_features[idx]
                        h = h + score * 0.1 * mem_feat
                except Exception:
                    pass

        # ── 5. SNN processing ────────────────────────────────────────────
        h_processed = self._safe_call(
            "snn_ffn",
            lambda: self.snn_ffn.forward(h.reshape(1, 1, -1)).reshape(-1),
        )
        if h_processed is None:
            h_processed = h  # fallback: pass through

        # ── 6. Neurochemistry modulation ─────────────────────────────────
        neuro_state: dict[str, Any] = {}
        if self._neurochemistry is not None:
            neuro_state = self._safe_call("neurochemistry", lambda: self._neuro_step(
                h_processed, retrieved,
            )) or {}

        # ── 7. Neurogenesis ──────────────────────────────────────────────
        neuro_info = self._safe_call(
            "neurogenesis",
            lambda: self.neurogenesis.step(h_processed),
        ) or {}

        # ── 8. Store in hippocampal memory ───────────────────────────────
        self._safe_call(
            "hippocampus.store",
            lambda: self.hippocampus.create_episodic_memory(features=h_processed),
        )

        # ── 9. Project back to VSA space ─────────────────────────────────
        output_flat = (self._proj_out @ h_processed).astype(np.float32)
        output_hv = self.bc.discretize(self.bc.from_flat(output_flat, self.k))

        # ── 10. Confidence ───────────────────────────────────────────────
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

    # ── Recall ───────────────────────────────────────────────────────────

    def recall(self, query: str | np.ndarray, k: int = 5) -> list:
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

    # ── LLM Attachment ───────────────────────────────────────────────────

    def attach_llm(
        self,
        model_path: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        inject_layers: bool = True,
        injection_strength: float = 0.1,
        use_mindforge: bool = False,
        **kwargs: Any,
    ) -> None:
        """Attach an LLM for language generation with brain-state injection."""
        from cubemind._archive.brain.llm_interface import LLMInterface
        self._llm = LLMInterface(
            model_path=model_path, api_url=api_url,
            api_key=api_key, **kwargs,
        )
        if inject_layers:
            from cubemind._archive.brain.llm_injector import LLMInjector
            self._injector = LLMInjector(
                brain=self, n_layers=32, d_model=4096,
                d_brain=self.d_hidden,
                injection_strength=injection_strength,
                use_mindforge=use_mindforge,
                k=self.k, l=self.l,
            )
            self._llm.attach_injector(self._injector)

    # ── Think (perceive → reason → speak) ────────────────────────────────

    def think(
        self,
        prompt: str,
        text: str | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
        location: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Full cognitive cycle: perceive → reason → remember → speak."""
        result = self.forward(
            text=text or prompt, image=image,
            audio=audio, location=location,
        )
        if self._injector is not None:
            self._safe_call("llm_injector", lambda: self._injector.update_brain_state(
                brain_hidden=result.get("hidden"),
                brain_hv=result.get("input_hv"),
                neurochemistry=result.get("neurochemistry"),
            ))

        response = ""
        if self._llm is not None and self._llm.available:
            response = self._safe_call("llm_generate", lambda: self._llm.generate(
                prompt=prompt, context=result,
                memories=[f"Memory {m} (score={s:.2f})" for m, s in self.recall(prompt, k=3)],
                neurochemistry=result.get("neurochemistry"),
            )) or ""
        result["response"] = response
        return result

    # ── Training ─────────────────────────────────────────────────────────

    def train_step(
        self,
        text: str | None = None,
        image: np.ndarray | None = None,
        audio: np.ndarray | None = None,
        target_hv: np.ndarray | None = None,
        location: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """One training step — forward + STDP + neurogenesis + memory."""
        result = self.forward(text=text, image=image, audio=audio, location=location)
        loss = 0.0
        similarity = 0.0
        if target_hv is not None:
            similarity = float(self.bc.similarity(result["output_hv"], target_hv))
            loss = 1.0 - similarity
        result["loss"] = loss
        result["similarity"] = similarity
        return result

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "hippocampus": self._safe_call("hippocampus.stats",
                                           lambda: self.hippocampus.stats()) or {},
            "neurogenesis": self._safe_call("neurogenesis.stats",
                                            lambda: self.neurogenesis.stats()) or {},
            "d_vsa": self.d_vsa,
            "d_hidden": self.d_hidden,
            "harrier": self._harrier is not None,
            "vision": self._vision is not None,
            "audio": self._audio is not None,
            "neurochemistry": self._neurochemistry is not None,
        }

    # ── Fault Isolation ──────────────────────────────────────────────────

    def _safe_call(self, module_name: str, fn, default=None):
        """Call fn() in an error boundary. Log and return default on failure."""
        try:
            return fn()
        except Exception as e:
            logger.warning("Module %s failed: %s", module_name, e)
            return default

    def _neuro_step(self, h: np.ndarray, retrieved: list) -> dict:
        """Run neurochemistry step. Isolated from forward() for clarity."""
        novelty = float(1.0 - max(s for _, s in retrieved) if retrieved else 1.0)
        intensity = float(np.linalg.norm(h))
        self._neurochemistry.step(
            novelty=novelty, threat=0.0,
            focus=min(intensity / 10.0, 1.0),
            valence=0.5, dt=0.05,
        )
        return self._neurochemistry.get_state()


# ── Factory Function ─────────────────────────────────────────────────────


def create_cubemind(
    k: int = K_BLOCKS,
    l: int = L_BLOCK,
    d_hidden: int = 128,
    seed: int = 42,
    **config_overrides: Any,
) -> CubeMind:
    """Create a CubeMind with default wiring via DI container.

    This is the simplest way to get a working CubeMind instance.
    Pass config_overrides to customize any container config value.

    Args:
        k: VSA block count.
        l: VSA block length.
        d_hidden: Hidden dimension.
        seed: Random seed.
        **config_overrides: Additional config values (e.g., enable_vision=False).

    Returns:
        Fully wired CubeMind instance.
    """
    from cubemind.container import CubeMindContainer

    config = {"k": k, "l": l, "d_hidden": d_hidden, "seed": seed}
    config.update(config_overrides)

    container = CubeMindContainer()
    container.config.from_dict(config)
    return container.cubemind()
