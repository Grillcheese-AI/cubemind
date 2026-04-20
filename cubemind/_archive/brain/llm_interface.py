"""LLM Interface — Hook external language model into CubeMind's brain.

Bridges CubeMind's VSA/SNN cognitive architecture with an LLM for
language generation. The LLM handles text generation while CubeMind
handles perception, memory, reasoning, and neurochemistry.

Pipeline:
  CubeMind perception → VSA encoding → context retrieval from hippocampus
  → LLM generates with context → response fed back into memory

Supports:
  - llama-cpp-python (GGUF models, local GPU/CPU)
  - Any OpenAI-compatible API

Usage:
    llm = LLMInterface(model_path="path/to/model.gguf")
    response = llm.generate("What do you see?", context=brain_context)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class LLMInterface:
    """External LLM hook for CubeMind language capabilities.

    Args:
        model_path: Path to GGUF model file.
        n_ctx: Context window size.
        n_gpu_layers: Layers to offload to GPU (-1 = all).
        temperature: Generation temperature.
        max_tokens: Max tokens per generation.
        system_prompt: System prompt for the LLM.
        api_url: OpenAI-compatible API URL (if using API instead of local).
        api_key: API key (if using API).
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = api_url
        self.api_key = api_key

        self.system_prompt = system_prompt or (
            "You are CubeMind, a neuro-vector-symbolic cognitive architecture. "
            "You perceive the world through bio-inspired vision, hearing, and "
            "spiking neural networks. You have episodic memory with place cells "
            "and time cells. Your reasoning uses vector symbolic algebra. "
            "Respond naturally based on your perceptions and memories."
        )

        self._llm = None
        self._mode = "none"
        self._injector = None

        # Try llama-cpp-python first
        if model_path is not None:
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
                self._mode = "llama_cpp"
            except ImportError:
                pass
            except Exception as e:
                print(f"[LLM] Failed to load GGUF: {e}")

        # Fallback to API
        if self._llm is None and api_url is not None:
            self._mode = "api"

    def attach_injector(self, injector) -> None:
        """Attach a LLMInjector for brain-state layer injection.

        Args:
            injector: LLMInjector instance.
        """
        self._injector = injector

    @property
    def available(self) -> bool:
        return self._mode != "none"

    def generate(
        self,
        prompt: str,
        context: Dict[str, Any] | None = None,
        memories: List[str] | None = None,
        neurochemistry: Dict[str, float] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text with CubeMind context injection.

        Args:
            prompt: User prompt.
            context: CubeMind brain state (spatial, temporal, perception).
            memories: Retrieved episodic memories as text.
            neurochemistry: Current hormone levels (modulate generation).
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Generated text.
        """
        if not self.available:
            return "[LLM not available — install llama-cpp-python or configure API]"

        temp = temperature or self.temperature
        max_tok = max_tokens or self.max_tokens

        # Modulate temperature by neurochemistry
        if neurochemistry:
            # High dopamine → more creative (higher temp)
            # High cortisol → more cautious (lower temp)
            da = neurochemistry.get("dopamine", 0.5)
            cortisol = neurochemistry.get("cortisol", 0.3)
            temp = temp * (0.8 + 0.4 * da) * (1.2 - 0.4 * cortisol)
            temp = max(0.1, min(2.0, temp))

        # Build context-enriched prompt
        full_prompt = self._build_prompt(prompt, context, memories)

        if self._mode == "llama_cpp":
            return self._generate_llama(full_prompt, temp, max_tok)
        elif self._mode == "api":
            return self._generate_api(full_prompt, temp, max_tok)
        return ""

    def _build_prompt(
        self,
        prompt: str,
        context: Dict[str, Any] | None,
        memories: List[str] | None,
    ) -> str:
        """Build context-enriched prompt for the LLM."""
        parts = []

        if context:
            ctx_str = []
            if "spatial_context" in context:
                loc = context["spatial_context"].get("current_location", [0, 0])
                n_mem = context["spatial_context"].get("n_memories", 0)
                ctx_str.append(f"Location: ({loc[0]:.1f}, {loc[1]:.1f}), "
                               f"Memories: {n_mem}")
            if "neurochemistry" in context and context["neurochemistry"]:
                nc = context["neurochemistry"]
                ctx_str.append(
                    f"State: DA={nc.get('dopamine', 0):.2f} "
                    f"5HT={nc.get('serotonin', 0):.2f} "
                    f"C={nc.get('cortisol', 0):.2f}")
            if "neurogenesis" in context:
                ng = context["neurogenesis"]
                ctx_str.append(f"Neurons: {ng.get('neuron_count', '?')}")
            if ctx_str:
                parts.append(f"[Brain State: {' | '.join(ctx_str)}]")

        if memories:
            parts.append(f"[Episodic Memories: {'; '.join(memories[:5])}]")

        parts.append(prompt)

        return "\n".join(parts)

    def _generate_llama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate via llama-cpp-python."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM error: {e}]"

    def _generate_api(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate via OpenAI-compatible API."""
        try:
            import httpx
            response = httpx.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[API error: {e}]"

    def get_logits(self, text: str) -> np.ndarray | None:
        """Get raw logits from the LLM for a given text.

        Used for MoQE distillation — extract teacher logits on the fly.

        Args:
            text: Input text.

        Returns:
            Logits array (seq_len, vocab_size) or None if unavailable.
        """
        if self._mode != "llama_cpp" or self._llm is None:
            return None
        try:
            tokens = self._llm.tokenize(text.encode())
            self._llm.eval(tokens)
            # Get logits for each position
            # llama-cpp-python stores logits after eval
            n_vocab = self._llm.n_vocab()
            logits = np.array(self._llm._scores, dtype=np.float32)
            if logits.ndim == 1:
                logits = logits.reshape(-1, n_vocab)
            return logits
        except Exception:
            return None
