"""VSA-LM: Vector Symbolic Architecture Language Model.

Architecture:
  token → embed + pos → [VSALayer × N] → output projection → logits

  VSALayer: LayerNorm → FFN(AdditionLinear) + MindForge LoRA → residual

Training uses grilly_core GPU ops when available:
  vsa_lm_upload    → upload all weights to Vulkan GPU
  vsa_lm_forward   → fused forward pass (all layers)
  vsa_lm_backward  → fused backward pass (all layers)
  vsa_lm_update_weights → re-upload after optimizer step

Falls back to CPU numpy when GPU unavailable.

Run: python -m cubemind train vsa-lm --steps 50000
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

# Heavy legacy imports (HippocampalFormation, MindForge, GIFNeuron,
# AdditionLinear, SignActivation) are required only by the Phase-2+ VSA-LM
# path below, not by the Phase-1 MinGRU backbone. They're imported lazily
# inside ``VSALayer.__init__`` so ``from cubemind.training.vsa_lm import
# MinGRUModel`` stays fast and — importantly — doesn't pull grilly's
# ``utils/visualization.py`` matplotlib import chain, which crashes on
# Colab where ``MPLBACKEND`` points at a backend that isn't installed in
# a fresh venv.
from cubemind.ops.block_codes import BlockCodes
from cubemind.functional.math import softmax

# ── GPU bridge ───────────────────────────────────────────────────────────

_gc = None
_gc_distill = False
_adamw = None
try:
    import grilly_core as _gc_mod
    if hasattr(_gc_mod, "vsa_lm_forward"):
        _gc = _gc_mod
    if hasattr(_gc_mod, "distillation_loss"):
        _gc_distill = True
except Exception:
    pass

try:
    from grilly.backend._bridge import adamw_update as _adamw
except Exception:
    pass


class AdamW:
    """AdamW optimizer using grilly GPU update when available, numpy fallback."""

    def __init__(self, lr: float = 5e-4, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def begin_step(self) -> None:
        """Call once per training step (before all param updates)."""
        self.t += 1

    def step(self, param: np.ndarray, grad: np.ndarray) -> None:
        """In-place AdamW update on param. Call begin_step() first each iteration."""
        pid = id(param)

        if pid not in self._m:
            self._m[pid] = np.zeros_like(param)
            self._v[pid] = np.zeros_like(param)

        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t

        if _adamw is not None:
            result = _adamw(
                param, grad, self._m[pid], self._v[pid],
                lr=self.lr, beta1=self.beta1, beta2=self.beta2,
                eps=self.eps, weight_decay=self.wd,
                beta1_t=beta1_t, beta2_t=beta2_t,
            )
            if result is not None:
                np.copyto(param, np.asarray(result["weights"], dtype=np.float32))
                np.copyto(self._m[pid], np.asarray(result["m"], dtype=np.float32))
                np.copyto(self._v[pid], np.asarray(result["v"], dtype=np.float32))
                return

        # Numpy fallback
        self._m[pid] = self.beta1 * self._m[pid] + (1 - self.beta1) * grad
        self._v[pid] = self.beta2 * self._v[pid] + (1 - self.beta2) * (grad ** 2)
        m_hat = self._m[pid] / (1 - beta1_t)
        v_hat = self._v[pid] / (1 - beta2_t)
        param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * param)


CAPSULE_DIM = 32
CAP_NOVELTY = 5
CAP_PLASTICITY = 14
CAP_CONSOLIDATION = 18


@dataclass
class VSALMConfig:
    k: int = 16
    l: int = 24
    d_model: int = 384
    n_layers: int = 18
    d_ffn: int = 1152
    forge_rank: int = 8
    forge_basis: int = 16
    n_place: int = 32
    n_time: int = 16
    n_grid: int = 24
    vocab_size: int = 8192
    seq_len: int = 256
    batch_size: int = 1
    lr: float = 5e-4
    lr_min: float = 1e-5
    train_steps: int = 200000
    val_every: int = 10
    save_every: int = 5000
    max_stories: int = 0
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════
# MinGRU Coherence-Baseline Backbone (Phase 1.2)
# ═══════════════════════════════════════════════════════════════════════════
#
# Per TASKS.md Phase 1 architectural decisions:
#   - MinGRU gated recurrence (Feng & Tegmark 2024), standard float32 weights.
#     Qwen3.5 (Gated Delta Networks + sparse MoE) validates this direction at
#     production scale. MinGRU is the proven small-scale formulation.
#   - TinyStories coherence (Eldan & Li 2023) is the evaluation gate, not PPL.
#   - grilly autograd only — no manual backward.
#   - AdditionLinear stays in the FFN (channel mix), never the sequence mixer.
#
# Recurrence (matches prefix_scan_causal form h_t = a_t · h_{t-1} + x_t):
#   [g, v, d] = three grilly.nn.Linear projections of x
#   x_scan    = sigmoid(g) · tanh(v)                 candidate hidden
#   a         = 0.05 + 0.9 · sigmoid(d)              bounded decay gate
#   h         = prefix_scan_causal(x_scan, a)        GPU linear-RNN scan
#
# Stack (pre-norm, residual) per block:
#   x = x + MinGRULayer(RMSNorm(x))
#   x = x + GLUChannelMix(RMSNorm(x))
#
# This is the Phase 1 baseline. Phase 2 will extend it into VSALMModel by
# wrapping CubeMind extensions (SNN gating, MindForge, hippocampal memory)
# around the same backbone — config-gated, disabled by default.


@dataclass
class MinGRUConfig:
    """Config for the MinGRU coherence baseline.

    Defaults match TASKS.md Phase 1.3: d=256, L=6, d_ffn=768, vocab=4000.
    That produces ~5M params on TinyStories per the task.
    """
    vocab_size: int = 4000
    d_model: int = 256
    d_ffn: int = 768
    n_layers: int = 6
    seq_len: int = 256
    tie_embeddings: bool = True
    decay_bias_init: float = 1.0  # sigmoid(1) ≈ 0.73 retention at t=0
    dropout: float = 0.0
    seed: int = 42


class MinGRULayer:
    """MinGRU sequence mixer, GPU autograd-wired via prefix_scan_causal.

    Bounded decay (``a ∈ [0.05, 0.95]``) keeps ``log(a)`` in a numerically
    safe range for the scan's log-space accumulation — same trick as the
    existing ``CausalSequenceMixer`` in grilly.
    """

    def __init__(self, d_model: int, bias_init: float = 1.0):
        from grilly import nn
        self.d_model = d_model
        self.proj_g = nn.Linear(d_model, d_model, bias=True)
        self.proj_v = nn.Linear(d_model, d_model, bias=True)
        self.proj_d = nn.Linear(d_model, d_model, bias=True)

        # Initialize decay-gate bias so sigmoid starts at ≈0.73 (mild
        # retention at step 0 — the LiquidCell-era default).
        try:
            b = self.proj_d.bias
            b_arr = b.data if hasattr(b, "data") else b
            b_arr[:] = float(bias_init)
        except Exception:
            pass

    def parameters(self):
        yield from self.proj_g.parameters()
        yield from self.proj_v.parameters()
        yield from self.proj_d.parameters()

    def __call__(self, x):
        """x: Variable or ndarray of shape (B, S, D). Returns Variable (B, S, D).

        Activation ops use grilly.nn.autograd's free functions (``sigmoid``,
        ``tanh``) instead of Variable methods so the forward works in both
        grad-enabled training and ``no_grad`` eval/generation — under
        ``no_grad``, grilly's ``Linear.forward`` returns a raw ndarray
        rather than a Variable, and ``ndarray.tanh()`` does not exist.
        """
        from grilly.nn.autograd import sigmoid, tanh
        from grilly.nn.prefix_scan import prefix_scan_causal

        g = self.proj_g(x)
        v = self.proj_v(x)
        d = self.proj_d(x)

        # Candidate input: sigmoid(g) · tanh(v), bounded in (-1, 1).
        x_scan = sigmoid(g) * tanh(v)

        # Bounded decay: 0.05 + 0.9 · sigmoid(d) ∈ [0.05, 0.95] — keeps
        # log(a) ≥ -3 for the prefix scan's log-space accumulation.
        a = 0.05 + sigmoid(d) * 0.9

        return prefix_scan_causal(x_scan, a)


class GLUChannelMix:
    """Gated linear unit FFN: ``y = W_o(silu(W_g x) ⊙ W_u x)``.

    Standard float32 projections via grilly.nn.Linear. A follow-up issue
    may swap the projections onto an autograd-wired AdditionLinear to
    realize the "matmul-free FFN" architectural intent (see TASKS.md) —
    the computation stays the same, only the weight kernel changes.
    """

    def __init__(self, d_model: int, d_ffn: int):
        from grilly import nn
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.W_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.W_up   = nn.Linear(d_model, d_ffn, bias=False)
        self.W_out  = nn.Linear(d_ffn, d_model, bias=False)

    def parameters(self):
        yield from self.W_gate.parameters()
        yield from self.W_up.parameters()
        yield from self.W_out.parameters()

    def __call__(self, x):
        from grilly.nn.autograd import sigmoid
        g = self.W_gate(x)
        silu_g = g * sigmoid(g)             # SiLU = x · sigmoid(x)
        up = self.W_up(x)
        return self.W_out(silu_g * up)


class MinGRUBlock:
    """Pre-norm residual block: mixer (MinGRU) + channel mix (GLU)."""

    def __init__(self, cfg: MinGRUConfig):
        from grilly import nn
        self.rms_mix = nn.RMSNorm(cfg.d_model)
        self.rms_ffn = nn.RMSNorm(cfg.d_model)
        self.mix = MinGRULayer(cfg.d_model, bias_init=cfg.decay_bias_init)
        self.ffn = GLUChannelMix(cfg.d_model, cfg.d_ffn)

    def parameters(self):
        yield from self.rms_mix.parameters()
        yield from self.rms_ffn.parameters()
        yield from self.mix.parameters()
        yield from self.ffn.parameters()

    def __call__(self, x):
        x = x + self.mix(self.rms_mix(x))
        x = x + self.ffn(self.rms_ffn(x))
        return x


class MinGRUModel:
    """Standalone MinGRU language model — the Phase 1 coherence baseline.

    ``token_ids (B, S) -> embed -> [MinGRUBlock × N] -> RMSNorm -> head``

    Head is tied to the embedding when ``cfg.tie_embeddings`` (default).
    Both reference the same Parameter object so ``loss.backward()``
    accumulates the embedding-lookup gradient and the output-projection
    gradient into a single ``.grad`` slot — standard tied-weight semantics.

    No CubeMind extensions here (SNN gating, MindForge, hippocampal memory
    land in Phase 2's VSALMModel, which extends this backbone).
    """

    def __init__(self, cfg: "MinGRUConfig | None" = None, **kwargs):
        from grilly import nn
        if cfg is None:
            cfg = MinGRUConfig(**kwargs)
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # grilly.nn.Embedding defaults to N(0, 1), which is catastrophic
        # for tied-weight language models — the head's logit scale becomes
        # O(sqrt(d_model)) at init, so softmax is near one-hot and CE
        # starts at ~log(V) · sqrt(d_model)/2 instead of log(V). Standard
        # transformer init is N(0, 0.02).
        try:
            from grilly.nn._helpers import _get_param_array
            emb_arr = _get_param_array(self.embed.weight)
            rng = np.random.default_rng(cfg.seed)
            emb_arr[:] = rng.standard_normal(emb_arr.shape).astype(np.float32) * 0.02
        except Exception:
            pass

        self.blocks = [MinGRUBlock(cfg) for _ in range(cfg.n_layers)]
        self.rms_f = nn.RMSNorm(cfg.d_model)

        # Output head. Linear does ``x @ weight.T`` so for tied weights we
        # point head.weight at the embedding's Parameter — same object,
        # shared ``.grad`` slot.
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

    def parameters(self):
        """Yield each underlying Parameter once (tied weights de-duped)."""
        seen: set = set()

        def _new(p) -> bool:
            pid = id(p)
            if pid in seen:
                return False
            seen.add(pid)
            return True

        for p in self.embed.parameters():
            if _new(p):
                yield p
        for block in self.blocks:
            for p in block.parameters():
                if _new(p):
                    yield p
        for p in self.rms_f.parameters():
            if _new(p):
                yield p
        for p in self.head.parameters():
            if _new(p):
                yield p

    def num_parameters(self) -> int:
        """Total trainable parameter count (tied weights counted once)."""
        from grilly.nn._helpers import _get_param_array
        total = 0
        seen: set = set()
        for p in self.parameters():
            arr = _get_param_array(p)
            if id(arr) in seen:
                continue
            seen.add(id(arr))
            total += int(arr.size)
        return total

    def __call__(self, tokens):
        """tokens: int (B, S) or (S,). Returns logits Variable (B, S, V)."""
        tokens = np.asarray(tokens)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h)
        h = self.rms_f(h)
        return self.head(h)


# ═══════════════════════════════════════════════════════════════════════════
# Legacy VSA-LM path (Phase 2+ extensions live downstream of MinGRUBlock)
# ═══════════════════════════════════════════════════════════════════════════

# ── VSA Layer (CPU path — used when GPU unavailable) ─────────────────────


class LiquidCell:
    def __init__(self, d: int, dt: float = 0.02):
        self.h = np.zeros(d, dtype=np.float32)
        self.dt = dt

    def step(self, x: np.ndarray) -> np.ndarray:
        self.h = ((1.0 - self.dt) * self.h + self.dt * x).astype(np.float32)
        return self.h.copy()


def _layernorm_bwd(dout, x, mean, var, weight, eps=1e-5):
    """LayerNorm backward — returns (dx, dweight, dbias)."""
    N = x.shape[-1]
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * std_inv
    dx_hat = dout * weight
    dvar = np.sum(dx_hat * (x - mean) * -0.5 * (std_inv ** 3), axis=-1, keepdims=True)
    dmean = (np.sum(dx_hat * -std_inv, axis=-1, keepdims=True)
             + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True))
    dx = dx_hat * std_inv + dvar * 2.0 * (x - mean) / N + dmean / N
    dweight = np.sum(dout * x_hat, axis=0)
    dbias = np.sum(dout, axis=0)
    return dx.astype(np.float32), dweight.astype(np.float32), dbias.astype(np.float32)


class VSALayer:
    def __init__(self, cfg: VSALMConfig, layer_id: int, forge: "MindForge",
                 hippo: "HippocampalFormation", bc: BlockCodes, seed: int = 42):
        # Lazy imports — keeps the MinGRU path free of the heavy legacy
        # dependency chain (grilly.utils.visualization -> matplotlib).
        from cubemind.brain.addition_linear import AdditionLinear, SignActivation
        from cubemind.brain.gif_neuron import GIFNeuron

        self.cfg = cfg
        self.layer_id = layer_id
        self.forge = forge
        self.bc = bc
        d = cfg.d_model

        self.ffn_up = AdditionLinear(d, cfg.d_ffn, bias=True, seed=seed)
        self.ffn_down = AdditionLinear(cfg.d_ffn, d, bias=True, seed=seed + 1)
        self.sign = SignActivation()
        self.liquid = LiquidCell(d)
        self.gif = GIFNeuron(input_dim=d, hidden_dim=d, L=8, seed=seed + 3)

        self.ln_g = np.ones(d, dtype=np.float32)
        self.ln_b = np.zeros(d, dtype=np.float32)
        self.grads = {
            "ln_g": np.zeros_like(self.ln_g),
            "ln_b": np.zeros_like(self.ln_b),
            "ffn_up_w": np.zeros_like(self.ffn_up.weight_patterns),
            "ffn_down_w": np.zeros_like(self.ffn_down.weight_patterns),
        }

    def forward_with_cache(self, x: np.ndarray, novelty: float = 0.5,
                           plasticity: float = 0.5, **kwargs):
        """Forward with cache for backward. Returns (output, cache)."""
        S, d = x.shape

        # LayerNorm
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        h = self.ln_g * (x - mean) / np.sqrt(var + 1e-5) + self.ln_b

        # Temporal context
        x_mean = h.mean(axis=0)
        temporal_ctx = self.liquid.step(x_mean)

        # GIF gating
        spikes, _ = self.gif.forward(temporal_ctx.reshape(1, d))
        gate = spikes.ravel()

        # MindForge LoRA (with cache for backward)
        ctx_flat = temporal_ctx[:self.cfg.k * self.cfg.l]
        if len(ctx_flat) < self.cfg.k * self.cfg.l:
            ctx_flat = np.pad(ctx_flat, (0, self.cfg.k * self.cfg.l - len(ctx_flat)))
        A, B, forge_cache = self.forge.forge_with_cache(
            self.bc.discretize(ctx_flat.reshape(self.cfg.k, self.cfg.l)),
            self.layer_id,
        )

        # FFN
        h_up_raw = self.ffn_up.forward(h) / np.sqrt(self.ffn_up.in_features)
        h_up = self.sign.forward(h_up_raw)
        h_ffn = self.ffn_down.forward(h_up) / np.sqrt(self.ffn_down.in_features)

        # LoRA: h @ B @ A
        M = h @ B
        h_lora = (M @ A).astype(np.float32)

        lora_scale = 0.5 + novelty
        ffn_scale = 0.5 + plasticity
        y = ffn_scale * h_ffn + lora_scale * gate * h_lora

        cache = (x, h, mean, var, h_up, M, A, B, lora_scale, ffn_scale, gate, forge_cache)
        return x + y, cache

    def forward(self, x: np.ndarray, novelty: float = 0.5,
                plasticity: float = 0.5, **kwargs) -> np.ndarray:
        """Forward without cache (inference only)."""
        out, _ = self.forward_with_cache(x, novelty, plasticity, **kwargs)
        return out

    def backward(self, d_out: np.ndarray, cache: tuple) -> np.ndarray:
        """Backward pass — accumulates gradients for FFN, LN, and MindForge."""
        x, h, mean, var, h_up, M, A, B, lora_scale, ffn_scale, gate, forge_cache = cache
        S, d = x.shape

        d_x = d_out.copy()  # residual
        d_y = d_out

        # 1. LoRA backward
        d_h_lora = d_y * lora_scale * gate
        d_A = M.T @ d_h_lora
        d_M = d_h_lora @ A.T
        d_B = h.T @ d_M
        d_h_from_lora = d_M @ B.T

        # 2. MindForge backward (accumulates into forge.grads)
        self.forge.backward(d_A, d_B, self.layer_id, forge_cache)

        # 3. FFN backward (STE for SignActivation)
        # ffn_down: (out=d, in=d_ffn), forward ≈ h_up @ W.T → (S, d)
        # ffn_up:   (out=d_ffn, in=d), forward ≈ h @ W.T → (S, d_ffn)
        d_h_ffn = d_y * ffn_scale  # (S, d)
        # grad_down_w = d_h_ffn.T @ h_up → (d, S) @ (S, d_ffn) = (d, d_ffn) ✓ matches (256, 768)
        self.grads["ffn_down_w"] += (d_h_ffn.T @ h_up) / S
        # d_h_up = d_h_ffn @ W_down → (S, d) @ (d, d_ffn) = (S, d_ffn)
        d_h_up = d_h_ffn @ self.ffn_down.weight_patterns
        d_h_up_pre = d_h_up  # STE through SignActivation
        # grad_up_w = d_h_up.T @ h → (d_ffn, S) @ (S, d) = (d_ffn, d) ✓ matches (768, 256)
        self.grads["ffn_up_w"] += (d_h_up_pre.T @ h) / S
        # d_h = d_h_up @ W_up → (S, d_ffn) @ (d_ffn, d) = (S, d) ✓
        d_h_from_ffn = d_h_up_pre @ self.ffn_up.weight_patterns

        # 4. LayerNorm backward
        d_h_total = d_h_from_lora + d_h_from_ffn
        d_x_ln, d_g, d_b = _layernorm_bwd(d_h_total, x, mean, var, self.ln_g)
        self.grads["ln_g"] += d_g.ravel()
        self.grads["ln_b"] += d_b.ravel()

        d_x += d_x_ln
        return d_x


# ── Model ────────────────────────────────────────────────────────────────


class VSALM:
    def __init__(self, cfg: VSALMConfig):
        # Lazy imports — see VSALayer.__init__ for rationale.
        from cubemind.execution.mindforge import MindForge
        from cubemind.memory.formation import HippocampalFormation

        self.cfg = cfg
        d = cfg.d_model
        rng = np.random.default_rng(cfg.seed)

        self.bc = BlockCodes(k=cfg.k, l=cfg.l)
        self.embed = rng.normal(0, 0.02, (cfg.vocab_size, d)).astype(np.float32)
        self.capsule_embed = rng.normal(0, 0.01, (cfg.vocab_size, CAPSULE_DIM)).astype(np.float32)
        self.capsule_proj = rng.normal(0, 0.02, (d, CAPSULE_DIM)).astype(np.float32)
        self.out_w = rng.normal(0, 0.02, (cfg.vocab_size, d)).astype(np.float32)

        pe = np.zeros((cfg.seq_len + 16, d), dtype=np.float32)
        pos = np.arange(cfg.seq_len + 16).reshape(-1, 1).astype(np.float32)
        div = np.exp(np.arange(0, d, 2).astype(np.float32) * -(math.log(10000) / d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.pe = pe

        self.hippo = HippocampalFormation(
            feature_dim=d, max_memories=500,
            n_place_cells=cfg.n_place, n_time_cells=cfg.n_time,
            n_grid_cells=cfg.n_grid, seed=cfg.seed,
        )
        self.forge = MindForge(
            k=cfg.k, l=cfg.l, n_layers=cfg.n_layers,
            d_target=d, rank=cfg.forge_rank, n_basis=cfg.forge_basis,
            d_hidden=256, seed=cfg.seed,
        )
        self.layers = [
            VSALayer(cfg, i, self.forge, self.hippo, self.bc, seed=cfg.seed + i)
            for i in range(cfg.n_layers)
        ]

        # GPU state
        self._gpu_dev = None
        self._gpu_handle = None
        self._last_hidden = None

    def _init_gpu(self) -> bool:
        """Try to upload weights to GPU."""
        if _gc is None:
            return False
        try:
            import pathlib
            spv = pathlib.Path(__file__).resolve().parent.parent.parent / "grilly" / "shaders" / "spv"
            if not spv.exists():
                # Try installed grilly
                import grilly
                spv = pathlib.Path(grilly.__file__).parent / "shaders" / "spv" # type: ignore

            self._gpu_dev = _gc.Device()
            self._gpu_dev.load_shaders(str(spv))

            ffn_up_w = [l.ffn_up.weight_patterns for l in self.layers]
            ffn_up_b = [l.ffn_up.bias if l.ffn_up.bias is not None
                        else np.zeros(l.ffn_up.out_features, dtype=np.float32)
                        for l in self.layers]
            ffn_down_w = [l.ffn_down.weight_patterns for l in self.layers]
            ffn_down_b = [l.ffn_down.bias if l.ffn_down.bias is not None
                          else np.zeros(l.ffn_down.out_features, dtype=np.float32)
                          for l in self.layers]
            ln_g = [l.ln_g for l in self.layers]
            ln_b = [l.ln_b for l in self.layers]

            self._gpu_handle = _gc.vsa_lm_upload(
                self._gpu_dev,
                self.embed, self.pe, ffn_up_w, ffn_up_b,
                ffn_down_w, ffn_down_b, ln_g, ln_b,
                self.out_w, self.cfg.n_layers, self.cfg.d_model, self.cfg.d_ffn,
            )
            logger.info("VSA-LM GPU initialized (handle={})", self._gpu_handle)
            return True
        except Exception as e:
            logger.warning("GPU init failed: {}. Using CPU.", e)
            self._gpu_dev = None
            self._gpu_handle = None
            return False

    def _reupload_gpu(self):
        """Re-upload weights after optimizer step."""
        if self._gpu_handle is None:
            return
        try:
            _gc.vsa_lm_update_weights(
                self._gpu_dev, self._gpu_handle,
                self.embed, self.pe,
                [l.ffn_up.weight_patterns for l in self.layers],
                [l.ffn_up.bias if l.ffn_up.bias is not None
                 else np.zeros(l.ffn_up.out_features, dtype=np.float32)
                 for l in self.layers],
                [l.ffn_down.weight_patterns for l in self.layers],
                [l.ffn_down.bias if l.ffn_down.bias is not None
                 else np.zeros(l.ffn_down.out_features, dtype=np.float32)
                 for l in self.layers],
                [l.ln_g for l in self.layers],
                [l.ln_b for l in self.layers],
                self.out_w,
            )
        except Exception:
            pass

    def forward(self, input_ids: np.ndarray, prev_loss: float = 0.0) -> np.ndarray:
        """Forward pass — GPU if available, else CPU."""
        S = len(input_ids)

        # GPU path: fused forward
        if self._gpu_handle is not None:
            try:
                logits = _gc.vsa_lm_forward(
                    self._gpu_dev, self._gpu_handle,
                    input_ids.astype(np.int32),
                )
                self._last_hidden = None  # GPU doesn't return hidden
                return np.asarray(logits, dtype=np.float32)
            except Exception:
                pass  # fall through to CPU

        # CPU path
        x = self.embed[input_ids] + self.pe[:S]
        caps = self.capsule_embed[input_ids]
        x = x + (caps @ self.capsule_proj.T).astype(np.float32)

        caps_mean = caps.mean(axis=0)
        novelty = float(np.clip(caps_mean[CAP_NOVELTY], 0, 1))
        plasticity = float(np.clip(caps_mean[CAP_PLASTICITY], 0, 1))

        for layer in self.layers:
            x = layer.forward(x, novelty=novelty, plasticity=plasticity)

        self._last_hidden = x.copy()
        return (x @ self.out_w.T / np.sqrt(self.cfg.d_model)).astype(np.float32)

    def param_count(self) -> int:
        n = self.embed.size + self.pe.size + self.out_w.size
        n += self.capsule_embed.size + self.capsule_proj.size
        n += sum(p.size for p in [
            self.forge.W_proj, self.forge.W_h, self.forge.b_h,
            self.forge.W_coeff, self.forge.b_coeff,
        ])
        n += self.forge.A_basis.size + self.forge.B_basis.size
        n += self.forge.layer_embeddings.size
        for layer in self.layers:
            n += layer.ffn_up.weight_patterns.size
            if layer.ffn_up.bias is not None:
                n += layer.ffn_up.bias.size
            n += layer.ffn_down.weight_patterns.size
            if layer.ffn_down.bias is not None:
                n += layer.ffn_down.bias.size
            n += layer.ln_g.size + layer.ln_b.size
            n += layer.liquid.h.size
        return n


# ── Data loading ─────────────────────────────────────────────────────────


def load_data(cfg: VSALMConfig, data_dir: str = "sandbox/vsa_lm/data"):
    tokens = np.load(os.path.join(data_dir, "tokens.npy"))
    actual_vocab = int(np.load(os.path.join(data_dir, "vocab_size.npy"))[0])
    logger.info("Loaded {} tokens, vocab={}", len(tokens), actual_vocab)

    n = len(tokens)
    train_tok = tokens[:int(0.8 * n)]
    val_tok = tokens[int(0.8 * n):int(0.9 * n)]

    def make_seqs(tok):
        x, y = [], []
        for i in range(0, len(tok) - cfg.seq_len - 1, cfg.seq_len // 2):
            x.append(tok[i:i + cfg.seq_len])
            y.append(tok[i + 1:i + cfg.seq_len + 1])
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    tx, ty = make_seqs(train_tok)
    vx, vy = make_seqs(val_tok)
    logger.info("Train: {} seqs, Val: {} seqs", len(tx), len(vx))
    return tx, ty, vx, vy, actual_vocab


def compute_ppl(model: VSALM, x_data, y_data, max_samples: int = 100) -> float:
    total_loss, n = 0.0, 0
    for i in range(min(len(x_data), max_samples)):
        logits = model.forward(x_data[i])
        p = np.asarray(softmax(logits), dtype=np.float32)
        for t in range(len(y_data[i])):
            tok = int(y_data[i][t])
            if 0 <= tok < p.shape[1]:
                total_loss -= math.log(max(float(p[t, tok]), 1e-8))
                n += 1
    return float(math.exp(min(total_loss / max(n, 1), 20)))


# ── Training ─────────────────────────────────────────────────────────────


def main(train_steps: int = 10000, n_layers: int = 6, d_model: int = 256,
         seq_len: int = 64, data_dir: str = "sandbox/vsa_lm/data_sp"):
    cfg = VSALMConfig(
        train_steps=train_steps, n_layers=n_layers, seq_len=seq_len,
        val_every=50, d_model=d_model, d_ffn=d_model * 3,
        k=16, l=16, forge_rank=8, forge_basis=16,
        n_place=32, n_time=16, n_grid=24, lr=5e-4,
    )

    train_x, train_y, val_x, val_y, actual_vocab = load_data(cfg, data_dir=data_dir)
    cfg.vocab_size = actual_vocab

    model = VSALM(cfg)
    logger.info("VSA-LM: d={}, layers={}, vocab={}, params={:.1f}M",
                cfg.d_model, cfg.n_layers, cfg.vocab_size, model.param_count() / 1e6)

    # Try GPU
    gpu_ok = model._init_gpu()
    logger.info("GPU: {}", "enabled" if gpu_ok else "CPU fallback")

    # Sanity check
    test_logits = model.forward(train_x[0])
    logger.info("Forward OK: {} finite={}", test_logits.shape, np.all(np.isfinite(test_logits)))

    ppl = compute_ppl(model, val_x, val_y, max_samples=20)
    logger.info("Initial PPL: {:.1f}", ppl)

    t0 = time.time()
    best_ppl = ppl
    step = 0
    loss = 0.0

    opt = AdamW(lr=cfg.lr)
    logger.info("Training {} steps (AdamW lr={})...", cfg.train_steps, cfg.lr)

    while step < cfg.train_steps:
        perm = np.random.permutation(len(train_x))
        train_x_s, train_y_s = train_x[perm], train_y[perm]

        for i in range(len(train_x_s)):
            if step >= cfg.train_steps:
                break

            ids = train_x_s[i]
            labels = train_y_s[i].ravel()
            S = len(ids)

            # ── Forward ──────────────────────────────────────────────
            prev_loss = loss if step > 0 else 0.0
            logits = model.forward(ids, prev_loss=prev_loss)

            # ── Loss + gradient ──────────────────────────────────────
            # Standard CE over vocab logits
            if _gc_distill and model._gpu_dev is not None:
                dl = _gc.distillation_loss(
                    model._gpu_dev,
                    logits.astype(np.float32),
                    logits.astype(np.float32),
                    labels.astype(np.int32),
                    temperature=1.0, alpha=0.0,
                )
                loss = float(np.mean(dl["loss"]))
                grad_logits = np.asarray(dl["grad"], dtype=np.float32)
            else:
                probs = np.asarray(softmax(logits), dtype=np.float32)
                loss = -float(np.mean(np.log(probs[np.arange(S), labels] + 1e-8)))
                grad_logits = probs.copy()
                grad_logits[np.arange(S), labels] -= 1.0
                grad_logits /= S

            # NOTE: Per-block CE disabled — embedding target distribution is not
            # a meaningful supervision signal and causes loss oscillation.
            # Will re-enable with proper block-code codebook target.

            # ── Backward: zero grads + advance optimizer step ─────────
            opt.begin_step()
            model.forge.grads = model.forge.zero_grads()
            for layer in model.layers:
                for k in layer.grads:
                    layer.grads[k].fill(0.0)

            # ── Forward with caches (needed for backward) ────────────
            x_fwd = model.embed[ids] + model.pe[:S]
            caps = model.capsule_embed[ids]
            x_fwd = x_fwd + (caps @ model.capsule_proj.T).astype(np.float32)
            caps_mean = caps.mean(axis=0)
            novelty = float(np.clip(caps_mean[CAP_NOVELTY], 0, 1))
            plasticity = float(np.clip(caps_mean[CAP_PLASTICITY], 0, 1))

            caches = []
            for layer in model.layers:
                x_fwd, cache = layer.forward_with_cache(
                    x_fwd, novelty=novelty, plasticity=plasticity)
                caches.append(cache)

            # ── Backward: output projection ──────────────────────────
            dx = (grad_logits @ model.out_w / np.sqrt(cfg.d_model)).astype(np.float32)
            grad_out_w = (grad_logits.T @ x_fwd / np.sqrt(cfg.d_model)).astype(np.float32)
            opt.step(model.out_w, grad_out_w)

            # ── Backward: layers in reverse (chain rule) ─────────────
            for layer, cache in reversed(list(zip(model.layers, caches))):
                dx = layer.backward(dx, cache)

            # ── Apply layer gradients via AdamW ──────────────────────
            for layer in model.layers:
                opt.step(layer.ffn_up.weight_patterns, layer.grads["ffn_up_w"])
                opt.step(layer.ffn_down.weight_patterns, layer.grads["ffn_down_w"])
                opt.step(layer.ln_g, layer.grads["ln_g"])
                opt.step(layer.ln_b, layer.grads["ln_b"])

            # ── Apply MindForge gradients via AdamW ──────────────────
            for fk, fg in model.forge.grads.items():
                param = getattr(model.forge, fk)
                opt.step(param, fg)

            # ── Embedding gradients (sparse SGD — per-token, can't batch) ─
            for t in range(S):
                tok = int(ids[t])
                if 0 <= tok < cfg.vocab_size:
                    model.embed[tok] -= cfg.lr * np.clip(dx[t], -0.1, 0.1)
                    dcaps = (dx[t] @ model.capsule_proj).astype(np.float32)
                    model.capsule_embed[tok] -= cfg.lr * np.clip(dcaps, -0.1, 0.1)

            # ── Re-upload to GPU if active ───────────────────────────
            if model._gpu_handle is not None:
                model._reupload_gpu()

            # ── Logging + checkpoint ─────────────────────────────────
            if step % cfg.val_every == 0:
                val_ppl = compute_ppl(model, val_x, val_y, max_samples=20)
                elapsed = time.time() - t0
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                gpu_tag = " [GPU]" if model._gpu_handle is not None else ""
                logger.info("step={:5d} | loss={:.3f} | PPL={:.1f} | {:.1f} stp/s{}",
                            step, float(loss), val_ppl, sps, gpu_tag)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl

            if step % cfg.save_every == 0 and step > 0:
                ckpt = f"data/checkpoints/vsa_lm_step{step}.npz"
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                np.savez_compressed(
                    ckpt, embed=model.embed, out_w=model.out_w,
                    capsule_embed=model.capsule_embed,
                    capsule_proj=model.capsule_proj,
                    forge_A_basis=model.forge.A_basis,
                    forge_B_basis=model.forge.B_basis,
                    forge_W_proj=model.forge.W_proj,
                    forge_W_h=model.forge.W_h,
                    forge_W_coeff=model.forge.W_coeff,
                    forge_layer_emb=model.forge.layer_embeddings,
                    step=step, best_ppl=best_ppl,
                )
                logger.info("Checkpoint: {} (PPL={:.1f})", ckpt, best_ppl)

            step += 1

    # Cleanup GPU
    if model._gpu_handle is not None:
        _gc.vsa_lm_release(model._gpu_dev, model._gpu_handle)

    elapsed = time.time() - t0
    logger.info("Done: {} steps in {:.0f}s, best PPL={:.1f}", step, elapsed, best_ppl)


def train_distill(
    teacher_dir: str = "data/teacher/gemma4_26b",
    train_steps: int = 10000,
    n_layers: int = 6,
    d_model: int = 256,
    seq_len: int = 128,
    lr: float = 3e-4,
    accum_steps: int = 1,
    loader_max_seq_len: int = 1024,
):
    """Train VSA-LM from pre-extracted teacher logits (offline distillation).

    Uses OfflineDistillationLoader to stream teacher .npz files.
    Loss = 0.3*CE(hard labels) + 0.6*KL(soft teacher).

    Phase 1 — window sweep: each .npz file is decoded once, then swept
    with non-overlapping windows of length `seq_len`. For Qwen3's 1013
    token sequences at seq_len=128 that's 7 training windows per file
    load instead of 1 — amortizes the ~270MB decompression cost.

    Phase 2 — gradient accumulation: gradients from `accum_steps` windows
    are summed before opt.step() fires. Effective batch = seq_len *
    accum_steps. Default `accum_steps=1` preserves the previous single-
    window behaviour; bump it when you want a larger effective batch
    without increasing seq_len.
    """
    from cubemind.training.moqe_distillation import (
        OfflineDistillationLoader, _cross_entropy_with_grad,
        _kl_divergence_with_grad, _sparse_kl_divergence_with_grad,
    )

    cfg = VSALMConfig(
        train_steps=train_steps, n_layers=n_layers, seq_len=seq_len,
        val_every=100, d_model=d_model, d_ffn=d_model * 3,
        k=16, l=16, forge_rank=8, forge_basis=16,
        n_place=32, n_time=16, n_grid=24, lr=lr,
    )

    # Load teacher logits. Ask the loader for the full sequence (up to
    # loader_max_seq_len) so we can sweep multiple windows out of each
    # yielded sample instead of throwing the tail away.
    loader_cap = max(seq_len, int(loader_max_seq_len))
    loader = OfflineDistillationLoader(teacher_dir, max_seq_len=loader_cap)

    # Detect vocab and format from first file
    import glob
    first_file = sorted(glob.glob(os.path.join(teacher_dir, "*.npz")))[0]
    sample = np.load(first_file)
    if "logits" in sample:
        teacher_vocab = int(sample["logits"].shape[-1])
        teacher_format = "dense"
    elif "top_k_indices" in sample:
        teacher_vocab = int(sample["top_k_indices"].max()) + 1
        teacher_format = "top_k"
    else:
        teacher_vocab = 262144  # Gemma default fallback
        teacher_format = "unknown"
    if "identity_len" in sample.files:
        ident_len = int(sample["identity_len"][0])
        logger.info("Teacher: vocab={} format={} identity_len={}",
                    teacher_vocab, teacher_format, ident_len)
    else:
        logger.info("Teacher: vocab={} format={}", teacher_vocab, teacher_format)

    # Student vocab: cap at 131072 — large enough that Qwen3 (151K) loses
    # < 0.001% tokens to OOV, while keeping embed + out_w under ~270MB fp32.
    # For Gemma 262K this still leaves some OOV tail, but Gemma is deprecated
    # here in favour of Qwen3 which has a proper frequency-ordered BPE.
    student_vocab = min(teacher_vocab, 131072)
    cfg.vocab_size = student_vocab
    logger.info("Student vocab: {} (from teacher {})", student_vocab, teacher_vocab)

    model = VSALM(cfg)
    logger.info("VSA-LM distill: d={}, layers={}, vocab={}, params={:.1f}M",
                cfg.d_model, cfg.n_layers, cfg.vocab_size, model.param_count() / 1e6)

    # GPU forward/backward disabled for distillation — vsa_lm_forward/backward
    # don't include MindForge LoRA layers, producing wrong outputs/gradients.
    # TODO: add MindForge to the C++ vsa_lm kernel.
    logger.info("GPU: disabled (MindForge not in C++ kernel yet, using CPU)")

    opt = AdamW(lr=lr)
    t0 = time.time()
    best_loss = float("inf")
    loss_ema = 0.0
    step = 0

    # ── Gradient accumulators (Phase 2) ──────────────────────────────
    # layer.grads and forge.grads already accumulate via += inside their
    # backward methods, so we only need extra buffers for the two pieces
    # that the old loop applied in place: the out_w matmul grad, and the
    # sparse embedding row updates.
    accum_grad_out_w = np.zeros_like(model.out_w)
    accum_embed_grad: dict[int, np.ndarray] = {}
    windows_in_accum = 0

    def _fire_optimizer() -> None:
        """Apply accumulated gradients via AdamW and zero everything."""
        nonlocal windows_in_accum
        if windows_in_accum == 0:
            return
        opt.begin_step()
        opt.step(model.out_w, accum_grad_out_w)
        for layer in model.layers:
            opt.step(layer.ffn_up.weight_patterns, layer.grads["ffn_up_w"])
            opt.step(layer.ffn_down.weight_patterns, layer.grads["ffn_down_w"])
            opt.step(layer.ln_g, layer.grads["ln_g"])
            opt.step(layer.ln_b, layer.grads["ln_b"])
        for fk, fg in model.forge.grads.items():
            opt.step(getattr(model.forge, fk), fg)
        # Sparse embedding rows — direct SGD (per-row slices defeat the
        # AdamW moment accumulators, same reason as the original loop).
        for tok, g in accum_embed_grad.items():
            model.embed[tok] -= lr * np.clip(g, -0.1, 0.1)
        # Zero every accumulator for the next accum batch.
        accum_grad_out_w.fill(0.0)
        accum_embed_grad.clear()
        for layer in model.layers:
            for k in layer.grads:
                layer.grads[k].fill(0.0)
        model.forge.grads = model.forge.zero_grads()
        windows_in_accum = 0

    logger.info(
        "Distillation training: {} steps, accum_steps={}, "
        "loader_cap={}, teacher_dir={}",
        train_steps, accum_steps, loader_cap, teacher_dir,
    )

    for epoch in range(100):  # enough epochs to hit train_steps
        for input_ids, labels, teacher_data in loader:
            if step >= train_steps:
                break

            # ── Phase 1: sweep non-overlapping windows from this sample ──
            full_len = int(len(input_ids))
            n_windows = full_len // seq_len
            if n_windows == 0:
                continue  # sample shorter than one window, skip

            for wi in range(n_windows):
                if step >= train_steps:
                    break
                ws = wi * seq_len
                we = ws + seq_len

                ids = input_ids[ws:we].astype(np.int32)
                labs = labels[ws:we].astype(np.int32)

                # Slice teacher_data to this window
                if isinstance(teacher_data, dict) and "top_k_indices" in teacher_data:
                    window_teacher = {
                        "top_k_indices": teacher_data["top_k_indices"][ws:we],
                        "top_k_logprobs": teacher_data["top_k_logprobs"][ws:we],
                    }
                elif isinstance(teacher_data, np.ndarray):
                    window_teacher = teacher_data[ws:we]
                else:
                    window_teacher = teacher_data

                # Remap OOV input IDs to UNK (token 1) so the model sees an
                # identifiable placeholder instead of a clamped valid-looking
                # token. Labels OOV are masked as -1 so CE/KD skip those.
                ids = np.where(ids >= cfg.vocab_size, 1, ids).astype(np.int32)
                ids = np.where(ids < 0, 1, ids).astype(np.int32)
                labs_oov = (labs >= cfg.vocab_size) | (labs < 0)
                labs = np.where(labs_oov, 0, labs).astype(np.int32)
                labs[labs_oov] = -1  # CE ignores -1 labels

                S = seq_len

                # ── Forward ──────────────────────────────────────────
                logits = model.forward(ids)  # (S, vocab)

                # ── Loss: CE + KL from teacher ───────────────────────
                loss_ce, grad_ce = _cross_entropy_with_grad(logits, labs)

                if window_teacher is None:
                    loss_kd, grad_kd = 0.0, np.zeros_like(logits)
                elif isinstance(window_teacher, dict) and "top_k_indices" in window_teacher:
                    t_indices = window_teacher["top_k_indices"].copy()
                    t_logprobs = window_teacher["top_k_logprobs"].astype(np.float32)
                    oov_mask = t_indices >= student_vocab
                    t_indices[oov_mask] = 0
                    t_logprobs[oov_mask] = -1e9  # zero after softmax
                    valid_count = np.sum(~oov_mask, axis=-1)
                    if np.min(valid_count) > 0:
                        loss_kd, grad_kd = _sparse_kl_divergence_with_grad(
                            logits, t_indices, t_logprobs, temperature=2.0,
                        )
                    else:
                        loss_kd, grad_kd = 0.0, np.zeros_like(logits)
                elif isinstance(window_teacher, np.ndarray):
                    min_v = min(logits.shape[-1], window_teacher.shape[-1])
                    loss_kd, grad_kd = _kl_divergence_with_grad(
                        logits[:, :min_v],
                        window_teacher[:, :min_v].astype(np.float32),
                        temperature=2.0,
                    )
                    if min_v < logits.shape[-1]:
                        pad = np.zeros((S, logits.shape[-1] - min_v), dtype=np.float32)
                        grad_kd = np.concatenate([grad_kd, pad], axis=-1)
                else:
                    loss_kd, grad_kd = 0.0, np.zeros_like(logits)

                loss = 0.3 * loss_ce + 0.6 * loss_kd
                grad_logits = (0.3 * grad_ce + 0.6 * grad_kd).astype(np.float32)

                # ── Backward (CPU path; GPU backward still lacks
                #    MindForge, so vsa_lm_backward is bypassed) ────────
                # Re-forward with caches because model.forward() above
                # didn't return them.
                x_fwd = model.embed[ids] + model.pe[:S]
                caps = model.capsule_embed[ids]
                x_fwd = x_fwd + (caps @ model.capsule_proj.T).astype(np.float32)
                caps_mean = caps.mean(axis=0)
                novelty = float(np.clip(caps_mean[CAP_NOVELTY], 0, 1))
                plasticity = float(np.clip(caps_mean[CAP_PLASTICITY], 0, 1))

                caches = []
                for layer in model.layers:
                    x_fwd, cache = layer.forward_with_cache(
                        x_fwd, novelty=novelty, plasticity=plasticity)
                    caches.append(cache)

                dx = (grad_logits @ model.out_w / np.sqrt(cfg.d_model)).astype(np.float32)
                grad_out_w = (grad_logits.T @ x_fwd / np.sqrt(cfg.d_model)).astype(np.float32)
                accum_grad_out_w += grad_out_w  # ← Phase 2: accumulate

                for layer, cache in reversed(list(zip(model.layers, caches))):
                    dx = layer.backward(dx, cache)
                    # layer.grads and forge.grads accumulate via += inside

                for t in range(S):
                    tok = int(ids[t])
                    if 0 <= tok < cfg.vocab_size:
                        delta = np.clip(dx[t], -0.1, 0.1)
                        buf = accum_embed_grad.get(tok)
                        if buf is None:
                            accum_embed_grad[tok] = delta.astype(np.float32).copy()
                        else:
                            buf += delta

                windows_in_accum += 1

                # ── Logging ──────────────────────────────────────────
                loss_ema = 0.99 * loss_ema + 0.01 * loss if step > 0 else loss
                if step % cfg.val_every == 0:
                    elapsed = time.time() - t0
                    sps = (step + 1) / elapsed if elapsed > 0 else 0
                    logger.info(
                        "step={:5d} | CE={:.3f} | KD={:.3f} | loss={:.3f} | "
                        "ema={:.3f} | {:.1f} stp/s",
                        step, loss_ce, loss_kd, loss, loss_ema, sps,
                    )
                    if loss_ema < best_loss:
                        best_loss = loss_ema

                if step > 0 and step % cfg.save_every == 0:
                    # Flush pending grads before snapshotting so the ckpt
                    # reflects the just-applied weights, not a mid-batch
                    # partial update.
                    _fire_optimizer()
                    ckpt = f"data/checkpoints/vsa_lm_distill_step{step}.npz"
                    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                    np.savez_compressed(
                        ckpt, embed=model.embed, out_w=model.out_w,
                        forge_A_basis=model.forge.A_basis,
                        forge_B_basis=model.forge.B_basis,
                        step=step, loss_ema=loss_ema,
                    )
                    logger.info("Checkpoint: {} (loss={:.3f})", ckpt, loss_ema)

                step += 1

                # ── Phase 2: fire optimizer every accum_steps windows ──
                if windows_in_accum >= accum_steps:
                    _fire_optimizer()
            # end of per-sample window sweep

        if step >= train_steps:
            break

    # Flush any leftover grads from a partial accum batch.
    _fire_optimizer()

    # Final checkpoint — always save at end of training so we never
    # lose a completed run (the in-loop save only fires at multiples
    # of save_every, which can be skipped if train_steps isn't aligned).
    ckpt = f"data/checkpoints/vsa_lm_distill_step{step}.npz"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    np.savez_compressed(
        ckpt, embed=model.embed, out_w=model.out_w,
        forge_A_basis=model.forge.A_basis,
        forge_B_basis=model.forge.B_basis,
        step=step, loss_ema=loss_ema,
    )
    logger.info("Final checkpoint: {} (loss={:.3f})", ckpt, loss_ema)

    elapsed = time.time() - t0
    logger.info("Distillation done: {} steps in {:.0f}s, best_loss={:.3f}",
                step, elapsed, best_loss)


if __name__ == "__main__":
    main()
