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

from cubemind.ops.block_codes import BlockCodes
from cubemind.brain.addition_linear import AdditionLinear, SignActivation
from cubemind.brain.gif_neuron import GIFNeuron
from cubemind.memory.formation import HippocampalFormation
from cubemind.execution.mindforge import MindForge
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
    def __init__(self, cfg: VSALMConfig, layer_id: int, forge: MindForge,
                 hippo: HippocampalFormation, bc: BlockCodes, seed: int = 42):
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
):
    """Train VSA-LM from pre-extracted teacher logits (offline distillation).

    Uses OfflineDistillationLoader to stream teacher .npz files.
    Loss = 0.3*CE(hard labels) + 0.6*KL(soft teacher).
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

    # Load teacher logits
    loader = OfflineDistillationLoader(teacher_dir, max_seq_len=seq_len)

    # Detect vocab from first file
    import glob
    first_file = sorted(glob.glob(os.path.join(teacher_dir, "*.npz")))[0]
    sample = np.load(first_file)
    if "logits" in sample:
        teacher_vocab = sample["logits"].shape[-1]
    else:
        teacher_vocab = 262144  # Gemma default
    logger.info("Teacher vocab: {}", teacher_vocab)

    # Use a reasonable student vocab — much smaller than teacher.
    # Sparse KL only matches positions where both have coverage.
    # 32K covers common tokens; rare tokens in top-k get clamped.
    student_vocab = min(teacher_vocab, 8192)
    cfg.vocab_size = student_vocab
    logger.info("Student vocab: {} (clamped from teacher {})", student_vocab, teacher_vocab)

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

    logger.info("Distillation training: {} steps, teacher_dir={}", train_steps, teacher_dir)

    for epoch in range(100):  # enough epochs to hit train_steps
        for input_ids, labels, teacher_data in loader:
            if step >= train_steps:
                break

            S = min(len(input_ids), seq_len)
            ids = input_ids[:S].astype(np.int32)
            labs = labels[:S].astype(np.int32)

            # Clamp token IDs to model vocab
            ids = np.clip(ids, 0, cfg.vocab_size - 1)
            labs = np.clip(labs, 0, cfg.vocab_size - 1)

            # ── Forward ──────────────────────────────────────────────
            logits = model.forward(ids)  # (S, vocab)

            # ── Loss: CE + KL from teacher ───────────────────────────
            loss_ce, grad_ce = _cross_entropy_with_grad(logits, labs)

            if teacher_data is None:
                loss_kd, grad_kd = 0.0, np.zeros_like(logits)
            elif isinstance(teacher_data, dict) and "top_k_indices" in teacher_data:
                t_indices = teacher_data["top_k_indices"][:S].copy()
                t_logprobs = teacher_data["top_k_logprobs"][:S].astype(np.float32)
                # Clamp teacher indices to student vocab
                mask = t_indices < student_vocab
                t_indices = np.where(mask, t_indices, 0)
                t_logprobs = np.where(mask, t_logprobs, -100.0)  # suppress OOV
                loss_kd, grad_kd = _sparse_kl_divergence_with_grad(
                    logits, t_indices, t_logprobs, temperature=2.0,
                )
            elif isinstance(teacher_data, np.ndarray):
                min_v = min(logits.shape[-1], teacher_data.shape[-1])
                loss_kd, grad_kd = _kl_divergence_with_grad(
                    logits[:, :min_v],
                    teacher_data[:S, :min_v].astype(np.float32),
                    temperature=2.0,
                )
                if min_v < logits.shape[-1]:
                    pad = np.zeros((S, logits.shape[-1] - min_v), dtype=np.float32)
                    grad_kd = np.concatenate([grad_kd, pad], axis=-1)
            else:
                loss_kd, grad_kd = 0.0, np.zeros_like(logits)

            loss = 0.3 * loss_ce + 0.6 * loss_kd
            grad_logits = (0.3 * grad_ce + 0.6 * grad_kd).astype(np.float32)

            # ── Backward ─────────────────────────────────────────────
            opt.begin_step()

            # GPU backward disabled — vsa_lm_backward doesn't handle MindForge
            # LoRA layers, produces wrong gradients. Use CPU backward which works.
            # TODO: add MindForge backward to the C++ vsa_lm_backward kernel.
            gpu_backward_ok = False
            if False:  # model._gpu_handle is not None and _gc is not None:
                try:
                    grads = _gc.vsa_lm_backward(
                        model._gpu_dev, model._gpu_handle,
                        ids.astype(np.int32),
                        grad_logits.astype(np.float32),
                    )
                    # Apply all gradients via AdamW
                    opt.step(model.embed, grads["grad_embed"])
                    opt.step(model.out_w, grads["grad_out_w"])
                    if "grad_pos" in grads:
                        opt.step(model.pe, grads["grad_pos"])
                    for li in range(cfg.n_layers):
                        for key, attr in [
                            (f"grad_ffn_up_{li}", "ffn_up"),
                            (f"grad_ffn_down_{li}", "ffn_down"),
                        ]:
                            if key in grads:
                                wp = getattr(model.layers[li], attr).weight_patterns
                                opt.step(wp, grads[key])
                        for key, attr in [
                            (f"grad_ln_g_{li}", "ln_g"),
                            (f"grad_ln_b_{li}", "ln_b"),
                        ]:
                            if key in grads:
                                opt.step(getattr(model.layers[li], attr), grads[key])
                    # MindForge grads from GPU (if returned)
                    for fk in ["A_basis", "B_basis", "W_coeff", "b_coeff",
                               "W_h", "b_h", "W_proj", "layer_embeddings",
                               "ln_g", "ln_b"]:
                        gkey = f"grad_forge_{fk}"
                        if gkey in grads:
                            opt.step(getattr(model.forge, fk), grads[gkey])
                    model._reupload_gpu()
                    gpu_backward_ok = True
                except Exception as e:
                    logger.warning("GPU backward failed: {}, using CPU", e)

            # CPU backward fallback
            if not gpu_backward_ok:
                model.forge.grads = model.forge.zero_grads()
                for layer in model.layers:
                    for k in layer.grads:
                        layer.grads[k].fill(0.0)

                # Forward with caches
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
                opt.step(model.out_w, grad_out_w)

                for layer, cache in reversed(list(zip(model.layers, caches))):
                    dx = layer.backward(dx, cache)

                for layer in model.layers:
                    opt.step(layer.ffn_up.weight_patterns, layer.grads["ffn_up_w"])
                    opt.step(layer.ffn_down.weight_patterns, layer.grads["ffn_down_w"])
                    opt.step(layer.ln_g, layer.grads["ln_g"])
                    opt.step(layer.ln_b, layer.grads["ln_b"])

                for fk, fg in model.forge.grads.items():
                    opt.step(getattr(model.forge, fk), fg)

                for t in range(S):
                    tok = int(ids[t])
                    if 0 <= tok < cfg.vocab_size:
                        model.embed[tok] -= lr * np.clip(dx[t], -0.1, 0.1)

            # ── Logging ──────────────────────────────────────────────
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

            if step % cfg.save_every == 0 and step > 0:
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

        if step >= train_steps:
            break

    elapsed = time.time() - t0
    logger.info("Distillation done: {} steps in {:.0f}s, best_loss={:.3f}",
                step, elapsed, best_loss)


if __name__ == "__main__":
    main()
