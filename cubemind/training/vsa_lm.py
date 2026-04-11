"""VSA-LM: Vector Symbolic Architecture Language Model.

Beat FlashLM's 1.36 PPL on TinyStories with:
- BlockCode embeddings (discrete, compositional)
- AdditionLinear (no matmul, L1 distance)
- MindForge adaptive weights (context-dependent LoRA)
- HippocampalFormation temporal memory (place/grid/time cells)
- GIFNeuron gating (spiking dynamics for temporal flow)

Target: PPL < 1.36, ~30M params, CPU trainable.

Architecture:
  token → BlockCode embed → [VSA Layer × N] → output projection → logits

  VSA Layer:
    1. Hippocampal update: h_state = hippo.step(x_mean) → temporal context
    2. Context block-code: bind(x_mean_bc, h_state_bc) → context
    3. MindForge: context → forged LoRA (A, B) for this layer
    4. Feed-forward: AdditionLinear(x) + LoRA(x) → y
    5. Residual: x = x + y

Run: python -u sandbox/vsa_lm/experiment.py
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


CAPSULE_DIM = 32  # Cognitive feature dimensions (from Novel_GNN_Arch capsules)

# Capsule feature indices
CAP_VALENCE = 0
CAP_AROUSAL = 1
CAP_DOMINANCE = 2
CAP_THREAT = 3
CAP_REWARD = 4
CAP_NOVELTY = 5
CAP_UNCERTAINTY = 6
CAP_SALIENCE = 7
CAP_TRUST = 8
CAP_PLASTICITY = 14
CAP_STABILITY = 16
CAP_CONSOLIDATION = 18


@dataclass
class VSALMConfig:
    # VSA dimensions
    k: int = 16              # blocks (smaller than production K=80 for speed)
    l: int = 24              # block length
    # Model
    d_model: int = 384       # matches FlashLM
    n_layers: int = 18       # matches FlashLM
    d_ffn: int = 1152        # matches FlashLM (3x d_model)
    # MindForge
    forge_rank: int = 8      # LoRA rank for forged adapters
    forge_basis: int = 16    # number of basis adapters
    # Hippocampal
    n_place: int = 32
    n_time: int = 16
    n_grid: int = 24
    # Training
    vocab_size: int = 8192   # matches FlashLM BPE
    seq_len: int = 256       # matches FlashLM
    batch_size: int = 1      # online learning
    lr: float = 5e-4
    lr_min: float = 1e-5
    warmup: int = 1000
    train_steps: int = 200000
    val_every: int = 10
    save_every: int = 5000
    max_stories: int = 0     # 0 = all (958M tokens)
    seed: int = 42


class LiquidCell:
    """Continuous-time liquid state machine with input-dependent time constants.

    From AURA_GENESIS: tau(x) = tau_min + softplus(V@x + c),
    dh/dt = -h/tau + tanh(W@h + U@x + b). Euler integrated.
    Equivalent to FlashLM's ParallelGatedRecurrence but physics-based.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dt: float = 0.02,
                 tau_min: float = 0.02, tau_max: float = 2.0, seed: int = 42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / (hidden_dim + hidden_dim)),
                            (hidden_dim, hidden_dim)).astype(np.float32)
        self.U = rng.normal(0, np.sqrt(2.0 / (in_dim + hidden_dim)),
                            (hidden_dim, in_dim)).astype(np.float32)
        self.b = np.zeros(hidden_dim, dtype=np.float32)
        self.V = rng.normal(0, np.sqrt(2.0 / (in_dim + hidden_dim)),
                            (hidden_dim, in_dim)).astype(np.float32)
        self.c = rng.normal(0, 0.1, (hidden_dim,)).astype(np.float32)
        self.h = np.zeros(hidden_dim, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        """One step of liquid dynamics. x: (d,) → h: (d,)."""
        x = x.astype(np.float32).ravel()[:self.in_dim]
        if len(x) < self.in_dim:
            x = np.pad(x, (0, self.in_dim - len(x)))

        # Input-dependent time constant
        vx = self.V @ x + self.c
        tau = self.tau_min + np.log1p(np.exp(-np.abs(vx))) + np.maximum(vx, 0)  # softplus
        tau = np.minimum(tau, self.tau_max)

        # Continuous-time dynamics: dh/dt = -h/tau + tanh(W@h + U@x + b)
        a = np.tanh(self.W @ self.h + self.U @ x + self.b)
        dh = -self.h / np.maximum(tau, 1e-6) + a

        # Euler integration
        self.h = (self.h + self.dt * dh).astype(np.float32)
        return self.h.copy()

    def param_count(self) -> int:
        return self.W.size + self.U.size + self.b.size + self.V.size + self.c.size


class VSALayer:
    """One VSA-LM layer: LiquidCell temporal memory + forged weights + addition-only FFN."""

    def __init__(self, cfg: VSALMConfig, layer_id: int, forge: MindForge,
                 hippo: HippocampalFormation, bc: BlockCodes, seed: int = 42):
        self.cfg = cfg
        self.layer_id = layer_id
        self.forge = forge
        self.hippo = hippo
        self.bc = bc
        d = cfg.d_model

        # Addition-only FFN: d → d_ffn → d (no matmul)
        self.ffn_up = AdditionLinear(d, cfg.d_ffn, bias=True, seed=seed)
        self.ffn_down = AdditionLinear(cfg.d_ffn, d, bias=True, seed=seed + 1)
        self.sign = SignActivation()

        # LiquidCell: continuous-time recurrence (replaces EMA)
        self.liquid = LiquidCell(d, d, dt=0.02, seed=seed + 2)

        # GIF neuron for temporal gating
        self.gif = GIFNeuron(input_dim=d, hidden_dim=d, L=8, seed=seed + 3)

        # Layer norm
        self.ln_g = np.ones(d, dtype=np.float32)
        self.ln_b = np.zeros(d, dtype=np.float32)

    def layernorm(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.ln_g * (x - mean) / np.sqrt(var + 1e-5) + self.ln_b

    def forward(self, x: np.ndarray, novelty: float = 0.5,
                plasticity: float = 0.5, consolidation: float = 0.5,
                loss: float = 0.0) -> np.ndarray:
        """x: (seq, d) → (seq, d). Capsule signals modulate behavior."""
        S, d = x.shape

        # 1. LayerNorm
        h = self.layernorm(x)

        # 2. Hippocampal recall: retrieve similar past context → prime LiquidCell
        x_mean = h.mean(axis=0)
        try:
            retrieved = self.hippo.retrieve_similar_memories(x_mean, k=3)
            if retrieved and len(retrieved) > 0:
                # Blend retrieved memories into a recall vector
                recall = np.zeros(d, dtype=np.float32)
                n_valid = 0
                for r in retrieved:
                    feats = None
                    if isinstance(r, dict):
                        feats = r.get("features")
                    elif isinstance(r, (tuple, list)) and len(r) >= 2:
                        candidate = r[1] if isinstance(r[1], np.ndarray) else r[0]
                        if isinstance(candidate, np.ndarray) and candidate.shape == (d,):
                            feats = candidate
                    if feats is not None and feats.shape == (d,):
                        recall += feats.astype(np.float32)
                        n_valid += 1
                if n_valid > 0:
                    recall /= n_valid
                    # Inject recall into LiquidCell state (gentle blend)
                    recall_strength = 0.1 + 0.2 * consolidation  # [0.1, 0.3]
                    self.liquid.h = (1 - recall_strength) * self.liquid.h + recall_strength * recall
        except Exception:
            pass  # Recall failure is non-fatal

        # 3. Store current context for future recall (high-loss = high priority)
        if loss > 1.0:  # Only store interesting (high-loss) contexts
            try:
                self.hippo.create_episodic_memory(features=x_mean)
            except Exception:
                pass

        # 4. LiquidCell: continuous-time temporal dynamics
        # Capsule endocrine modulation
        self.liquid.dt = 0.01 + 0.03 * plasticity
        self.liquid.tau_min = 0.02 + 0.08 * consolidation
        temporal_ctx = self.liquid.step(x_mean)

        # 3. GIF gating: temporal modulation
        gif_input = temporal_ctx.reshape(1, d)
        spikes, _ = self.gif.forward(gif_input)
        gate = spikes.ravel()  # (d,) — spike-based gate

        # 4. MindForge: forge context-dependent LoRA adapter
        # Create block-code from temporal context
        ctx_flat = temporal_ctx[:self.cfg.k * self.cfg.l]
        if len(ctx_flat) < self.cfg.k * self.cfg.l:
            ctx_flat = np.pad(ctx_flat, (0, self.cfg.k * self.cfg.l - len(ctx_flat)))
        ctx_bc = self.bc.discretize(ctx_flat.reshape(self.cfg.k, self.cfg.l))
        A, B = self.forge.forge(ctx_bc, self.layer_id)  # (rank, d), (d, rank)

        # 5. Feed-forward: AdditionLinear (no matmul!) + forged LoRA
        # Normalize AdditionLinear output by sqrt(in_features) to control scale
        h_up_raw = self.ffn_up.forward(h) / np.sqrt(self.ffn_up.in_features)
        h_up = self.sign.forward(h_up_raw)                   # (S, d_ffn)
        h_ffn = self.ffn_down.forward(h_up) / np.sqrt(self.ffn_down.in_features)  # (S, d)

        # LoRA path: h @ B @ A (this uses matmul — but it's low-rank)
        h_lora = (h @ B @ A).astype(np.float32)  # (S, d)

        # 6. Capsule-modulated combination + residual
        # Novelty scales LoRA strength: high novelty → stronger adaptation
        lora_scale = 0.5 + novelty  # range [0.5, 1.5]
        # Plasticity scales FFN contribution: high plasticity → more change
        ffn_scale = 0.5 + plasticity  # range [0.5, 1.5]
        y = ffn_scale * h_ffn + lora_scale * gate * h_lora
        return x + y


class VSALM:
    """VSA Language Model."""

    def __init__(self, cfg: VSALMConfig):
        self.cfg = cfg
        d = cfg.d_model
        rng = np.random.default_rng(cfg.seed)

        # Block-code VSA engine
        self.bc = BlockCodes(k=cfg.k, l=cfg.l)

        # Token embedding: vocab → d_model
        self.embed = rng.normal(0, 0.02, (cfg.vocab_size, d)).astype(np.float32)

        # Capsule embedding: vocab → CAPSULE_DIM (cognitive features per token)
        # Each token has learned affect/control/domain features
        self.capsule_embed = rng.normal(0, 0.01, (cfg.vocab_size, CAPSULE_DIM)).astype(np.float32)
        # Capsule projection: CAPSULE_DIM → d_model (fuse capsule into hidden state)
        self.capsule_proj = rng.normal(0, 0.02, (d, CAPSULE_DIM)).astype(np.float32)

        # Positional encoding (sinusoidal)
        pe = np.zeros((cfg.seq_len + 16, d), dtype=np.float32)
        pos = np.arange(cfg.seq_len + 16).reshape(-1, 1).astype(np.float32)
        div = np.exp(np.arange(0, d, 2).astype(np.float32) * -(math.log(10000) / d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.pe = pe

        # Output projection
        self.out_w = rng.normal(0, 0.02, (cfg.vocab_size, d)).astype(np.float32)

        # Hippocampal formation (shared across layers)
        self.hippo = HippocampalFormation(
            feature_dim=d, max_memories=500,
            n_place_cells=cfg.n_place, n_time_cells=cfg.n_time,
            n_grid_cells=cfg.n_grid, seed=cfg.seed)

        # MindForge (shared, generates per-layer adapters)
        self.forge = MindForge(
            k=cfg.k, l=cfg.l, n_layers=cfg.n_layers,
            d_target=d, rank=cfg.forge_rank,
            n_basis=cfg.forge_basis, d_hidden=256,
            seed=cfg.seed)

        # VSA layers
        self.layers = [
            VSALayer(cfg, l, self.forge, self.hippo, self.bc, seed=cfg.seed + l)
            for l in range(cfg.n_layers)
        ]

        self._last_hidden = None

    def forward(self, input_ids: np.ndarray, prev_loss: float = 0.0) -> np.ndarray:
        """input_ids: (seq,) → logits: (seq, vocab). prev_loss enables hippocampal storage."""
        S = len(input_ids)

        # Token embedding + positional encoding
        x = self.embed[input_ids] + self.pe[:S]

        # Capsule embedding: cognitive features per token
        caps = self.capsule_embed[input_ids]  # (S, CAPSULE_DIM)
        caps_proj = (caps @ self.capsule_proj.T).astype(np.float32)  # (S, d)
        x = x + caps_proj

        # Extract per-sequence capsule signals for layer modulation
        caps_mean = caps.mean(axis=0)
        novelty = float(np.clip(caps_mean[CAP_NOVELTY], 0, 1))
        plasticity = float(np.clip(caps_mean[CAP_PLASTICITY], 0, 1))
        consolidation = float(np.clip(caps_mean[CAP_CONSOLIDATION], 0, 1))

        for layer in self.layers:
            x = layer.forward(x, novelty=novelty, plasticity=plasticity,
                               consolidation=consolidation, loss=prev_loss)

        self._last_hidden = x.copy()  # save for gradient computation
        logits = (x @ self.out_w.T / np.sqrt(self.cfg.d_model)).astype(np.float32)
        return logits

    def param_count(self) -> int:
        """Count total parameters."""
        n = self.embed.size + self.pe.size + self.out_w.size + self.capsule_embed.size + self.capsule_proj.size
        n += sum(p.size for p in [self.forge.W_proj, self.forge.W_h, self.forge.b_h,
                                   self.forge.W_coeff, self.forge.b_coeff])
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
            n += layer.liquid.param_count()
        return n


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(cfg: VSALMConfig, data_dir="sandbox/vsa_lm/data"):
    """Load pre-tokenized TinyStories from .npy (instant)."""
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


# ── Loss + eval ──────────────────────────────────────────────────────────────

def compute_ppl(model, x_data, y_data, max_samples=100):
    total_loss, n = 0.0, 0
    for i in range(min(len(x_data), max_samples)):
        logits = model.forward(x_data[i])
        p = softmax(logits)
        for t in range(len(y_data[i])):
            tok = int(y_data[i][t])
            if 0 <= tok < p.shape[1]:
                total_loss -= math.log(max(float(p[t, tok]), 1e-8))
                n += 1
    return float(math.exp(min(total_loss / max(n, 1), 20)))


# ── Main ─────────────────────────────────────────────────────────────────────

def main(train_steps: int = 10000, n_layers: int = 6, d_model: int = 256,
         seq_len: int = 64):
    cfg = VSALMConfig(
        train_steps=train_steps,
        n_layers=n_layers,
        seq_len=seq_len,
        val_every=50,
        d_model=d_model,
        d_ffn=d_model * 3,
        k=16,
        l=16,                # d_vsa = 256 = d_model
        forge_rank=8,
        forge_basis=16,
        n_place=32,
        n_time=16,
        n_grid=24,
        lr=5e-4,
    )

    train_x, train_y, val_x, val_y, actual_vocab = load_data(cfg)
    cfg.vocab_size = actual_vocab

    model = VSALM(cfg)
    n_params = model.param_count()
    logger.info("VSA-LM: d={}, layers={}, vocab={}, params={:.1f}M",
                cfg.d_model, cfg.n_layers, cfg.vocab_size, n_params / 1e6)
    logger.info("  VSA: K={}, L={}, d_vsa={}", cfg.k, cfg.l, cfg.k * cfg.l)
    logger.info("  MindForge: rank={}, basis={}", cfg.forge_rank, cfg.forge_basis)

    # Validate forward works
    logger.info("Sanity check...")
    test_logits = model.forward(train_x[0])
    logger.info("Forward OK: {} finite={}", test_logits.shape, np.all(np.isfinite(test_logits)))

    # Initial PPL
    ppl = compute_ppl(model, val_x, val_y, max_samples=20)
    logger.info("Initial PPL: {:.1f} (random weights, should be ~vocab={})", ppl, cfg.vocab_size)

    # ── Quick training loop: Hebbian + error-driven updates ──
    # AdditionLinear weights updated via error-driven delta rule (no backprop)
    # Embed + out_w updated via simple gradient (the only matmul parts)
    logger.info("Starting training (1000 steps, small scale)...")

    t0 = time.time()
    best_ppl = ppl
    step = 0
    loss = 0.0

    while step < cfg.train_steps:
        perm = np.random.permutation(len(train_x))
        train_x_s, train_y_s = train_x[perm], train_y[perm]

        for i in range(len(train_x_s)):
            if step >= cfg.train_steps:
                break

            ids = train_x_s[i]
            labels = train_y_s[i].ravel()
            S = len(ids)

            # Forward with hippocampal recall + capsule modulation
            prev_loss = loss if step > 0 else 0.0
            logits = model.forward(ids, prev_loss=prev_loss)
            x_final = model._last_hidden  # saved by forward()

            # CE loss + softmax grad
            probs = softmax(logits)
            loss = -np.mean(np.log(probs[np.arange(S), labels] + 1e-8))
            grad_logits = probs.copy()
            grad_logits[np.arange(S), labels] -= 1.0
            grad_logits /= S
            grad_out_w = grad_logits.T @ x_final / S
            model.out_w -= cfg.lr * np.clip(grad_out_w, -0.1, 0.1)

            # ── Update embedding (scatter-add) ──
            dx = (grad_logits @ model.out_w).astype(np.float32)
            for t in range(S):
                tok = int(ids[t])
                if 0 <= tok < cfg.vocab_size:
                    model.embed[tok] -= cfg.lr * np.clip(dx[t], -0.1, 0.1)
                    # Update capsule embedding (backprop through capsule_proj)
                    dcaps = (dx[t] @ model.capsule_proj).astype(np.float32)
                    model.capsule_embed[tok] -= cfg.lr * np.clip(dcaps, -0.1, 0.1)

            # ── Update AdditionLinear weights (error-driven, no backprop) ──
            # Approximate: push weight patterns toward input for correct tokens
            error_signal = dx.mean(axis=0)  # (d,) average error direction
            for layer in model.layers:
                # FFN up: push patterns toward input mean
                h_mean = x_final.mean(axis=0)[:layer.ffn_up.in_features]
                delta = np.outer(error_signal[:layer.ffn_up.out_features],
                                  h_mean[:layer.ffn_up.in_features])
                if delta.shape == layer.ffn_up.weight_patterns.shape:
                    layer.ffn_up.weight_patterns -= cfg.lr * 0.1 * np.clip(delta, -0.05, 0.05)

            if step % cfg.val_every == 0:
                val_ppl = compute_ppl(model, val_x, val_y, max_samples=20)
                elapsed = time.time() - t0
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                logger.info("step={:5d} | loss={:.3f} | PPL={:.1f} | {:.1f} stp/s",
                            step, float(loss), val_ppl, sps)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl

            step += 1

    elapsed = time.time() - t0
    logger.info("Training done: {} steps in {:.0f}s, best PPL={:.1f}", step, elapsed, best_ppl)


if __name__ == "__main__":
    main()
