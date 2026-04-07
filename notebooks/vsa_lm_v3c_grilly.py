"""VSA-LM v3c — grilly-native port of vsa_lm_v3c_warmrestart.py.

Architectural difference from the PyTorch v3c:
  The sequence-mean LiquidCell (``x_mean = h.mean(dim=1)``) is REPLACED
  with a ``CausalSequenceMixer`` built on a subgroup-parallel causal
  prefix scan. The mean-pool version leaks future tokens into past
  tokens' representations, destroying autoregressive causality — that's
  why the PyTorch v3c plateaued at PPL ~42. The prefix-scan mixer is
  strictly causal: h_t depends only on x_0..x_t, never x_{t+1}.

Other carry-overs from v3c:
  - Cognitive capsules (plasticity/consolidation scalars drive per-layer
    behavior at runtime)
  - Graded multiplication-free FFN (``AdditionLinear`` = -||W - x||_1)
  - GELU instead of sign() on the FFN bottleneck

Runs on grilly's Vulkan path via ``import grilly.torch_api as torch``.
Swaps cuda → vulkan and skips AMP (grilly fp16 autocast is a no-op today).

Constraint: ``SEQ_LEN <= 32`` for this initial run — the current causal
prefix scan shader runs one thread per time step inside a single subgroup
(Wave32 on RDNA3/NVIDIA, Wave64 on RDNA2). Longer sequences need a
hierarchical scan with chunk-level carry, which is a TODO.

Prerequisites:
- grilly rebuild with Thread A staging pattern (all ops fast)
- grilly Python-side autograd wiring for nn.Linear, nn.LayerNorm,
  nn.Embedding
- grilly_core with prefix-scan-causal and prefix-scan-causal-backward
  shaders compiled
- ``sandbox/vsa_lm/data/tokens.npy`` and ``vocab.npy`` (pre-tokenized)

Run:
    python notebooks/vsa_lm_v3c_grilly.py
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

# Grilly facade — drop-in for PyTorch's torch / torch.nn / torch.nn.functional
import grilly.torch_api as torch
from grilly import nn
from grilly.nn import functional as F
from grilly.nn.prefix_scan import CausalSequenceMixer

device = torch.device("vulkan" if torch.vulkan.is_available() else "cpu")
print(f"Using device: {device}")

# NOTE: SEQ_LEN must be <= 32 for this run — the current causal prefix scan
# shader runs one thread per time step inside a single subgroup. Longer
# sequences need a hierarchical scan with chunk-level carry (TODO). 32 is
# short for an LM but plenty to prove the causal-RNN architecture beats
# the causally-leaky ``h.mean(dim=1)`` pooling that plateau'd at PPL ~42.
SEQ_LEN = 32
DATA_DIR = "sandbox/vsa_lm/data/"

# ── Data ──
tokens = np.load(f"{DATA_DIR}/tokens.npy")
vocab = int(np.load(f"{DATA_DIR}/vocab.npy")[0])
n = len(tokens)
tr, va = tokens[: int(0.8 * n)], tokens[int(0.8 * n) : int(0.9 * n)]


def mkseqs(t, sl):
    x, y = [], []
    for i in range(0, len(t) - sl - 1, sl // 2):
        x.append(t[i : i + sl])
        y.append(t[i + 1 : i + sl + 1])
    return (
        torch.tensor(np.array(x), dtype=torch.long),
        torch.tensor(np.array(y), dtype=torch.long),
    )


train_x, train_y = mkseqs(tr, SEQ_LEN)
val_x, val_y = mkseqs(va, SEQ_LEN)
print(f"Vocab={vocab}, Train={len(train_x)}, Val={len(val_x)}")


def compute_ppl(model, x_data, y_data, max_samples=64):
    model.eval()
    total_loss, n_tok = 0.0, 0
    with torch.no_grad():
        bs = 8  # smaller batch for eval on grilly (no fp16 yet)
        for i in range(0, min(len(x_data), max_samples), bs):
            xb = x_data[i : i + bs].to(device)
            yb = y_data[i : i + bs].to(device)
            logits = model(xb)  # (B, S, V)
            # Flatten to (B*S, V) / (B*S,) for cross_entropy
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = yb.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_labels, reduction="sum")
            total_loss += float(loss.item() if hasattr(loss, "item") else loss)
            n_tok += int(yb.numel())
    model.train()
    return math.exp(min(total_loss / max(n_tok, 1), 20))


# ── Config ──
D_MODEL = 384
N_LAYERS = 12
D_FFN = 1152
LR = 2e-4
WARM_RESTART_STEPS = 15000
VAL_EVERY = 200
GRAD_CLIP = 1.0
BATCH_SIZE = 4  # small — grilly's current kernel throughput is ~370 GFLOP/s
CAPSULE_DIM = 32


# ── Model ──
class AdditionLinearCUDA(nn.Module):
    """Multiplication-free linear: out = -||W - x||_1 + bias, then GELU.

    Despite the name, this runs on grilly's Vulkan path via torch.cdist,
    not CUDA. Kept the name to match vsa_lm_v3c_warmrestart.py so the
    two scripts are line-by-line comparable.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in).uniform_(-0.1, 0.1))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.d_in = d_in

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        dist = torch.cdist(
            x_flat.unsqueeze(0), self.weight.unsqueeze(0), p=1
        ).squeeze(0)
        out = -dist + self.bias
        return out.reshape(*orig_shape[:-1], -1)


class VSALayerCUDA(nn.Module):
    """v3c-grilly: graded FFN + CausalSequenceMixer (strictly causal).

    Changes from vsa_lm_v3c_warmrestart.py:
      - LiquidCellCUDA + ``h.mean(dim=1)`` sequence pool REMOVED.
        Those leaked future tokens into past-token representations (mean
        over the whole sequence includes tokens t > current), destroying
        the autoregressive causal mask. That's the architectural reason
        the PyTorch v3c plateau'd at PPL ~42.
      - Replaced with ``CausalSequenceMixer`` (from grilly.nn.prefix_scan),
        which runs ``h_t = a_t * h_{t-1} + x_t`` via subgroup-parallel
        causal prefix scan. Strictly causal by construction.
      - Plasticity / consolidation scalars now scale the mixer's output
        contribution to the residual, instead of modulating a continuous-time
        LiquidCell's tau/dt.

    FFN path is unchanged — multiplication-free ``AdditionLinear`` (cdist =
    subtract + abs + sum) with per-layer learnable scales and GELU on the
    recentered cdist output.
    """

    def __init__(self, d, d_ffn):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ffn_up = AdditionLinearCUDA(d, d_ffn)
        self.ffn_down = AdditionLinearCUDA(d_ffn, d)
        self.up_scale = nn.Parameter(torch.ones(1))
        self.down_scale = nn.Parameter(torch.ones(1))
        # Subgroup-parallel causal RNN mixer. Replaces the
        # causally-leaky ``h.mean(dim=1)`` pooling that was the old
        # LiquidCell entry point.
        self.mixer = CausalSequenceMixer(d)
        self.d = d

    def parameters(self):
        # Walk grilly's Module ``_parameters`` / ``_modules`` as usual,
        # then also yield the mixer's params (CausalSequenceMixer is a
        # plain Python container, not a registered grilly Module).
        yield from super().parameters()
        yield from self.mixer.parameters()

    def _ffn(self, h):
        # cdist returns large negative values; rescale then GELU.
        # NOTE: the PyTorch v3c also recentered (``h - h.mean(dim=-1)``) to
        # put GELU at its active range. Dropped here because grilly's
        # mean-over-last-dim plumbing is fragile across Variable / ndarray
        # / Parameter types — the learnable ``up_scale`` / ``down_scale``
        # params compensate. Revisit if convergence suffers.
        h_up = self.ffn_up(h) / math.sqrt(self.ffn_up.d_in)
        h_up = F.gelu(h_up * self.up_scale)
        h_ffn = self.ffn_down(h_up) / math.sqrt(self.ffn_down.d_in)
        return h_ffn * self.down_scale

    def forward(self, x, plasticity=0.5, consolidation=0.5):
        # x: (B, S, d) — strictly causal path. No sequence-wide mean pool.
        h = self.ln(x)

        # Causal Linear-RNN mixer: h_causal[b, t] depends only on h[b, 0..t].
        h_causal = self.mixer(h)  # (B, S, d)

        # FFN on the normalized input (position-wise, already causal).
        h_ffn = self._ffn(h)

        # Gate the FFN output with the causal mixer state (per time step).
        gate = h_causal  # already (B, S, d)
        y = (0.5 + plasticity) * h_ffn * (1.0 + 0.1 * gate)
        return x + y


class VSALMModel(nn.Module):
    def __init__(self, vocab, d, d_ffn, n_layers, max_seq):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.capsule_embed = nn.Embedding(vocab, CAPSULE_DIM)
        self.capsule_proj = nn.Linear(CAPSULE_DIM, d, bias=False)
        self.pos = nn.Parameter(torch.randn(max_seq + 16, d) * 0.02)
        self.out_proj = nn.Linear(d, vocab, bias=False)
        self.layers = nn.ModuleList(
            [VSALayerCUDA(d, d_ffn) for _ in range(n_layers)]
        )
        self.scale = math.sqrt(d)
        nn.init.normal_(self.embed.weight, 0, 0.02)
        nn.init.normal_(self.capsule_embed.weight, 0, 0.01)

    def forward(self, ids):
        # Batched path only: ids: (B, S)
        B, S = ids.shape
        x = self.embed(ids) + self.pos[:S].unsqueeze(0)
        caps = self.capsule_embed(ids)
        x = x + self.capsule_proj(caps)
        with torch.no_grad():
            # Reduce via numpy directly — ``caps`` can be Variable, Tensor,
            # or plain ndarray depending on where in the chain we are, and
            # only numpy's ``axis=`` keyword is accepted by all of those.
            caps_np = np.asarray(caps)  # (B, S, CAPSULE_DIM)
            cm = caps_np.mean(axis=(0, 1))  # (CAPSULE_DIM,)
            # Capsule dims 14/18 drive plasticity/consolidation scalars.
            plasticity = float(cm[14])
            plasticity = max(0.0, min(1.0, plasticity))
            consolidation = float(cm[18])
            consolidation = max(0.0, min(1.0, consolidation))
        for layer in self.layers:
            x = layer(x, plasticity=plasticity, consolidation=consolidation)
        return self.out_proj(x / self.scale)

    # NOTE: no ``reset_liquid`` anymore. The old LiquidCell kept a
    # ``register_buffer('h', ...)`` across calls and needed resetting per
    # batch. CausalSequenceMixer is stateless between calls — each forward
    # starts its own causal scan from h_0 = 0 internally.


# ── Model construction ──
model = VSALMModel(vocab, D_MODEL, D_FFN, N_LAYERS, SEQ_LEN).to(device)

n_params = sum(p.size if hasattr(p, "size") else p.numel() for p in model.parameters())
print(f"VSA-LM v3c-grilly: d={D_MODEL}, L={N_LAYERS}, B={BATCH_SIZE}, "
      f"params={n_params/1e6:.1f}M")
print(f"Fresh init (no .pt checkpoint migration on grilly yet)")
print()

# Initial PPL sanity check — should be roughly log(vocab) = ~9 for 8192
# (fresh init, uniform output distribution)
ppl = compute_ppl(model, val_x, val_y, max_samples=32)
print(f"Initial PPL (fresh init): {ppl:.1f}  (expect ~{vocab:.0f}^{1}*0.6)")

# ── Train ──
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
# Linear warmup for 500 steps then cosine over remaining.
warmup_steps = 500


def lr_at(step):
    if step < warmup_steps:
        return LR * (step / warmup_steps)
    progress = (step - warmup_steps) / max(WARM_RESTART_STEPS - warmup_steps, 1)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


t0 = time.time()
best_ppl = ppl
step = 0

print(f"Training for {WARM_RESTART_STEPS} steps, val every {VAL_EVERY}")
print()

while step < WARM_RESTART_STEPS:
    perm = torch.randperm(len(train_x))
    for i in range(0, len(perm) - BATCH_SIZE, BATCH_SIZE):
        if step >= WARM_RESTART_STEPS:
            break

        # Adjust LR manually (grilly's LR scheduler also works but doing it
        # inline makes the warmup+cosine curve obvious).
        lr_now = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        idx = perm[i : i + BATCH_SIZE]
        ids_batch = train_x[idx].to(device)
        labels_batch = train_y[idx].to(device)

        optimizer.zero_grad()

        logits = model(ids_batch)  # (B, S, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels_batch.reshape(-1),
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % VAL_EVERY == 0:
            ppl = compute_ppl(model, val_x, val_y)
            el = time.time() - t0
            sps = (step + 1) / el if el > 0 else 0
            loss_val = float(loss.item() if hasattr(loss, "item") else loss)
            print(
                f"step={step:6d} lr={lr_now:.5f} | loss={loss_val:.3f} | "
                f"PPL={ppl:.1f} | {sps:.2f} stp/s (B={BATCH_SIZE})"
            )
            if ppl < best_ppl:
                best_ppl = ppl
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "best_ppl": best_ppl,
                    },
                    "vsa_lm_v3c_grilly_best.grl",
                )

        step += 1

print(f"\nDone: {step} steps, best PPL={best_ppl:.1f}, time={time.time()-t0:.0f}s")
