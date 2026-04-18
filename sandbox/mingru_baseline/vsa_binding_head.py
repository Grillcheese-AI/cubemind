"""VSA binding head — replaces the dense LM vocab projection with a
fixed MAP-bipolar codebook lookup.

Rationale (see the Colab validation run plan):

    Standard LM head : Linear(d_model, V) = d_model * V params
                       (e.g. 512 * 32768 = 16.7M; or 1024 * 32768 = 33.6M)

    VSA binding head : Linear(d_model, D) + fixed codebook (V, D)
                       = d_model * D params (codebook has ZERO params)
                       (e.g. 512 * 10240 = 5.2M; or 1024 * 10240 = 10.5M)

    → ~3× fewer learned params for the same vocab throughput.

The codebook is a fixed ``{-1, +1}`` matrix derived deterministically
from a seed + token_id so every rank/run sees the same codes. At the
head we:

  1. project the hidden state h into the VSA space (continuous — NO
     sign() anywhere, the "destroyed gradient" pitfall from the prior
     STE-sign VSA experiments);
  2. cosine-similar vs the codebook — both the query and each row of
     the codebook are L2-normalized, so the dot product is cos θ ∈ [-1, 1];
  3. scale by τ = sqrt(D) so logits have typical unit variance at init
     (pre-softmax activations roughly match the scale Linear heads
     produce);
  4. softmax + CE as usual.

The embedding layer is **left untied** — callers keep their learned
``nn.Embedding(V, d_model)``. The binding head replaces only the output
side. Tying the embedding to the codebook (full decoupling, ~57M
saved) is a follow-up ablation; we don't do it on the first pass
because the model loses the ability to adjust input-side token
representations, which historically costs a few PPL points on small
training budgets.

Usage::

    from vsa_binding_head import VSABindingHead

    head = VSABindingHead(d_model=512, vocab_size=32768, d_vsa=10240)
    logits = head(h)                 # (B, S, V) — plug into your CE loss
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Codebook construction ───────────────────────────────────────────────

def build_vsa_codebook(
    vocab_size: int,
    d_vsa: int,
    seed: int = 0xC0DEB00C,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a deterministic ``(V, D)`` MAP-bipolar codebook as a
    float tensor with values in ``{-1, +1}``.

    Deterministic across runs and ranks given the same ``seed`` — uses
    ``numpy.random.default_rng(seed)`` which is platform-stable. We
    generate in one shot rather than per-token because the codebook is
    static across training and contiguous GPU memory matters more than
    construction speed.

    For strict cross-repo codebook sharing with
    ``cubemind/utils/stable_hash.py`` / ``opcode-vsa-rs/src/vsa_hash.rs``,
    swap the PRNG line for BLAKE3-per-token. Runtime-equivalent, bytes
    differ. That's a separate concern from the training-head design so
    we keep the simple PRNG here and flip later if the trained head
    needs to round-trip through the Rust VM.
    """
    rng = np.random.default_rng(seed)
    # Bernoulli(0.5) → {0, 1} → {-1, +1}.
    bits = rng.integers(0, 2, size=(vocab_size, d_vsa), dtype=np.int8)
    codes = (bits.astype(np.int8) * 2 - 1).astype(np.float32)
    t = torch.from_numpy(codes).to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


# ── Binding head module ─────────────────────────────────────────────────

class VSABindingHead(nn.Module):
    """LM head that reads out tokens via cosine similarity to a fixed
    bipolar VSA codebook.

    Args:
        d_model:   backbone hidden size (e.g. 512, 1024)
        vocab_size: number of token classes
        d_vsa:     VSA hypervector dimension. Defaults to 10240
                   (K_BLOCKS=80 × L_BLOCK=128, the CubeMind constants).
        temperature: logit scale. ``None`` → ``sqrt(d_vsa)`` (default),
                   a float for a fixed override, or ``"learned"`` for a
                   single trainable scalar.
        seed:      codebook seed for determinism across runs.

    Parameters:
        query_proj : Linear(d_model, d_vsa, bias=False)

    Buffers (NOT parameters — no gradients):
        codebook  : (V, d_vsa) float32 ∈ {-1, +1}
        codebook_norm_inv : (V,) float32   precomputed 1/||row||

    Forward:
        h (B, S, d_model) → logits (B, S, V)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        d_vsa: int = 10240,
        temperature: float | str | None = None,
        seed: int = 0xC0DEB00C,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_vsa = d_vsa

        # Query projection: h → continuous VSA vector. No bias — the
        # codebook is zero-mean {-1,+1}, so any bias shifts every logit
        # equally and is absorbed by softmax.
        self.query_proj = nn.Linear(d_model, d_vsa, bias=False)

        # Codebook + its row-norm inverse. Buffers (not Parameters)
        # so they're saved with the state_dict but receive no grads.
        codebook = build_vsa_codebook(vocab_size, d_vsa, seed=seed)
        self.register_buffer("codebook", codebook)
        # All MAP-bipolar rows have the same norm sqrt(d_vsa) by
        # construction, but computing once is free and guards against
        # future variants (partial rows, masked tokens, etc.).
        row_norms = codebook.norm(dim=-1, keepdim=False).clamp_min(1e-8)
        self.register_buffer("codebook_norm_inv", 1.0 / row_norms)

        # Logit scale.
        if temperature is None:
            # τ = sqrt(D) → logits at init have ≈ unit variance
            # (Linear init gives h @ W with variance ≈ 1, and
            #  cos(h, bipolar) scales as 1/sqrt(D), so multiplying by
            #  sqrt(D) puts us back in the usual LM-logit regime).
            self.register_buffer(
                "temperature_scalar",
                torch.tensor(math.sqrt(d_vsa), dtype=torch.float32),
            )
            self._learned_temp = False
        elif temperature == "learned":
            # Single scalar, trainable. Initialize at sqrt(D).
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(math.sqrt(d_vsa)), dtype=torch.float32)
            )
            self._learned_temp = True
        else:
            self.register_buffer(
                "temperature_scalar",
                torch.tensor(float(temperature), dtype=torch.float32),
            )
            self._learned_temp = False

    @property
    def temperature(self) -> torch.Tensor:
        if self._learned_temp:
            return self.log_temperature.exp()
        return self.temperature_scalar

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, S, d_model) or (B, d_model). Returns logits (..., V)."""
        # 1. Project to VSA space — continuous, bounded by the linear
        #    init. No sign() so gradients flow naturally.
        z = self.query_proj(h)
        # 2. L2-normalize both sides so the inner product IS cos θ.
        z_norm = F.normalize(z, dim=-1, eps=1e-8)
        # codebook has per-row norm sqrt(D) by construction; multiply
        # by norm_inv rather than re-normalizing each forward.
        codebook_unit = self.codebook * self.codebook_norm_inv.unsqueeze(-1)
        # 3. Cosine similarity vs vocab. matmul is the right shape:
        #    z_norm (..., D) · codebook_unit.T (D, V) → (..., V)
        cos_sim = torch.matmul(z_norm, codebook_unit.t())
        # 4. Temperature scale — τ = sqrt(D) at default.
        return cos_sim * self.temperature

    def num_params_saved_vs_linear(self) -> int:
        """Reporting helper. Returns how many params a plain
        ``Linear(d_model, V)`` would have had vs this head."""
        linear_params = self.d_model * self.vocab_size + self.vocab_size  # + bias
        our_params = self.d_model * self.d_vsa                             # query_proj, no bias
        if self._learned_temp:
            our_params += 1
        return linear_params - our_params


# ── Self-test ───────────────────────────────────────────────────────────

def _smoketest():
    torch.manual_seed(0)
    d_model, vocab, d_vsa = 128, 1024, 10240
    head = VSABindingHead(d_model, vocab, d_vsa=d_vsa)

    # Shapes
    h = torch.randn(4, 7, d_model)
    logits = head(h)
    assert logits.shape == (4, 7, vocab), logits.shape

    # Logits should be roughly unit-variance at init (τ = sqrt(D)).
    std = logits.std().item()
    assert 0.2 < std < 3.0, f"logit std {std:.3f} outside expected range"

    # Gradient check — loss should be differentiable w.r.t. query_proj.
    targets = torch.randint(0, vocab, (4, 7))
    loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
    loss.backward()
    assert head.query_proj.weight.grad is not None
    assert head.codebook.grad is None  # buffer, no grad

    # Determinism across instances.
    head2 = VSABindingHead(d_model, vocab, d_vsa=d_vsa)
    assert torch.equal(head.codebook, head2.codebook), "codebook not deterministic"

    saved = head.num_params_saved_vs_linear()
    linear = d_model * vocab + vocab
    ours = d_model * d_vsa
    print(f"  smoketest PASS — logit std {std:.3f}, "
          f"params ours={ours:,} linear={linear:,} "
          f"saved={saved:,} ({saved / linear * 100:.1f}%)")


if __name__ == "__main__":
    _smoketest()
