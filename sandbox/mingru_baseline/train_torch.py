#!/usr/bin/env python3
"""MinGRU coherence-baseline training on TinyStories — PyTorch mirror.

Same architecture as ``sandbox/mingru_baseline/train.py`` (grilly backend),
same CLI surface, same data prep — but uses PyTorch so it runs natively
on Colab A100 (and any other CUDA GPU) without a Vulkan ICD.

Phase 1.5 will re-validate the same architecture on the grilly stack once
Vulkan is sorted on the target GPU. For Phase 1.3 the coherence gate is
architecture-level, not backend-level, so PyTorch on CUDA is the fastest
route to "does MinGRU tell a TinyStory?".

Architecture (identical math to the grilly version in
``cubemind.training.vsa_lm.MinGRUModel``):

    MinGRULayer:    [g, v, d] = three Linear projections of x
                    x_scan    = sigmoid(g) · tanh(v)
                    a         = 0.05 + 0.9 · sigmoid(d)   (decay in [0.05, 0.95])
                    h         = prefix_scan_causal(x_scan, a)

    GLUChannelMix:  y = W_out(silu(W_gate x) * W_up x)

    MinGRUBlock:    x = x + MinGRULayer(RMSNorm(x))
                    x = x + GLUChannelMix(RMSNorm(x))

    MinGRUModel:    embed (tied with head) -> N × MinGRUBlock -> RMSNorm -> head

Run::

    python -u sandbox/mingru_baseline/train_torch.py --steps 6000 \\
        --batch-size 128 --lr 1e-3 --warmup 200 \\
        --eval-every 500 --ckpt-every 500
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# UTF-8 stdout so Windows consoles don't choke on BPE byte markers.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# HuggingFace tokenizers (real package, not grilly's shadow subpackage —
# explicit imports before any grilly touch, defensive even though grilly
# is not imported at all in this file).
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders as hf_decoders
from tokenizers import models as hf_models
from tokenizers import pre_tokenizers as hf_pre
from tokenizers import trainers as hf_trainers


# ── Paths ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "results_torch"
PROMPTS_PATH = SCRIPT_DIR / "prompts.txt"

REPO_ROOT = SCRIPT_DIR.parents[1]
LOCAL_JSON = REPO_ROOT / "data" / "tinystories_50k.json"
TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")


# ── Config ───────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 4000
    d_model: int = 256
    n_layers: int = 6
    d_ffn: int = 768
    seq_len: int = 256

    # Training
    batch_size: int = 128
    grad_accum: int = 1
    max_lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Data subset
    subset_tokens: int = 10_979_128  # full TinyStories-50k corpus by default

    # Budget
    max_steps: int = 6_000
    max_minutes: float = 0.0

    # Schedule
    log_every: int = 20
    eval_every: int = 500
    ckpt_every: int = 500

    # Generation
    gen_max_tokens: int = 120
    gen_temperature: float = 0.8
    gen_top_p: float = 0.9
    gen_greedy_too: bool = True

    # ── Hybrid architecture (experimental) ─────────────────────────────
    # MoE recurrence: M small MinGRU experts + top-K routing per token
    enable_moe: bool = False
    moe_n_experts: int = 4
    moe_top_k: int = 2

    # Local sliding-window attention (supplements recurrence every N layers)
    enable_attention: bool = False
    attn_n_heads: int = 4
    attn_window: int = 128
    attn_every_n: int = 3       # attention on every Nth layer (0, 3, 6, ...)

    # Runtime
    device: str = "auto"        # auto / cuda / cpu
    amp_dtype: str = "bf16"     # fp32 / bf16 / fp16
    compile_model: bool = False  # torch.compile
    seed: int = 42


# ── Data prep (shares the ``data/`` cache with train.py) ─────────────────

def _download(url: str, dst: Path) -> None:
    import urllib.request
    print(f"  downloading {url.rsplit('/', 1)[-1]} -> {dst}")
    urllib.request.urlretrieve(url, str(dst))
    print(f"    {dst.stat().st_size / 1e6:.1f} MB")


def _load_local_stories() -> list[str]:
    with LOCAL_JSON.open("r", encoding="utf-8") as f:
        stories = json.load(f)
    assert isinstance(stories, list) and stories
    return stories


def prepare_data(cfg: TrainConfig):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = "local50k" if LOCAL_JSON.exists() else "hf"
    tok_path  = DATA_DIR / f"tokenizer_{source}_v{cfg.vocab_size}.json"
    train_bin = DATA_DIR / f"train_{source}_v{cfg.vocab_size}.bin"
    val_bin   = DATA_DIR / f"val_{source}_v{cfg.vocab_size}.bin"
    meta_path = DATA_DIR / f"meta_{source}_v{cfg.vocab_size}.json"

    needs_build = not (
        tok_path.exists() and train_bin.exists()
        and val_bin.exists() and meta_path.exists()
    )

    if needs_build:
        corpus_txt = DATA_DIR / f"corpus_{source}.txt"
        if source == "local50k":
            print(f"  local corpus: {LOCAL_JSON}")
            stories = _load_local_stories()
            corpus_txt.write_text("\n\n".join(stories), encoding="utf-8")
        else:
            print("  downloading TinyStories V2 GPT-4 split")
            train_txt = DATA_DIR / "train.txt"
            val_txt   = DATA_DIR / "valid.txt"
            if not train_txt.exists(): _download(TRAIN_URL, train_txt)
            if not val_txt.exists():   _download(VALID_URL, val_txt)
            corpus_txt.write_bytes(
                train_txt.read_bytes() + b"\n\n" + val_txt.read_bytes()
            )

        print(f"  training BPE tokenizer (vocab={cfg.vocab_size})")
        tokenizer = HFTokenizer(hf_models.BPE())
        tokenizer.pre_tokenizer = hf_pre.ByteLevel()
        tokenizer.decoder = hf_decoders.ByteLevel()
        tokenizer.train(
            files=[str(corpus_txt)],
            trainer=hf_trainers.BpeTrainer(
                vocab_size=cfg.vocab_size, min_frequency=2,
                special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            ),
        )
        tokenizer.save(str(tok_path))

        def _tokenize_text(text: str, dst: Path, label: str) -> int:
            tmp = Path(tempfile.mktemp(suffix=".bin"))
            total = 0
            with tmp.open("wb") as f:
                step = 500_000
                for i in range(0, len(text), step):
                    ids = tokenizer.encode(text[i : i + step]).ids
                    np.asarray(ids, dtype=np.uint16).tofile(f)
                    total += len(ids)
                    if (i // step) % 20 == 0:
                        print(f"    {label}: {total:,} tokens", end="\r")
                    gc.collect()
            shutil.copy2(tmp, dst); tmp.unlink()
            print(f"    {label}: {total:,} tokens            ")
            return total

        if source == "local50k":
            stories = _load_local_stories()
            split_n = max(1, int(len(stories) * 0.95))
            train_text = "\n\n".join(stories[:split_n])
            val_text   = "\n\n".join(stories[split_n:])
        else:
            train_txt = DATA_DIR / "train.txt"
            val_txt   = DATA_DIR / "valid.txt"
            train_text = train_txt.read_text(encoding="utf-8", errors="ignore")
            val_text   = val_txt.read_text(encoding="utf-8", errors="ignore")

        train_n = _tokenize_text(train_text, train_bin, "train")
        val_n   = _tokenize_text(val_text,   val_bin,   "val")
        meta_path.write_text(json.dumps({
            "source": source, "vocab": tokenizer.get_vocab_size(),
            "train_tokens": train_n, "val_tokens": val_n,
        }, indent=2))

    tokenizer = HFTokenizer.from_file(str(tok_path))
    if tokenizer.decoder is None:
        tokenizer.decoder = hf_decoders.ByteLevel()
    vocab = tokenizer.get_vocab_size()

    train_mm = np.memmap(str(train_bin), dtype=np.uint16, mode="r")
    n_total = int(train_mm.shape[0])
    n_train = min(n_total, cfg.subset_tokens)
    train_data = np.asarray(train_mm[:n_train]).astype(np.int64)
    val_data   = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int64)

    print(f"  source: {source}  | train {n_train:,} / {n_total:,} tokens "
          f"(~{n_train/1e6:.1f}M) | val {len(val_data):,}")
    return tokenizer, vocab, train_data, val_data


def load_prompts() -> list[str]:
    return [
        line.strip() for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── Model ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Standard RMSNorm: y = x · rsqrt(mean(x²) + eps) · weight."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def prefix_scan_causal(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Causal linear-RNN scan: ``h_t = a_t · h_{t-1} + x_t``.

    Sequential time loop — simple and numerically stable at any seq_len
    in any dtype (bf16, fp16, fp32). Each step is a cheap ``(B, D)``
    multiply-add; kernel launch overhead dominates but stays under a ms
    per layer on a modern GPU at seq=256.

    The parallel log-space version (grilly shader / earlier revision
    here) is faster but underflows when the cumulative sum of ``log(a)``
    passes the dtype's min exponent. For ``a ∈ [0.05, 0.95]``,
    ``log(a) ≥ -3``, so at seq=256 the worst-case cumulative log can hit
    -768 — ``exp(-768) = 0`` in both bf16 and fp32, producing a division
    by zero and NaN loss within the first step.

    A proper parallel associative scan (Heinsen 2023 / Chen 2024) can
    restore the O(log T) depth while staying numerically stable — wiring
    that in is a Phase 1.5+ follow-up.
    """
    B, S, D = x.shape
    h = torch.zeros(B, D, dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for t in range(S):
        h = a[:, t] * h + x[:, t]
        out[:, t] = h
    return out


class MinGRULayer(nn.Module):
    """MinGRU sequence mixer — see module docstring for math."""

    def __init__(self, d_model: int, decay_bias_init: float = 1.0):
        super().__init__()
        self.proj_g = nn.Linear(d_model, d_model, bias=True)
        self.proj_v = nn.Linear(d_model, d_model, bias=True)
        self.proj_d = nn.Linear(d_model, d_model, bias=True)
        # Decay gate bias → sigmoid ≈ 0.73 retention at t=0.
        nn.init.constant_(self.proj_d.bias, decay_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.proj_g(x)
        v = self.proj_v(x)
        d = self.proj_d(x)
        x_scan = torch.sigmoid(g) * torch.tanh(v)
        a = 0.001 + 0.998 * torch.sigmoid(d)
        return prefix_scan_causal(x_scan, a)


class GLUChannelMix(nn.Module):
    """SwiGLU FFN: ``y = W_out(silu(W_gate x) · W_up x)``."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.W_up   = nn.Linear(d_model, d_ffn, bias=False)
        self.W_out  = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(F.silu(self.W_gate(x)) * self.W_up(x))


class MoEMinGRULayer(nn.Module):
    """Mixture of MinGRU experts — each expert has its own proj_g/v/d.

    Different experts can specialize: one learns high-decay "character
    register" behavior, another learns fast-forget "verb dynamics." The
    router selects top-K experts per token, so specialization is explicit.

    All experts run on the full ``d_model`` (not split dims). With M=4
    and top_k=2, compute is 4× the recurrence but routing adds
    specialization that a single MinGRU can't express.
    """

    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2,
                 decay_bias_init: float = 1.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            MinGRULayer(d_model, decay_bias_init) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Router: per-token expert selection
        gate_logits = self.gate(x)                          # (B, S, M)
        top_vals, top_idx = gate_logits.topk(self.top_k, dim=-1)  # (B, S, K)
        weights = torch.softmax(top_vals, dim=-1)           # (B, S, K)

        # Run ALL experts (simple prototype; token-dispatch version later)
        expert_outs = torch.stack([exp(x) for exp in self.experts], dim=-1)
        # expert_outs: (B, S, D, M)

        # Gather top-K and weighted sum
        # Expand indices for gather: (B, S, D, K)
        idx = top_idx.unsqueeze(2).expand(B, S, D, self.top_k)
        selected = expert_outs.gather(-1, idx)              # (B, S, D, K)
        weights_exp = weights.unsqueeze(2)                  # (B, S, 1, K)
        return (selected * weights_exp).sum(dim=-1)         # (B, S, D)


class LocalCausalAttention(nn.Module):
    """Sliding-window causal self-attention.

    Supplements MinGRU's recurrence with precise local token-to-token
    lookups within a window of W positions. The recurrence handles
    global context decay; attention handles "who said what" precision.

    Uses PyTorch's ``scaled_dot_product_attention`` with ``is_causal=True``
    — efficient on A100/Blackwell via FlashAttention-2 kernel fusion.
    """

    def __init__(self, d_model: int, n_heads: int = 4, window: int = 128):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                        # each (B, S, H, Dh)
        q = q.transpose(1, 2)                               # (B, H, S, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Sliding-window: mask positions beyond W tokens back.
        # Build additive attention mask where out-of-window = -inf.
        if S <= self.window:
            # Full causal — window covers everything
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Build sliding-window causal mask
            mask = torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype)
            for i in range(S):
                start = max(0, i - self.window + 1)
                mask[i, start:i + 1] = 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(B, S, D)          # (B, S, D)
        return self.out_proj(out)


class HybridBlock(nn.Module):
    """MinGRU (or MoE-MinGRU) + optional local attention + GLU.

    Architecture per block::

        x = x + Mixer(RMSNorm(x))         # MinGRU or MoE-MinGRU
        x = x + Attn(RMSNorm(x))          # local attention (if enabled)
        x = x + GLU(RMSNorm(x))           # channel mixing

    Attention activates only on layers where ``layer_idx % attn_every_n == 0``.
    """

    def __init__(self, cfg: "TrainConfig", layer_idx: int):
        super().__init__()
        d = cfg.d_model

        # Sequence mixer
        self.rms_mix = RMSNorm(d)
        if cfg.enable_moe:
            self.mix = MoEMinGRULayer(d, cfg.moe_n_experts, cfg.moe_top_k)
        else:
            self.mix = MinGRULayer(d)

        # Local attention (on selected layers)
        self.attn = None
        self.rms_attn = None
        if cfg.enable_attention and layer_idx % cfg.attn_every_n == 0:
            self.rms_attn = RMSNorm(d)
            self.attn = LocalCausalAttention(d, cfg.attn_n_heads, cfg.attn_window)

        # FFN
        self.rms_ffn = RMSNorm(d)
        self.ffn = GLUChannelMix(d, cfg.d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.rms_mix(x))
        if self.attn is not None:
            x = x + self.attn(self.rms_attn(x))
        x = x + self.ffn(self.rms_ffn(x))
        return x


class MinGRUBlock(nn.Module):
    """Original pure-MinGRU block (no MoE, no attention)."""
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.rms_mix = RMSNorm(cfg.d_model)
        self.rms_ffn = RMSNorm(cfg.d_model)
        self.mix = MinGRULayer(cfg.d_model)
        self.ffn = GLUChannelMix(cfg.d_model, cfg.d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.rms_mix(x))
        x = x + self.ffn(self.rms_ffn(x))
        return x


class MinGRUModel(nn.Module):
    """PyTorch MinGRU language model — selects pure or hybrid blocks."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        use_hybrid = cfg.enable_moe or cfg.enable_attention
        if use_hybrid:
            self.blocks = nn.ModuleList([
                HybridBlock(cfg, i) for i in range(cfg.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                MinGRUBlock(cfg) for i in range(cfg.n_layers)
            ])

        self.rms_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h)
        return self.head(self.rms_f(h))


# ── Training utilities ───────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def sample_batch(data: np.ndarray, batch_size: int, seq_len: int, rng):
    """Contiguous-window random sampling. Returns int64 tensors of shape
    ``(batch_size, seq_len - 1)`` for both input and target (shifted)."""
    n_windows = (len(data) - 1) // seq_len
    starts = rng.integers(0, n_windows, size=batch_size) * seq_len
    x = np.empty((batch_size, seq_len - 1), dtype=np.int64)
    y = np.empty((batch_size, seq_len - 1), dtype=np.int64)
    for i, s in enumerate(starts):
        chunk = data[s : s + seq_len]
        x[i] = chunk[:-1]
        y[i] = chunk[1:]
    return torch.from_numpy(x), torch.from_numpy(y)


def _resolve_dtype(name: str):
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def _autocast_ctx(device: torch.device, dtype_name: str):
    if dtype_name == "fp32" or device.type != "cuda":
        from contextlib import nullcontext
        return nullcontext
    dtype = _resolve_dtype(dtype_name)
    def _ctx():
        return torch.autocast(device_type=device.type, dtype=dtype)
    return _ctx


# ── Loss ─────────────────────────────────────────────────────────────────

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits: (B, S, V) float, targets: (B, S) int64. Returns scalar mean."""
    B, S, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * S, V),
        targets.reshape(B * S),
    )


# ── Generation ───────────────────────────────────────────────────────────

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Return logits with tokens outside the top-p nucleus set to -inf."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = cum > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(0, sorted_idx, remove)
    return logits.masked_fill(mask, float("-inf"))


@dataclass
class GenParams:
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 120


@torch.no_grad()
def generate(model: MinGRUModel, tokenizer, prompt: str, params: GenParams,
             device: torch.device, greedy: bool = False) -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = model.cfg.seq_len

    for _ in range(params.max_new_tokens):
        context = tokens[:, -seq_len:]
        logits = model(context)[:, -1, :]  # (1, V)
        if greedy:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            logits = logits / max(params.temperature, 1e-5)
            filt = _top_p_filter(logits[0], params.top_p)
            probs = torch.softmax(filt, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        tokens = torch.cat([tokens, tokens.new_tensor([[next_id]])], dim=1)

    model.train()
    text = tokenizer.decode(tokens[0].tolist())
    return text.replace("\u0120", " ").replace("\u010a", "\n")


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: MinGRUModel, val_data: np.ndarray, cfg: TrainConfig,
             device: torch.device, autocast_ctx, max_batches: int = 50) -> float:
    model.eval()
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(max_batches):
        x, y = sample_batch(val_data, cfg.batch_size, cfg.seq_len, rng)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast_ctx():
            logits = model(x)
            loss = compute_loss(logits, y)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


# ── Checkpoint ───────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model: MinGRUModel, optimizer, step: int,
                    best_val: float, tokens_seen: int, elapsed: float,
                    cfg: TrainConfig,
                    tokenizer_json: str | None = None) -> None:
    """Save checkpoint with optional inline tokenizer JSON.

    Embedding the tokenizer in the checkpoint avoids the foot-gun where
    eval-time tokenizer-source detection (e.g. ``LOCAL_JSON.exists()``)
    diverges from train-time tokenizer choice — different BPE merges
    produce different token IDs and the embedding lookup maps every
    token to garbage.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "step": step, "best_val": best_val,
        "tokens_seen": tokens_seen, "elapsed": elapsed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
        "tokenizer_json": tokenizer_json,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: MinGRUModel, optimizer,
                    device: torch.device) -> dict:
    data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    return {
        "step": int(data["step"]), "best_val": float(data["best_val"]),
        "tokens_seen": int(data["tokens_seen"]),
        "elapsed": float(data["elapsed"]),
    }


# ── Generation file writer ───────────────────────────────────────────────

def write_generations(model: MinGRUModel, tokenizer, path: Path,
                      cfg: TrainConfig, device: torch.device, step: int,
                      val_ce: float, val_ppl: float,
                      include_greedy: bool = False) -> None:
    prompts = load_prompts()
    params = GenParams(
        temperature=cfg.gen_temperature, top_p=cfg.gen_top_p,
        max_new_tokens=cfg.gen_max_tokens,
    )

    lines = [
        f"# Generations (PyTorch) — step {step:,}",
        "",
        f"- val CE: **{val_ce:.4f}**  · val PPL: **{val_ppl:.2f}**",
        f"- sampling: T={cfg.gen_temperature}, top_p={cfg.gen_top_p}, "
        f"max={cfg.gen_max_tokens}",
        "",
    ]
    for i, prompt in enumerate(prompts, 1):
        text = generate(model, tokenizer, prompt, params, device, greedy=False)
        lines.append(f"## {i:02d}. {prompt}")
        lines.append("")
        lines.append("**sampled:**  " + text.replace("\n", " "))
        if include_greedy and cfg.gen_greedy_too:
            text_g = generate(model, tokenizer, prompt, params, device, greedy=True)
            lines.append("")
            lines.append("**greedy:**  " + text_g.replace("\n", " "))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Training loop ────────────────────────────────────────────────────────

def train(cfg: TrainConfig, args) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)
    print(f"\n{'=' * 72}")
    print(f"  MinGRU Phase 1.3 -- PyTorch backend")
    print(f"{'=' * 72}")
    print(f"  device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"  vram  : {total_mb:,.0f} MB total")
    print(f"  model : d={cfg.d_model} L={cfg.n_layers} d_ffn={cfg.d_ffn} "
          f"vocab={cfg.vocab_size} seq_len={cfg.seq_len}")
    print(f"  optim : lr={cfg.max_lr} -> {cfg.min_lr} cosine | "
          f"warmup={cfg.warmup_steps} | wd={cfg.weight_decay} | "
          f"clip={cfg.grad_clip} | dtype={cfg.amp_dtype}")
    max_min_str = f"{cfg.max_minutes}" if cfg.max_minutes > 0 else "inf"
    print(f"  sched : max_steps={cfg.max_steps} | max_min={max_min_str} | "
          f"log={cfg.log_every} eval={cfg.eval_every} ckpt={cfg.ckpt_every}")
    print(f"  batch : {cfg.batch_size} x accum={cfg.grad_accum} "
          f"= {cfg.batch_size * cfg.grad_accum * cfg.seq_len:,} toks/step")

    print("\n--- data ---")
    tokenizer, vocab, train_data, val_data = prepare_data(cfg)
    if vocab != cfg.vocab_size:
        print(f"  note: actual vocab {vocab} != configured {cfg.vocab_size}")
        cfg.vocab_size = vocab

    print("\n--- model ---")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = MinGRUModel(cfg).to(device)
    n_params = sum(p.numel() for p in set(p for p in model.parameters()))
    # De-duplicate tied weights for reporting
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen: continue
        seen.add(id(p)); total += p.numel()
    print(f"  params: {total:,} ({total/1e6:.2f}M)")

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            print("  torch.compile: on")
        except Exception as e:
            print(f"  torch.compile failed, continuing eager: {e}")

    print("\n--- optimizer ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr, betas=(0.9, 0.95),
        eps=1e-8, weight_decay=cfg.weight_decay,
    )
    autocast_ctx = _autocast_ctx(device, cfg.amp_dtype)
    scaler = (torch.amp.GradScaler(device.type)
              if cfg.amp_dtype == "fp16" and device.type == "cuda"
              else None)

    # Resume
    ckpt_path = OUT_DIR / "checkpoint.pt"
    step, best_val, tokens_seen, elapsed_prev = 0, float("inf"), 0, 0.0
    if args.resume and ckpt_path.exists():
        meta = load_checkpoint(ckpt_path, model, optimizer, device)
        step = meta["step"]; best_val = meta["best_val"]
        tokens_seen = meta["tokens_seen"]; elapsed_prev = meta["elapsed"]
        print(f"\n  *** resumed from step {step:,} "
              f"({elapsed_prev/60:.1f}m done) ***")

    rng = np.random.default_rng(cfg.seed + step)

    header = (f"  {'step':>6} {'lr':>9} {'CE':>8} {'PPL':>9} "
              f"{'gn':>7} {'tok/s':>8} {'elapsed':>8}")
    divider = "  " + "-" * 72
    print(); print(header); print(divider)
    header_every = 20
    log_count = 0
    history: list = []

    log_ce, log_n = 0.0, 0
    t_start = time.time()
    max_time = cfg.max_minutes * 60.0 if cfg.max_minutes > 0 else math.inf

    while step < cfg.max_steps:
        if elapsed_prev + (time.time() - t_start) >= max_time:
            print("  [time budget reached]")
            break

        optimizer.zero_grad(set_to_none=True)

        loss_val = 0.0
        for _ in range(cfg.grad_accum):
            x, y = sample_batch(train_data, cfg.batch_size, cfg.seq_len, rng)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_ctx():
                logits = model(x)
                loss = compute_loss(logits, y) / cfg.grad_accum
            if not torch.isfinite(loss):
                print(f"  WARN non-finite loss at step {step}, skipping")
                continue
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_val += float(loss.item()) * cfg.grad_accum
            tokens_seen += x.numel()

        if scaler is not None:
            scaler.unscale_(optimizer)
        gn = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip))

        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        step += 1

        log_ce += loss_val / cfg.grad_accum
        log_n += 1

        if step % cfg.log_every == 0:
            avg_ce = log_ce / max(log_n, 1)
            now = elapsed_prev + (time.time() - t_start)
            tps = tokens_seen / max(now, 1)
            print(f"  {step:>6d} {lr:>9.2e} {avg_ce:>8.4f} "
                  f"{math.exp(min(avg_ce, 20)):>9.2f} {gn:>7.3f} "
                  f"{tps:>8,.0f} {now/60:>7.1f}m")
            history.append({
                "step": step, "lr": lr, "ce": avg_ce,
                "ppl": math.exp(min(avg_ce, 20)),
                "grad_norm": gn, "tokens_per_sec": tps, "elapsed": now,
            })
            log_ce, log_n = 0.0, 0
            log_count += 1
            if log_count % header_every == 0:
                print(divider); print(header); print(divider)

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val_ce = evaluate(model, val_data, cfg, device, autocast_ctx,
                              max_batches=25)
            val_ppl = math.exp(min(val_ce, 20))
            tag = ""
            if val_ce < best_val:
                best_val = val_ce
                best_path = OUT_DIR / "best.pt"
                save_checkpoint(best_path, model, optimizer, step, best_val,
                                tokens_seen,
                                elapsed_prev + (time.time() - t_start), cfg,
                                tokenizer_json=tokenizer.to_str())
                tag = " *"
            print(f"\n  === step {step:,} ===  val CE {val_ce:.4f}  "
                  f"PPL {val_ppl:.2f}{tag}")
            gen_path = OUT_DIR / f"gen_step_{step:06d}.md"
            write_generations(model, tokenizer, gen_path, cfg, device,
                              step, val_ce, val_ppl)
            print(f"  [gen] wrote {gen_path.name}")
            print(divider); print(header); print(divider)

        if step % cfg.ckpt_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, step, best_val,
                            tokens_seen,
                            elapsed_prev + (time.time() - t_start), cfg,
                            tokenizer_json=tokenizer.to_str())

        if step % 200 == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    elapsed_total = elapsed_prev + (time.time() - t_start)
    final_val = evaluate(model, val_data, cfg, device, autocast_ctx,
                         max_batches=50)
    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))

    final_gen_path = OUT_DIR / "generations_final.md"
    write_generations(model, tokenizer, final_gen_path, cfg, device, step,
                      final_val, final_ppl, include_greedy=True)

    summary = {
        "model": "MinGRU-phase1.3-torch",
        "params": total,
        "config": asdict(cfg),
        "final_val_ce": final_val,
        "final_val_ppl": final_ppl,
        "best_val_ce": best_val,
        "best_val_ppl": best_ppl,
        "steps": step,
        "tokens_seen": tokens_seen,
        "elapsed_min": elapsed_total / 60,
        "tokens_per_sec": tokens_seen / max(elapsed_total, 1),
        "device": str(device),
        "history": history[-200:],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  {'=' * 72}")
    print(f"  DONE. steps={step:,}  tokens={tokens_seen/1e6:.1f}M  "
          f"time={elapsed_total/60:.1f}m")
    print(f"  final PPL {final_ppl:.2f}  (best {best_ppl:.2f})")
    print(f"  saved -> {OUT_DIR}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description="MinGRU Phase 1.3 (PyTorch)")
    ap.add_argument("--steps", type=int, default=cfg.max_steps)
    ap.add_argument("--minutes", type=float, default=cfg.max_minutes)
    ap.add_argument("--d-model", type=int, default=cfg.d_model)
    ap.add_argument("--n-layers", type=int, default=cfg.n_layers)
    ap.add_argument("--d-ffn", type=int, default=cfg.d_ffn)
    ap.add_argument("--vocab", type=int, default=cfg.vocab_size)
    ap.add_argument("--seq-len", type=int, default=cfg.seq_len)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--grad-accum", type=int, default=cfg.grad_accum)
    ap.add_argument("--lr", type=float, default=cfg.max_lr)
    ap.add_argument("--min-lr", type=float, default=cfg.min_lr)
    ap.add_argument("--warmup", type=int, default=cfg.warmup_steps)
    ap.add_argument("--wd", type=float, default=cfg.weight_decay)
    ap.add_argument("--clip", type=float, default=cfg.grad_clip)
    ap.add_argument("--subset-tokens", type=int, default=cfg.subset_tokens)
    ap.add_argument("--log-every", type=int, default=cfg.log_every)
    ap.add_argument("--eval-every", type=int, default=cfg.eval_every)
    ap.add_argument("--ckpt-every", type=int, default=cfg.ckpt_every)
    ap.add_argument("--gen-tokens", type=int, default=cfg.gen_max_tokens)
    ap.add_argument("--gen-temp", type=float, default=cfg.gen_temperature)
    ap.add_argument("--gen-top-p", type=float, default=cfg.gen_top_p)
    # Hybrid architecture
    ap.add_argument("--moe", action="store_true",
                    help="Enable MoE-MinGRU (4 experts, top-2 routing)")
    ap.add_argument("--moe-experts", type=int, default=cfg.moe_n_experts)
    ap.add_argument("--moe-top-k", type=int, default=cfg.moe_top_k)
    ap.add_argument("--attention", action="store_true",
                    help="Enable local sliding-window attention every N layers")
    ap.add_argument("--attn-heads", type=int, default=cfg.attn_n_heads)
    ap.add_argument("--attn-window", type=int, default=cfg.attn_window)
    ap.add_argument("--attn-every-n", type=int, default=cfg.attn_every_n)
    # Runtime
    ap.add_argument("--device", default=cfg.device,
                    help="auto / cuda / cuda:0 / cpu")
    ap.add_argument("--dtype", default=cfg.amp_dtype,
                    choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        vocab_size=args.vocab, d_model=args.d_model, n_layers=args.n_layers,
        d_ffn=args.d_ffn, seq_len=args.seq_len,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup,
        weight_decay=args.wd, grad_clip=args.clip,
        subset_tokens=args.subset_tokens,
        max_steps=args.steps, max_minutes=args.minutes,
        log_every=args.log_every, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        gen_max_tokens=args.gen_tokens,
        gen_temperature=args.gen_temp, gen_top_p=args.gen_top_p,
        enable_moe=args.moe, moe_n_experts=args.moe_experts,
        moe_top_k=args.moe_top_k,
        enable_attention=args.attention, attn_n_heads=args.attn_heads,
        attn_window=args.attn_window, attn_every_n=args.attn_every_n,
        device=args.device, amp_dtype=args.dtype,
        compile_model=args.compile, seed=args.seed,
    )
    train(cfg, args)


if __name__ == "__main__":
    main()
