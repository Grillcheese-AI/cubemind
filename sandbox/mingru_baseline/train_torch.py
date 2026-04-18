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

# Ensure sibling modules (vsa_binding_head.py, vm_opcodes.py) are
# importable regardless of how the script was launched. Colab / Jupyter
# invoke via a kernel that doesn't always put the script's directory on
# sys.path[0], so deferred imports inside classes fail with
# ModuleNotFoundError. Inserting the parent directory here is
# idempotent and cheap.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

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
from torch.utils.data import DataLoader, Dataset

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
PROMPTS_PATH = SCRIPT_DIR / "prompts_tinystories.txt"  # default for TinyStories runs

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

    # Data
    subset_tokens: int = 10_979_128  # full TinyStories-50k corpus by default
    data_path: str = ""             # direct path to a .txt file (bypasses TinyStories download)
    val_split: float = 0.05         # fraction held out for validation when using --data-path

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
    # Prompt file for gen_step_*.md output. Default: TinyStories prompts.
    # For non-fiction corpora (e.g. C4 news), pass prompts_news.txt.
    prompts_path: str = ""

    # VSA binding head for LM vocab (ablation — replaces the tied
    # Linear(d_model, V) head with Linear(d_model, D) + fixed bipolar
    # codebook cosine lookup). Untied from the embedding so the
    # embedding stays learned. See vsa_binding_head.py.
    vsa_binding_head: bool = False
    vsa_binding_d: int = 10240       # VSA dimension (K_BLOCKS × L_BLOCK)
    vsa_binding_seed: int = 0xC0DEB00C

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

    # Hippocampal episodic memory — dopamine-gated write, reward-weighted read
    enable_memory: bool = False
    mem_max: int = 200          # max memories in the store
    mem_write_threshold: float = 0.4  # dopamine must exceed this to write
    mem_every_n: int = 4        # inject memory every Nth layer
    mem_consolidate_every: int = 1000  # sleep/consolidation interval (steps)

    # Neuromodulated input scaling (Yerkes-Dodson / OSGM). Pairs with
    # AutoHypergradientAdamW: the optimizer reports a ``current_surprise_gain``
    # S ≥ 0 each step; each MinGRULayer owns a learnable α and scales its
    # input as ``x' = x · (1 + α·S)``. Disable for the plain baseline.
    enable_hypergrad: bool = False
    hypergrad_scale_init: float = 0.1

    # Structured JSONL multitask mode — adds opcode/intent/act/rule/validity
    # heads on top of the LM head, supervised from row-wise labeled JSONL.
    # When False, the script behaves exactly as the flat next-token baseline.
    #
    # Class counts are sized for the full CubeMind label space:
    #   - 58 opcodes  = position-based contiguous IDs from vm_opcodes.py
    #                   (matches opcode-vsa-rs/src/ir.rs::CubeMindOpcode order)
    #   - 16 schemas  = generous headroom; LabelRegistry assigns at runtime
    #   - 32 rules    = bounded learn() pattern bank, expand if Gemini exceeds
    #   - 6 intents   = classify_intent() output in the conversation contract
    #   - 2 validity  = verify() boolean
    use_jsonl_dataset: bool = False
    jsonl_path: str = ""
    aux_opcode_loss_weight:   float = 0.4   # paper's strongest signal
    aux_intent_loss_weight:   float = 0.0   # off by default; turn on once opcode head trains
    aux_schema_loss_weight:   float = 0.0
    aux_rule_loss_weight:     float = 0.0
    aux_validity_loss_weight: float = 0.0
    num_opcode_classes: int = 55
    num_schema_classes: int = 16
    num_rule_classes:   int = 32
    num_intent_classes: int = 6

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


def prepare_data_from_file(cfg: TrainConfig):
    """Load training data from a raw .txt file at ``cfg.data_path``.

    Trains a BPE tokenizer on the file, tokenizes to uint16 bins, and
    splits into train/val by the ``cfg.val_split`` ratio. Caches
    everything in ``DATA_DIR`` keyed by the file basename + vocab size
    so subsequent runs with the same file skip tokenization.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(cfg.data_path).resolve()
    assert src.exists(), f"--data-path not found: {src}"
    tag = src.stem.replace(".", "_")[:40]

    tok_path  = DATA_DIR / f"tokenizer_{tag}_v{cfg.vocab_size}.json"
    train_bin = DATA_DIR / f"train_{tag}_v{cfg.vocab_size}.bin"
    val_bin   = DATA_DIR / f"val_{tag}_v{cfg.vocab_size}.bin"
    meta_path = DATA_DIR / f"meta_{tag}_v{cfg.vocab_size}.json"

    needs_build = not (
        tok_path.exists() and train_bin.exists()
        and val_bin.exists() and meta_path.exists()
    )

    if needs_build:
        file_size = src.stat().st_size
        print(f"  source: {src.name} ({file_size / 1e9:.2f} GB)")

        # Train BPE on the source file directly
        print(f"  training BPE tokenizer (vocab={cfg.vocab_size})")
        tokenizer = HFTokenizer(hf_models.BPE())
        tokenizer.pre_tokenizer = hf_pre.ByteLevel()
        tokenizer.decoder = hf_decoders.ByteLevel()
        tokenizer.train(
            files=[str(src)],
            trainer=hf_trainers.BpeTrainer(
                vocab_size=cfg.vocab_size, min_frequency=2,
                special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            ),
        )
        tokenizer.save(str(tok_path))
        print(f"  vocab: {tokenizer.get_vocab_size()}")

        # Tokenize the entire file in streaming chunks
        print(f"  tokenizing {src.name}...")
        all_ids: list[int] = []
        with src.open("r", encoding="utf-8", errors="ignore") as f:
            chunk_idx = 0
            while True:
                chunk = f.read(1_000_000)  # 1 MB chunks
                if not chunk:
                    break
                ids = tokenizer.encode(chunk).ids
                all_ids.extend(ids)
                chunk_idx += 1
                if chunk_idx % 100 == 0:
                    print(f"    {len(all_ids):,} tokens ({chunk_idx} MB read)",
                          end="\r")
                gc.collect()
        total = len(all_ids)
        print(f"    total: {total:,} tokens                        ")

        # Split into train / val
        split_idx = int(total * (1.0 - cfg.val_split))
        train_ids = np.array(all_ids[:split_idx], dtype=np.uint16)
        val_ids   = np.array(all_ids[split_idx:], dtype=np.uint16)
        train_ids.tofile(str(train_bin))
        val_ids.tofile(str(val_bin))

        meta_path.write_text(json.dumps({
            "source": str(src), "tag": tag,
            "vocab": tokenizer.get_vocab_size(),
            "train_tokens": len(train_ids),
            "val_tokens": len(val_ids),
        }, indent=2))
        print(f"  train: {len(train_ids):,} | val: {len(val_ids):,}")

    tokenizer = HFTokenizer.from_file(str(tok_path))
    if tokenizer.decoder is None:
        tokenizer.decoder = hf_decoders.ByteLevel()
    meta = json.loads(meta_path.read_text())
    vocab = tokenizer.get_vocab_size()

    train_mm = np.memmap(str(train_bin), dtype=np.uint16, mode="r")
    n_total = int(train_mm.shape[0])
    n_train = min(n_total, cfg.subset_tokens)
    train_data = np.asarray(train_mm[:n_train]).astype(np.int64)
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int64)

    print(f"  source: {meta.get('tag', '?')} | "
          f"train {n_train:,} / {n_total:,} tokens | "
          f"val {len(val_data):,}")
    return tokenizer, vocab, train_data, val_data


def prepare_data(cfg: TrainConfig):
    # Route to direct-file loader if --data-path is set
    if cfg.data_path:
        return prepare_data_from_file(cfg)

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


# ── Structured JSONL multitask data ─────────────────────────────────────
#
# The flat LM path above feeds one long token stream to ``sample_batch``.
# The multitask path below feeds row-wise labeled examples (text + opcode
# + rule + schema + intent + validity) to a DataLoader, so the multitask
# heads in MinGRUMultiTask see a real label per row instead of pooling
# across an arbitrary contiguous window.

def load_jsonl_rows(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (skips blank lines)."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_text_corpus(rows: list[dict], dst: Path) -> None:
    """Concatenate the ``text`` field of each row into a flat .txt file
    so the existing BPE tokenizer training path can consume it unchanged."""
    dst.write_text(
        "\n".join(str(r["text"]) for r in rows),
        encoding="utf-8",
    )


class StructuredTraceDataset(Dataset):
    """Row-wise labeled dataset for the multitask heads.

    Each item is ``{x, y, opcode_id, rule_id, schema_id, intent_id, validity}``
    where ``x`` and ``y`` are the standard left-shifted next-token pair
    (so the LM head trains identically to the flat path) and the rest are
    pooled-label classification targets the auxiliary heads supervise on.

    Rows shorter than ``seq_len`` are right-padded with ``pad_id``; rows
    longer are truncated. ``intent_id`` is optional in the JSONL — falls
    back to 0 (open_dialogue) when absent so older samples without intent
    labels still load.
    """

    def __init__(self, rows: list[dict], tokenizer, seq_len: int, pad_id: int = 0):
        self.items: list[dict] = []
        self.seq_len = seq_len
        self.pad_id = pad_id

        for row in rows:
            ids = tokenizer.encode(str(row["text"])).ids[:seq_len]
            if len(ids) < 2:
                continue
            real_len = len(ids)  # before padding
            if len(ids) < seq_len:
                ids = ids + [pad_id] * (seq_len - len(ids))
            ids = np.asarray(ids, dtype=np.int64)
            # ``last_idx`` is the index of the rightmost non-pad token in
            # the input slice (x = ids[:-1]). Heads pool from here so they
            # see the actual end of the labeled row, not a pad token.
            last_idx = max(0, min(real_len - 2, seq_len - 2))
            self.items.append({
                "x": torch.tensor(ids[:-1], dtype=torch.long),
                "y": torch.tensor(ids[1:], dtype=torch.long),
                "last_idx":   torch.tensor(last_idx, dtype=torch.long),
                "opcode_id":  torch.tensor(int(row.get("opcode_id", 0)),  dtype=torch.long),
                "rule_id":    torch.tensor(int(row.get("rule_id", 0)),    dtype=torch.long),
                "schema_id":  torch.tensor(int(row.get("schema_id", 0)),  dtype=torch.long),
                "intent_id":  torch.tensor(int(row.get("intent_id", 0)),  dtype=torch.long),
                "validity":   torch.tensor(int(row.get("validity", 1)),   dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def prepare_structured_data(cfg: "TrainConfig"):
    """Build tokenizer + train/val datasets from a multitask JSONL.

    The tokenizer is trained from the JSONL's ``text`` fields concatenated
    into a single corpus, so special tags like ``<TASK:SCHEMA2RULE>`` and
    ``<OPCODE>`` get learned as merges rather than split into bytes.
    """
    assert cfg.jsonl_path, "--jsonl-path is required when --use-jsonl-dataset is set"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(cfg.jsonl_path).resolve()
    assert src.exists(), f"--jsonl-path not found: {src}"

    rows = load_jsonl_rows(src)
    assert rows, f"No rows found in {src}"

    tag = src.stem.replace(".", "_")[:40]
    corpus_txt = DATA_DIR / f"corpus_{tag}.txt"
    write_jsonl_text_corpus(rows, corpus_txt)

    # Reuse the BPE-from-text path.
    text_cfg = TrainConfig(**asdict(cfg))
    text_cfg.data_path = str(corpus_txt)
    tokenizer, vocab, _, _ = prepare_data_from_file(text_cfg)

    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    split_n = max(1, int(len(rows) * (1.0 - cfg.val_split)))
    train_rows = [rows[i] for i in idx[:split_n]]
    val_rows   = [rows[i] for i in idx[split_n:]]
    if not val_rows:
        val_rows = train_rows[: max(1, len(train_rows) // 10)]

    train_ds = StructuredTraceDataset(train_rows, tokenizer, cfg.seq_len)
    val_ds   = StructuredTraceDataset(val_rows,   tokenizer, cfg.seq_len)
    print(f"  multitask jsonl: {len(train_ds):,} train rows / "
          f"{len(val_ds):,} val rows from {src.name}")
    return tokenizer, vocab, train_ds, val_ds


def load_prompts(path: Path | None = None) -> list[str]:
    p = Path(path) if path else PROMPTS_PATH
    return [
        line.strip() for line in p.read_text(encoding="utf-8").splitlines()
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
    """MinGRU sequence mixer — see module docstring for math.

    When ``enable_hypergrad`` is set, the layer owns a learnable scalar
    ``surprise_scale`` (α) and scales the input as ``x' = x · (1 + α·S)``
    before projection. S is the optimizer's ``current_surprise_gain``
    from AutoHypergradientAdamW (or 0.0 when not paired).
    """

    def __init__(
        self,
        d_model: int,
        decay_bias_init: float = 1.0,
        enable_hypergrad: bool = False,
        hypergrad_scale_init: float = 0.1,
    ):
        super().__init__()
        self.proj_g = nn.Linear(d_model, d_model, bias=True)
        self.proj_v = nn.Linear(d_model, d_model, bias=True)
        self.proj_d = nn.Linear(d_model, d_model, bias=True)
        # Decay gate bias → sigmoid ≈ 0.73 retention at t=0.
        nn.init.constant_(self.proj_d.bias, decay_bias_init)

        self.enable_hypergrad = bool(enable_hypergrad)
        if self.enable_hypergrad:
            self.surprise_scale = nn.Parameter(
                torch.tensor([float(hypergrad_scale_init)])
            )
        else:
            self.register_parameter("surprise_scale", None)

    def forward(self, x: torch.Tensor, surprise_gain: float = 0.0) -> torch.Tensor:
        if self.surprise_scale is not None and surprise_gain != 0.0:
            x = x * (1.0 + self.surprise_scale * float(surprise_gain))
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
                 decay_bias_init: float = 1.0,
                 enable_hypergrad: bool = False,
                 hypergrad_scale_init: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            MinGRULayer(
                d_model, decay_bias_init,
                enable_hypergrad=enable_hypergrad,
                hypergrad_scale_init=hypergrad_scale_init,
            )
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor, surprise_gain: float = 0.0) -> torch.Tensor:
        B, S, D = x.shape
        # Router: per-token top-K expert selection
        gate_logits = self.gate(x)                          # (B, S, M)
        top_vals, top_idx = gate_logits.topk(self.top_k, dim=-1)  # (B, S, K)
        weights = torch.softmax(top_vals, dim=-1)           # (B, S, K)

        # Run all experts — each sees the same surprise_gain
        expert_outs = [exp(x, surprise_gain=surprise_gain) for exp in self.experts]      # M × (B, S, D)

        # Weighted sum via explicit loop — no gather/scatter, works on
        # any backend (CUDA, CPU, DirectML). K=2 × M=4 = 8 iterations
        # of cheap elementwise ops.
        result = torch.zeros_like(x)
        for k in range(self.top_k):
            w_k = weights[:, :, k : k + 1]                 # (B, S, 1)
            idx_k = top_idx[:, :, k]                        # (B, S)
            for m in range(self.n_experts):
                mask = (idx_k == m).unsqueeze(-1).to(x.dtype)  # (B, S, 1)
                result = result + w_k * mask * expert_outs[m]
        return result


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


class EpisodicMemory:
    """Dopamine-gated hippocampal memory — stores hidden states that led to
    coherent output, retrieves them to bias future generation.

    Biological loop:
      1. Model gets it right → loss drops → reward signal
      2. Reward → dopamine spikes → memory write gate opens
      3. Memory stores the hidden state that LED to the good output,
         tagged with the dopamine level at write time
      4. On retrieval, memories are ranked by ``cosine_sim × utility``
         — high-dopamine memories get retrieved more aggressively
      5. During sleep (consolidation), high-utility memories survive,
         low-utility ones get pruned, near-duplicates get merged

    Not differentiable — the gradient flows through the integration gate
    (``mem_gate`` in ``HybridBlock``), not through the memory itself.
    """

    def __init__(self, d_model: int, max_memories: int = 200,
                 write_threshold: float = 0.4, merge_threshold: float = 0.95):
        self.d = d_model
        self.max = max_memories
        self.write_threshold = write_threshold
        self.merge_threshold = merge_threshold

        # Storage: parallel lists for efficiency (no class overhead)
        self.keys: list[torch.Tensor] = []      # (D,) detached
        self.values: list[torch.Tensor] = []     # (D,) detached
        self.utilities: list[float] = []         # dopamine at write time
        self.ages: list[int] = []                # steps since write
        self._step = 0

    @property
    def size(self) -> int:
        return len(self.keys)

    @torch.no_grad()
    def write(self, key: torch.Tensor, value: torch.Tensor,
              utility: float) -> bool:
        """Write a memory tagged with its utility (loss improvement gap).

        Bigger gap = more important memory = survives consolidation longer.
        When at capacity, evicts the lowest-utility entry — naturally
        creates a "hall of fame" of the biggest training breakthroughs.
        """
        if utility <= 0:
            return False
        self.keys.append(key.detach().cpu().float())
        self.values.append(value.detach().cpu().float())
        self.utilities.append(utility)
        self.ages.append(0)
        # Evict lowest-utility if at capacity
        if len(self.keys) > self.max:
            min_idx = min(range(len(self.utilities)),
                          key=lambda i: self.utilities[i])
            for lst in (self.keys, self.values, self.utilities, self.ages):
                lst.pop(min_idx)
        return True

    @torch.no_grad()
    def read(self, query: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Retrieve top-K memories ranked by ``cosine_sim × utility``.
        Returns (D,) tensor on the same device as query."""
        if not self.keys:
            return torch.zeros(self.d, device=query.device, dtype=query.dtype)
        keys_t = torch.stack(self.keys)                     # (N, D)
        q = query.detach().cpu().float()
        sims = F.cosine_similarity(q.unsqueeze(0), keys_t, dim=-1)  # (N,)
        utils = torch.tensor(self.utilities, dtype=torch.float32)
        scores = sims * utils                               # reward-weighted
        top_k = min(k, len(self.keys))
        vals, idx = scores.topk(top_k)
        weights = F.softmax(vals, dim=-1)                   # (K,)
        retrieved = torch.stack([self.values[i] for i in idx])  # (K, D)
        result = (weights.unsqueeze(-1) * retrieved).sum(dim=0)  # (D,)
        return result.to(device=query.device, dtype=query.dtype)

    def tick(self) -> None:
        """Advance age for all memories (call once per training step)."""
        self._step += 1
        for i in range(len(self.ages)):
            self.ages[i] += 1

    @torch.no_grad()
    def consolidate(self) -> dict:
        """Sleep-phase consolidation:
          1. Merge near-duplicate memories (cosine > merge_threshold)
          2. Decay utilities by age (older → less utility unless refreshed)
          3. Prune lowest-utility memories down to 80% capacity

        Returns stats dict for logging.
        """
        if not self.keys:
            return {"merged": 0, "pruned": 0, "remaining": 0}

        # 1. Merge near-duplicates
        merged = 0
        keys_t = torch.stack(self.keys)
        i = 0
        while i < len(self.keys):
            if i + 1 >= len(self.keys):
                break
            sims = F.cosine_similarity(
                self.keys[i].unsqueeze(0),
                torch.stack(self.keys[i + 1:]), dim=-1,
            )
            dups = (sims > self.merge_threshold).nonzero(as_tuple=True)[0]
            for j in sorted(dups.tolist(), reverse=True):
                real_j = i + 1 + j
                # Merge: keep higher utility, average the vectors
                self.keys[i] = (self.keys[i] + self.keys[real_j]) / 2
                self.values[i] = (self.values[i] + self.values[real_j]) / 2
                self.utilities[i] = max(self.utilities[i],
                                        self.utilities[real_j])
                for lst in (self.keys, self.values, self.utilities, self.ages):
                    lst.pop(real_j)
                merged += 1
            i += 1

        # 2. Age-based utility decay (half-life = 500 steps)
        for i in range(len(self.utilities)):
            decay = 0.5 ** (self.ages[i] / 500.0)
            self.utilities[i] *= decay
            self.ages[i] = 0  # reset age after consolidation

        # 3. Prune to 80% capacity
        target = int(self.max * 0.8)
        pruned = 0
        while len(self.keys) > target:
            min_idx = min(range(len(self.utilities)),
                          key=lambda i: self.utilities[i])
            for lst in (self.keys, self.values, self.utilities, self.ages):
                lst.pop(min_idx)
            pruned += 1

        return {"merged": merged, "pruned": pruned,
                "remaining": len(self.keys)}


class HybridBlock(nn.Module):
    """MinGRU (or MoE-MinGRU) + optional local attention + optional
    hippocampal memory injection + GLU.

    Architecture per block::

        x = x + Mixer(RMSNorm(x))         # MinGRU or MoE-MinGRU
        x = x + Attn(RMSNorm(x))          # local attention (if enabled)
        x = x + MemGate(hippo.read(x))    # memory injection (if enabled)
        x = x + GLU(RMSNorm(x))           # channel mixing

    Memory injection: the block's mean hidden state queries the episodic
    memory; retrieved memories are projected through a learned gate and
    broadcast to all positions. This doesn't affect gradients on the
    memory itself — only the gate is trainable.
    """

    def __init__(self, cfg: "TrainConfig", layer_idx: int,
                 memory: "EpisodicMemory | None" = None):
        super().__init__()
        d = cfg.d_model
        self.layer_idx = layer_idx

        # Sequence mixer
        self.rms_mix = RMSNorm(d)
        if cfg.enable_moe:
            self.mix = MoEMinGRULayer(
                d, cfg.moe_n_experts, cfg.moe_top_k,
                enable_hypergrad=cfg.enable_hypergrad,
                hypergrad_scale_init=cfg.hypergrad_scale_init,
            )
        else:
            self.mix = MinGRULayer(
                d,
                enable_hypergrad=cfg.enable_hypergrad,
                hypergrad_scale_init=cfg.hypergrad_scale_init,
            )

        # Local attention (on selected layers)
        self.attn = None
        self.rms_attn = None
        if cfg.enable_attention and layer_idx % cfg.attn_every_n == 0:
            self.rms_attn = RMSNorm(d)
            self.attn = LocalCausalAttention(d, cfg.attn_n_heads, cfg.attn_window)

        # Hippocampal memory injection (on selected layers)
        self.memory = None
        self.mem_gate = None
        if memory is not None and cfg.enable_memory and layer_idx % cfg.mem_every_n == 0:
            self.memory = memory
            self.mem_gate = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.mem_gate.weight)  # start as no-op

        # FFN
        self.rms_ffn = RMSNorm(d)
        self.ffn = GLUChannelMix(d, cfg.d_ffn)

    def forward(self, x: torch.Tensor, surprise_gain: float = 0.0) -> torch.Tensor:
        x = x + self.mix(self.rms_mix(x), surprise_gain=surprise_gain)
        if self.attn is not None:
            x = x + self.attn(self.rms_attn(x))
        if self.memory is not None and self.memory.size > 0:
            # Query memory with sequence-mean hidden state
            x_mean = x.mean(dim=1)                          # (B, D)
            retrieved = self.memory.read(x_mean[0])          # (D,) from batch[0]
            mem_inject = self.mem_gate(retrieved)             # (D,) — learned gate
            x = x + mem_inject.unsqueeze(0).unsqueeze(0)     # broadcast (1, 1, D)
        x = x + self.ffn(self.rms_ffn(x))
        return x


class MinGRUBlock(nn.Module):
    """Original pure-MinGRU block (no MoE, no attention)."""
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.rms_mix = RMSNorm(cfg.d_model)
        self.rms_ffn = RMSNorm(cfg.d_model)
        self.mix = MinGRULayer(
            cfg.d_model,
            enable_hypergrad=cfg.enable_hypergrad,
            hypergrad_scale_init=cfg.hypergrad_scale_init,
        )
        self.ffn = GLUChannelMix(cfg.d_model, cfg.d_ffn)

    def forward(self, x: torch.Tensor, surprise_gain: float = 0.0) -> torch.Tensor:
        x = x + self.mix(self.rms_mix(x), surprise_gain=surprise_gain)
        x = x + self.ffn(self.rms_ffn(x))
        return x


class MinGRUModel(nn.Module):
    """PyTorch MinGRU language model — selects pure or hybrid blocks.

    When ``enable_memory`` is True, an ``EpisodicMemory`` instance is
    shared across selected layers. The training loop is responsible for
    calling ``dopamine_write()`` after each step and ``consolidate()``
    every ``mem_consolidate_every`` steps.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # Shared episodic memory (if enabled)
        self.memory: EpisodicMemory | None = None
        if cfg.enable_memory:
            self.memory = EpisodicMemory(
                cfg.d_model, max_memories=cfg.mem_max,
                write_threshold=cfg.mem_write_threshold,
            )

        use_hybrid = cfg.enable_moe or cfg.enable_attention or cfg.enable_memory
        if use_hybrid:
            self.blocks = nn.ModuleList([
                HybridBlock(cfg, i, memory=self.memory)
                for i in range(cfg.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                MinGRUBlock(cfg) for i in range(cfg.n_layers)
            ])

        self.rms_f = RMSNorm(cfg.d_model)
        if getattr(cfg, "vsa_binding_head", False):
            # Untied VSA binding head — embedding stays learned, output
            # side uses a fixed MAP-bipolar codebook cosine lookup.
            from vsa_binding_head import VSABindingHead
            self.head = VSABindingHead(
                d_model=cfg.d_model,
                vocab_size=cfg.vocab_size,
                d_vsa=getattr(cfg, "vsa_binding_d", 10240),
                seed=getattr(cfg, "vsa_binding_seed", 0xC0DEB00C),
            )
            # No tying — the binding head has no weight matrix to tie to.
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.head.weight = self.embed.weight

        # Memory write state — simple loss-improvement gating.
        # Every step where loss improves, the hidden state is stored
        # with utility = gap size. Biggest breakthroughs get highest
        # utility and survive consolidation; small improvements get
        # evicted when memory fills.
        self._prev_loss: float = 10.0

    def forward_features(self, tokens: torch.Tensor,
                         surprise_gain: float = 0.0) -> torch.Tensor:
        """Backbone hidden states (B, S, D) — no LM projection.

        Split out so the multitask wrapper can attach auxiliary heads
        on the same representation the LM head uses, without re-running
        the embedding + N blocks + final RMSNorm pipeline twice.
        """
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h, surprise_gain=surprise_gain)
        h = self.rms_f(h)
        # Cache the sequence-mean hidden state for dopamine_write.
        # Detached — no gradient through the memory system.
        self._cached_h_mean = h.detach().mean(dim=1).mean(dim=0)  # (D,)
        return h

    def forward(self, tokens: torch.Tensor, surprise_gain: float = 0.0) -> torch.Tensor:
        return self.head(self.forward_features(tokens, surprise_gain=surprise_gain))

    def dopamine_write(self, loss: float) -> dict:
        """Call after each training step. If loss improved, store the
        hidden state that led to the improvement, tagged with the gap
        size as utility. Bigger improvements → higher utility → survive
        consolidation longer → get recalled more aggressively.

        No prediction-error complexity — just: did loss improve? By how
        much? Store it. The memory naturally becomes a "hall of fame" of
        the model's biggest training breakthroughs.
        """
        if self.memory is None:
            return {"improvement": 0.0, "wrote": False, "mem_size": 0}

        improvement = max(0.0, self._prev_loss - loss)
        self._prev_loss = loss

        wrote = False
        if improvement > 0 and self._cached_h_mean is not None:
            wrote = self.memory.write(
                key=self._cached_h_mean,
                value=self._cached_h_mean,
                utility=improvement,
            )

        self.memory.tick()
        return {
            "improvement": improvement,
            "wrote": wrote,
            "mem_size": self.memory.size,
        }

    def consolidate(self) -> dict:
        """Sleep-phase consolidation. Call every ``mem_consolidate_every`` steps."""
        if self.memory is None:
            return {}
        return self.memory.consolidate()


class MindForgeLoRAHead(nn.Module):
    """Hypernetwork-forged LoRA head — PyTorch port of cubemind.execution.mindforge.

    Each forward forges a per-sample (A, B) low-rank adapter from a
    *context* vector and adds the LoRA delta on top of a frozen-style
    base projection::

        ctx_h    = LayerNorm(ctx_proj(context))
        coeffs   = MLP(ctx_h)                                   # (B, n_basis)
        A        = sum(coeffs[i] * basis_A[i])                  # (B, rank, d_in)
        B        = sum(coeffs[i] * basis_B[i])                  # (B, d_out, rank)
        output   = base(x) + scale * (x @ A.T) @ B.T            # per-sample LoRA

    Plasticity split (matches Oja/NLMS pattern from the prior live demo):

        basis_A          — structural prior, offline training only
        basis_B          — PLASTIC, zero-init, eligible for online NLMS
        base / ctx_proj  — structural prior, offline training only
        ctx_norm / coeff — structural prior, offline training only

    ``basis_B`` is zero-initialized so the adapter contributes 0 at init
    (head behaves like a plain ``nn.Linear``); it's the only parameter
    updated during inference-time ``online_update`` so online learning
    can't destabilize the base projection. Offline training updates
    every parameter normally via the main optimizer.

    Classification only in this prototype — returns raw logits, loss is
    cross-entropy. Regression mode (cosine head for future-VSA targets)
    will land alongside the future_vsa head once precomputed targets are
    available.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        n_basis: int = 8,
        rank: int = 4,
        hidden_dim: int = 128,
        scale: float = 1.0,
        context_dim: int | None = None,
    ):
        super().__init__()
        ctx_d = context_dim or d_model
        self.d_model = d_model
        self.num_classes = num_classes
        self.rank = rank
        self.n_basis = n_basis
        self.scale = scale

        # Trainable base projection — equivalent to a plain head at init.
        self.base = nn.Linear(d_model, num_classes, bias=True)
        # Hypernet: context → mixing coefficients over the basis.
        self.ctx_proj = nn.Linear(ctx_d, hidden_dim)
        self.ctx_norm = nn.LayerNorm(hidden_dim)
        self.coeff    = nn.Linear(hidden_dim, n_basis)
        # Shared basis. A is small (rank × d_in), B is wider (d_out × rank).
        basis_std = math.sqrt(2.0 / (d_model + rank))
        self.basis_A = nn.Parameter(
            torch.randn(n_basis, rank, d_model) * basis_std
        )
        # Zero-init B → the LoRA delta is exactly 0 at step 0 → identity start.
        # This is the PLASTIC parameter — only one updated by online_update.
        self.basis_B = nn.Parameter(torch.zeros(n_basis, num_classes, rank))

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: pooled features (B, d_model). ``context``: optional
        conditioning vector (B, ctx_d). Defaults to ``x`` (self-conditioned)."""
        if context is None:
            context = x
        h = F.gelu(self.ctx_norm(self.ctx_proj(context)))  # (B, hidden_dim)
        coeffs = self.coeff(h)                              # (B, n_basis)
        # Per-sample basis mix — einsum keeps it differentiable + batched.
        A = torch.einsum("bn,nrd->brd", coeffs, self.basis_A)  # (B, rank, d_in)
        B = torch.einsum("bn,ncr->bcr", coeffs, self.basis_B)  # (B, d_out, rank)
        # LoRA forward: (x @ A.T) @ B.T per sample.
        z = torch.einsum("bd,brd->br", x, A)                   # (B, rank)
        delta = torch.einsum("br,bcr->bc", z, B)               # (B, num_classes)
        return self.base(x) + self.scale * delta

    @torch.no_grad()
    def online_update(self, pooled: torch.Tensor, target: torch.Tensor,
                      lr: float = 1e-3, context: torch.Tensor | None = None) -> float:
        """Single-step NLMS update on ``basis_B`` only.

        Called at inference time when a teacher/verifier signal arrives.
        All other parameters (base, ctx_proj, coeff, basis_A) stay frozen
        so the base projection remains stable across long sessions.

        Returns the pre-update loss (so the caller can log plasticity
        pressure over time).
        """
        with torch.enable_grad():
            basis_B = self.basis_B.detach().clone().requires_grad_(True)
            if context is None:
                context = pooled
            h = F.gelu(self.ctx_norm(self.ctx_proj(context)))
            coeffs = self.coeff(h)
            A = torch.einsum("bn,nrd->brd", coeffs, self.basis_A.detach())
            B = torch.einsum("bn,ncr->bcr", coeffs, basis_B)
            z = torch.einsum("bd,brd->br", pooled, A)
            delta = torch.einsum("br,bcr->bc", z, B)
            pred = self.base(pooled).detach() + self.scale * delta
            loss = F.cross_entropy(pred, target.long())
            (grad_B,) = torch.autograd.grad(loss, basis_B, retain_graph=False)
        # NLMS-style normalized update — prevents runaway steps when
        # the gradient norm spikes on a single correction.
        norm = grad_B.norm() + 1e-8
        self.basis_B.data -= lr * grad_B / norm
        return float(loss.item())


# ── Default head registry — declarative, mutable from CLI / config ──────
#
# Each entry is a dict the multitask wrapper consumes at construction
# time. Add or remove rows here (or via add_head at runtime) without
# editing the wrapper class. Loss + accuracy logic is dispatched off
# ``output_mode`` and ``loss_type`` so new heads don't need code changes.
#
#   label_key:        the per-row JSONL field this head supervises on
#   num_classes:      output dimension (vocab for classification, D for VSA)
#   output_mode:      "classification" | "regression"
#   loss_type:        "ce" | "cosine"
#   weight:           scalar multiplier on the head's loss
#   task_code_source: "self" | a label_key (for cross-head conditioning)
#   rank / n_basis:   MindForge LoRA capacity
#
# The future_vsa head is included but has weight=0 by default so it
# only activates when the JSONL provides a precomputed target AND the
# user enables it explicitly via --aux-future-vsa-loss-weight.

DEFAULT_HEAD_SPECS: list[dict] = [
    {"name": "opcode",     "label_key": "opcode_id", "num_classes_attr": "num_opcode_classes",
     "loss_type": "ce", "weight_attr": "aux_opcode_loss_weight",
     "task_code_source": "self",       "rank": 4, "n_basis": 8},
    {"name": "intent",     "label_key": "intent_id", "num_classes_attr": "num_intent_classes",
     "loss_type": "ce", "weight_attr": "aux_intent_loss_weight",
     "task_code_source": "self",       "rank": 2, "n_basis": 4},
    {"name": "schema",     "label_key": "schema_id", "num_classes_attr": "num_schema_classes",
     "loss_type": "ce", "weight_attr": "aux_schema_loss_weight",
     "task_code_source": "self",       "rank": 2, "n_basis": 4},
    {"name": "rule",       "label_key": "rule_id",   "num_classes_attr": "num_rule_classes",
     "loss_type": "ce", "weight_attr": "aux_rule_loss_weight",
     "task_code_source": "self",       "rank": 4, "n_basis": 8},
    {"name": "validity",   "label_key": "validity",  "num_classes_attr": None,  # always 2
     "num_classes": 2,
     "loss_type": "ce", "weight_attr": "aux_validity_loss_weight",
     "task_code_source": "self",       "rank": 2, "n_basis": 2},
    # NOTE: future_vsa head is deferred — it needs precomputed 10240-d
    # MAP-bipolar targets per row, which the data pipeline doesn't emit
    # yet. Re-add as {"name": "future_vsa", ..., "loss_type": "cosine"}
    # with the regression dispatch path once the precompute task lands.
]


class MinGRUMultiTask(nn.Module):
    """Backbone + flexible registry of auxiliary heads.

    Heads live in ``self.heads`` (an ``nn.ModuleDict``) so they can be
    added or removed at any time without subclassing — call ``add_head``
    to register a new one mid-training (e.g. when a new schema appears
    in the data stream).

    The forward returns ``{token_logits, <head>_logits for each head}``.
    The train loop iterates ``cfg.head_specs`` to build per-head losses
    and accuracies, so adding a head is purely a config change.

    Real-time learning: ``online_update(head_name, pooled, target, lr)``
    runs a single NLMS step on that head's plastic ``basis_B`` only,
    leaving everything else frozen. Use this when a teacher/verifier
    signal arrives mid-conversation.
    """

    def __init__(
        self,
        backbone: MinGRUModel,
        d_model: int,
        head_specs: list[dict],
        cfg: "TrainConfig",
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.head_specs: list[dict] = []          # active specs (order matters for logging)
        self.heads = nn.ModuleDict()
        for spec in head_specs:
            num_classes = spec.get("num_classes")
            if num_classes is None and spec.get("num_classes_attr"):
                num_classes = getattr(cfg, spec["num_classes_attr"], None)
            if num_classes is None:
                continue  # spec couldn't be resolved (missing config attr) — skip
            self.add_head(
                name=spec["name"],
                num_classes=int(num_classes),
                rank=spec.get("rank", 4),
                n_basis=spec.get("n_basis", 8),
                spec=spec,
            )

    def add_head(self, name: str, num_classes: int,
                 rank: int = 4, n_basis: int = 8,
                 spec: dict | None = None) -> None:
        """Register a new head. Safe to call any time — the train loop
        picks it up on the next iteration via ``self.head_specs``."""
        head = MindForgeLoRAHead(
            d_model=self.d_model,
            num_classes=num_classes,
            n_basis=n_basis,
            rank=rank,
        )
        # Move the new head to the same device as the backbone.
        try:
            head = head.to(next(self.backbone.parameters()).device)
        except StopIteration:
            pass
        self.heads[name] = head
        if spec is None:
            spec = {"name": name, "label_key": f"{name}_id" if output_mode == "classification" else name,
                    "num_classes": num_classes, "output_mode": output_mode,
                    "loss_type": "ce" if output_mode == "classification" else "cosine",
                    "weight": 1.0, "task_code_source": "self",
                    "rank": rank, "n_basis": n_basis}
        self.head_specs.append(spec)

    def remove_head(self, name: str) -> None:
        if name in self.heads:
            del self.heads[name]
        self.head_specs = [s for s in self.head_specs if s["name"] != name]

    def forward(self, tokens: torch.Tensor,
                surprise_gain: float = 0.0,
                last_idx: torch.Tensor | None = None,
                ) -> dict[str, torch.Tensor]:
        h = self.backbone.forward_features(tokens, surprise_gain=surprise_gain)
        if last_idx is None:
            pooled = h[:, -1, :]
        else:
            B = h.size(0)
            pooled = h[torch.arange(B, device=h.device), last_idx]
        out = {"token_logits": self.backbone.head(h), "_pooled": pooled}
        for name, head in self.heads.items():
            out[f"{name}_logits"] = head(pooled)
        return out

    @torch.no_grad()
    def online_update(self, head_name: str, pooled: torch.Tensor,
                      target: torch.Tensor, lr: float = 1e-3) -> float:
        """One NLMS step on ``heads[head_name].basis_B`` only.

        ``pooled`` is the last-position hidden state from a previous
        forward (call ``model(tokens)["_pooled"]`` to get it). ``target``
        is the ground-truth label (long for CE) or VSA vector (float for
        cosine). Returns the pre-update loss for plasticity logging.
        """
        return self.heads[head_name].online_update(pooled, target, lr=lr)

    # ── Backbone passthrough hooks (dopamine + consolidation) ──────────────

    def dopamine_write(self, loss: float) -> dict:
        return self.backbone.dopamine_write(loss)

    def consolidate(self) -> dict:
        return self.backbone.consolidate()


# ── Training utilities ───────────────────────────────────────────────────

class SurpriseTracker:
    """Minimal port of AutoHypergradientAdamW's surprise signal.

    The grilly optimizer uses per-parameter gradient-direction prediction
    error; here we use the scalar global grad-norm that the train loop
    already computes, which is a reasonable proxy (captures "gradient
    magnitude shifted unexpectedly") without flattening every parameter.

    S_instant = tanh((||g|| - EMA_norm)^2 / (EMA_sq + eps))
    S_bar     = alpha * S_instant + (1 - alpha) * S_bar_prev
    gain      = S_bar * exp(-S_bar / trauma_threshold)

    Reads from ``after_step()`` into ``current_surprise_gain`` for the
    next forward pass. Returns 0.0 during warmup.
    """

    def __init__(
        self,
        warmup_steps: int = 10,
        gamma: float = 0.9,       # EMA decay for gradient tracking
        alpha: float = 0.1,       # EMA decay for accumulated surprise
        trauma_threshold: float = 0.5,
        eps: float = 1e-8,
    ):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.alpha = alpha
        self.trauma_threshold = trauma_threshold
        self.eps = eps
        self._step = 0
        self._ema_norm = 0.0
        self._ema_sq = 0.0
        self._s_bar = 0.0
        self.current_surprise_gain: float = 0.0

    def after_step(self, grad_norm: float) -> float:
        self._step += 1
        gn = float(grad_norm)
        # Defensive: a non-finite grad-norm (exploding gradient, skipped
        # batch) would poison the EMA forever. Drop the update and reuse
        # the previous gain.
        if not math.isfinite(gn):
            return self.current_surprise_gain
        if self._step < self.warmup_steps:
            # Build the baseline during warmup; no gain yet.
            self._ema_norm = self.gamma * self._ema_norm + (1 - self.gamma) * gn
            self._ema_sq   = self.gamma * self._ema_sq   + (1 - self.gamma) * gn * gn
            self.current_surprise_gain = 0.0
            return 0.0
        diff = gn - self._ema_norm
        s_instant = math.tanh((diff * diff) / (self._ema_sq + self.eps))
        self._s_bar = self.alpha * s_instant + (1 - self.alpha) * self._s_bar
        # Yerkes-Dodson inverted-U: gain rises, then trauma suppresses it.
        gain = self._s_bar * math.exp(-self._s_bar / max(self.trauma_threshold, 1e-6))
        # Update the EMA for next step AFTER using the prior baseline.
        self._ema_norm = self.gamma * self._ema_norm + (1 - self.gamma) * gn
        self._ema_sq   = self.gamma * self._ema_sq   + (1 - self.gamma) * gn * gn
        self.current_surprise_gain = float(gain)
        return self.current_surprise_gain


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


def _head_weight(spec: dict, cfg: "TrainConfig") -> float:
    """Resolve a head's loss weight from the cfg attr named in its spec.

    Falls back to ``spec["weight"]`` (used by hot-added heads that
    weren't present at TrainConfig construction time)."""
    attr = spec.get("weight_attr")
    if attr and hasattr(cfg, attr):
        return float(getattr(cfg, attr))
    return float(spec.get("weight", 0.0))


def _head_loss(spec: dict, logits: torch.Tensor,
               target: torch.Tensor) -> torch.Tensor:
    """Per-head loss — CE only in this prototype. Cosine path returns
    when the future_vsa head re-lands with precomputed targets."""
    return F.cross_entropy(logits, target.long())


def compute_multitask_loss(
    out: dict[str, torch.Tensor],
    targets: torch.Tensor,
    head_targets: dict[str, torch.Tensor],
    head_specs: list[dict],
    cfg: "TrainConfig",
) -> torch.Tensor:
    """LM CE + Σ (head weight) × (head loss).

    ``head_targets`` is keyed by head name; missing entries skip that
    head (so a JSONL without future_vsa_target won't break the graph).
    Heads with weight=0 are also skipped — saves a CE / einsum we'd
    otherwise multiply by zero.
    """
    loss = compute_loss(out["token_logits"], targets)
    for spec in head_specs:
        name = spec["name"]
        target = head_targets.get(name)
        weight = _head_weight(spec, cfg)
        if target is None or weight == 0.0:
            continue
        loss = loss + weight * _head_loss(spec, out[f"{name}_logits"], target)
    return loss


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


@torch.no_grad()
def evaluate_structured(
    model: "MinGRUMultiTask",
    val_loader,
    cfg: TrainConfig,
    device: torch.device,
    autocast_ctx,
) -> dict:
    """Per-head metric + LM CE on a structured DataLoader.

    Iterates ``model.head_specs`` so adding a new head shows up in the
    eval log without touching this function. Classification heads
    report top-1 accuracy; regression heads report mean cosine
    similarity to the target (1.0 = perfect, 0.0 = orthogonal).
    """
    model.eval()
    lm_losses: list[float] = []
    metric_buckets: dict[str, list[float]] = {s["name"]: [] for s in model.head_specs}
    for batch in val_loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        last_idx = batch["last_idx"].to(device, non_blocking=True)
        with autocast_ctx():
            out = model(x, last_idx=last_idx)
            loss_lm = compute_loss(out["token_logits"], y)
        if torch.isfinite(loss_lm):
            lm_losses.append(float(loss_lm.item()))
        for spec in model.head_specs:
            name = spec["name"]
            if spec["label_key"] not in batch:
                continue
            tgt = batch[spec["label_key"]].to(device, non_blocking=True)
            logits = out[f"{name}_logits"]
            metric = (logits.argmax(dim=-1) == tgt.long()).float().mean()
            metric_buckets[name].append(float(metric.item()))
    model.train()
    metrics: dict = {"val_ce": float(np.mean(lm_losses)) if lm_losses else float("inf")}
    for name, vals in metric_buckets.items():
        metrics[f"{name}_metric"] = float(np.mean(vals)) if vals else 0.0
    return metrics


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
    prompts = load_prompts(cfg.prompts_path or None)
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

    # Device — supports cuda, cpu, dml (DirectML for AMD/Intel GPUs on Windows)
    _dml_device = None
    if cfg.device in ("auto", "dml"):
        if cfg.device == "auto" and torch.cuda.is_available():
            pass  # fall through to cuda
        else:
            try:
                import torch_directml
                _dml_device = torch_directml.device()
                print(f"  DirectML: {torch_directml.device_name(0)}")
            except ImportError:
                if cfg.device == "dml":
                    raise RuntimeError("--device dml requires: pip install torch-directml")

    if _dml_device is not None:
        device = _dml_device
    elif cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
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
    if cfg.use_jsonl_dataset:
        tokenizer, vocab, train_data, val_data = prepare_structured_data(cfg)
    else:
        tokenizer, vocab, train_data, val_data = prepare_data(cfg)
    if vocab != cfg.vocab_size:
        print(f"  note: actual vocab {vocab} != configured {cfg.vocab_size}")
        cfg.vocab_size = vocab

    print("\n--- model ---")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    backbone = MinGRUModel(cfg)
    if cfg.use_jsonl_dataset:
        model = MinGRUMultiTask(
            backbone=backbone,
            d_model=cfg.d_model,
            head_specs=DEFAULT_HEAD_SPECS,
            cfg=cfg,
        ).to(device)
        head_dims = ", ".join(
            f"{s['name']}={s.get('num_classes') or getattr(cfg, s.get('num_classes_attr', ''), '?')}"
            for s in model.head_specs
        )
        print(f"  multitask heads: {head_dims}")
    else:
        model = backbone.to(device)
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
    surprise_tracker = SurpriseTracker() if cfg.enable_hypergrad else None
    if surprise_tracker is not None:
        print("  SurpriseTracker: on (gradient-norm EMA, Yerkes-Dodson gain)")
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

    train_loader = val_loader = train_iter = None
    if cfg.use_jsonl_dataset:
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                                  shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_data,   batch_size=cfg.batch_size,
                                  shuffle=False)
        train_iter = iter(train_loader)

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

        surprise_gain = (surprise_tracker.current_surprise_gain
                         if surprise_tracker is not None else 0.0)

        loss_val = 0.0
        for _ in range(cfg.grad_accum):
            if cfg.use_jsonl_dataset:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                last_idx = batch["last_idx"].to(device, non_blocking=True)
                # Build the head_targets dict from the spec registry —
                # heads added via add_head() at runtime get picked up
                # automatically as long as their label_key is in the batch.
                head_targets = {}
                for spec in model.head_specs:
                    key = spec["label_key"]
                    if key in batch:
                        head_targets[spec["name"]] = batch[key].to(
                            device, non_blocking=True)
                with autocast_ctx():
                    out = model(x, surprise_gain=surprise_gain, last_idx=last_idx)
                    loss = compute_multitask_loss(
                        out, y, head_targets, model.head_specs, cfg,
                    ) / cfg.grad_accum
            else:
                x, y = sample_batch(train_data, cfg.batch_size, cfg.seq_len, rng)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast_ctx():
                    logits = model(x, surprise_gain=surprise_gain)
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

        if surprise_tracker is not None:
            surprise_tracker.after_step(gn)

        # ── Dopamine-gated memory write ─────────────────────────────────
        mem_info = model.dopamine_write(loss_val / cfg.grad_accum)

        # ── Sleep consolidation ─────────────────────────────────────────
        if (cfg.enable_memory and cfg.mem_consolidate_every > 0
                and step % cfg.mem_consolidate_every == 0):
            cons = model.consolidate()
            print(f"  [sleep] step {step:,}: "
                  f"merged={cons.get('merged', 0)} "
                  f"pruned={cons.get('pruned', 0)} "
                  f"remaining={cons.get('remaining', 0)}")

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
            if cfg.use_jsonl_dataset:
                metrics = evaluate_structured(model, val_loader, cfg,
                                              device, autocast_ctx)
                val_ce = metrics["val_ce"]
            else:
                val_ce = evaluate(model, val_data, cfg, device, autocast_ctx,
                                  max_batches=25)
                metrics = None
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
            if metrics is not None:
                parts = []
                for s in model.head_specs:
                    v = metrics.get(f"{s['name']}_metric", 0.0)
                    parts.append(f"{s['name']} {v:.3f}")
                head_str = " | ".join(parts)
                print(f"\n  === step {step:,} ===  val CE {val_ce:.4f}  "
                      f"PPL {val_ppl:.2f}{tag} | {head_str}")
            else:
                print(f"\n  === step {step:,} ===  val CE {val_ce:.4f}  "
                      f"PPL {val_ppl:.2f}{tag}")
            # Skip generation in structured mode — gen prompts assume
            # free-text continuation, not labeled multitask rows.
            if not cfg.use_jsonl_dataset:
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
    if cfg.use_jsonl_dataset:
        final_metrics = evaluate_structured(model, val_loader, cfg,
                                            device, autocast_ctx)
        final_val = final_metrics["val_ce"]
    else:
        final_val = evaluate(model, val_data, cfg, device, autocast_ctx,
                             max_batches=50)
        final_metrics = None
    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))

    if not cfg.use_jsonl_dataset:
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
    ap.add_argument("--data-path", type=str, default="",
                    help="Path to a raw .txt file. Bypasses TinyStories "
                         "download — trains BPE on the file, tokenizes, "
                         "caches bins. Use with --vocab for larger vocabs.")
    ap.add_argument("--val-split", type=float, default=cfg.val_split,
                    help="Fraction held out for val when using --data-path")
    ap.add_argument("--log-every", type=int, default=cfg.log_every)
    ap.add_argument("--eval-every", type=int, default=cfg.eval_every)
    ap.add_argument("--ckpt-every", type=int, default=cfg.ckpt_every)
    ap.add_argument("--gen-tokens", type=int, default=cfg.gen_max_tokens)
    ap.add_argument("--gen-temp", type=float, default=cfg.gen_temperature)
    ap.add_argument("--gen-top-p", type=float, default=cfg.gen_top_p)
    # Hybrid architecture
    # Hippocampal memory
    ap.add_argument("--memory", action="store_true",
                    help="Enable dopamine-gated hippocampal episodic memory")
    ap.add_argument("--mem-max", type=int, default=cfg.mem_max)
    ap.add_argument("--mem-write-threshold", type=float,
                    default=cfg.mem_write_threshold)
    ap.add_argument("--mem-every-n", type=int, default=cfg.mem_every_n,
                    help="Inject memory every Nth layer")
    ap.add_argument("--mem-consolidate-every", type=int,
                    default=cfg.mem_consolidate_every,
                    help="Sleep/consolidation interval in steps")
    # MoE recurrence
    ap.add_argument("--moe", action="store_true",
                    help="Enable MoE-MinGRU (4 experts, top-2 routing)")
    ap.add_argument("--moe-experts", type=int, default=cfg.moe_n_experts)
    ap.add_argument("--moe-top-k", type=int, default=cfg.moe_top_k)
    ap.add_argument("--attention", action="store_true",
                    help="Enable local sliding-window attention every N layers")
    ap.add_argument("--attn-heads", type=int, default=cfg.attn_n_heads)
    ap.add_argument("--attn-window", type=int, default=cfg.attn_window)
    ap.add_argument("--attn-every-n", type=int, default=cfg.attn_every_n)
    # Prompts for gen_step_*.md (eval samples written during training)
    ap.add_argument("--prompts", default=cfg.prompts_path,
                    help="Path to a .txt file of newline-separated prompts. "
                         "Default: sandbox/mingru_baseline/prompts_tinystories.txt. "
                         "For news-style corpora, pass prompts_news.txt.")
    # Multitask: structured JSONL with opcode/intent/schema/rule/validity labels
    ap.add_argument("--use-jsonl-dataset", action="store_true",
                    help="Train on structured JSONL rows instead of flat LM bins")
    ap.add_argument("--jsonl-path", type=str, default=cfg.jsonl_path,
                    help="Path to JSONL with text + opcode_id/intent_id/"
                         "schema_id/rule_id/validity labels")
    ap.add_argument("--aux-opcode-loss-weight",   type=float, default=cfg.aux_opcode_loss_weight)
    ap.add_argument("--aux-intent-loss-weight",   type=float, default=cfg.aux_intent_loss_weight)
    ap.add_argument("--aux-schema-loss-weight",   type=float, default=cfg.aux_schema_loss_weight)
    ap.add_argument("--aux-rule-loss-weight",     type=float, default=cfg.aux_rule_loss_weight)
    ap.add_argument("--aux-validity-loss-weight", type=float, default=cfg.aux_validity_loss_weight)
    ap.add_argument("--num-opcode-classes", type=int, default=cfg.num_opcode_classes)
    ap.add_argument("--num-intent-classes", type=int, default=cfg.num_intent_classes)
    ap.add_argument("--num-schema-classes", type=int, default=cfg.num_schema_classes)
    ap.add_argument("--num-rule-classes",   type=int, default=cfg.num_rule_classes)
    # VSA binding head (LM vocab replacement via bipolar codebook cosine)
    ap.add_argument("--vsa-binding-head", action="store_true",
                    help="Replace the tied Linear LM head with a VSA binding "
                         "head — Linear(d_model, D) + fixed bipolar codebook. "
                         "Embedding stays learned (untied).")
    ap.add_argument("--vsa-binding-d", type=int, default=cfg.vsa_binding_d,
                    help="VSA hypervector dimension (default 10240)")
    # Hypergradient / neuromodulated input scaling
    ap.add_argument("--hypergrad", action="store_true",
                    help="Feed a Yerkes-Dodson surprise_gain (from "
                         "grad-norm EMA) into each MinGRULayer's learnable α")
    ap.add_argument("--hypergrad-scale-init", type=float,
                    default=cfg.hypergrad_scale_init,
                    help="Initial value for per-layer α (default 0.1)")
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
        data_path=args.data_path, val_split=args.val_split,
        max_steps=args.steps, max_minutes=args.minutes,
        log_every=args.log_every, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        gen_max_tokens=args.gen_tokens,
        gen_temperature=args.gen_temp, gen_top_p=args.gen_top_p,
        enable_memory=args.memory, mem_max=args.mem_max,
        mem_write_threshold=args.mem_write_threshold,
        mem_every_n=args.mem_every_n,
        mem_consolidate_every=args.mem_consolidate_every,
        enable_moe=args.moe, moe_n_experts=args.moe_experts,
        moe_top_k=args.moe_top_k,
        enable_attention=args.attention, attn_n_heads=args.attn_heads,
        attn_window=args.attn_window, attn_every_n=args.attn_every_n,
        enable_hypergrad=args.hypergrad,
        hypergrad_scale_init=args.hypergrad_scale_init,
        prompts_path=args.prompts,
        use_jsonl_dataset=args.use_jsonl_dataset,
        jsonl_path=args.jsonl_path,
        aux_opcode_loss_weight=args.aux_opcode_loss_weight,
        aux_intent_loss_weight=args.aux_intent_loss_weight,
        aux_schema_loss_weight=args.aux_schema_loss_weight,
        aux_rule_loss_weight=args.aux_rule_loss_weight,
        aux_validity_loss_weight=args.aux_validity_loss_weight,
        num_opcode_classes=args.num_opcode_classes,
        num_intent_classes=args.num_intent_classes,
        num_schema_classes=args.num_schema_classes,
        num_rule_classes=args.num_rule_classes,
        vsa_binding_head=args.vsa_binding_head,
        vsa_binding_d=args.vsa_binding_d,
        device=args.device, amp_dtype=args.dtype,
        compile_model=args.compile, seed=args.seed,
    )
    train(cfg, args)


if __name__ == "__main__":
    main()
