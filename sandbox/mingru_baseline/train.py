#!/usr/bin/env python3
"""MinGRU coherence-baseline training on TinyStories — Phase 1.3.

Local-first (RX 6750 XT) via the grilly Vulkan backend. Colab A100 run is
the same code under Phase 1.5 once grilly's Linux wheel is available.

Architecture: ``MinGRUModel`` from ``cubemind.training.vsa_lm`` — the
Phase 1.2 scaffold. No CubeMind extensions; those land in Phase 2.
Defaults match TASKS.md 1.3: d=256, L=6, d_ffn=768, vocab=4000, lr=3e-4,
cosine decay, warmup 1000, grad_clip=1.0.

Loss: ``grilly.nn.autograd.cross_entropy`` — logits flattened
``(B, S-1, V) -> (B*(S-1), V)`` for next-token prediction on shifted
targets.

Dataset: TinyStories V2 GPT-4 split. Downloaded once; BPE tokenizer
(vocab=4000) trained once; tokenized into uint16 bin files. 5M-token
subset matches the CPU-baseline experiments.

Generation: every ``--eval-every`` steps we sample 20 stories from the
fixed prompt set at ``sandbox/mingru_baseline/prompts.txt`` under
temperature / top-p / greedy and log them. Phase 1.4's GPT-4 judge
runs against these artefacts.

Run::

    uv run python sandbox/mingru_baseline/train.py                    # defaults
    uv run python sandbox/mingru_baseline/train.py --steps 1000       # smoke
    uv run python sandbox/mingru_baseline/train.py --minutes 60       # time-boxed
    uv run python sandbox/mingru_baseline/train.py --resume           # checkpoint resume
"""

from __future__ import annotations

import os as _os
# Colab sets ``MPLBACKEND=module://matplotlib_inline.backend_inline`` in
# its shell env. grilly's ``utils/__init__.py`` eagerly imports matplotlib
# via ``visualization.py``; if ``matplotlib_inline`` isn't installed in
# the current venv, matplotlib throws ``ValueError: Key backend`` at
# import time and grilly never loads. Clear the env var so matplotlib
# picks a valid default.
_os.environ.pop("MPLBACKEND", None)

# IMPORTANT: import the HuggingFace ``tokenizers`` package BEFORE touching
# grilly — grilly's own ``grilly/tokenizers/`` subpackage gets put on the
# module search path by ``grilly/__init__.py`` and would otherwise shadow
# the HF tokenizers package we need for BPE training.
import tokenizers as _hf_tokenizers  # noqa: F401 — side-effect import only
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders as hf_decoders
from tokenizers import models as hf_models
from tokenizers import pre_tokenizers as hf_pre
from tokenizers import trainers as hf_trainers

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

import numpy as np

# Windows console defaults to cp1252 and chokes on unicode in training
# logs (em-dash, infinity, BPE byte markers in decoded text). Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from cubemind.training.vsa_lm import MinGRUConfig, MinGRUModel


# ── Paths ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "results"
PROMPTS_PATH = SCRIPT_DIR / "prompts.txt"

# Primary source: local 50k-story TinyStories JSON in the repo root
# (cubemind/data/tinystories_50k.json). Fallback: HuggingFace V2 split.
REPO_ROOT = SCRIPT_DIR.parents[1]
LOCAL_JSON = REPO_ROOT / "data" / "tinystories_50k.json"
TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")


# ── Config ───────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model (matches TASKS.md Phase 1.3 spec)
    vocab_size: int = 4000
    d_model: int = 256
    n_layers: int = 6
    d_ffn: int = 768
    seq_len: int = 256

    # Training
    batch_size: int = 16
    grad_accum: int = 1
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Data
    subset_tokens: int = 5_000_000
    data_path: str = ""             # direct path to a .txt file (bypasses TinyStories)
    val_split: float = 0.05

    # Budget
    max_steps: int = 20_000
    max_minutes: float = 0.0  # 0 = no wall-time cap

    # Schedule
    log_every: int = 10
    eval_every: int = 1_000
    ckpt_every: int = 1_000

    # Generation
    gen_max_tokens: int = 120
    gen_temperature: float = 0.8
    gen_top_p: float = 0.9
    gen_greedy_too: bool = True

    seed: int = 42

    # Neuromodulated input scaling via AutoHypergradientAdamW
    # (x' = x · (1 + α·S), where α is per-layer learnable and S is the
    # optimizer's ``current_surprise_gain``).
    enable_hypergrad: bool = False
    hypergrad_scale_init: float = 0.1
    hypergrad_trauma_threshold: float = 0.5

    # ── Derived ─────────────────────────────────────────────────────────
    def model_cfg(self) -> MinGRUConfig:
        return MinGRUConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            n_layers=self.n_layers,
            seq_len=self.seq_len,
            seed=self.seed,
            enable_hypergrad=self.enable_hypergrad,
            hypergrad_scale_init=self.hypergrad_scale_init,
        )


# ── Data prep ────────────────────────────────────────────────────────────

def prepare_data_from_file(cfg: TrainConfig):
    """Load training data from a raw .txt file at ``cfg.data_path``.

    Trains BPE on the file, tokenizes to uint16 bins, splits train/val.
    Caches everything in DATA_DIR keyed by filename + vocab size.
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

        print(f"  tokenizing {src.name}...")
        all_ids: list = []
        with src.open("r", encoding="utf-8", errors="ignore") as f:
            chunk_idx = 0
            while True:
                chunk = f.read(1_000_000)
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

        split_idx = int(total * (1.0 - cfg.val_split))
        np.array(all_ids[:split_idx], dtype=np.uint16).tofile(str(train_bin))
        np.array(all_ids[split_idx:], dtype=np.uint16).tofile(str(val_bin))

        meta_path.write_text(json.dumps({
            "source": str(src), "tag": tag,
            "vocab": tokenizer.get_vocab_size(),
            "train_tokens": split_idx,
            "val_tokens": total - split_idx,
        }, indent=2))
        print(f"  train: {split_idx:,} | val: {total - split_idx:,}")

    tokenizer = HFTokenizer.from_file(str(tok_path))
    if tokenizer.decoder is None:
        tokenizer.decoder = hf_decoders.ByteLevel()
    vocab = tokenizer.get_vocab_size()

    train_mm = np.memmap(str(train_bin), dtype=np.uint16, mode="r")
    n_total = int(train_mm.shape[0])
    n_train = min(n_total, cfg.subset_tokens)
    train_data = np.asarray(train_mm[:n_train]).astype(np.int64)
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int64)

    print(f"  source: {tag} | train {n_train:,} / {n_total:,} tokens | "
          f"val {len(val_data):,}")
    return tokenizer, vocab, train_data, val_data


def _download(url: str, dst: Path) -> None:
    import urllib.request
    print(f"  downloading {url.rsplit('/', 1)[-1]} -> {dst}")
    urllib.request.urlretrieve(url, str(dst))
    print(f"    {dst.stat().st_size / 1e6:.1f} MB")


def _load_local_stories() -> list[str]:
    """Load the cubemind-vendored TinyStories JSON (list of story strings)."""
    with LOCAL_JSON.open("r", encoding="utf-8") as f:
        stories = json.load(f)
    assert isinstance(stories, list) and stories, (
        f"{LOCAL_JSON} must be a non-empty list of story strings"
    )
    return stories


def _load_hf_txt() -> tuple[Path, Path]:
    train_txt = DATA_DIR / "train.txt"
    val_txt   = DATA_DIR / "valid.txt"
    if not train_txt.exists():
        _download(TRAIN_URL, train_txt)
    if not val_txt.exists():
        _download(VALID_URL, val_txt)
    return train_txt, val_txt


def prepare_data(cfg: TrainConfig):
    """Prepare training data.

    Routes to ``prepare_data_from_file`` if ``--data-path`` is set.
    Otherwise prefers the local 50k-story JSON at ``data/tinystories_50k.json``.
    Falls back to the HuggingFace V2 split if the local file is missing.

    Caches: BPE tokenizer, uint16 token bins, and a small meta.json in
    ``sandbox/mingru_baseline/data/`` keyed by (source, vocab_size).
    """
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
        # Build a text corpus (newline-separated stories) for BPE training.
        corpus_txt = DATA_DIR / f"corpus_{source}.txt"
        if source == "local50k":
            print(f"  using local corpus: {LOCAL_JSON}")
            stories = _load_local_stories()
            corpus_txt.write_text("\n\n".join(stories), encoding="utf-8")
            n_stories = len(stories)
        else:
            print(f"  local corpus not found, falling back to HuggingFace")
            train_txt, val_txt = _load_hf_txt()
            corpus_txt.write_bytes(
                train_txt.read_bytes() + b"\n\n" + val_txt.read_bytes()
            )
            n_stories = -1

        print(f"  training BPE tokenizer (vocab={cfg.vocab_size})")
        tokenizer = HFTokenizer(hf_models.BPE())
        tokenizer.pre_tokenizer = hf_pre.ByteLevel()
        # Pair the byte-level pre-tokenizer with the matching decoder so
        # decode() reassembles byte pairs back into clean UTF-8 instead of
        # leaving ``Ġ``/``Ċ``/``Ĥ``-style marker characters in the output.
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
            print(f"  tokenizing {label} -> {dst.name}")
            tmp_path = Path(tempfile.mktemp(suffix=".bin"))
            total = 0
            # Encode in 500KB chunks to keep memory reasonable.
            with tmp_path.open("wb") as out_f:
                step = 500_000
                for i in range(0, len(text), step):
                    chunk = text[i : i + step]
                    ids = tokenizer.encode(chunk).ids
                    np.asarray(ids, dtype=np.uint16).tofile(out_f)
                    total += len(ids)
                    if (i // step) % 20 == 0:
                        print(f"    {total:,} tokens", end="\r")
                    gc.collect()
            shutil.copy2(tmp_path, dst)
            tmp_path.unlink()
            print(f"    {label}: {total:,} tokens            ")
            return total

        # Train/val split
        if source == "local50k":
            stories = _load_local_stories()
            # 95/5 story-level split — avoids leakage across stories
            split_n = max(1, int(len(stories) * 0.95))
            train_text = "\n\n".join(stories[:split_n])
            val_text   = "\n\n".join(stories[split_n:])
            meta_n_stories = len(stories)
        else:
            train_txt, val_txt = _load_hf_txt()
            train_text = train_txt.read_text(encoding="utf-8", errors="ignore")
            val_text   = val_txt.read_text(encoding="utf-8", errors="ignore")
            meta_n_stories = -1

        train_n = _tokenize_text(train_text, train_bin, "train")
        val_n   = _tokenize_text(val_text,   val_bin,   "val")

        meta_path.write_text(json.dumps({
            "source": source,
            "vocab": tokenizer.get_vocab_size(),
            "train_tokens": train_n, "val_tokens": val_n,
            "n_stories": meta_n_stories,
        }, indent=2))
        # Keep corpus on disk in case we rebuild with a different vocab later.

    tokenizer = HFTokenizer.from_file(str(tok_path))
    # Older cached tokenizer.json files may not include the decoder — set
    # it defensively so decode() works on byte-level pre-tokenized text.
    if tokenizer.decoder is None:
        tokenizer.decoder = hf_decoders.ByteLevel()
    meta = json.loads(meta_path.read_text())
    vocab = tokenizer.get_vocab_size()

    train_mm = np.memmap(str(train_bin), dtype=np.uint16, mode="r")
    n_total = int(train_mm.shape[0])
    n_train = min(n_total, cfg.subset_tokens)
    train_data = np.asarray(train_mm[:n_train])
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int64)

    print(f"  source: {meta['source']}  | "
          f"train subset: {n_train:,} / {n_total:,} tokens "
          f"(~{n_train / 1e6:.1f}M) | val: {len(val_data):,}")
    return tokenizer, vocab, train_data, val_data


# ── Training utilities ───────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def _param_grad(p):
    """Return the grad ndarray on a grilly Parameter / ParameterWrapper,
    or None if not yet populated."""
    g = getattr(p, "grad", None)
    if g is None:
        return None
    if hasattr(g, "data") and not isinstance(g, np.ndarray):
        g = g.data
    return np.asarray(g, dtype=np.float32)


def clip_grad_norm(params, max_norm: float) -> float:
    total_sq = 0.0
    for p in params:
        g = _param_grad(p)
        if g is None:
            continue
        total_sq += float((g * g).sum())
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if getattr(p, "grad", None) is None:
                continue
            g = p.grad
            if hasattr(g, "data") and not isinstance(g, np.ndarray):
                g.data *= scale
            else:
                p.grad = g * scale
    return total_norm


def zero_grads(params) -> None:
    for p in params:
        if hasattr(p, "zero_grad"):
            try:
                p.zero_grad()
                continue
            except Exception:
                pass
        p.grad = None


def sample_batch(data: np.ndarray, batch_size: int, seq_len: int, rng) -> tuple:
    """Uniform random contiguous-window sampling. Returns (x, y) int64,
    each of shape (batch_size, seq_len-1) — shifted next-token targets."""
    n_windows = (len(data) - 1) // seq_len
    starts = rng.integers(0, n_windows, size=batch_size) * seq_len
    x = np.empty((batch_size, seq_len - 1), dtype=np.int64)
    y = np.empty((batch_size, seq_len - 1), dtype=np.int64)
    for i, s in enumerate(starts):
        chunk = data[s : s + seq_len]
        x[i] = chunk[:-1].astype(np.int64)
        y[i] = chunk[1:].astype(np.int64)
    return x, y


# ── Loss ─────────────────────────────────────────────────────────────────

def compute_loss(logits, targets: np.ndarray):
    """logits Variable (B, S, V), targets ndarray (B, S) int64."""
    from grilly.nn.autograd import cross_entropy
    B, S, V = logits.data.shape
    # Flatten for 2D cross_entropy (grilly's autograd CE takes 2D hard-targets)
    flat_logits = logits.reshape(B * S, V)
    flat_targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    return cross_entropy(flat_logits, flat_targets)


# ── Generation ───────────────────────────────────────────────────────────

def _top_p_logits(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus filter: keep smallest set whose cumulative prob >= top_p."""
    if top_p >= 1.0:
        return logits
    order = np.argsort(-logits)
    sorted_logits = logits[order]
    probs = np.exp(sorted_logits - sorted_logits.max())
    probs /= probs.sum()
    cum = np.cumsum(probs)
    cut = int(np.searchsorted(cum, top_p)) + 1
    cut = max(cut, 1)
    keep = set(order[:cut].tolist())
    mask = np.full_like(logits, -np.inf)
    for i in keep:
        mask[i] = logits[i]
    return mask


@dataclass
class GenParams:
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 120


def generate(model, tokenizer, prompt: str, params: GenParams,
             rng: np.random.Generator, greedy: bool = False) -> str:
    from grilly.nn.autograd import no_grad

    ids = tokenizer.encode(prompt).ids
    tokens = list(ids)
    seq_len = model.cfg.seq_len

    with no_grad():
        for _ in range(params.max_new_tokens):
            context = tokens[-seq_len:]
            x = np.asarray([context], dtype=np.int64)
            logits_var = model(x)
            # Variable.data can be a VulkanTensor wrapper that rejects
            # sub-view indexing — materialize to ndarray first.
            logits_full = np.asarray(logits_var.data, dtype=np.float32)
            logits = logits_full[0, -1]

            if greedy:
                next_id = int(np.argmax(logits))
            else:
                logits = logits / max(params.temperature, 1e-5)
                logits = _top_p_logits(logits, params.top_p)
                shifted = logits - np.max(logits)
                probs = np.exp(shifted)
                probs /= probs.sum()
                next_id = int(rng.choice(len(probs), p=probs))

            tokens.append(next_id)

    # Byte-level decoder (set in prepare_data) restores spaces/newlines
    # automatically. The .replace() calls guard against a stale cached
    # tokenizer that was saved without the decoder.
    text = tokenizer.decode(tokens)
    return text.replace("\u0120", " ").replace("\u010a", "\n")


def load_prompts() -> list[str]:
    return [
        line.strip() for line in PROMPTS_PATH.read_text().splitlines()
        if line.strip()
    ]


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(model, val_data: np.ndarray, cfg: TrainConfig,
             max_batches: int = 50) -> float:
    from grilly.nn.autograd import no_grad

    rng = np.random.default_rng(0)
    losses = []
    with no_grad():
        for _ in range(max_batches):
            x, y = sample_batch(val_data, cfg.batch_size, cfg.seq_len, rng)
            logits = model(x)
            loss = compute_loss(logits, y)
            if np.isfinite(loss.data):
                losses.append(float(loss.data))
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


# ── Checkpoint ───────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model, optimizer, step: int,
                    best_val: float, tokens_seen: int,
                    elapsed: float) -> None:
    from grilly.nn._helpers import _get_param_array

    # Flatten params to a name-indexed dict. We rebuild on load by
    # walking the same iteration order.
    params_list = []
    for p in model.parameters():
        params_list.append(_get_param_array(p).copy())

    state = {
        "step": step,
        "best_val": best_val,
        "tokens_seen": tokens_seen,
        "elapsed": elapsed,
        "params": params_list,
        "optimizer_state": {
            k: {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in st.items()}
            for k, st in optimizer.state.items()
        },
        "step_count": getattr(optimizer, "_step_count", 0),
    }
    # np.savez_compressed auto-appends ``.npz``. Use a base path without
    # an extension so the written file matches what we then rename from.
    tmp_base = path.with_name(path.stem + "_tmp")  # e.g. results/best_tmp
    tmp_written = tmp_base.with_suffix(".npz")      #       results/best_tmp.npz
    np.savez_compressed(
        str(tmp_base),
        step=np.array(step),
        best_val=np.array(best_val),
        tokens_seen=np.array(tokens_seen),
        elapsed=np.array(elapsed),
        step_count=np.array(state["step_count"]),
        **{f"p{i}": arr for i, arr in enumerate(params_list)},
    )
    os.replace(tmp_written, path)


def load_checkpoint(path: Path, model, optimizer) -> dict:
    data = np.load(path, allow_pickle=False)
    from grilly.nn._helpers import _get_param_array
    for i, p in enumerate(model.parameters()):
        arr = _get_param_array(p)
        src = data[f"p{i}"]
        assert arr.shape == src.shape, (
            f"param {i} shape mismatch: ckpt {src.shape} vs model {arr.shape}"
        )
        arr[:] = src
    return {
        "step": int(data["step"].item()),
        "best_val": float(data["best_val"].item()),
        "tokens_seen": int(data["tokens_seen"].item()),
        "elapsed": float(data["elapsed"].item()),
    }


# ── Training loop ────────────────────────────────────────────────────────

def train(cfg: TrainConfig, args) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"  MinGRU Phase 1.3 -- TinyStories coherence baseline")
    print(f"{'=' * 72}")
    print(f"  model : d={cfg.d_model} L={cfg.n_layers} d_ffn={cfg.d_ffn} "
          f"vocab={cfg.vocab_size} seq_len={cfg.seq_len}")
    print(f"  optim : lr={cfg.max_lr} -> {cfg.min_lr} cosine | "
          f"warmup={cfg.warmup_steps} | wd={cfg.weight_decay} | "
          f"clip={cfg.grad_clip}")
    max_min_str = f"{cfg.max_minutes}" if cfg.max_minutes > 0 else "inf"
    print(f"  sched : max_steps={cfg.max_steps} | max_min={max_min_str} | "
          f"log={cfg.log_every} eval={cfg.eval_every} ckpt={cfg.ckpt_every}")

    print("\n--- data ---")
    tokenizer, vocab, train_data, val_data = prepare_data(cfg)
    if vocab != cfg.vocab_size:
        print(f"  note: actual vocab {vocab} vs configured {cfg.vocab_size} "
              "(tokenizer picked fewer merges than requested)")
        cfg.vocab_size = vocab

    print("\n--- model ---")
    model_cfg = cfg.model_cfg()
    model_cfg.vocab_size = vocab  # reflect tokenizer-actual vocab
    model = MinGRUModel(model_cfg)
    model.gpu_mode(True)  # keep tensors in VRAM between ops (no PCIe round-trip)
    n_params = model.num_parameters()
    print(f"  params: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  gpu_mode: ON (DEVICE_LOCAL resident tensors)")

    print("\n--- optimizer ---")
    params = list(model.parameters())
    if cfg.enable_hypergrad:
        from grilly.optim import AutoHypergradientAdamW
        optimizer = AutoHypergradientAdamW(
            params,
            lr=cfg.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=cfg.weight_decay,
            track_surprise=True,
            trauma_threshold=cfg.hypergrad_trauma_threshold,
            warmup_steps=max(10, cfg.warmup_steps // 50),
        )
        print(f"  AutoHypergradientAdamW: track_surprise=True, "
              f"trauma={cfg.hypergrad_trauma_threshold}")
    else:
        from grilly.optim import AdamW
        optimizer = AdamW(
            params,
            lr=cfg.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=cfg.weight_decay,
        )

    # Resume
    ckpt_path = OUT_DIR / "checkpoint.npz"
    step, best_val, tokens_seen, elapsed_prev = 0, float("inf"), 0, 0.0
    if args.resume and ckpt_path.exists():
        meta = load_checkpoint(ckpt_path, model, optimizer)
        step = meta["step"]; best_val = meta["best_val"]
        tokens_seen = meta["tokens_seen"]; elapsed_prev = meta["elapsed"]
        print(f"\n  *** resumed from step {step:,} ({elapsed_prev / 60:.1f}m done) ***")

    rng = np.random.default_rng(cfg.seed + step)
    history: list = []

    header = (f"  {'step':>6} {'lr':>9} {'CE':>8} {'PPL':>9} "
              f"{'gn':>7} {'tok/s':>8} {'elapsed':>8}")
    divider = "  " + "-" * 72
    print()
    print(header)
    print(divider)
    header_every = 20  # reprint header every N log lines

    log_ce, log_n = 0.0, 0
    log_count = 0
    t_start = time.time()
    max_time = cfg.max_minutes * 60.0 if cfg.max_minutes > 0 else math.inf

    while step < cfg.max_steps:
        if elapsed_prev + (time.time() - t_start) >= max_time:
            print("  [time budget reached]")
            break

        zero_grads(params)

        # Read the neuromodulatory gain set by the previous optimizer.step().
        # AutoHypergradientAdamW returns 0.0 during warmup or when track_surprise
        # is off; plain AdamW lacks the attribute entirely, so we default to 0.0.
        surprise_gain = float(getattr(optimizer, "current_surprise_gain", 0.0))

        loss_val_accum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = sample_batch(train_data, cfg.batch_size, cfg.seq_len, rng)
            logits = model(x, surprise_gain=surprise_gain)
            loss = compute_loss(logits, y)
            # Scale for grad accumulation
            loss_scaled_data = float(loss.data)
            if not math.isfinite(loss_scaled_data):
                print(f"  WARN non-finite loss at step {step}, skipping batch")
                continue
            # grilly autograd's cross_entropy already returns scalar mean;
            # for grad-accum we want gradients averaged, which means
            # scaling the backward's grad_output by 1/accum.
            loss.backward(
                np.asarray(1.0 / cfg.grad_accum, dtype=np.float32)
            )
            loss_val_accum += loss_scaled_data
            tokens_seen += x.size

        # LR schedule
        lr = get_lr(step, cfg)
        optimizer.defaults["lr"] = lr
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Grad clipping
        gn = clip_grad_norm(params, cfg.grad_clip)

        optimizer.step()
        step += 1

        log_ce += loss_val_accum / cfg.grad_accum
        log_n += 1

        if step % cfg.log_every == 0:
            avg_ce = log_ce / max(log_n, 1)
            now = elapsed_prev + (time.time() - t_start)
            tps = tokens_seen / max(now, 1)
            print(f"  {step:>6d} {lr:>9.2e} {avg_ce:>8.4f} "
                  f"{math.exp(min(avg_ce, 20)):>9.2f} {gn:>7.3f} "
                  f"{tps:>8,.0f} {now / 60:>7.1f}m")
            history.append({
                "step": step, "lr": lr, "ce": avg_ce,
                "ppl": math.exp(min(avg_ce, 20)),
                "grad_norm": gn, "tokens_per_sec": tps,
                "elapsed": now,
            })
            log_ce, log_n = 0.0, 0
            log_count += 1
            if log_count % header_every == 0:
                print(divider)
                print(header)
                print(divider)

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val_ce = evaluate(model, val_data, cfg, max_batches=25)
            val_ppl = math.exp(min(val_ce, 20))
            tag = ""
            if val_ce < best_val:
                best_val = val_ce
                best_path = OUT_DIR / "best.npz"
                save_checkpoint(best_path, model, optimizer, step,
                                best_val, tokens_seen,
                                elapsed_prev + (time.time() - t_start))
                tag = " *"
            print(f"\n  === step {step:,} ===  val CE {val_ce:.4f}  "
                  f"PPL {val_ppl:.2f}{tag}")

            # Generate from fixed prompts
            gen_path = OUT_DIR / f"gen_step_{step:06d}.md"
            write_generations(model, tokenizer, gen_path, cfg, step,
                              val_ce, val_ppl)
            print(f"  [gen] wrote {gen_path.name}")

            # Reprint header after eval block so the next training lines
            # are clearly labeled again.
            print(divider)
            print(header)
            print(divider)

        if step % cfg.ckpt_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, step, best_val,
                            tokens_seen,
                            elapsed_prev + (time.time() - t_start))

        if step % 200 == 0:
            gc.collect()

    elapsed_total = elapsed_prev + (time.time() - t_start)
    final_val = evaluate(model, val_data, cfg, max_batches=50)
    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))

    # Final generation with both greedy and sampling
    final_gen_path = OUT_DIR / "generations_final.md"
    write_generations(model, tokenizer, final_gen_path, cfg, step,
                      final_val, final_ppl, include_greedy=True)

    # Results summary
    summary = {
        "model": "MinGRU-phase1.3",
        "params": n_params,
        "config": asdict(cfg),
        "final_val_ce": final_val,
        "final_val_ppl": final_ppl,
        "best_val_ce": best_val,
        "best_val_ppl": best_ppl,
        "steps": step,
        "tokens_seen": tokens_seen,
        "elapsed_min": elapsed_total / 60,
        "tokens_per_sec": tokens_seen / max(elapsed_total, 1),
        "history": history[-200:],  # tail of the log
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  {'=' * 72}")
    print(f"  DONE. steps={step:,}  tokens={tokens_seen / 1e6:.1f}M  "
          f"time={elapsed_total / 60:.1f}m")
    print(f"  final PPL {final_ppl:.2f}  (best {best_ppl:.2f})")
    print(f"  saved -> {OUT_DIR}")
    return summary


def write_generations(model, tokenizer, path: Path, cfg: TrainConfig,
                      step: int, val_ce: float, val_ppl: float,
                      include_greedy: bool = False) -> None:
    prompts = load_prompts()
    rng = np.random.default_rng(cfg.seed + step * 17)
    params = GenParams(
        temperature=cfg.gen_temperature, top_p=cfg.gen_top_p,
        max_new_tokens=cfg.gen_max_tokens,
    )

    lines = [
        f"# Generations — step {step:,}",
        "",
        f"- val CE: **{val_ce:.4f}**  · val PPL: **{val_ppl:.2f}**",
        f"- sampling: T={cfg.gen_temperature}, top_p={cfg.gen_top_p}, "
        f"max={cfg.gen_max_tokens}",
        "",
    ]
    for i, prompt in enumerate(prompts, 1):
        text = generate(model, tokenizer, prompt, params, rng, greedy=False)
        lines.append(f"## {i:02d}. {prompt}")
        lines.append("")
        lines.append("**sampled:**  " + text.replace("\n", " "))
        if include_greedy and cfg.gen_greedy_too:
            text_g = generate(model, tokenizer, prompt, params, rng, greedy=True)
            lines.append("")
            lines.append("**greedy:**  " + text_g.replace("\n", " "))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser(description="MinGRU Phase 1.3 training")
    # Core schedule
    ap.add_argument("--steps", type=int, default=cfg.max_steps,
                    help="Max training steps (default 20000)")
    ap.add_argument("--minutes", type=float, default=cfg.max_minutes,
                    help="Wall-time budget in minutes (default: no cap)")
    # Model
    ap.add_argument("--d-model", type=int, default=cfg.d_model)
    ap.add_argument("--n-layers", type=int, default=cfg.n_layers)
    ap.add_argument("--d-ffn", type=int, default=cfg.d_ffn)
    ap.add_argument("--vocab", type=int, default=cfg.vocab_size)
    ap.add_argument("--seq-len", type=int, default=cfg.seq_len)
    # Optim
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--grad-accum", type=int, default=cfg.grad_accum)
    ap.add_argument("--lr", type=float, default=cfg.max_lr)
    ap.add_argument("--min-lr", type=float, default=cfg.min_lr)
    ap.add_argument("--warmup", type=int, default=cfg.warmup_steps)
    ap.add_argument("--wd", type=float, default=cfg.weight_decay)
    ap.add_argument("--clip", type=float, default=cfg.grad_clip)
    # Data
    ap.add_argument("--subset-tokens", type=int, default=cfg.subset_tokens)
    ap.add_argument("--data-path", type=str, default="",
                    help="Path to a raw .txt file — bypasses TinyStories, "
                         "trains BPE on the file, tokenizes, caches bins.")
    ap.add_argument("--val-split", type=float, default=cfg.val_split)
    # Schedule
    ap.add_argument("--log-every", type=int, default=cfg.log_every)
    ap.add_argument("--eval-every", type=int, default=cfg.eval_every)
    ap.add_argument("--ckpt-every", type=int, default=cfg.ckpt_every)
    # Gen
    ap.add_argument("--gen-tokens", type=int, default=cfg.gen_max_tokens)
    ap.add_argument("--gen-temp", type=float, default=cfg.gen_temperature)
    ap.add_argument("--gen-top-p", type=float, default=cfg.gen_top_p)
    # Hypergradient (AutoHypergradientAdamW + per-layer α modulation)
    ap.add_argument("--hypergrad", action="store_true",
                    help="Use AutoHypergradientAdamW and feed "
                         "current_surprise_gain into each MinGRULayer's "
                         "learnable α (Yerkes-Dodson / OSGM).")
    ap.add_argument("--hypergrad-scale-init", type=float,
                    default=cfg.hypergrad_scale_init,
                    help="Initial value for the per-layer α (default 0.1).")
    ap.add_argument("--hypergrad-trauma", type=float,
                    default=cfg.hypergrad_trauma_threshold,
                    help="S_bar level where gain peaks before suppression.")
    # Misc
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from results/checkpoint.npz if present")
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
        gen_max_tokens=args.gen_tokens, gen_temperature=args.gen_temp,
        gen_top_p=args.gen_top_p,
        seed=args.seed,
        enable_hypergrad=args.hypergrad,
        hypergrad_scale_init=args.hypergrad_scale_init,
        hypergrad_trauma_threshold=args.hypergrad_trauma,
    )
    train(cfg, args)


if __name__ == "__main__":
    main()
