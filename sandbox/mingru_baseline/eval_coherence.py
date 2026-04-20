"""GPT-4 coherence judge for the MinGRU baseline — Phase 1.4.

Loads a PyTorch MinGRU checkpoint from Phase 1.3
(``results_torch/best.pt`` by default), samples a story from each prompt
in ``prompts.txt`` at matched decoding parameters, and asks an LLM judge
(GPT-4o or a Claude model) to score each story on three axes from the
Eldan & Li (2023) TinyStories rubric: **grammar**, **creativity**,
**consistency** (each 1-5).

Run::

    python -u sandbox/mingru_baseline/eval_coherence.py \\
        --checkpoint sandbox/mingru_baseline/results_torch/best.pt \\
        --judge gpt-4o \\
        --out sandbox/mingru_baseline/results_torch/eval_run1.json

The script re-uses ``train_torch``'s model class, tokenizer cache, and
``generate`` function — the architecture / sampling / decoding match
the training run exactly so scores reflect the model, not drift in the
eval harness.

Env: set ``OPENAI_API_KEY`` for GPT judges, ``ANTHROPIC_API_KEY`` for
Claude judges. The judge import is deferred — running without the key
and without a network connection still parses/loads the checkpoint so
you can smoke-test the plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# UTF-8 stdout for Windows consoles
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

# HuggingFace tokenizers (not grilly's subpackage)
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders as hf_decoders

# Allow ``python sandbox/mingru_baseline/eval_coherence.py`` to resolve
# the sibling ``train_torch`` module regardless of whether sandbox/ is on
# the Python package path. Put the script dir on sys.path BEFORE the
# import so script-mode and module-mode invocation both work.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse training-time model, config, data paths, and generation helper.
from train_torch import (  # noqa: E402 — after sys.path tweak
    DATA_DIR,
    LOCAL_JSON,
    GenParams,
    MinGRUModel,
    TrainConfig,
    generate,
    load_prompts,
)


RUBRIC = """\
You are grading a short story produced by a small language model on a
TinyStories-style prompt. Score each axis on a 1-5 integer scale.

- grammar: sentence structure, agreement, verb tense, punctuation.
- creativity: variety of vocabulary and plot elements, within the simple
  tiny-story register.
- consistency: story stays on the prompt; characters and objects
  introduced at the start do not vanish or swap identities.

Respond with a single JSON object on one line and nothing else:
{"grammar": <1-5>, "creativity": <1-5>, "consistency": <1-5>,
 "notes": "<one short sentence>"}"""


# ── Score records ────────────────────────────────────────────────────────

@dataclass
class Score:
    prompt: str
    completion: str
    grammar: int
    creativity: int
    consistency: int
    notes: str = ""

    @property
    def total(self) -> float:
        return (self.grammar + self.creativity + self.consistency) / 3.0


@dataclass
class EvalRun:
    model_tag: str
    judge: str
    checkpoint: str
    scores: list[Score] = field(default_factory=list)

    def summary(self) -> dict:
        n = max(len(self.scores), 1)
        mean = lambda k: sum(getattr(s, k) for s in self.scores) / n
        return {
            "model": self.model_tag,
            "judge": self.judge,
            "checkpoint": self.checkpoint,
            "n_samples": len(self.scores),
            "grammar_mean": mean("grammar"),
            "creativity_mean": mean("creativity"),
            "consistency_mean": mean("consistency"),
            "total_mean": sum(s.total for s in self.scores) / n,
            # Per-axis pass criteria from HYPOTHESES.md
            "h1_grammar_pass": mean("grammar") >= 4.0,
            "h2_consistency_pass": mean("consistency") >= 3.0,
        }


# ── Checkpoint + tokenizer loading ───────────────────────────────────────

def _resolve_tokenizer_path(cfg: TrainConfig) -> Path:
    """Mirror ``train_torch.prepare_data``'s cache-key convention so the
    same tokenizer the model trained against is picked up."""
    source = "local50k" if LOCAL_JSON.exists() else "hf"
    return DATA_DIR / f"tokenizer_{source}_v{cfg.vocab_size}.json"


def load_torch_checkpoint(path: Path, device: torch.device) -> tuple[MinGRUModel, TrainConfig, str | None]:
    """Rebuild ``MinGRUModel`` from a saved Phase 1.3 checkpoint.

    Returns ``(model, cfg, tokenizer_json)`` — the third item is the
    inline tokenizer JSON saved alongside the checkpoint, or ``None``
    for older checkpoints that pre-date the embed.
    """
    path = Path(path).resolve()
    data = torch.load(str(path), map_location=device, weights_only=False)

    cfg_dict = data.get("config", {})
    allowed = {f for f in TrainConfig.__dataclass_fields__.keys()}
    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in allowed}
    cfg = TrainConfig(**cfg_kwargs) if cfg_kwargs else TrainConfig()

    model = MinGRUModel(cfg).to(device)
    state = data.get("model_state", data.get("state_dict", data))
    model.load_state_dict(state)
    model.eval()

    tokenizer_json = data.get("tokenizer_json")

    print(f"  loaded checkpoint: {path.name}")
    if "step" in data:
        print(f"    step: {int(data['step']):,}")
    if "best_val" in data:
        import math
        bv = float(data["best_val"])
        print(f"    best val CE: {bv:.4f}  PPL: {math.exp(min(bv, 20)):.2f}")
    print(f"    model: d={cfg.d_model} L={cfg.n_layers} d_ffn={cfg.d_ffn} "
          f"vocab={cfg.vocab_size} seq_len={cfg.seq_len}")
    print(f"    embedded tokenizer: "
          f"{'yes' if tokenizer_json else 'no (will fall back to disk cache)'}")
    return model, cfg, tokenizer_json


def load_tokenizer(cfg: TrainConfig,
                   embedded_json: str | None = None,
                   override_path: Path | None = None) -> HFTokenizer:
    """Load the tokenizer in this priority order:

    1. ``override_path`` (CLI ``--tokenizer``).
    2. Inline ``embedded_json`` from the checkpoint (preferred — guaranteed
       to match the trained model's BPE merges).
    3. Disk cache at ``data/tokenizer_<source>_v<vocab>.json``.

    Raises if all three fail. Mismatched tokenizer = the embedding
    lookup maps every token id to garbage even with a valid checkpoint.
    """
    if override_path is not None:
        path = Path(override_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"--tokenizer path does not exist: {path}")
        print(f"  tokenizer: {path} (CLI override)")
        tokenizer = HFTokenizer.from_file(str(path))
    elif embedded_json:
        print(f"  tokenizer: <embedded in checkpoint> "
              f"({len(embedded_json):,} chars)")
        tokenizer = HFTokenizer.from_str(embedded_json)
    else:
        tok_path = _resolve_tokenizer_path(cfg)
        print(f"  tokenizer: {tok_path} (disk cache fallback)")
        if not tok_path.exists():
            raise FileNotFoundError(
                f"No tokenizer source available.\n"
                f"  - Checkpoint has no embedded tokenizer (older format).\n"
                f"  - --tokenizer override not provided.\n"
                f"  - Disk cache missing at {tok_path}.\n"
                "Either pass --tokenizer path/to/tokenizer.json, or copy\n"
                "the matching tokenizer file from the training environment.\n"
                "Tokenizer mismatch silently produces garbage generations."
            )
        tokenizer = HFTokenizer.from_file(str(tok_path))

    if tokenizer.decoder is None:
        tokenizer.decoder = hf_decoders.ByteLevel()

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size mismatch: "
            f"checkpoint expects {cfg.vocab_size}, tokenizer has {actual_vocab}.\n"
            "This will produce garbage generations — refuse to continue."
        )
    return tokenizer


# ── LLM judge ────────────────────────────────────────────────────────────

def _grade_gpt(prompt: str, completion: str, judge: str) -> dict:
    from openai import OpenAI  # requires OPENAI_API_KEY
    client = OpenAI()
    user = (
        f"Prompt: {prompt}\n\n"
        f"Completion: {completion}\n\n"
        "Grade per the rubric above."
    )
    resp = client.chat.completions.create(
        model=judge,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    return _parse_json(resp.choices[0].message.content.strip())


def _grade_claude(prompt: str, completion: str, judge: str) -> dict:
    import anthropic  # requires ANTHROPIC_API_KEY
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=judge,
        max_tokens=120,
        system=RUBRIC,
        messages=[{
            "role": "user",
            "content": (
                f"Prompt: {prompt}\n\n"
                f"Completion: {completion}\n\n"
                "Grade per the rubric above."
            ),
        }],
    )
    return _parse_json(resp.content[0].text.strip())


def _parse_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Judge did not return JSON: {raw!r}")
    return json.loads(raw[start : end + 1])


def grade(prompt: str, completion: str, judge: str) -> dict:
    if judge.startswith("gpt"):
        return _grade_gpt(prompt, completion, judge)
    if judge.startswith("claude"):
        return _grade_claude(prompt, completion, judge)
    raise ValueError(f"Unknown judge model: {judge}")


# ── Orchestration ────────────────────────────────────────────────────────

def run(checkpoint: Path, prompts_path: Path, judge: str,
        out_path: Path, model_tag: str, temperature: float,
        top_p: float, max_new_tokens: int, greedy: bool,
        device_str: str, skip_grade: bool,
        tokenizer_path: Path | None = None) -> EvalRun:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))

    model, cfg, embedded_tok = load_torch_checkpoint(checkpoint, device)
    tokenizer = load_tokenizer(
        cfg, embedded_json=embedded_tok, override_path=tokenizer_path,
    )
    prompts = [p.strip() for p in Path(prompts_path).read_text(
        encoding="utf-8").splitlines() if p.strip()]
    print(f"  prompts: {len(prompts)} from {prompts_path}")
    print(f"  judge  : {judge}  (skip_grade={skip_grade})")
    print()

    gen = GenParams(
        temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
    )
    run = EvalRun(
        model_tag=model_tag, judge=judge, checkpoint=str(checkpoint),
    )
    for i, prompt in enumerate(prompts, 1):
        completion = generate(model, tokenizer, prompt, gen, device,
                              greedy=greedy)
        print(f"  [{i:2d}/{len(prompts)}] {prompt}")
        print(f"        -> {completion[:120].replace(chr(10), ' ')}...")

        if skip_grade:
            # Dry-run: record completion with zero scores so the artefact
            # contains the generations even without an API key present.
            run.scores.append(Score(
                prompt=prompt, completion=completion,
                grammar=0, creativity=0, consistency=0,
                notes="skipped (--skip-grade)",
            ))
            continue

        try:
            graded = grade(prompt, completion, judge)
        except Exception as e:
            print(f"        ! judge error: {e}")
            graded = {"grammar": 0, "creativity": 0, "consistency": 0,
                      "notes": f"judge error: {e}"}
        run.scores.append(Score(
            prompt=prompt, completion=completion,
            grammar=int(graded.get("grammar", 0)),
            creativity=int(graded.get("creativity", 0)),
            consistency=int(graded.get("consistency", 0)),
            notes=str(graded.get("notes", "")),
        ))
        print(f"        grammar={graded.get('grammar')} "
              f"creativity={graded.get('creativity')} "
              f"consistency={graded.get('consistency')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": run.summary(),
        "scores": [asdict(s) for s in run.scores],
    }, indent=2), encoding="utf-8")
    print()
    print(json.dumps(run.summary(), indent=2))
    print(f"\n  saved -> {out_path}")
    return run


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    default_ckpt = SCRIPT_DIR / "results_torch" / "best.pt"
    ap = argparse.ArgumentParser(description="MinGRU Phase 1.4 coherence judge")
    ap.add_argument("--checkpoint", type=Path, default=default_ckpt,
                    help="PyTorch MinGRU checkpoint (default: "
                         "sandbox/mingru_baseline/results_torch/best.pt)")
    ap.add_argument("--prompts", type=Path,
                    default=SCRIPT_DIR / "prompts.txt")
    ap.add_argument("--judge", default="gpt-4o",
                    help="gpt-4o, gpt-4o-mini, claude-opus-4-6, claude-sonnet-4-6, ...")
    ap.add_argument("--out", type=Path,
                    default=SCRIPT_DIR / "results_torch" / "eval_run1.json")
    ap.add_argument("--model-tag", default="mingru_baseline_phase1_torch")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--greedy", action="store_true",
                    help="Deterministic argmax decoding for cross-run "
                         "reproducibility instead of nucleus sampling.")
    ap.add_argument("--device", default="auto",
                    help="auto / cuda / cuda:0 / cpu")
    ap.add_argument("--skip-grade", action="store_true",
                    help="Generate + save completions without hitting "
                         "the LLM judge (smoke-test the pipeline).")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="Override the tokenizer path. Use this when the "
                         "checkpoint pre-dates inline-tokenizer embedding "
                         "and the disk cache doesn't match the trained "
                         "BPE (e.g. checkpoint trained on Colab HF corpus, "
                         "evaluated locally where local50k cache exists).")
    args = ap.parse_args()

    run(
        checkpoint=args.checkpoint,
        prompts_path=args.prompts,
        judge=args.judge,
        out_path=args.out,
        model_tag=args.model_tag,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        device_str=args.device,
        skip_grade=args.skip_grade,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
