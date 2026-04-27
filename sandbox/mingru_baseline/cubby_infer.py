"""Stage-1.6 Cubby inference — bypasses LiveAdapter.

Reads TrainConfig embedded in the checkpoint and rebuilds the EXACT
architecture that produced best.pt, so state_dict loads with 0 missing /
0 unexpected instead of the 478/428 mismatch LiveAdapter produces against
its Stage-2 multitask assumption.

Usage:
    python sandbox/mingru_baseline/cubby_infer.py \
        --checkpoint sandbox/mingru_baseline/cuby-213M/best.pt \
        --tokenizer  /path/to/grillcheese_spm32k_v2.model

    # One-shot prompts from prompts_cubby.txt:
    python sandbox/mingru_baseline/cubby_infer.py \
        --checkpoint ... --tokenizer ... \
        --prompts sandbox/mingru_baseline/prompts_cubby.txt \
        --max-new-tokens 80
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_torch import (  # noqa: E402
    GenParams,
    MinGRUModel,
    SpmAdapter,
    TrainConfig,
    generate,
)

_STOP_MARKERS = ("<|user|>", "<|tool|>", "<|system|>", "<|assistant|>")


def _truncate_at_stop(text: str, prompt: str) -> str:
    """Return just the assistant turn — drop anything past the next role token."""
    tail = text[len(prompt):] if text.startswith(prompt) else text
    cut = len(tail)
    for m in _STOP_MARKERS:
        i = tail.find(m)
        if i != -1 and i < cut:
            cut = i
    return (prompt + tail[:cut]).rstrip()


def _cfg_from_dict(d: dict) -> TrainConfig:
    known = {f.name for f in fields(TrainConfig)}
    kept = {k: v for k, v in d.items() if k in known}
    return TrainConfig(**kept)


def load_cubby(checkpoint: Path, device: torch.device) -> tuple[MinGRUModel, TrainConfig, dict]:
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(data["config"])
    model = MinGRUModel(cfg).to(device)
    missing, unexpected = model.load_state_dict(data["model_state"], strict=False)
    print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"    first missing: {list(missing)[:5]}")
    if unexpected:
        print(f"    first unexpected: {list(unexpected)[:5]}")
    model.eval()
    return model, cfg, data


def run_prompts(model: MinGRUModel, tokenizer, prompts: list[str],
                device: torch.device, gen: GenParams) -> None:
    for i, p in enumerate(prompts, 1):
        t0 = time.time()
        out = generate(model, tokenizer, p, gen, device, greedy=False)
        out = _truncate_at_stop(out, p)
        dt = time.time() - t0
        print(f"\n--- [{i}/{len(prompts)}]  ({dt:.1f}s) ---")
        print(f"prompt: {p}")
        print(f"output: {out}")


def repl(model: MinGRUModel, tokenizer, device: torch.device, gen: GenParams) -> None:
    print("\nCubby REPL. Wrap user text with <|user|>…<|assistant|> for best results.")
    print("Commands: /quit  /temp <f>  /top_p <f>  /max <n>  /greedy  /sample\n")
    greedy = False
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line.startswith("/temp "):
            gen.temperature = float(line.split()[1]); print(f"  temperature={gen.temperature}"); continue
        if line.startswith("/top_p "):
            gen.top_p = float(line.split()[1]); print(f"  top_p={gen.top_p}"); continue
        if line.startswith("/max "):
            gen.max_new_tokens = int(line.split()[1]); print(f"  max_new_tokens={gen.max_new_tokens}"); continue
        if line == "/greedy":
            greedy = True; print("  greedy=on"); continue
        if line == "/sample":
            greedy = False; print("  greedy=off (sampling)"); continue
        prompt = line if line.startswith("<|user|>") else f"<|user|>{line}<|assistant|>"
        t0 = time.time()
        out = generate(model, tokenizer, prompt, gen, device, greedy=greedy)
        out = _truncate_at_stop(out, prompt)
        print(f"\n{out}\n  ({time.time()-t0:.1f}s)\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prompts", type=Path, default=None,
                    help="one-shot: run every prompt in this file then exit")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  device: {device}")

    tokenizer = SpmAdapter(str(args.tokenizer))
    model, cfg, _ = load_cubby(args.checkpoint, device)
    print(f"  cfg: d={cfg.d_model} L={cfg.n_layers} vocab={cfg.vocab_size} seq={cfg.seq_len}")
    print(f"  hybrid: moe={cfg.enable_moe} attn={cfg.enable_attention} "
          f"mem={cfg.enable_memory} vsa_binding={cfg.vsa_binding_head}")

    gen = GenParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    if args.prompts:
        lines = [l.strip() for l in args.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
        run_prompts(model, tokenizer, lines, device,
                    gen if not args.greedy else GenParams(max_new_tokens=args.max_new_tokens))
    else:
        repl(model, tokenizer, device, gen)


if __name__ == "__main__":
    main()
