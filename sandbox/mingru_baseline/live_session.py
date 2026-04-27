"""Live REPL for the trained CubeMind multitask model.

Wraps ``LiveAdapter`` in a keyboard-driven loop — type a prompt, see the
backbone's per-head predictions + an optional generated continuation,
then optionally TEACH the model with corrective labels (single NLMS step
on the relevant head's plastic ``basis_B`` only). This is the
single-modality (text-only) version of ``scripts/live_brain.py``: same
online-learning idea, no vision/audio yet.

Commands inside the prompt:

    <text>                          run forward and show heads
    /gen <text>                     run forward + autoregressive continuation
    /teach <head> <target>          NLMS step on that head (target=name or int)
    /recall                         retrieve from hippocampal memory using last pooled
    /write [utility]                store last pooled state as an episode
    /stats                          adapter stats (step / memory / params / dtype)
    /save <path>                    persist updated heads to disk
    /help                           show this list
    /quit (or empty + Ctrl-D)       exit

Run::

    python sandbox/mingru_baseline/live_session.py \\
        --checkpoint results_5090/stage2_multitask/best.pt \\
        --tokenizer  /workspace/tokenizer/grillcheese_spm32k_v2.model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sibling modules importable without -e install.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from live_adapter import LiveAdapter  # noqa: E402
from train_torch import GenParams      # noqa: E402


HELP = (__doc__ or "").split("Commands inside the prompt:")[1].split("Run::")[0]


def _resolve_target(spec_label_names: list[str] | None,
                    raw: str) -> int | None:
    """Accept either a class name (looked up in ``spec_label_names``) or an
    integer id. Returns None on failure."""
    raw = raw.strip()
    if raw.lstrip("-").isdigit():
        return int(raw)
    if spec_label_names:
        try:
            return spec_label_names.index(raw)
        except ValueError:
            pass
    return None


def _show_heads(out: dict) -> None:
    for name, h in out["heads"].items():
        print(f"    {name:<10} top1={h['top1_name']:<20} p={h['top1_prob']:.3f}")


def repl(bot: LiveAdapter, gen_max: int) -> None:
    print("Live session ready. /help for commands, /quit to exit.\n")
    last_pooled = None
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in ("/quit", "/exit", "/q"):
            break

        if line == "/help":
            print(HELP); continue

        if line == "/stats":
            for k, v in bot.stats().items():
                print(f"  {k}: {v}")
            continue

        if line.startswith("/save"):
            parts = line.split(None, 1)
            if len(parts) < 2:
                print("usage: /save <path>"); continue
            bot.save(parts[1].strip())
            continue

        if line == "/recall":
            if last_pooled is None:
                print("  no prior pooled state — run a prompt first"); continue
            r = bot.recall(last_pooled, k=5)
            norm = float(r.norm())
            print(f"  recalled (D,) tensor, ||r||={norm:.3f}, "
                  f"memories={bot.stats()['memory_size']}")
            continue

        if line.startswith("/write"):
            if last_pooled is None:
                print("  no prior pooled state — run a prompt first"); continue
            parts = line.split()
            utility = float(parts[1]) if len(parts) > 1 else 1.0
            ok = bot.write_memory(last_pooled, utility=utility)
            print(f"  write {'ok' if ok else 'rejected (utility<=0)'} "
                  f"— memory now {bot.stats()['memory_size']}")
            continue

        if line.startswith("/teach"):
            parts = line.split(None, 2)
            if len(parts) < 3:
                print("usage: /teach <head> <target_name_or_id>"); continue
            head, raw_tgt = parts[1], parts[2]
            spec = next((s for s in bot.model.head_specs if s["name"] == head),
                        None)
            if spec is None:
                names = [s["name"] for s in bot.model.head_specs]
                print(f"  unknown head {head!r}, choose from {names}"); continue
            tgt = _resolve_target(spec.get("label_names"), raw_tgt)
            if tgt is None:
                print(f"  could not resolve target {raw_tgt!r} for head "
                      f"{head!r}"); continue
            if last_pooled is None:
                print("  no prior pooled state — run a prompt first"); continue
            loss = bot.online_update(head, target_id=tgt,
                                     pooled=last_pooled, lr=1e-3)
            print(f"  online_update {head}→{tgt} pre-loss {loss:.4f}")
            continue

        # Default: forward (with optional /gen prefix for autoregressive continuation)
        do_gen = False
        text = line
        if line.startswith("/gen"):
            do_gen = True
            text = line[len("/gen"):].strip()
            if not text:
                print("usage: /gen <text>"); continue

        out = bot.forward(text=text, generate_continuation=do_gen,
                          gen_params=GenParams(max_new_tokens=gen_max))
        last_pooled = out["pooled"]
        if do_gen:
            print(f"\n  generated:\n  {out['generated']}\n")
        _show_heads(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a stage-2 best.pt")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to grillcheese_spm32k_v2.model")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu (default: auto)")
    ap.add_argument("--gen-max", type=int, default=80,
                    help="Max tokens for /gen continuations")
    args = ap.parse_args()

    bot = LiveAdapter.load(args.checkpoint, args.tokenizer, device=args.device)
    repl(bot, gen_max=args.gen_max)
    print(f"\nsession done — {bot.stats()}")


if __name__ == "__main__":
    main()
