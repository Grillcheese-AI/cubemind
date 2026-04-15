"""GPT-4 coherence judge for the MinGRU baseline.

Scores generated TinyStories samples on grammar / creativity / consistency
(1–5 each) per the Eldan & Li (2023) rubric. Eval harness stub — the
actual model call is plugged in from Phase 1.2 onward.

Usage (after Phase 1.3 training produces a model checkpoint)::

    python eval_coherence.py \\
        --checkpoint sandbox/mingru_baseline/out/best.pt \\
        --prompts sandbox/mingru_baseline/prompts.txt \\
        --judge gpt-4o \\
        --out sandbox/mingru_baseline/results/eval_run1.json

Env: `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` if `--judge` is a claude
model). The judge call is deferred and the module is importable without
the key — hard-fail only happens at run time if you try to actually
grade.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


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
    scores: list[Score] = field(default_factory=list)

    def summary(self) -> dict:
        n = max(len(self.scores), 1)
        avg = lambda k: sum(getattr(s, k) for s in self.scores) / n
        return {
            "model": self.model_tag,
            "judge": self.judge,
            "n_samples": len(self.scores),
            "grammar_mean": avg("grammar"),
            "creativity_mean": avg("creativity"),
            "consistency_mean": avg("consistency"),
            "total_mean": sum(s.total for s in self.scores) / n,
        }


def load_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150,
             temperature: float = 0.8, top_p: float = 0.9) -> str:
    """Plug in the MinGRU model.generate() here once Phase 1.2 lands.

    Keep the decoding defaults matched to the TinyStories paper so
    cross-run and cross-model comparisons are apples-to-apples.
    """
    raise NotImplementedError(
        "Phase 1.2 will provide the MinGRU model + tokenizer. Wire "
        "model.generate(prompt_ids, max_new_tokens, temperature, top_p) "
        "here and return the decoded text."
    )


def grade(prompt: str, completion: str, judge: str) -> dict:
    """Call the LLM judge. Deferred import so this module loads without
    the SDK installed."""
    if judge.startswith("gpt"):
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
        raw = resp.choices[0].message.content.strip()
    elif judge.startswith("claude"):
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
        raw = resp.content[0].text.strip()
    else:
        raise ValueError(f"Unknown judge model: {judge}")

    # Parse the JSON object the judge was asked for.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Judge did not return JSON: {raw!r}")
    return json.loads(raw[start:end + 1])


def run(checkpoint: Path, prompts_path: Path, judge: str,
        out_path: Path, model_tag: str) -> EvalRun:
    # Phase 1.2 wires the real model + tokenizer; this is the contract.
    from cubemind.training.vsa_lm import load_mingru_checkpoint
    model, tokenizer = load_mingru_checkpoint(checkpoint)

    prompts = load_prompts(prompts_path)
    run = EvalRun(model_tag=model_tag, judge=judge)
    for i, prompt in enumerate(prompts, 1):
        completion = generate(model, tokenizer, prompt)
        graded = grade(prompt, completion, judge)
        run.scores.append(Score(
            prompt=prompt,
            completion=completion,
            grammar=int(graded["grammar"]),
            creativity=int(graded["creativity"]),
            consistency=int(graded["consistency"]),
            notes=graded.get("notes", ""),
        ))
        print(f"  [{i:2d}/{len(prompts)}] grammar={graded['grammar']} "
              f"creativity={graded['creativity']} "
              f"consistency={graded['consistency']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": run.summary(),
        "scores": [s.__dict__ for s in run.scores],
    }, indent=2))
    print(json.dumps(run.summary(), indent=2))
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prompts", type=Path,
                    default=Path("sandbox/mingru_baseline/prompts.txt"))
    ap.add_argument("--judge", default="gpt-4o",
                    help="e.g. gpt-4o, claude-opus-4-6, claude-sonnet-4-6")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-tag", default="mingru_baseline_phase1")
    args = ap.parse_args()
    run(args.checkpoint, args.prompts, args.judge, args.out, args.model_tag)


if __name__ == "__main__":
    main()
