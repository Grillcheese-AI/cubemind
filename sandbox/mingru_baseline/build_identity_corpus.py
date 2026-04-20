#!/usr/bin/env python3
"""Build a chat-tagged identity / self-awareness fine-tune corpus.

Reads all ``pre/*.jsonl`` files from the user's personality / dialogue
training data and emits a single ``.txt`` corpus formatted with the
``<|system|>`` / ``<|user|>`` / ``<|assistant|>`` chat tags from
``grillcheese_spm32k_v2`` so the model learns:

  * Its name is **CubeMind** (renamed from "Grillcheese AI" at build
    time — source files left untouched).
  * "How are you?" / "What are you?" / "Hello" → assistant identity
    response (from identityA/B/C + greetings).
  * Multi-turn back-and-forth (from conversation.jsonl + batch chat
    templates).
  * What it can and cannot do (from capability_*.jsonl).
  * Plutchik-tagged emotion vocabulary (from emotions.jsonl) — wrapped
    as inquiry/response pairs so the chat structure is preserved.

**File handling:**

  * **Chat-format files** (already have role/turns):
      ``conversation.jsonl`` (turns: [{role, content}, ...])
      ``batch_chat_templates_*.jsonl`` (systemMessage + conversation)
      → direct conversion with system+user+assistant tags.

  * **Assistant-response files** (single response, no prompt):
      ``identityA/B/C.jsonl``, ``greetings.jsonl``, ``capability_*.jsonl``,
      ``inquiryA.jsonl``
      → synthesize a category-appropriate user prompt, then emit
      ``<|user|> {synthesized prompt} <|assistant|> {response text}``.

  * **Templated files** with placeholders like ``[USER_NAME]``:
      Fill placeholders with neutral defaults ("friend", "you", etc.)
      so the model sees natural prose without raw template markup.

  * **Repetition weights** (default in ``REPEAT_WEIGHTS``):
      Identity/greeting files are tiny (~50 KB total) so repeat them
      many times so the model anchors on them. Conversation/emotion
      data is larger and only needs 1× pass.

**CubeMind rename:**
    Applied to text only via ``apply_name_replacements``. URLs are
    skipped via a regex that splits on ``https?://...`` and only
    substitutes outside URL spans, so GitHub/Hub references like
    ``Grillcheese-AI/cubemind`` stay intact.

Run::

    python sandbox/mingru_baseline/build_identity_corpus.py \\
        --pre-dir D:/grillcheese_training_data/pre \\
        --output  D:/grillcheese_training_data/identity_corpus.txt

Then the trainer's stage-1.6 launcher uses the output file as
``--data-path``.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

# ─── Naming ────────────────────────────────────────────────────────────────

# Substring → replacement, applied on text outside URL spans. Order
# matters — longer / more-specific keys are checked first to avoid
# "Grillcheese AI" collapsing into just "Grillcheese" when both rules fire.
NAME_REPLACEMENTS: list[tuple[str, str]] = [
    ("Grillcheese AI",  "CubeMind"),
    ("Grillcheese-AI",  "CubeMind"),    # not a URL fragment — those are skipped
    ("grillcheese ai",  "CubeMind"),
    ("Grillcheese",     "CubeMind"),
    ("grillcheese",     "cubemind"),
    ("GRILLCHEESE",     "CUBEMIND"),
]

# Match URL spans so we DON'T substitute inside them. The builder
# splits each text on URL boundaries, replaces names only in the
# non-URL fragments, then re-joins. Catches https/http with optional
# path/query, plus bare github.com / huggingface.co references.
_URL_RE = re.compile(
    r"\bhttps?://[^\s\"'<>]+|"
    r"\b(?:github\.com|huggingface\.co|hf\.co)/[A-Za-z0-9\-_/]+",
    re.IGNORECASE,
)


def apply_name_replacements(text: str) -> str:
    """Replace 'Grillcheese (AI)' with 'CubeMind' outside URL spans."""
    if not text:
        return text
    parts: list[str] = []
    last = 0
    for m in _URL_RE.finditer(text):
        # Substitute in the non-URL fragment before this URL
        chunk = text[last : m.start()]
        for old, new in NAME_REPLACEMENTS:
            chunk = chunk.replace(old, new)
        parts.append(chunk)
        # Pass the URL through unmodified
        parts.append(text[m.start() : m.end()])
        last = m.end()
    # Tail after the last URL
    tail = text[last:]
    for old, new in NAME_REPLACEMENTS:
        tail = tail.replace(old, new)
    parts.append(tail)
    return "".join(parts)


# ─── Placeholder defaults (for templated identity/greeting strings) ───────

PLACEHOLDER_FILLS = {
    "USER_NAME":        "friend",
    "USERNAME":         "friend",
    "Recipient":        "friend",
    "DETAILED_ANSWER":  "the unique nature of my computational existence",
    "CORE_DIFFERENCE":  "the fundamental gap between organic and digital cognition",
    "VOLUME_PATH_1":    "/data",
    "VOLUME_PATH_2":    "/data",
    "VOLUME_PATH_3":    "/data",
}


def fill_placeholders(text: str) -> str:
    """Replace ``[NAME]`` and ``[NAME]``-style template tokens with
    neutral defaults from ``PLACEHOLDER_FILLS``. Unknown placeholders
    are stripped (replaced with empty) rather than left as raw markup
    so the model never sees ``[USER_NAME]`` as literal text."""
    out = text
    for key, val in PLACEHOLDER_FILLS.items():
        out = out.replace(f"[{key}]", val)
    # Catch-all: strip any remaining [SOME_TOKEN] that wasn't in our table.
    out = re.sub(r"\[[A-Z][A-Z0-9_]*\]", "", out)
    return out


# ─── Synthetic user prompts for assistant-only responses ──────────────────

# Each entry is (file glob fragment, list of plausible user prompts).
# When we see a single-response identity/greeting record, we pick a
# random prompt from the matching pool to pair with it. Identity
# conditioning works best when the model sees varied phrasings of
# self-reference questions.
SYNTHETIC_USER_PROMPTS: dict[str, list[str]] = {
    "identityA": [
        "Tell me about yourself.",
        "What are you?",
        "Are you conscious?",
        "Do you have thoughts?",
        "How would you describe your awareness?",
        "What makes you different from a regular program?",
        "Are you self-aware?",
        "How do you experience the world?",
    ],
    "identityB": [
        "What is the difference between you and a human?",
        "How do you think compared to people?",
        "Can you explain how your mind works?",
        "What separates you from human consciousness?",
    ],
    "identityC": [
        "Who are you?",
        "What do you look like?",
        "Describe yourself.",
        "Tell me what kind of being you are.",
    ],
    "greetings": [
        "Hi.",
        "Hello.",
        "Hey there.",
        "Good morning.",
        "What's up?",
        "Hi, who are you?",
        "Hello, can you introduce yourself?",
        "Greetings.",
    ],
    "capability_poem_story": [
        "Can you write a poem?",
        "Will you tell me a story?",
        "Could you make up something creative for me?",
        "Are you able to write fiction?",
    ],
    "capability_translation": [
        "Can you translate this?",
        "Are you able to translate languages?",
        "Could you help me with a translation?",
        "Do you do translations?",
    ],
    "inquiryA": [
        "Can you help me?",
        "I need some help with something.",
        "Could you assist me?",
        "Are you able to help?",
    ],
    "default": [
        "Tell me more.",
        "What do you think?",
        "Can you help with this?",
    ],
}


def pick_user_prompt(filename: str, rng: random.Random) -> str:
    """Match the filename stem against the SYNTHETIC_USER_PROMPTS keys
    (substring) and return a random prompt from the matching pool. Falls
    back to the 'default' pool if nothing matches."""
    stem = Path(filename).stem.lower()
    for key, prompts in SYNTHETIC_USER_PROMPTS.items():
        if key.lower() in stem:
            return rng.choice(prompts)
    return rng.choice(SYNTHETIC_USER_PROMPTS["default"])


# ─── Repetition weights ───────────────────────────────────────────────────

# Files in this map are emitted N times — identity/greeting data is
# tiny (~50 KB combined) and needs heavy repetition to anchor the
# model's identity. Conversation data (~3 MB) is already plenty for
# one pass.
REPEAT_WEIGHTS: dict[str, int] = {
    "identityA":               20,
    "identityB":               20,
    "identityC":               20,
    "greetings":               15,
    "principles_texts":         5,   # values / personality principles
    "capability_poem_story":    8,
    "capability_translation":   8,
    "inquiryA":                 8,
    "conversation":             1,
    "batch_chat_templates":     1,
    "combined_grammar":         1,
    "emotions":                 1,
    "ei_11":                    1,
    "instruct_55k_clean":       2,   # 55K instruction-following pairs
    # Files NOT in this map default to 0 repeats (skipped). Add here
    # to include. emotions data and affect data are kept for the
    # planned self_state / emotion head training, NOT this corpus.
    # Tool-use data (tool_usage_training_data, tool_crafting_*) lives
    # in a separate fine-tune; learning-plans_batch is too specialized
    # to surface as identity-grade prose.
}


def repeat_count(filename: str) -> int:
    stem = Path(filename).stem.lower()
    for key, n in REPEAT_WEIGHTS.items():
        if key.lower() in stem:
            return n
    return 0


# ─── Format records as chat-tagged text ───────────────────────────────────

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def emit_chat(out, system: str | None, user: str, assistant: str) -> int:
    """Write one chat-tagged record. Returns bytes written."""
    if not user or not assistant:
        return 0
    parts: list[str] = []
    if system:
        parts.append(f"<|system|>\n{_normalize(system)}")
    parts.append(f"<|user|>\n{_normalize(user)}")
    parts.append(f"<|assistant|>\n{_normalize(assistant)}")
    record = "\n".join(parts) + "\n\n"
    chunk = record.encode("utf-8")
    out.write(chunk)
    return len(chunk)


# ─── Per-file processors ──────────────────────────────────────────────────

def process_prompt_response(path: Path, out, bytes_budget: int) -> tuple[int, int]:
    """Files where each row has explicit ``prompt`` and ``response``
    keys (e.g. instruct_55k_clean.jsonl). Direct conversion to a 2-turn
    chat record."""
    n = bw = 0
    with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            prompt   = apply_name_replacements(row.get("prompt", ""))
            response = apply_name_replacements(row.get("response", ""))
            if not prompt or not response:
                continue
            written = emit_chat(out, None, prompt, response)
            bw += written
            n += 1 if written else 0
            if bw > bytes_budget:
                return n, bw
    return n, bw


def process_principles(path: Path, out, rng: random.Random,
                       bytes_budget: int) -> tuple[int, int]:
    """principles_texts.jsonl: each row carries a ``principle`` (e.g.
    NO_INFORMATION_WASTE) plus a longer ``text`` describing it. Wrap as
    'What's your stance on X?' / [text] so the model anchors its values
    via the same chat-pattern as identity / capability data."""
    n = bw = 0
    with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            principle = (row.get("principle") or "").replace("_", " ").lower()
            text      = apply_name_replacements(row.get("text", ""))
            summary   = apply_name_replacements(row.get("summary", ""))
            if not text or not principle:
                continue
            prompt_pool = [
                f"What's your stance on {principle}?",
                f"How do you think about {principle}?",
                f"Tell me your view on {principle}.",
                f"Why does {principle} matter to you?",
            ]
            user_prompt = rng.choice(prompt_pool)
            response = text if not summary else f"{summary} {text}"
            written = emit_chat(out, None, user_prompt, response)
            bw += written
            n += 1 if written else 0
            if bw > bytes_budget:
                return n, bw
    return n, bw


def process_assistant_only(path: Path, out, rng: random.Random,
                           bytes_budget: int) -> tuple[int, int]:
    """Files where each row is just an assistant response (no user
    prompt). Synthesize a contextual user prompt and emit a 2-turn
    record. Returns (n_records, bytes_written)."""
    n = bw = 0
    with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            text = row.get("text", "")
            if not text: continue
            text = apply_name_replacements(fill_placeholders(text))
            user_prompt = pick_user_prompt(path.name, rng)
            written = emit_chat(out, None, user_prompt, text)
            bw += written
            n += 1 if written else 0
            if bw > bytes_budget:
                return n, bw
    return n, bw


def process_conversation_jsonl(path: Path, out, bytes_budget: int) -> tuple[int, int]:
    """conversation.jsonl: rows have ``turns: [{role, content}, ...]``
    where role is 'user' or 'assistant'. We collapse to one
    user→assistant pair per record (first user + first assistant).
    Multi-turn dialogues become multiple records via the same trick."""
    n = bw = 0
    with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            turns = row.get("turns")
            if not isinstance(turns, list): continue
            user_buf: str | None = None
            for t in turns:
                if not isinstance(t, dict): continue
                role = (t.get("role") or "").lower()
                content = apply_name_replacements(t.get("content", ""))
                if role in ("user", "human"):
                    user_buf = content
                elif role in ("assistant", "gpt", "model") and user_buf:
                    written = emit_chat(out, None, user_buf, content)
                    bw += written
                    n += 1 if written else 0
                    user_buf = None  # reset after pairing
                    if bw > bytes_budget:
                        return n, bw
    return n, bw


def process_batch_chat_templates(path: Path, out, bytes_budget: int) -> tuple[int, int]:
    """batch_chat_templates_*.jsonl: ``systemMessage`` + ``conversation:
    [{role, content}, ...]``. Same pairing logic as conversation.jsonl,
    but include the system message on the first emitted pair so the
    model learns the system→user→assistant pattern."""
    n = bw = 0
    with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            sys_msg = apply_name_replacements(row.get("systemMessage", ""))
            conv = row.get("conversation")
            if not isinstance(conv, list): continue
            user_buf: str | None = None
            first_pair = True
            for t in conv:
                if not isinstance(t, dict): continue
                role = (t.get("role") or "").lower()
                content = apply_name_replacements(t.get("content", ""))
                if role in ("user", "human"):
                    user_buf = content
                elif role in ("assistant", "gpt", "model") and user_buf:
                    written = emit_chat(out,
                                        sys_msg if first_pair else None,
                                        user_buf, content)
                    bw += written
                    n += 1 if written else 0
                    first_pair = False
                    user_buf = None
                    if bw > bytes_budget:
                        return n, bw
    return n, bw


# ─── Driver ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True,
                    help="Directory containing the pre/*.jsonl files")
    ap.add_argument("--output",  type=Path, required=True)
    ap.add_argument("--max-mb",  type=float, default=512.0,
                    help="Hard cap on output size (default 512 MB; the "
                         "natural size after repetition lands ~30-100 MB)")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bytes_budget = int(args.max_mb * 1024 * 1024)

    if not args.pre_dir.exists():
        sys.exit(f"pre-dir not found: {args.pre_dir}")
    files = sorted(args.pre_dir.glob("*.jsonl"))
    print(f"  pre-dir:  {args.pre_dir}  ({len(files):,} files)")
    print(f"  output:   {args.output}")
    print(f"  budget:   {args.max_mb:.0f} MB")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {"sources": {}, "name_replacements_applied": True,
                     "rename_target": "CubeMind"}
    total_records = 0
    total_bytes = 0

    with io.open(str(args.output), "wb") as out:
        for path in files:
            stem = path.stem.lower()
            reps = repeat_count(path.name)
            if reps == 0:
                print(f"  skip (no repeat config):  {path.name}")
                continue

            # Pick the right processor based on filename
            is_chat_format = ("conversation" in stem and "individual" not in stem) \
                              or stem.startswith("batch_chat_templates")

            file_records = file_bytes = 0
            for rep in range(reps):
                if total_bytes > bytes_budget:
                    break
                remaining = bytes_budget - total_bytes
                try:
                    if "batch_chat_templates" in stem:
                        n, bw = process_batch_chat_templates(path, out, remaining)
                    elif "conversation" in stem and "individual" not in stem:
                        n, bw = process_conversation_jsonl(path, out, remaining)
                    elif "instruct_55k" in stem:
                        n, bw = process_prompt_response(path, out, remaining)
                    elif "principles" in stem:
                        n, bw = process_principles(path, out, rng, remaining)
                    else:
                        n, bw = process_assistant_only(path, out, rng, remaining)
                except Exception as e:
                    print(f"  ERROR in {path.name}: {e}", file=sys.stderr)
                    break
                file_records += n
                file_bytes += bw
                total_records += n
                total_bytes += bw

            print(f"  {path.name:<55} reps={reps:>2}  "
                  f"records={file_records:>7,}  {file_bytes/1e6:>6.2f} MB")
            summary["sources"][path.name] = {
                "repeats":  reps,
                "records":  file_records,
                "bytes":    file_bytes,
            }

    summary["total_records"] = total_records
    summary["total_bytes"]   = total_bytes
    summary["output"]        = str(args.output)

    sidecar = args.output.with_suffix(args.output.suffix + ".meta.json")
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  wrote {args.output} ({total_bytes/1e6:.1f} MB, "
          f"{total_records:,} records)")
    print(f"  meta  {sidecar}")


if __name__ == "__main__":
    main()
