"""Generate rich JSONL historical events using an LLM via OpenRouter.

Usage:
    uv run python scripts/generate_historical_events.py --start 1850 --end 1900 --count 20
    uv run python scripts/generate_historical_events.py --start -3000 --end -2500 --count 50

Requires OPENROUTER_API_KEY in .env at project root.
Output is written to data/historical_events/historical_events_{start}-{end}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid

# ---------------------------------------------------------------------------
# Load .env from project root (same pattern used by augment_test_events.py)
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _val = _val.strip().strip('"').strip("'")
                os.environ.setdefault(_key.strip(), _val)

import httpx  # noqa: E402  (after env load so key is available)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0.3
MAX_TOKENS = 8192

_SCHEMA_EXAMPLE = """\
{
  "event_id": "<uuid>",
  "title": "Short title",
  "summary": "2-3 sentence description.",
  "source_text": "Same as summary or longer.",
  "earliest_date_year": 1830,
  "latest_date_year": 1830,
  "category": "Transportation Infrastructure",
  "event_type": "technology_innovation",
  "sentiment": "positive",
  "sentiment_score": 0.8,
  "energy": 0.8,
  "pleasantness": 0.7,
  "latitude": 53.4,
  "longitude": -2.9,
  "participants": [{"name": "George Stephenson", "entity_type": "PERSON"}],
  "affect_tags": [{"id": "innovation", "score": 0.9}],
  "topic_tags": ["transportation", "industrial revolution"],
  "precursor_events": [{"description": "...", "year": 1825, "causal_strength": 0.8}],
  "causal_link": {
    "effect_score": 0.9,
    "influence_factors": [
      {"factor": "Technological Advancement", "description": "...", "score": 0.95}
    ],
    "next_event_summary": "..."
  },
  "similar_events": [
    {"event_summary": "...", "reasoning": "...", "similarity_score": 0.85}
  ]
}"""


def build_prompt(start: int, end: int, count: int) -> str:
    """Build the OpenRouter prompt requesting historical events.

    Args:
        start: Start year (inclusive). Negative values denote BCE.
        end: End year (inclusive). Negative values denote BCE.
        count: Number of events to generate.

    Returns:
        Prompt string to send to the LLM.
    """
    start_label = f"{abs(start)} BCE" if start < 0 else str(start)
    end_label = f"{abs(end)} BCE" if end < 0 else str(end)

    return f"""\
You are a historian generating structured training data. Generate exactly {count} real \
historical events that occurred between {start_label} and {end_label} (inclusive).

Return ONLY a JSONL block — one JSON object per line, no prose, no commentary.
Do NOT wrap in a markdown code block.

Each event must follow this exact schema (fill every field with real data):
{_SCHEMA_EXAMPLE}

Rules:
- event_id: a fresh UUID v4 for each event
- earliest_date_year / latest_date_year: integer year (negative = BCE)
- sentiment: one of "positive", "negative", "neutral", "mixed"
- sentiment_score / energy / pleasantness: float in [0, 1]
- latitude / longitude: approximate geographic coordinates of the event (float)
- participants: list of {{name, entity_type}} where entity_type is PERSON, ORG, or PLACE
- affect_tags: list of {{id (snake_case word), score}}
- topic_tags: list of strings
- precursor_events: list of {{description, year, causal_strength}} — what led to this
- causal_link: {{effect_score, influence_factors, next_event_summary}} — downstream impact
- similar_events: list of {{event_summary, reasoning, similarity_score}} — analogous events

Output {count} lines, one JSON object per line.
"""


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrapping if present.

    Args:
        text: Raw LLM response text.

    Returns:
        Text with markdown code fences stripped.
    """
    # Strip leading/trailing whitespace first
    text = text.strip()
    # Remove ```json or ``` at the start
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    # Remove ``` at the end
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def call_openrouter(prompt: str, api_key: str, model: str) -> str:
    """Send a prompt to OpenRouter and return the assistant's reply text.

    Args:
        prompt: The user message to send.
        api_key: OpenRouter API key.
        model: Model identifier string.

    Returns:
        Raw text content of the assistant reply.

    Raises:
        RuntimeError: If the API call fails or returns a non-200 status.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cubemind",
        "X-Title": "CubeMind historical event generator",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response shape: {data!r}") from exc


def parse_events(raw: str, start: int, end: int) -> list[dict]:
    """Parse LLM response into a list of event dicts.

    Handles:
    - Markdown code fence wrapping
    - Blank lines between JSON objects
    - Invalid lines (logged and skipped)

    Args:
        raw: Raw text from the LLM.
        start: Start year (used to fill missing event_id / year fields).
        end: End year.

    Returns:
        List of parsed event dicts (may be shorter than requested if parsing fails).
    """
    cleaned = strip_markdown_fences(raw)
    events: list[dict] = []

    for lineno, line in enumerate(cleaned.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"  [WARN] Line {lineno}: JSON parse error — {exc}", file=sys.stderr)
            continue

        # Ensure event_id is present
        if not obj.get("event_id"):
            obj["event_id"] = str(uuid.uuid4())

        events.append(obj)

    return events


def output_path(data_dir: str, start: int, end: int) -> str:
    """Compute the output file path for a given time window.

    Args:
        data_dir: Base data directory.
        start: Start year.
        end: End year.

    Returns:
        Absolute path to the output JSONL file.
    """
    filename = f"historical_events_{start}-{end}.jsonl"
    return os.path.join(data_dir, filename)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate rich JSONL historical events via OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", type=int, required=True, help="Start year (negative = BCE)")
    parser.add_argument("--end", type=int, required=True, help="End year (negative = BCE)")
    parser.add_argument(
        "--count", type=int, default=20, help="Number of events to generate per batch"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model identifier"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt and exit without calling the API",
    )
    args = parser.parse_args(argv)

    if args.start >= args.end:
        parser.error(f"--start ({args.start}) must be less than --end ({args.end})")

    # Resolve output path
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "historical_events")
    out_file = output_path(data_dir, args.start, args.end)

    print("CubeMind historical event generator")
    print(f"  Window : {args.start} to {args.end}")
    print(f"  Count  : {args.count}")
    print(f"  Model  : {args.model}")
    print(f"  Output : {os.path.normpath(out_file)}")
    print()

    # Skip if output already exists
    if os.path.exists(out_file):
        print("[SKIP] Output file already exists — delete it to regenerate.")
        return

    # Build prompt
    prompt = build_prompt(args.start, args.end, args.count)

    if args.dry_run:
        print("[DRY RUN] Prompt that would be sent:")
        print("-" * 60)
        print(prompt)
        print("-" * 60)
        return

    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY not set. Add it to .env or export it as an env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create output directory
    os.makedirs(data_dir, exist_ok=True)

    # Call API
    print(f"Calling OpenRouter ({args.model})...")
    raw = call_openrouter(prompt, api_key, args.model)

    # Parse
    print("Parsing response…")
    events = parse_events(raw, args.start, args.end)
    print(f"Parsed {len(events)} events (requested {args.count})")

    if not events:
        print("ERROR: No events parsed — check raw response below.", file=sys.stderr)
        print(raw[:2000], file=sys.stderr)
        sys.exit(1)

    # Write JSONL
    with open(out_file, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"Saved {len(events)} events → {os.path.normpath(out_file)}")


if __name__ == "__main__":
    main()
