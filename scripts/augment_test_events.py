"""Augment 1,000 test events with 32 LLM-extracted attributes via OpenRouter.

Usage:
    uv run python scripts/augment_test_events.py

Requires OPENROUTER_API_KEY in .env or environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Load .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cubemind.execution.attribute_extractor import extract_batch


async def main():
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "test_events_1000.json",
    )
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run Task 2 first.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    print(f"Augmenting {len(events)} events via OpenRouter...")
    print("Model: alibaba/tongyi-deepresearch-30b-a3b")
    print("Estimated cost: ~$0.05")
    print()

    batch = [
        {"text": e.get("summary", ""), "category": e.get("category", "")}
        for e in events
    ]
    results = await extract_batch(batch, batch_size=10, delay=1.0)

    # Merge attributes back
    for event, attrs in zip(events, results):
        event["augmented_attributes"] = attrs

    augmented_count = sum(1 for r in results if r)
    print(f"\nSuccessfully augmented: {augmented_count}/{len(events)}")

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "test_events_1000_augmented.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

    # Quick quality check
    if augmented_count > 0:
        sample = next(r for r in results if r)
        print(f"\nSample attributes ({len(sample)} keys):")
        for k, v in sorted(sample.items())[:10]:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
