"""Train the WorldManager on 1800-1900 historical events.

Pipeline:
1. Load JSONL events from two files (38 total)
2. Sort by earliest_date_year
3. Encode each event to a block-code via EventEncoder
4. Feed sequential transitions (event_i -> event_i+1) to WorldManager
5. Run VSATranslator on discovered specialists to generate descriptions
6. Print results

Run with:
    uv run python -u scripts/train_world_manager.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Ensure stdout uses UTF-8 on Windows so VSATranslator arrow chars print cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cubemind.execution.event_encoder import EventEncoder
from cubemind.execution.vsa_translator import VSATranslator
from cubemind.execution.world_manager import WorldManager
from cubemind.ops.block_codes import BlockCodes

# Use small dims for CPU speed
K = 16
L = 64

# Paths to JSONL data files
_DATA_DIR = (
    r"C:\Users\grill\Desktop\GrillCheese\data_learning\temporal\historical"
)
_JSONL_FILES = [
    os.path.join(_DATA_DIR, "historical_events_1800-1850.jsonl"),
    os.path.join(_DATA_DIR, "historical_events_1800-1900.jsonl"),
]


def load_events(paths: list[str]) -> list[dict]:
    """Load and deduplicate events from JSONL files.

    Args:
        paths: List of file paths to JSONL files.

    Returns:
        Deduplicated list of event dicts (unique by event_id).
    """
    seen: set[str] = set()
    events: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                event_id = event.get("event_id", "")
                if event_id and event_id in seen:
                    continue
                seen.add(event_id)
                events.append(event)
    return events


def build_codebook(encoder: EventEncoder, events: list[dict]) -> dict[str, "np.ndarray"]:  # noqa: F821
    """Build a concept codebook from event titles and topic/affect tags.

    Args:
        encoder: EventEncoder with a warm text-vector cache.
        events: List of event dicts.

    Returns:
        Dict mapping concept name to (k, l) block-code vector.
    """
    codebook: dict[str, object] = {}

    for event in events:
        # Title as a top-level concept
        title = event.get("title", "").strip()
        if title and title not in codebook:
            codebook[title] = encoder._hash_vec(title)

        # Topic tags
        for tag in event.get("topic_tags", []):
            tag = tag.strip()
            if tag and tag not in codebook:
                codebook[tag] = encoder._hash_vec(tag)

        # Affect tag ids
        for aff in event.get("affect_tags", []):
            tag_id = aff.get("id", "").strip()
            if tag_id and tag_id not in codebook:
                codebook[tag_id] = encoder._hash_vec(tag_id)

    return codebook


def main() -> None:
    t_start = time.perf_counter()

    # ── 1. Load events ────────────────────────────────────────────────────
    print("=" * 64)
    print("STEP 1: Loading historical events")
    events = load_events(_JSONL_FILES)
    print(f"  Loaded {len(events)} unique events from {len(_JSONL_FILES)} files")

    # ── 2. Sort by year ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("STEP 2: Sorting events by year")
    events.sort(key=lambda e: e.get("earliest_date_year", 0))
    years = [e.get("earliest_date_year") for e in events if e.get("earliest_date_year")]
    if years:
        print(f"  Year range: {min(years)} – {max(years)}")
    print(f"  Events in order:")
    for ev in events:
        year = ev.get("earliest_date_year", "?")
        title = ev.get("title", ev.get("event_id", "<unknown>"))
        print(f"    [{year}] {title}")

    # ── 3. Encode events to block-codes ───────────────────────────────────
    print("\n" + "=" * 64)
    print(f"STEP 3: Encoding events (k={K}, l={L})")
    encoder = EventEncoder(k=K, l=L)
    block_codes: list[object] = []
    for i, ev in enumerate(events):
        bc = encoder.encode_event(ev)
        block_codes.append(bc)
        if (i + 1) % 10 == 0 or (i + 1) == len(events):
            sys.stdout.write(f"  Encoded {i + 1}/{len(events)} events\r")
            sys.stdout.flush()
    print(f"  Encoded {len(block_codes)} events -> ({K}, {L}) block-codes")

    # ── 4. Train WorldManager on sequential transitions ───────────────────
    print("\n" + "=" * 64)
    print("STEP 4: Training WorldManager on sequential transitions")
    wm = WorldManager(k=K, l=L, max_worlds=64, tau=0.65, oja_lr=0.01)

    n_transitions = len(events) - 1
    for i in range(n_transitions):
        ev_before = events[i]
        ev_after = events[i + 1]
        title_before = ev_before.get("title", f"event_{i}")
        title_after = ev_after.get("title", f"event_{i + 1}")
        result = wm.process_transition(block_codes[i], block_codes[i + 1])
        action = result["action"]
        world_id = result["world_id"]
        sim = result["similarity"]

        if action == "spawned":
            print(
                f"  [SPAWN  ] specialist #{world_id:2d} | sim={sim:.4f}"
                f" | {title_before} -> {title_after}"
            )
        else:
            print(
                f"  [CONSOL ] specialist #{world_id:2d} | sim={sim:.4f}"
                f" | {title_before} -> {title_after}"
            )

    print(f"\n  Transitions processed: {n_transitions}")
    print(f"  Specialists discovered: {wm.active_worlds}")

    # ── 5. Translate specialists via VSATranslator ────────────────────────
    print("\n" + "=" * 64)
    print("STEP 5: Building concept codebook for VSATranslator")
    codebook = build_codebook(encoder, events)
    print(f"  Codebook size: {len(codebook)} concepts")

    if len(codebook) < 2:
        print("  Codebook too small — skipping translation")
    else:
        bc_ops = BlockCodes(k=K, l=L)
        translator = VSATranslator(bc=bc_ops, codebook=codebook)
        specialists = wm.get_specialists()

        print("\n" + "=" * 64)
        print(f"STEP 6: Translating {len(specialists)} specialists")
        for wid, spec in enumerate(specialists):
            obs_count = wm.get_obs_count(wid)
            translation = translator.translate(spec)
            summary = translation["summary"]
            print(f"\n  Specialist #{wid:2d}  (obs={obs_count}):")
            print(f"    {summary}")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 64)
    print("TRAINING COMPLETE")
    print(f"  Events loaded:          {len(events)}")
    print(f"  Transitions trained:    {n_transitions}")
    print(f"  Specialists emerged:    {wm.active_worlds}")
    print(f"  Codebook concepts:      {len(codebook)}")
    print(f"  Total time:             {elapsed:.2f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
