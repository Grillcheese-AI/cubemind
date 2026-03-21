"""Train the Decision Oracle on 1,000 augmented historical events.

Pipeline:
1. Load augmented events
2. Sample corpus embeddings from Qdrant for PCA
3. Build CausalCodebook (PCA + VQ)
4. Encode all events -> block-codes
5. Build CausalGraph with tiered linking
6. Train Oracle (direct pairs -> graph walks -> contrastive)
7. Demo: interactive prediction
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use smaller dims for CPU validation — scale up when grilly GPU is connected
# from cubemind.core import K_BLOCKS, L_BLOCK
K_BLOCKS = 16
L_BLOCK = 64
from cubemind.execution.causal_codebook import CausalCodebook
from cubemind.execution.causal_graph import CausalGraph
from cubemind.execution.data_normalizer import normalize_historical
from cubemind.execution.decision_tree import DecisionTree, Future
from cubemind.execution.future_decoder import FutureDecoder
from cubemind.execution.oracle_trainer import OracleTrainer


def load_augmented_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_qdrant_embeddings(n: int = 10000) -> np.ndarray | None:
    """Sample n embeddings from Qdrant corpus for PCA."""
    try:
        import httpx

        # Scroll through corpus to get embeddings
        embeddings = []
        offset = None
        batch_size = min(100, n)

        while len(embeddings) < n:
            payload: dict = {
                "limit": batch_size,
                "with_vector": True,
                "with_payload": False,
            }
            if offset:
                payload["offset"] = offset

            resp = httpx.post(
                "http://localhost:6333/collections/corpus/points/scroll",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            points = data["result"]["points"]
            if not points:
                break

            for p in points:
                vec = p.get("vector")
                if vec:
                    embeddings.append(vec)

            offset = data["result"].get("next_page_offset")
            if not offset:
                break

            print(f"  Sampled {len(embeddings)}/{n} embeddings...", end="\r")

        print(f"  Sampled {len(embeddings)} embeddings from Qdrant")
        return np.array(embeddings[:n], dtype=np.float32)
    except Exception as e:
        print(f"  Qdrant not available ({e}), using random embeddings")
        return None


def main():
    t_start = time.perf_counter()

    # ── 1. Load augmented events ─────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading augmented events")
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "test_events_1000_augmented.json"
    )
    raw_events = load_augmented_events(data_path)
    print(f"  Loaded {len(raw_events)} events")

    # Count how many have augmented attributes
    with_attrs = sum(1 for e in raw_events if e.get("augmented_attributes"))
    print(f"  With attributes: {with_attrs}/{len(raw_events)}")

    # ── 2. Sample corpus embeddings for PCA ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Sampling corpus embeddings for PCA")
    corpus_embeddings = sample_qdrant_embeddings(n=10000)

    if corpus_embeddings is None:
        print("  Falling back to random embeddings for PCA")
        rng = np.random.default_rng(42)
        corpus_embeddings = rng.standard_normal((10000, 384)).astype(np.float32)

    # ── 3. Build CausalCodebook ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Building CausalCodebook")
    n_explicit = min(32, K_BLOCKS - 2)  # leave room for learned axes
    n_learned = K_BLOCKS - n_explicit
    codebook = CausalCodebook(
        k=K_BLOCKS, l=L_BLOCK,
        n_explicit=n_explicit, n_learned=n_learned,
    )
    codebook.fit_pca(corpus_embeddings)
    print(f"  Codebook: {K_BLOCKS} axes ({n_explicit} explicit + {n_learned} learned)")
    print(f"  Block length: {L_BLOCK}")
    print(f"  D_VSA: {K_BLOCKS * L_BLOCK}")

    # Save codebook
    codebook_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "causal_codebook.npz"
    )
    codebook.save(codebook_path)
    print(f"  Saved codebook to {codebook_path}")

    # ── 4. Encode events -> block-codes ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Encoding events to block-codes")
    events = normalize_historical(raw_events)
    block_codes: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)

    for i, (event, raw) in enumerate(zip(events, raw_events)):
        attrs = raw.get("augmented_attributes", {})
        # Use a deterministic fake embedding based on event text hash
        seed = hash(event.text) % (2**31)
        fake_emb = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        bc = codebook.encode(attrs, fake_emb)
        block_codes[event.event_id] = bc

        if (i + 1) % 200 == 0:
            print(f"  Encoded {i + 1}/{len(events)} events")

    print(f"  Encoded {len(block_codes)} events -> ({K_BLOCKS}, {L_BLOCK}) block-codes")

    # ── 5. Build CausalGraph ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Building CausalGraph")
    graph = CausalGraph()
    graph.add_events(events)

    # Strong links from precursor chains
    strong_count = 0
    for i, raw in enumerate(raw_events):
        precursors = raw.get("precursor_events", [])
        if not precursors:
            continue
        # Link to temporally adjacent events with similar precursors
        event = events[i]
        for j, other_raw in enumerate(raw_events):
            if i == j:
                continue
            other = events[j]
            if event.year and other.year and event.year < other.year:
                # Check if any precursor description overlaps
                for p in precursors:
                    desc = p.get("description", "").lower()
                    if other.text and any(
                        word in other.text.lower()
                        for word in desc.split()[:3]
                        if len(word) > 4
                    ):
                        graph.add_strong_link(event.event_id, other.event_id)
                        strong_count += 1
                        break
            if strong_count > 5000:
                break
        if strong_count > 5000:
            break

    print(f"  Strong links (precursor-based): {strong_count}")

    # Entity links
    entity_links = graph.build_entity_links(max_year_gap=10)
    print(f"  Entity links (medium): {entity_links}")

    print(f"  Total: {graph.node_count()} nodes, {graph.edge_count()} edges")

    # ── 6. Train Oracle ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Training Oracle")

    trainer = OracleTrainer(
        k=K_BLOCKS, l=L_BLOCK,
        n_worlds=16, d_hidden=32,
        seed=42,
    )

    # Phase 1: Direct pairs
    print("\n  Phase 1: Direct pair training...")
    pairs = []
    for src_id, tgt_id, weight in graph.get_training_pairs():
        if src_id in block_codes and tgt_id in block_codes:
            pairs.append((block_codes[src_id], block_codes[tgt_id], weight))
    print(f"  Training pairs: {len(pairs)}")

    if pairs:
        # Use subset for speed (full training can be done later)
        train_pairs = pairs[:100]
        stats1 = trainer.train_direct_pairs(train_pairs, n_epochs=2, beta=0.95)
        print(f"  Phase 1 done: {stats1['updates']} updates")

    # Phase 2: Graph walks
    print("\n  Phase 2: Graph walk training...")
    walks_encoded = []
    event_ids = list(block_codes.keys())
    for start_id in event_ids[:100]:
        walk_ids = graph.walk(start_id, max_hops=4)
        if len(walk_ids) >= 2:
            walk_codes = [block_codes[eid] for eid in walk_ids if eid in block_codes]
            if len(walk_codes) >= 2:
                walks_encoded.append(walk_codes)

    print(f"  Walks generated: {len(walks_encoded)}")
    if walks_encoded:
        stats2 = trainer.train_graph_walks(walks_encoded[:20], n_epochs=1)
        print(f"  Phase 2 done: {stats2['updates']} updates")

    # Phase 3: Contrastive
    print("\n  Phase 3: Contrastive training...")
    pos_pairs = [(cause, effect) for cause, effect, _ in pairs[:200]]
    if pos_pairs:
        stats3 = trainer.train_contrastive(pos_pairs[:50], n_negatives=4, n_epochs=1)
        print(f"  Phase 3 done: {stats3['updates']} updates")

    # ── 7. Demo: Interactive prediction ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Demo — Interactive Decision Tree")

    # Pick a real event as starting state
    demo_event = raw_events[0]
    demo_attrs = demo_event.get("augmented_attributes", {})
    demo_text = demo_event.get("summary", "Unknown event")
    demo_state = list(block_codes.values())[0]
    demo_action = list(block_codes.values())[1]

    print(f"\n  Prompt: \"{demo_text[:100]}...\"")

    oracle = trainer.oracle
    futures_raw = oracle.top_k(demo_state, demo_action, world_prior=demo_state, k=5)

    tree = DecisionTree(
        state=demo_state, prompt=demo_text, k=K_BLOCKS, l=L_BLOCK,
    )
    decoder = FutureDecoder()
    futures = []
    for f in futures_raw:
        desc = decoder.decode(demo_attrs)
        futures.append(Future(
            state=f["future_state"],
            description=desc,
            plausibility=f["plausibility"],
            q_value=f["q_value"],
            grounding=[],
        ))
    tree.set_futures(futures)

    print(f"\n  Top 5 futures:")
    for i, f in enumerate(tree.current.futures):
        print(f"    {i + 1}. [plaus={f.plausibility:.4f} q={f.q_value:.4f}] {f.description[:80]}")

    # Branch into future #1
    tree.select(0)
    print(f"\n  Selected future #1 -> depth {tree.current.depth}")

    # Export tree
    export = tree.export()
    export_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "demo_tree.json"
    )
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"  Tree exported to {export_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Events: {len(events)}")
    print(f"  Block-codes: {len(block_codes)} × ({K_BLOCKS}, {L_BLOCK})")
    print(f"  Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
    print(f"  Training pairs: {len(pairs)}")
    print(f"  Walks: {len(walks_encoded)}")
    print(f"  Oracle worlds: {oracle.n_worlds}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
