"""End-to-end test of the causal oracle pipeline.
Validates: normalize → codebook → graph → train → predict → branch.
"""

import json
import os
import numpy as np
import pytest

from cubemind.execution.data_normalizer import normalize_historical
from cubemind.execution.causal_codebook import CausalCodebook
from cubemind.execution.causal_graph import CausalGraph
from cubemind.execution.oracle_trainer import OracleTrainer
from cubemind.execution.decision_tree import DecisionTree, Future
from cubemind.execution.future_decoder import FutureDecoder

K, L = 4, 8  # Small dims for CI speed


@pytest.fixture
def test_events():
    """Load from data/test_events_1000.json or generate synthetic."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "test_events_1000.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)[:20]
    # Synthetic fallback for CI
    return [
        {
            "event_id": f"TEST_{i}",
            "summary": f"Test event {i} happened in year {1900 + i}.",
            "earliest_date_year": float(1900 + i),
            "category": "political_event" if i % 2 == 0 else "war_conflict",
            "confidence": float(i + 1),
            "entities": '{"PERSON": ["Person_A"], "ORG": [], "GPE": ["City_A"], "LOC": []}',
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.5,
            "precursor_events": [
                {"description": f"Cause of event {i}", "year_parsed": float(1890 + i)},
            ],
        }
        for i in range(20)
    ]


def test_end_to_end_pipeline(test_events):
    # 1. Normalize
    events = normalize_historical(test_events)
    assert len(events) >= 10

    # 2. Build codebook (fake 384-dim embeddings for test)
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((100, 384)).astype(np.float32)
    codebook = CausalCodebook(k=K, l=L, n_explicit=2, n_learned=2)
    codebook.fit_pca(embeddings)

    # 3. Encode events → block-codes
    block_codes = {}
    for e in events:
        fake_emb = rng.standard_normal(384).astype(np.float32)
        attrs = {"reversibility": 0.5, "agency_type": 0.3}
        bc = codebook.encode(attrs, fake_emb)
        block_codes[e.event_id] = bc
        assert bc.shape == (K, L)

    # 4. Build graph
    graph = CausalGraph()
    graph.add_events(events)
    graph.build_entity_links(max_year_gap=10)
    event_ids = list(block_codes.keys())
    for i in range(len(event_ids) - 1):
        graph.add_strong_link(event_ids[i], event_ids[i + 1])
    assert graph.edge_count() > 0

    # 5. Train Oracle
    trainer = OracleTrainer(k=K, l=L, n_worlds=4, d_hidden=8)
    pairs = []
    for src_id, tgt_id, weight in graph.get_training_pairs()[:10]:
        if src_id in block_codes and tgt_id in block_codes:
            pairs.append((block_codes[src_id], block_codes[tgt_id], weight))
    if pairs:
        stats = trainer.train_direct_pairs(pairs, n_epochs=2)
        assert stats["updates"] > 0

    # 6. Predict
    state = list(block_codes.values())[0]
    action = list(block_codes.values())[1]
    oracle = trainer.oracle
    futures_raw = oracle.top_k(state, action, world_prior=state, k=3)

    # 7. Decision tree
    tree = DecisionTree(state=state, prompt="What happens next?", k=K, l=L)
    decoder = FutureDecoder()
    futures = [
        Future(
            state=f["future_state"],
            description=decoder.decode({"urgency": 0.5, "stakes": 0.7}),
            plausibility=f["plausibility"],
            q_value=f["q_value"],
            grounding=[],
        )
        for f in futures_raw
    ]
    tree.set_futures(futures)

    # 8. Branch
    tree.select(0)
    assert tree.current.depth == 1

    # 9. Backtrack
    tree.backtrack()
    assert tree.current.depth == 0

    # 10. Export
    export = tree.export()
    assert export["prompt"] == "What happens next?"
    assert len(export["children"]) == 1
