"""Tests for cubemind.execution.decoder.

Validates:
  - decode returns the best-matching codebook entry
  - decode_topk results are sorted by descending similarity
  - decode_soft produces a valid probability distribution
  - Decoding a codebook entry itself returns that entry
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.decoder import Decoder
from cubemind.ops import BlockCodes


# -- Fixtures ------------------------------------------------------------------

K = 4
L = 8
N_ENTRIES = 10


@pytest.fixture
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture
def codebook(bc: BlockCodes) -> np.ndarray:
    return bc.codebook_discrete(N_ENTRIES, seed=42)


@pytest.fixture
def decoder(codebook: np.ndarray) -> Decoder:
    labels = [f"class_{i}" for i in range(N_ENTRIES)]
    return Decoder(codebook=codebook, labels=labels)


# -- Tests ---------------------------------------------------------------------


def test_decode_returns_best_match(decoder: Decoder, codebook: np.ndarray, bc: BlockCodes):
    """decode should return the codebook entry with highest similarity."""
    # Query is the first codebook entry (should match itself perfectly)
    query = codebook[0]
    label, sim, idx = decoder.decode(query)
    assert idx == 0, f"Expected index 0, got {idx}"
    assert label == "class_0"
    assert sim > 0.5, f"Expected high similarity, got {sim}"


def test_decode_topk_sorted(decoder: Decoder, codebook: np.ndarray):
    """decode_topk should return results sorted by descending similarity."""
    query = codebook[3]
    results = decoder.decode_topk(query, k=5)

    assert len(results) == 5
    sims = [r[1] for r in results]
    for i in range(len(sims) - 1):
        assert sims[i] >= sims[i + 1], (
            f"Results not sorted: sim[{i}]={sims[i]:.4f} < sim[{i + 1}]={sims[i + 1]:.4f}"
        )

    # The top result should be the query itself
    assert results[0][2] == 3, f"Expected index 3 as top match, got {results[0][2]}"


def test_decode_soft_sums_to_one(decoder: Decoder, codebook: np.ndarray):
    """decode_soft should produce a probability distribution that sums to 1."""
    query = codebook[5]
    probs = decoder.decode_soft(query, temperature=40.0)

    assert probs.shape == (N_ENTRIES,), f"Expected ({N_ENTRIES},), got {probs.shape}"
    np.testing.assert_allclose(
        probs.sum(), 1.0, atol=1e-5,
        err_msg=f"Probabilities do not sum to 1: sum={probs.sum()}"
    )
    assert np.all(probs >= 0), "Probabilities contain negative values"


def test_decode_identity(decoder: Decoder, codebook: np.ndarray):
    """Decoding each codebook entry should return that same entry."""
    for i in range(N_ENTRIES):
        label, sim, idx = decoder.decode(codebook[i])
        assert idx == i, (
            f"Codebook entry {i} decoded to index {idx} (label={label}, sim={sim:.4f})"
        )


def test_decode_soft_peaks_at_correct_entry(decoder: Decoder, codebook: np.ndarray):
    """Soft decoding should assign highest probability to the matching entry."""
    for i in range(N_ENTRIES):
        probs = decoder.decode_soft(codebook[i], temperature=40.0)
        best_idx = int(np.argmax(probs))
        assert best_idx == i, (
            f"Entry {i}: argmax of soft decode is {best_idx}, not {i}"
        )


def test_decoder_integer_labels(codebook: np.ndarray):
    """Decoder with no labels should use integer indices."""
    dec = Decoder(codebook=codebook)
    label, sim, idx = dec.decode(codebook[0])
    assert label == 0
    assert idx == 0
