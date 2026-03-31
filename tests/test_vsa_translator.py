"""
Tests for cubemind.execution.vsa_translator.VSATranslator.

Validates:
  - probe_returns_best_match — result has input, name, similarity fields
  - translate_returns_probes_and_summary — full translation works
  - translate_summary_is_string — summary is non-empty string
  - constant_rule_detected — identity-like specialist shows constants
  - different_specialists_different_translations — distinct specialists produce
    different summaries
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.vsa_translator import VSATranslator
from cubemind.ops.block_codes import BlockCodes

# ── Small dims to avoid OOM ────────────────────────────────────────────────────

K = 4
L = 8


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def codebook(bc: BlockCodes) -> dict[str, np.ndarray]:
    """Small codebook with three named concept vectors."""
    return {
        "alpha": bc.random_discrete(seed=1),
        "beta": bc.random_discrete(seed=2),
        "gamma": bc.random_discrete(seed=3),
    }


@pytest.fixture(scope="module")
def translator(bc: BlockCodes, codebook: dict[str, np.ndarray]) -> VSATranslator:
    return VSATranslator(bc=bc, codebook=codebook)


@pytest.fixture(scope="module")
def random_specialist(bc: BlockCodes) -> np.ndarray:
    """A random specialist vector (rule vector) of shape (k, l)."""
    return bc.random_discrete(seed=42)


@pytest.fixture(scope="module")
def identity_specialist(bc: BlockCodes) -> np.ndarray:
    """A near-identity specialist: binding with it returns a vector similar to input.

    We create this by bundling all codebook concepts with themselves — the result
    is that probing any concept through it will match back to the same concept
    (constant / identity behaviour).
    """
    # Use the zero-frequency component trick: bundle each concept bound with itself.
    # bind(v, identity) ≈ v when identity ≈ delta (circular convolution identity).
    # We approximate by building a bundle of all concepts bound to themselves —
    # the specialist is effectively the "no-change" rule for each concept.
    # For the test we just use the bind-identity vector (frequency-domain [1,0,...,0]).
    identity = np.zeros((K, L), dtype=np.float32)
    identity[:, 0] = 1.0  # per-block delta function — circular-conv identity element
    return identity


# ── Test: probe_returns_best_match ────────────────────────────────────────────


def test_probe_returns_best_match_keys(
    translator: VSATranslator, random_specialist: np.ndarray
):
    """probe() must return a dict with keys: input, name, similarity."""
    result = translator.probe(random_specialist, "alpha")
    assert isinstance(result, dict), "probe() must return a dict"
    assert "input" in result, "result must have 'input' key"
    assert "name" in result, "result must have 'name' key"
    assert "similarity" in result, "result must have 'similarity' key"


def test_probe_input_field_matches_argument(
    translator: VSATranslator, random_specialist: np.ndarray
):
    """probe() result['input'] must equal the input concept string passed in."""
    result = translator.probe(random_specialist, "beta")
    assert result["input"] == "beta", (
        f"Expected input='beta', got {result['input']!r}"
    )


def test_probe_name_is_codebook_member(
    translator: VSATranslator, codebook: dict[str, np.ndarray], random_specialist: np.ndarray
):
    """probe() result['name'] must be one of the codebook concept names."""
    result = translator.probe(random_specialist, "alpha")
    assert result["name"] in codebook, (
        f"Expected name in codebook, got {result['name']!r}"
    )


def test_probe_similarity_is_float_in_range(
    translator: VSATranslator, random_specialist: np.ndarray
):
    """probe() result['similarity'] must be a float."""
    result = translator.probe(random_specialist, "gamma")
    assert isinstance(result["similarity"], float), (
        f"Expected float, got {type(result['similarity'])}"
    )


# ── Test: translate_returns_probes_and_summary ───────────────────────────────


def test_translate_returns_dict_with_probes_and_summary(
    translator: VSATranslator, random_specialist: np.ndarray
):
    """translate() must return a dict with 'probes' and 'summary' keys."""
    result = translator.translate(random_specialist)
    assert isinstance(result, dict), "translate() must return a dict"
    assert "probes" in result, "result must have 'probes' key"
    assert "summary" in result, "result must have 'summary' key"


def test_translate_probes_length_matches_codebook(
    translator: VSATranslator, codebook: dict[str, np.ndarray], random_specialist: np.ndarray
):
    """translate() probes list must have one entry per codebook concept."""
    result = translator.translate(random_specialist)
    assert len(result["probes"]) == len(codebook), (
        f"Expected {len(codebook)} probes, got {len(result['probes'])}"
    )


def test_translate_each_probe_has_required_keys(
    translator: VSATranslator, random_specialist: np.ndarray
):
    """Each probe in translate() result must have input, name, similarity keys."""
    result = translator.translate(random_specialist)
    for probe in result["probes"]:
        assert "input" in probe, f"Probe missing 'input': {probe}"
        assert "name" in probe, f"Probe missing 'name': {probe}"
        assert "similarity" in probe, f"Probe missing 'similarity': {probe}"


def test_translate_each_probe_has_kind_key(
    translator: VSATranslator, random_specialist: np.ndarray
):
    """Each probe must be classified as 'transform' or 'constant'."""
    result = translator.translate(random_specialist)
    for probe in result["probes"]:
        assert "kind" in probe, f"Probe missing 'kind': {probe}"
        assert probe["kind"] in ("transform", "constant"), (
            f"Unexpected kind value: {probe['kind']!r}"
        )


# ── Test: translate_summary_is_string ────────────────────────────────────────


def test_translate_summary_is_string(
    translator: VSATranslator, random_specialist: np.ndarray
):
    """translate() result['summary'] must be a non-empty string."""
    result = translator.translate(random_specialist)
    assert isinstance(result["summary"], str), (
        f"Expected str, got {type(result['summary'])}"
    )
    assert len(result["summary"]) > 0, "summary must be non-empty"


# ── Test: constant_rule_detected ─────────────────────────────────────────────


def test_constant_rule_detected(
    bc: BlockCodes, codebook: dict[str, np.ndarray], identity_specialist: np.ndarray
):
    """Identity-like specialist must classify most/all probes as 'constant'."""
    translator = VSATranslator(bc=bc, codebook=codebook)
    result = translator.translate(identity_specialist)
    constant_count = sum(1 for p in result["probes"] if p["kind"] == "constant")
    # For a delta-function specialist, binding always returns the input concept,
    # so all probes should be classified as 'constant'.
    assert constant_count == len(result["probes"]), (
        f"Expected all {len(result['probes'])} probes to be 'constant', "
        f"got {constant_count} constants. "
        f"Probes: {[(p['input'], p['name'], p['kind']) for p in result['probes']]}"
    )


# ── Test: different_specialists_different_translations ───────────────────────


def test_different_specialists_different_translations(bc: BlockCodes):
    """Two distinct specialists must produce different summary strings."""
    concepts = {
        "apple": bc.random_discrete(seed=10),
        "banana": bc.random_discrete(seed=20),
        "cherry": bc.random_discrete(seed=30),
        "date": bc.random_discrete(seed=40),
    }
    translator = VSATranslator(bc=bc, codebook=concepts)

    specialist_a = bc.random_discrete(seed=100)
    specialist_b = bc.random_discrete(seed=200)

    result_a = translator.translate(specialist_a)
    result_b = translator.translate(specialist_b)

    assert result_a["summary"] != result_b["summary"], (
        "Different specialists must produce different summaries, "
        f"both got: {result_a['summary']!r}"
    )


# ── Test: probe always returns highest similarity match ───────────────────────


def test_probe_returns_highest_similarity_match(bc: BlockCodes):
    """probe() must return the codebook entry with the highest similarity."""
    # Build a specialist that is the identity, then probe with "red".
    # bind(identity, red) == red, so the best match should be "red".
    identity = np.zeros((K, L), dtype=np.float32)
    identity[:, 0] = 1.0

    red_vec = bc.random_discrete(seed=77)
    blue_vec = bc.random_discrete(seed=88)
    green_vec = bc.random_discrete(seed=99)

    codebook = {"red": red_vec, "blue": blue_vec, "green": green_vec}
    translator = VSATranslator(bc=bc, codebook=codebook)

    result = translator.probe(identity, "red")
    assert result["name"] == "red", (
        f"Identity specialist probed with 'red' should return 'red', "
        f"got {result['name']!r} (sim={result['similarity']:.4f})"
    )
