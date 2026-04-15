"""
Tests for cubemind.execution.future_decoder.FutureDecoder.

Validates:
  - decode_returns_string — output is a non-empty string
  - decode_high_urgency — high urgency attrs produce urgency-related words
  - decode_low_impact — low impact attrs produce low-impact words
  - decode_empty_attrs — returns a sensible default string
  - decode_high_stakes — high stakes/tension produce crisis-level words
"""

from __future__ import annotations

import pytest

from cubemind.execution.future_decoder import FutureDecoder


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def decoder() -> FutureDecoder:
    return FutureDecoder()


def _high_urgency_attrs() -> dict[str, float]:
    return {
        "urgency": 0.95,
        "effect_magnitude": 0.9,
        "stakes": 0.85,
        "tension_level": 0.8,
        "geographic_scope": 0.7,
        "population_scale": 0.6,
        "duration": 0.5,
        "chain_length_potential": 0.4,
        "cause_certainty": 0.6,
        "pivotality": 0.5,
        "moral_complexity": 0.4,
        "reversibility": 0.2,
        "agency_type": 0.5,
    }


def _low_impact_attrs() -> dict[str, float]:
    return {
        "urgency": 0.1,
        "effect_magnitude": 0.1,
        "stakes": 0.1,
        "tension_level": 0.1,
        "geographic_scope": 0.1,
        "population_scale": 0.1,
        "duration": 0.1,
        "chain_length_potential": 0.1,
        "cause_certainty": 0.2,
        "pivotality": 0.1,
        "moral_complexity": 0.1,
        "reversibility": 0.8,
        "agency_type": 0.3,
    }


def _high_stakes_attrs() -> dict[str, float]:
    return {
        "stakes": 1.0,
        "tension_level": 0.95,
        "effect_magnitude": 0.9,
        "urgency": 0.6,
        "geographic_scope": 0.9,
        "population_scale": 0.9,
        "duration": 0.8,
        "chain_length_potential": 0.8,
        "cause_certainty": 0.5,
        "pivotality": 0.9,
        "moral_complexity": 0.8,
        "reversibility": 0.1,
        "agency_type": 0.5,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_decode_returns_string(decoder: FutureDecoder) -> None:
    """decode() must always return a non-empty string."""
    result = decoder.decode({"urgency": 0.5, "stakes": 0.5})
    assert isinstance(result, str)
    assert len(result) > 0


def test_decode_high_urgency(decoder: FutureDecoder) -> None:
    """High urgency attributes must produce urgency-related words in the output."""
    result = decoder.decode(_high_urgency_attrs()).lower()
    urgency_words = {"urgent", "immediate", "critical", "pressing"}
    assert any(word in result for word in urgency_words), (
        f"Expected urgency word in: {result!r}"
    )


def test_decode_low_impact(decoder: FutureDecoder) -> None:
    """Low impact attributes must produce low-impact words in the output."""
    result = decoder.decode(_low_impact_attrs()).lower()
    low_impact_words = {"local", "minor", "limited", "small", "contained", "gradual", "low"}
    assert any(word in result for word in low_impact_words), (
        f"Expected low-impact word in: {result!r}"
    )


def test_decode_empty_attrs(decoder: FutureDecoder) -> None:
    """decode({}) must return a non-empty default string without raising."""
    result = decoder.decode({})
    assert isinstance(result, str)
    assert len(result) > 0


def test_decode_high_stakes(decoder: FutureDecoder) -> None:
    """High stakes/tension attributes must produce crisis-level words."""
    result = decoder.decode(_high_stakes_attrs()).lower()
    crisis_words = {"existential", "crisis", "high stakes", "stakes", "peak"}
    assert any(word in result for word in crisis_words), (
        f"Expected crisis word in: {result!r}"
    )


def test_decode_partial_attrs(decoder: FutureDecoder) -> None:
    """decode() must handle a partial attribute dict without raising."""
    result = decoder.decode({"urgency": 0.9})
    assert isinstance(result, str)
    assert len(result) > 0


def test_decode_deterministic(decoder: FutureDecoder) -> None:
    """decode() must return the same result for the same input (no randomness)."""
    attrs = _high_urgency_attrs()
    assert decoder.decode(attrs) == decoder.decode(attrs)


def test_decode_default_message_on_empty(decoder: FutureDecoder) -> None:
    """decode({}) should contain 'moderate' or 'impact' as a sensible default."""
    result = decoder.decode({}).lower()
    default_words = {"moderate", "impact", "outcome"}
    assert any(word in result for word in default_words), (
        f"Expected default word in: {result!r}"
    )
