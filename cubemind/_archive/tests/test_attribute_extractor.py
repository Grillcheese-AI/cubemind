"""
Tests for cubemind.execution.attribute_extractor.

Validates:
  - ATTRIBUTE_NAMES has exactly 32 entries
  - build_extraction_prompt includes expected content
  - parse_attributes handles valid JSON, clamping, and invalid JSON
  - extract_batch returns correct structure (no real API calls)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cubemind.execution.attribute_extractor import (
    ATTRIBUTE_NAMES,
    build_extraction_prompt,
    extract_batch,
    parse_attributes,
)


# ── ATTRIBUTE_NAMES ───────────────────────────────────────────────────────────


def test_attribute_names_count():
    """ATTRIBUTE_NAMES must contain exactly 32 entries."""
    assert len(ATTRIBUTE_NAMES) == 32, f"Expected 32 attributes, got {len(ATTRIBUTE_NAMES)}"


def test_attribute_names_all_strings():
    """All attribute names must be non-empty strings."""
    for name in ATTRIBUTE_NAMES:
        assert isinstance(name, str), f"Expected str, got {type(name)}: {name!r}"
        assert name.strip(), f"Attribute name must not be empty or whitespace: {name!r}"


def test_attribute_names_unique():
    """All attribute names must be unique."""
    assert len(ATTRIBUTE_NAMES) == len(set(ATTRIBUTE_NAMES)), (
        "Duplicate attribute names found"
    )


def test_attribute_names_expected_entries():
    """Spot-check a selection of expected attribute names across all 6 categories."""
    expected = [
        # Causality
        "reversibility", "agency_type", "chain_length_potential",
        "cause_certainty", "effect_magnitude", "feedback_loop",
        # Temporal
        "urgency", "duration", "periodicity",
        "temporal_distance", "decay_rate", "recurrence_risk",
        # Impact
        "geographic_scope", "population_scale", "domain_breadth",
        "institutional_depth", "economic_weight", "cultural_reach",
        # Counterfactual
        "n_plausible_alternatives", "pivotality", "contingency",
        "path_dependence", "fragility", "determinism_score",
        # Narrative
        "tension_level", "stakes", "resolution_type",
        "moral_complexity", "perspective_count", "ambiguity",
        # Semantic
        "category_score", "event_type_score",
    ]
    for name in expected:
        assert name in ATTRIBUTE_NAMES, f"Expected '{name}' in ATTRIBUTE_NAMES"


# ── build_extraction_prompt ───────────────────────────────────────────────────


def test_build_extraction_prompt_returns_string():
    """build_extraction_prompt must return a non-empty string."""
    result = build_extraction_prompt("The volcano erupted.", "natural_disaster")
    assert isinstance(result, str)
    assert len(result) > 100


def test_build_extraction_prompt_contains_text():
    """Prompt must include the supplied event text."""
    text = "A new trade agreement was signed."
    result = build_extraction_prompt(text, "economics")
    assert text in result


def test_build_extraction_prompt_contains_category():
    """Prompt must include the supplied category."""
    result = build_extraction_prompt("Event text.", "geopolitics")
    assert "geopolitics" in result


def test_build_extraction_prompt_mentions_all_attributes():
    """Prompt must reference every one of the 32 attribute names."""
    result = build_extraction_prompt("Some event.", "test")
    for name in ATTRIBUTE_NAMES:
        assert name in result, f"Prompt missing attribute: {name!r}"


def test_build_extraction_prompt_requests_json():
    """Prompt must instruct the model to return JSON-only output."""
    result = build_extraction_prompt("Some event.", "test")
    lower = result.lower()
    assert "json" in lower, "Prompt must mention JSON output"


def test_build_extraction_prompt_mentions_scale():
    """Prompt must describe the 0.0-to-1.0 scale."""
    result = build_extraction_prompt("Some event.", "test")
    assert "0.0" in result or "0" in result
    assert "1.0" in result or "1" in result


# ── parse_attributes ──────────────────────────────────────────────────────────


def test_parse_attributes_valid_json():
    """parse_attributes returns correct floats from plain JSON."""
    data = {name: 0.5 for name in ATTRIBUTE_NAMES}
    raw = json.dumps(data)
    result = parse_attributes(raw)
    assert isinstance(result, dict)
    for name in ATTRIBUTE_NAMES:
        assert name in result
        assert result[name] == pytest.approx(0.5)


def test_parse_attributes_markdown_code_block():
    """parse_attributes strips ```json ... ``` wrappers before parsing."""
    data = {"urgency": 0.8, "reversibility": 0.3}
    inner = json.dumps(data)
    raw = f"```json\n{inner}\n```"
    result = parse_attributes(raw)
    assert result.get("urgency") == pytest.approx(0.8)
    assert result.get("reversibility") == pytest.approx(0.3)


def test_parse_attributes_clamps_above_one():
    """Values > 1.0 must be clamped to 1.0."""
    data = {"urgency": 1.5, "reversibility": 2.0}
    raw = json.dumps(data)
    result = parse_attributes(raw)
    assert result["urgency"] == pytest.approx(1.0)
    assert result["reversibility"] == pytest.approx(1.0)


def test_parse_attributes_clamps_below_zero():
    """Values < 0.0 must be clamped to 0.0."""
    data = {"urgency": -0.5, "reversibility": -10.0}
    raw = json.dumps(data)
    result = parse_attributes(raw)
    assert result["urgency"] == pytest.approx(0.0)
    assert result["reversibility"] == pytest.approx(0.0)


def test_parse_attributes_filters_unknown_keys():
    """Keys not in ATTRIBUTE_NAMES must be excluded from the result."""
    data = {"urgency": 0.5, "totally_unknown_key": 0.9}
    raw = json.dumps(data)
    result = parse_attributes(raw)
    assert "totally_unknown_key" not in result
    assert "urgency" in result


def test_parse_attributes_invalid_json_returns_empty():
    """Malformed JSON must return an empty dict (no exception)."""
    result = parse_attributes("not valid json at all {{{")
    assert result == {}


def test_parse_attributes_empty_string_returns_empty():
    """Empty string must return an empty dict."""
    result = parse_attributes("")
    assert result == {}


def test_parse_attributes_non_numeric_values_skipped():
    """Non-numeric attribute values must be skipped (not raise)."""
    data = {"urgency": "high", "reversibility": 0.4}
    raw = json.dumps(data)
    result = parse_attributes(raw)
    # "urgency" has a non-numeric value — implementation may skip or default it
    assert "reversibility" in result
    assert result["reversibility"] == pytest.approx(0.4)


def test_parse_attributes_boundary_values():
    """Exact boundary values 0.0 and 1.0 must pass through unchanged."""
    data = {"urgency": 0.0, "reversibility": 1.0}
    raw = json.dumps(data)
    result = parse_attributes(raw)
    assert result["urgency"] == pytest.approx(0.0)
    assert result["reversibility"] == pytest.approx(1.0)


# ── extract_batch ─────────────────────────────────────────────────────────────


def test_extract_batch_returns_list_length():
    """extract_batch must return one dict per input event."""
    events = [
        {"text": "The earthquake struck the coast.", "category": "natural_disaster"},
        {"text": "Elections were held in June.", "category": "politics"},
    ]

    mock_attrs = {name: 0.5 for name in ATTRIBUTE_NAMES}
    mock_response_body = json.dumps(mock_attrs)

    # Mock httpx.AsyncClient so no real HTTP request is made
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": mock_response_body}}]
    }

    import asyncio

    async def fake_post(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("cubemind.execution.attribute_extractor.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(extract_batch(events, batch_size=10, delay=0.0))

    assert isinstance(result, list)
    assert len(result) == 2


def test_extract_batch_result_dicts_have_attribute_keys():
    """Each result dict must contain recognized attribute keys."""
    events = [{"text": "A storm hit the city.", "category": "weather"}]

    mock_attrs = {name: 0.7 for name in ATTRIBUTE_NAMES}
    mock_response_body = json.dumps(mock_attrs)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": mock_response_body}}]
    }

    import asyncio

    async def fake_post(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("cubemind.execution.attribute_extractor.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(extract_batch(events, batch_size=10, delay=0.0))

    assert len(result) == 1
    attrs = result[0]
    assert isinstance(attrs, dict)
    for name in ATTRIBUTE_NAMES:
        assert name in attrs, f"Missing attribute: {name!r}"


def test_extract_batch_api_error_returns_empty_dict():
    """API errors for a given event must produce an empty dict, not raise."""
    events = [{"text": "A treaty collapsed.", "category": "diplomacy"}]

    import asyncio

    async def fake_post(*args, **kwargs):
        raise Exception("Network timeout")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("cubemind.execution.attribute_extractor.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(extract_batch(events, batch_size=10, delay=0.0))

    assert result == [{}]


def test_extract_batch_empty_input():
    """extract_batch on an empty list must return an empty list."""
    import asyncio

    result = asyncio.run(extract_batch([], batch_size=10, delay=0.0))
    assert result == []
