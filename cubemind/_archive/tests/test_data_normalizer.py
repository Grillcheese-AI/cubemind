"""Tests for cubemind.execution.data_normalizer.

Validates:
  - UnifiedEvent dataclass fields and defaults
  - normalize_historical() maps records correctly
  - normalize_nyt() maps records correctly
  - select_test_events() filters and sorts records
"""

from __future__ import annotations

import json

import pytest

from cubemind.execution.data_normalizer import (
    UnifiedEvent,
    normalize_historical,
    normalize_nyt,
    select_test_events,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def historical_records() -> list[dict]:
    return [
        {
            "event_id": "hist_001",
            "summary": "The fall of the Western Roman Empire.",
            "earliest_date_year": 476.0,
            "entities": json.dumps(
                {"PERSON": ["Romulus Augustulus"], "GPE": ["Rome"]}
            ),
            "category": "political",
            "source_ref": "encyclopedia_a",
            "precursor_events": ["hist_000"],
            "confidence": 0.92,
            "attributes": {"significance": 0.95},
            "sentiment": "negative",
            "sentiment_score": -0.6,
        },
        {
            "event_id": "hist_002",
            "source_text": "Magna Carta signed.",
            "earliest_date_year": 1215.0,
            "entities": "{}",
            "category": "",
            "source_ref": "encyclopedia_b",
            "precursor_events": [],
            "confidence": 0.88,
            "attributes": {},
            "sentiment": "positive",
            "sentiment_score": 0.4,
        },
        {
            "event_id": "hist_003",
            "summary": "",
            "source_text": "",
            "earliest_date_year": None,
            "entities": "not-valid-json",  # malformed — should default to {}
            "category": "science",
            "source_ref": "",
            "precursor_events": ["hist_001", "hist_002"],
            "confidence": 0.5,
            "attributes": {},
            "sentiment": "",
            "sentiment_score": 0.0,
        },
    ]


@pytest.fixture
def nyt_records() -> list[dict]:
    return [
        {
            "_id": "nyt_001",
            "headline": {"main": "Tech Giant Acquires Startup"},
            "abstract": "Abstract text.",
            "snippet": "Snippet text.",
            "pub_date": "2023-05-15T12:00:00Z",
            "section_name": "Business Day",
            "keywords": [
                {"name": "persons", "value": "Jane Doe"},
                {"name": "organizations", "value": "TechCorp"},
                {"name": "glocations", "value": "New York"},
                {"name": "subject", "value": "Mergers"},
            ],
            "source_ref": "nytimes.com/2023/05/15/business",
        },
        {
            "_id": "nyt_002",
            "headline": {},
            "abstract": "Fallback abstract text.",
            "snippet": "Fallback snippet.",
            "pub_date": "2024-11-01T08:30:00+00:00",
            "section_name": "World News",
            "keywords": [],
            "source_ref": "",
        },
        {
            "_id": "nyt_003",
            "headline": {"main": ""},
            "abstract": "",
            "snippet": "Only snippet available.",
            "pub_date": "2022-01-20",
            "section_name": "",
            "keywords": [
                {"name": "persons", "value": "John Smith"},
                {"name": "persons", "value": "Alice Brown"},
            ],
            "source_ref": "nytimes.com/2022",
        },
    ]


@pytest.fixture
def select_records() -> list[dict]:
    return [
        {
            "event_id": "e1",
            "summary": "Important event with precursors.",
            "precursor_events": ["e0"],
            "confidence": 0.9,
        },
        {
            "event_id": "e2",
            "summary": "Another event with many precursors.",
            "precursor_events": ["e0", "e1"],
            "confidence": 0.95,
        },
        {
            "event_id": "e3",
            "summary": "No precursors — should be filtered out.",
            "precursor_events": [],
            "confidence": 0.99,
        },
        {
            "event_id": "e4",
            "summary": "",  # empty summary — should be filtered out
            "precursor_events": ["e0"],
            "confidence": 0.85,
        },
        {
            "event_id": "e5",
            "summary": "Low confidence but valid.",
            "precursor_events": ["e0"],
            "confidence": 0.7,
        },
    ]


# ── UnifiedEvent defaults ─────────────────────────────────────────────────────


def test_unified_event_defaults():
    """UnifiedEvent should initialise with sensible defaults."""
    ev = UnifiedEvent(event_id="x", text="hello")
    assert ev.year is None
    assert ev.date is None
    assert ev.category == ""
    assert ev.entities == {}
    assert ev.source_type == ""
    assert ev.source_ref == ""
    assert ev.precursors == []
    assert ev.attributes == {}
    assert ev.confidence == 0.0
    assert ev.sentiment == ""
    assert ev.sentiment_score == 0.0


def test_unified_event_mutable_defaults_are_independent():
    """Mutable defaults must not be shared across instances."""
    a = UnifiedEvent(event_id="a", text="a")
    b = UnifiedEvent(event_id="b", text="b")
    a.precursors.append("x")
    assert b.precursors == [], "Mutable default leak detected"


# ── normalize_historical ──────────────────────────────────────────────────────


def test_normalize_historical_returns_list_of_unified_events(historical_records):
    result = normalize_historical(historical_records)
    assert isinstance(result, list)
    assert all(isinstance(e, UnifiedEvent) for e in result)
    assert len(result) == len(historical_records)


def test_normalize_historical_source_type(historical_records):
    result = normalize_historical(historical_records)
    for ev in result:
        assert ev.source_type == "historical"


def test_normalize_historical_event_id(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].event_id == "hist_001"
    assert result[1].event_id == "hist_002"


def test_normalize_historical_summary_mapped_to_text(historical_records):
    """summary field should be used as text."""
    result = normalize_historical(historical_records)
    assert result[0].text == "The fall of the Western Roman Empire."


def test_normalize_historical_source_text_fallback(historical_records):
    """source_text used when summary is absent."""
    result = normalize_historical(historical_records)
    assert result[1].text == "Magna Carta signed."


def test_normalize_historical_year_float_to_int(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].year == 476
    assert isinstance(result[0].year, int)
    assert result[1].year == 1215


def test_normalize_historical_year_none_when_missing(historical_records):
    result = normalize_historical(historical_records)
    assert result[2].year is None


def test_normalize_historical_entities_parsed(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].entities == {"PERSON": ["Romulus Augustulus"], "GPE": ["Rome"]}


def test_normalize_historical_entities_empty_dict_on_malformed_json(historical_records):
    """Malformed entity JSON should produce an empty dict, not raise."""
    result = normalize_historical(historical_records)
    assert result[2].entities == {}


def test_normalize_historical_precursors(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].precursors == ["hist_000"]
    assert result[1].precursors == []


def test_normalize_historical_confidence(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].confidence == pytest.approx(0.92)


def test_normalize_historical_sentiment(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].sentiment == "negative"
    assert result[0].sentiment_score == pytest.approx(-0.6)


def test_normalize_historical_attributes(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].attributes == {"significance": 0.95}


def test_normalize_historical_category(historical_records):
    result = normalize_historical(historical_records)
    assert result[0].category == "political"


def test_normalize_historical_empty_input():
    assert normalize_historical([]) == []


# ── normalize_nyt ─────────────────────────────────────────────────────────────


def test_normalize_nyt_returns_list_of_unified_events(nyt_records):
    result = normalize_nyt(nyt_records)
    assert isinstance(result, list)
    assert all(isinstance(e, UnifiedEvent) for e in result)
    assert len(result) == len(nyt_records)


def test_normalize_nyt_source_type(nyt_records):
    result = normalize_nyt(nyt_records)
    for ev in result:
        assert ev.source_type == "nyt"


def test_normalize_nyt_event_id(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[0].event_id == "nyt_001"


def test_normalize_nyt_headline_main_as_text(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[0].text == "Tech Giant Acquires Startup"


def test_normalize_nyt_text_fallback_abstract(nyt_records):
    """When headline.main is absent, use abstract."""
    result = normalize_nyt(nyt_records)
    assert result[1].text == "Fallback abstract text."


def test_normalize_nyt_text_fallback_snippet(nyt_records):
    """When headline.main is empty and abstract is empty, use snippet."""
    result = normalize_nyt(nyt_records)
    assert result[2].text == "Only snippet available."


def test_normalize_nyt_pub_date_year(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[0].year == 2023
    assert isinstance(result[0].year, int)
    assert result[1].year == 2024


def test_normalize_nyt_pub_date_date_field(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[0].date == "2023-05-15"
    assert result[1].date == "2024-11-01"
    assert result[2].date == "2022-01-20"


def test_normalize_nyt_keyword_mapping(nyt_records):
    result = normalize_nyt(nyt_records)
    assert "PERSON" in result[0].entities
    assert "Jane Doe" in result[0].entities["PERSON"]
    assert "ORG" in result[0].entities
    assert "TechCorp" in result[0].entities["ORG"]
    assert "GPE" in result[0].entities
    assert "New York" in result[0].entities["GPE"]


def test_normalize_nyt_keywords_unknown_name_excluded(nyt_records):
    """Keywords with name 'subject' should not appear as entity keys."""
    result = normalize_nyt(nyt_records)
    assert "subject" not in result[0].entities


def test_normalize_nyt_multiple_same_keyword_type(nyt_records):
    """Multiple persons should all appear in PERSON list."""
    result = normalize_nyt(nyt_records)
    assert set(result[2].entities.get("PERSON", [])) == {"John Smith", "Alice Brown"}


def test_normalize_nyt_empty_keywords(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[1].entities == {}


def test_normalize_nyt_section_name_to_category(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[0].category == "business_day"
    assert result[1].category == "world_news"


def test_normalize_nyt_empty_section_name(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[2].category == ""


def test_normalize_nyt_source_ref(nyt_records):
    result = normalize_nyt(nyt_records)
    assert result[0].source_ref == "nytimes.com/2023/05/15/business"


def test_normalize_nyt_empty_input():
    assert normalize_nyt([]) == []


# ── select_test_events ────────────────────────────────────────────────────────


def test_select_test_events_filters_no_precursors(select_records):
    """Records without precursors must be excluded."""
    result = select_test_events(select_records)
    ids = [r["event_id"] for r in result]
    assert "e3" not in ids


def test_select_test_events_filters_empty_summary(select_records):
    """Records with empty summary must be excluded."""
    result = select_test_events(select_records)
    ids = [r["event_id"] for r in result]
    assert "e4" not in ids


def test_select_test_events_sorted_by_confidence_descending(select_records):
    result = select_test_events(select_records)
    confidences = [r["confidence"] for r in result]
    assert confidences == sorted(confidences, reverse=True)


def test_select_test_events_top_n(select_records):
    result = select_test_events(select_records, n=2)
    assert len(result) == 2
    assert result[0]["event_id"] == "e2"
    assert result[1]["event_id"] == "e1"


def test_select_test_events_default_n_caps_at_available(select_records):
    """When fewer valid records exist than n, return all valid records."""
    result = select_test_events(select_records, n=1000)
    # Valid: e1, e2, e5 (e3 no precursors, e4 empty summary)
    assert len(result) == 3


def test_select_test_events_min_precursors_2(select_records):
    """min_precursors=2 should only include records with >=2 precursors."""
    result = select_test_events(select_records, min_precursors=2)
    ids = [r["event_id"] for r in result]
    assert ids == ["e2"]


def test_select_test_events_returns_dicts(select_records):
    result = select_test_events(select_records)
    assert all(isinstance(r, dict) for r in result)


def test_select_test_events_empty_input():
    assert select_test_events([]) == []


def test_select_test_events_n_zero():
    records = [{"event_id": "x", "summary": "s", "precursor_events": ["y"], "confidence": 1.0}]
    assert select_test_events(records, n=0) == []
