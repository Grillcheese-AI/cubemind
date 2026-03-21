"""DataNormalizer — UnifiedEvent dataclass and source adapters.

Converts heterogeneous event records from historical and NYT sources into a
canonical UnifiedEvent representation for use in the Decision Oracle causal
training pipeline.

Pipeline position: Layer 1 of 4 — raw JSON → UnifiedEvent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class UnifiedEvent:
    """Canonical normalized event record for the Decision Oracle pipeline.

    Args:
        event_id:       Unique identifier for the event.
        text:           Primary human-readable description of the event.
        year:           Four-digit year (int), or None if unknown.
        date:           ISO 8601 date string (YYYY-MM-DD), or None if unknown.
        category:       Thematic category (lowercase, underscored).
        entities:       Named-entity map — keys are NER labels (PERSON, ORG,
                        GPE, …), values are lists of entity strings.
        source_type:    Origin of the record: "historical" | "nyt" | "corpus"
                        | "movie" | "knowledge".
        source_ref:     Opaque reference string to the originating document.
        precursors:     event_id strings of causal precursor events.
        attributes:     Numeric attribute bag (e.g. {"significance": 0.95}).
        confidence:     Normalised confidence score in [0, 1].
        sentiment:      Coarse sentiment label ("positive" | "negative" | "").
        sentiment_score: Fine-grained sentiment value in [-1, 1].
    """

    event_id: str
    text: str
    year: int | None = None
    date: str | None = None
    category: str = ""
    entities: dict[str, list[str]] = field(default_factory=dict)
    source_type: str = ""
    source_ref: str = ""
    precursors: list[str] = field(default_factory=list)
    attributes: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    sentiment: str = ""
    sentiment_score: float = 0.0


# ── keyword name → NER label mapping for NYT records ─────────────────────────

_NYT_KEYWORD_MAP: dict[str, str] = {
    "persons": "PERSON",
    "organizations": "ORG",
    "glocations": "GPE",
}


# ── Source adapters ───────────────────────────────────────────────────────────


def normalize_historical(records: list[dict]) -> list[UnifiedEvent]:
    """Convert historical event JSON records to UnifiedEvent.

    Mapping rules:
    - text:       ``summary`` field; falls back to ``source_text`` when empty.
    - year:       ``earliest_date_year`` cast from float → int (None if absent).
    - entities:   JSON-decoded from ``entities`` string field; empty dict on error.
    - source_type is always ``"historical"``.

    Args:
        records: Raw historical event dicts.

    Returns:
        List of UnifiedEvent in the same order as *records*.
    """
    result: list[UnifiedEvent] = []
    for rec in records:
        # ── text ──────────────────────────────────────────────────────────────
        text = rec.get("summary") or rec.get("source_text") or ""

        # ── year ──────────────────────────────────────────────────────────────
        raw_year = rec.get("earliest_date_year")
        year: int | None = int(raw_year) if raw_year is not None else None

        # ── entities ──────────────────────────────────────────────────────────
        entities_raw = rec.get("entities", "{}")
        try:
            entities: dict[str, list[str]] = json.loads(entities_raw)
        except (json.JSONDecodeError, TypeError):
            entities = {}

        result.append(
            UnifiedEvent(
                event_id=rec.get("event_id", ""),
                text=text,
                year=year,
                category=rec.get("category", ""),
                entities=entities,
                source_type="historical",
                source_ref=rec.get("source_ref", ""),
                precursors=list(rec.get("precursor_events", [])),
                attributes=dict(rec.get("attributes", {})),
                confidence=float(rec.get("confidence", 0.0)),
                sentiment=rec.get("sentiment", ""),
                sentiment_score=float(rec.get("sentiment_score", 0.0)),
            )
        )
    return result


def normalize_nyt(records: list[dict]) -> list[UnifiedEvent]:
    """Convert NYT article JSON records to UnifiedEvent.

    Mapping rules:
    - event_id:   ``_id`` field.
    - text:       ``headline.main``; falls back to ``abstract`` then ``snippet``.
    - year / date: parsed from ``pub_date`` ISO string (YYYY-MM-DD).
    - entities:   built from ``keywords`` list using :data:`_NYT_KEYWORD_MAP`;
                  unknown keyword names are silently dropped.
    - category:   ``section_name`` lowercased with spaces replaced by ``_``.
    - source_type is always ``"nyt"``.

    Args:
        records: Raw NYT article dicts (as returned by the Article Search API).

    Returns:
        List of UnifiedEvent in the same order as *records*.
    """
    result: list[UnifiedEvent] = []
    for rec in records:
        # ── text ──────────────────────────────────────────────────────────────
        headline = rec.get("headline") or {}
        text = headline.get("main") or rec.get("abstract") or rec.get("snippet") or ""

        # ── date / year ───────────────────────────────────────────────────────
        pub_date: str = rec.get("pub_date") or ""
        date: str | None = None
        year: int | None = None
        if pub_date:
            # ISO strings may have a time component: "2023-05-15T12:00:00Z"
            date_part = pub_date[:10]  # always YYYY-MM-DD
            date = date_part
            try:
                year = int(date_part[:4])
            except (ValueError, IndexError):
                year = None

        # ── entities from keywords ────────────────────────────────────────────
        entities: dict[str, list[str]] = {}
        for kw in rec.get("keywords") or []:
            kw_name = kw.get("name", "")
            ner_label = _NYT_KEYWORD_MAP.get(kw_name)
            if ner_label is None:
                continue
            value = kw.get("value", "")
            if value:
                entities.setdefault(ner_label, []).append(value)

        # ── category ─────────────────────────────────────────────────────────
        section = rec.get("section_name") or ""
        category = section.lower().replace(" ", "_") if section else ""

        result.append(
            UnifiedEvent(
                event_id=rec.get("_id", ""),
                text=text,
                year=year,
                date=date,
                category=category,
                entities=entities,
                source_type="nyt",
                source_ref=rec.get("source_ref", ""),
            )
        )
    return result


def select_test_events(
    records: list[dict],
    n: int = 1000,
    min_precursors: int = 1,
) -> list[dict]:
    """Select the top *n* records that are suitable for causal training.

    A record is eligible when it has:
    - at least *min_precursors* entries in its ``precursor_events`` list, and
    - a non-empty ``summary`` field.

    Eligible records are sorted by ``confidence`` descending before slicing.

    Args:
        records:        Raw event dicts (pre-normalisation).
        n:              Maximum number of records to return.
        min_precursors: Minimum length of ``precursor_events``.

    Returns:
        Up to *n* filtered dicts ordered by confidence descending.
    """
    eligible = [
        r
        for r in records
        if len(r.get("precursor_events") or []) >= min_precursors
        and (r.get("summary") or "").strip()
    ]
    eligible.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    return eligible[:n]
