"""
AttributeExtractor — OpenRouter LLM extraction of 32 structured attributes.

Extracts causal, temporal, impact, counterfactual, narrative, and semantic
attributes from event text using an LLM via the OpenRouter API.

Part of the causal oracle training pipeline (Task 3).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Attribute names (32 total) ────────────────────────────────────────────────

ATTRIBUTE_NAMES: list[str] = [
    # Causality (1–6)
    "reversibility",
    "agency_type",
    "chain_length_potential",
    "cause_certainty",
    "effect_magnitude",
    "feedback_loop",
    # Temporal (7–12)
    "urgency",
    "duration",
    "periodicity",
    "temporal_distance",
    "decay_rate",
    "recurrence_risk",
    # Impact (13–18)
    "geographic_scope",
    "population_scale",
    "domain_breadth",
    "institutional_depth",
    "economic_weight",
    "cultural_reach",
    # Counterfactual (19–24)
    "n_plausible_alternatives",
    "pivotality",
    "contingency",
    "path_dependence",
    "fragility",
    "determinism_score",
    # Narrative (25–30)
    "tension_level",
    "stakes",
    "resolution_type",
    "moral_complexity",
    "perspective_count",
    "ambiguity",
    # Semantic (31–32)
    "category_score",
    "event_type_score",
]

_ATTRIBUTE_SET: frozenset[str] = frozenset(ATTRIBUTE_NAMES)

# ── Attribute descriptions with scale anchors ─────────────────────────────────

_ATTRIBUTE_DESCRIPTIONS: dict[str, str] = {
    # Causality
    "reversibility": (
        "How reversible are the effects? 0.0 = completely irreversible (death, "
        "extinction), 1.0 = fully reversible (temporary setback)."
    ),
    "agency_type": (
        "Degree of intentional human agency. 0.0 = purely natural/accidental, "
        "1.0 = deliberate planned action."
    ),
    "chain_length_potential": (
        "How many causal steps this event could trigger. 0.0 = isolated event, "
        "1.0 = long chain of cascading consequences."
    ),
    "cause_certainty": (
        "Confidence in the identified cause. 0.0 = highly ambiguous/unknown, "
        "1.0 = definitively established cause."
    ),
    "effect_magnitude": (
        "Scale of effects produced. 0.0 = negligible effect, "
        "1.0 = catastrophic or transformative effect."
    ),
    "feedback_loop": (
        "Degree of self-reinforcing feedback. 0.0 = no feedback, "
        "1.0 = strong positive or negative feedback loop."
    ),
    # Temporal
    "urgency": (
        "How time-sensitive is a response? 0.0 = no urgency (decades to act), "
        "1.0 = immediate action required (hours)."
    ),
    "duration": (
        "How long the event/effects persist. 0.0 = momentary (seconds/minutes), "
        "1.0 = permanent or multi-generational."
    ),
    "periodicity": (
        "How regularly the event recurs. 0.0 = unique one-time event, "
        "1.0 = highly regular periodic recurrence."
    ),
    "temporal_distance": (
        "Distance from the present. 0.0 = happening now, "
        "1.0 = distant past or far future."
    ),
    "decay_rate": (
        "How quickly impacts diminish over time. 0.0 = no decay (permanent), "
        "1.0 = very rapid decay (fades within days)."
    ),
    "recurrence_risk": (
        "Probability of recurrence. 0.0 = extremely unlikely to recur, "
        "1.0 = near-certain to recur."
    ),
    # Impact
    "geographic_scope": (
        "Geographic extent of impact. 0.0 = local/neighborhood, "
        "1.0 = global/planetary."
    ),
    "population_scale": (
        "Number of people affected. 0.0 = individual/family, "
        "1.0 = entire global population."
    ),
    "domain_breadth": (
        "Number of domains (social, economic, political, etc.) affected. "
        "0.0 = single narrow domain, 1.0 = all major societal domains."
    ),
    "institutional_depth": (
        "Depth of impact on institutions and governance. "
        "0.0 = no institutional impact, 1.0 = fundamental institutional restructuring."
    ),
    "economic_weight": (
        "Economic significance. 0.0 = negligible economic impact, "
        "1.0 = global economic transformation."
    ),
    "cultural_reach": (
        "Influence on culture, norms, or values. "
        "0.0 = no cultural impact, 1.0 = paradigm-shifting cultural transformation."
    ),
    # Counterfactual
    "n_plausible_alternatives": (
        "How many plausible alternative outcomes existed. "
        "0.0 = only one possible outcome, 1.0 = many equally plausible alternatives."
    ),
    "pivotality": (
        "How pivotal/decisive this event is. 0.0 = marginal event that changes little, "
        "1.0 = turning-point event that changes everything."
    ),
    "contingency": (
        "How contingent on chance/circumstance. 0.0 = inevitable regardless of circumstances, "
        "1.0 = highly contingent on a specific unlikely conjunction."
    ),
    "path_dependence": (
        "Degree to which prior history locks in outcomes. "
        "0.0 = outcome independent of history, 1.0 = strongly path-dependent."
    ),
    "fragility": (
        "How fragile/sensitive to small perturbations. "
        "0.0 = robust and stable, 1.0 = extremely fragile (butterfly-effect sensitive)."
    ),
    "determinism_score": (
        "How deterministic vs. random the outcome. "
        "0.0 = highly random/stochastic, 1.0 = fully deterministic."
    ),
    # Narrative
    "tension_level": (
        "Narrative tension and conflict intensity. "
        "0.0 = no tension (calm/routine), 1.0 = extreme conflict or crisis."
    ),
    "stakes": (
        "What is at stake. 0.0 = trivial stakes, "
        "1.0 = existential stakes (civilization/species-level)."
    ),
    "resolution_type": (
        "How clearly resolved the event is. "
        "0.0 = completely unresolved/ongoing, 1.0 = fully resolved with clear outcome."
    ),
    "moral_complexity": (
        "Degree of moral ambiguity. "
        "0.0 = clear moral valence (obviously good or bad), "
        "1.0 = deeply morally complex with no clear right answer."
    ),
    "perspective_count": (
        "Number of distinct stakeholder perspectives. "
        "0.0 = single uniform perspective, 1.0 = many conflicting perspectives."
    ),
    "ambiguity": (
        "Overall interpretive ambiguity. "
        "0.0 = clear unambiguous meaning, 1.0 = highly ambiguous/contested meaning."
    ),
    # Semantic
    "category_score": (
        "Confidence that the event fits its stated category. "
        "0.0 = very poor fit, 1.0 = perfect category match."
    ),
    "event_type_score": (
        "How prototypical this event is for its event type. "
        "0.0 = highly atypical, 1.0 = archetypal example of this event type."
    ),
}

# ── Prompt builder ────────────────────────────────────────────────────────────

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "minimax/minimax-m2.7"


def build_extraction_prompt(text: str, category: str) -> str:
    """Build an LLM prompt for structured attribute extraction.

    Asks the model to score all 32 attributes from 0.0 to 1.0 for the
    given event text and category. The prompt includes attribute descriptions
    with scale anchors and requests JSON-only output.

    Args:
        text: The event text to score.
        category: The event category (e.g. "natural_disaster", "politics").

    Returns:
        Formatted prompt string ready to send to an LLM.
    """
    attr_block_lines: list[str] = []
    for name in ATTRIBUTE_NAMES:
        desc = _ATTRIBUTE_DESCRIPTIONS.get(name, "Score from 0.0 to 1.0.")
        attr_block_lines.append(f'  "{name}": <float 0.0–1.0>  # {desc}')
    attr_block = "\n".join(attr_block_lines)

    return (
        f"You are an expert event analyst. Score the following event on 32 structured "
        f"attributes. Each attribute must be scored as a float from 0.0 to 1.0.\n\n"
        f"Event category: {category}\n\n"
        f"Event text:\n{text}\n\n"
        f"Score each attribute below. Return ONLY a JSON object with exactly these 32 keys "
        f"and float values between 0.0 and 1.0. Do not include any explanation, markdown, "
        f"or text outside the JSON object.\n\n"
        f"Required JSON schema (return only the JSON):\n"
        f"{{\n{attr_block}\n}}"
    )


# ── Response parser ───────────────────────────────────────────────────────────


def parse_attributes(raw: str) -> dict[str, float]:
    """Parse LLM JSON response into a validated attribute dict.

    Handles markdown code-block wrappers (```json ... ```), filters to
    recognized attribute names, clamps all values to [0.0, 1.0], and
    returns an empty dict on any parse error.

    Args:
        raw: Raw string response from the LLM.

    Returns:
        Dict mapping recognized attribute names to clamped float values.
        Returns {} on invalid or empty input.
    """
    if not raw or not raw.strip():
        return {}

    text = raw.strip()

    # Strip markdown code-block wrappers: ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # Try direct parse first
    data: Any = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Model may wrap JSON in reasoning text — extract first { ... } block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass
    # Handle truncated JSON — salvage key-value pairs via regex
    if data is None:
        pairs = re.findall(r'"(\w+)"\s*:\s*([\d.]+)', text)
        if pairs:
            data = {}
            for k, v in pairs:
                try:
                    data[k] = float(v)
                except ValueError:
                    pass
    if data is None:
        logger.warning("parse_attributes: no valid JSON found, returning empty dict")
        return {}

    if not isinstance(data, dict):
        logger.warning("parse_attributes: JSON root is not a dict, returning empty dict")
        return {}

    result: dict[str, float] = {}
    for key, value in data.items():
        if key not in _ATTRIBUTE_SET:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            logger.debug("parse_attributes: skipping non-numeric value for %r: %r", key, value)
            continue
        result[key] = max(0.0, min(1.0, fval))

    return result


# ── Batch extraction ──────────────────────────────────────────────────────────


async def extract_batch(
    events: list[dict[str, str]],
    model: str = _DEFAULT_MODEL,
    batch_size: int = 10,
    delay: float = 1.0,
) -> list[dict[str, float]]:
    """Extract 32 structured attributes for a batch of events via OpenRouter.

    Calls the OpenRouter chat completions API (one request per event),
    processes events in chunks of batch_size with a delay between chunks
    to avoid rate-limiting. API errors for individual events return an
    empty dict rather than raising.

    Args:
        events: List of dicts, each with 'text' and 'category' keys.
        model: OpenRouter model identifier. Defaults to tongyi-deepresearch-30b-a3b.
        batch_size: Number of events to process per chunk before sleeping.
        delay: Seconds to sleep between batch chunks.

    Returns:
        List of attribute dicts (one per input event), preserving order.
        Events that fail due to API errors return an empty dict.
    """
    if not events:
        return []

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results: list[dict[str, float]] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for chunk_start in range(0, len(events), batch_size):
            chunk = events[chunk_start : chunk_start + batch_size]

            for event in chunk:
                text = event.get("text", "")
                category = event.get("category", "")
                prompt = build_extraction_prompt(text, category)

                payload = {
                    "model": "x-ai/grok-4.20-multi-agent-beta",
                    "messages": [{"role": "user", "content": prompt}],
                    
                    "max_tokens": 2048,
                }

                try:
                    response = await client.post(
                        _OPENROUTER_URL,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    body = response.json()
                    
                    raw_content: str = body["choices"][0]["message"]["content"]
                    
                    attrs = parse_attributes(raw_content)
                except Exception as exc:
                    logger.warning(
                        "extract_batch: API error for event %r: %s", text[:60], exc
                    )
                    attrs = {}

                print(attrs)

                results.append(attrs)

            # Delay between chunks (skip after the final chunk)
            if chunk_start + batch_size < len(events) and delay > 0:
                await asyncio.sleep(delay)

    return results
