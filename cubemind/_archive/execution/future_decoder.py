"""
FutureDecoder — block-code attribute dict to natural language description.

Template-based MVP decoder that maps the 32 attribute scores to descriptive
text.  Production will replace this with a local LLM call; the interface is
stable.

Part of the Decision Oracle pipeline (Task 8).
"""

from __future__ import annotations

# ── Descriptor map ────────────────────────────────────────────────────────────
# Each attribute maps to a list of (threshold, descriptor) tuples.
# The descriptor for the *first* threshold that the score is <= to is used.

_DESCRIPTORS: dict[str, list[tuple[float, str]]] = {
    "reversibility": [
        (0.3, "irreversible"),
        (0.7, "partially reversible"),
        (1.0, "fully reversible"),
    ],
    "agency_type": [
        (0.25, "human-driven"),
        (0.5, "institutional"),
        (0.75, "systemic"),
        (1.0, "natural"),
    ],
    "chain_length_potential": [
        (0.3, "contained"),
        (0.7, "cascading"),
        (1.0, "massive chain reaction"),
    ],
    "cause_certainty": [
        (0.3, "speculative"),
        (0.7, "likely"),
        (1.0, "near-certain"),
    ],
    "effect_magnitude": [
        (0.3, "minor"),
        (0.7, "significant"),
        (1.0, "civilization-altering"),
    ],
    "urgency": [
        (0.3, "gradual"),
        (0.7, "pressing"),
        (1.0, "immediate and critical"),
    ],
    "duration": [
        (0.3, "short-lived"),
        (0.7, "sustained"),
        (1.0, "permanent"),
    ],
    "geographic_scope": [
        (0.25, "local"),
        (0.5, "regional"),
        (0.75, "continental"),
        (1.0, "global"),
    ],
    "population_scale": [
        (0.3, "small group"),
        (0.7, "large population"),
        (1.0, "billions affected"),
    ],
    "tension_level": [
        (0.3, "calm"),
        (0.7, "tense"),
        (1.0, "peak crisis"),
    ],
    "stakes": [
        (0.3, "low stakes"),
        (0.7, "high stakes"),
        (1.0, "existential"),
    ],
    "moral_complexity": [
        (0.3, "clear-cut"),
        (0.7, "nuanced"),
        (1.0, "deeply ambiguous"),
    ],
    "pivotality": [
        (0.3, "incremental"),
        (0.7, "pivotal"),
        (1.0, "historic turning point"),
    ],
}

_DEFAULT_DESCRIPTION = "An outcome with moderate impact across multiple dimensions."

# Sentence slots: which attribute fills which role in the composed sentence.
# Structure: scope+magnitude → urgency+duration → chain+certainty → stakes+tension → pivotality
_SLOT_ORDER: list[tuple[str, str]] = [
    # (slot_name, attribute_key)
    ("scope", "geographic_scope"),
    ("magnitude", "effect_magnitude"),
    ("urgency", "urgency"),
    ("duration", "duration"),
    ("chain", "chain_length_potential"),
    ("certainty", "cause_certainty"),
    ("stakes", "stakes"),
    ("tension", "tension_level"),
    ("pivotality", "pivotality"),
]


def _describe(attr: str, value: float) -> str | None:
    """Return the descriptor string for a given attribute and score value.

    Returns None if the attribute is not in _DESCRIPTORS.
    """
    thresholds = _DESCRIPTORS.get(attr)
    if thresholds is None:
        return None
    for threshold, label in thresholds:
        if value <= threshold:
            return label
    # Fallback to the last label (value > all thresholds, shouldn't happen if 1.0 is included)
    return thresholds[-1][1]


class FutureDecoder:
    """Template-based decoder from attribute dicts to natural language.

    Converts a dict of attribute scores (floats in [0, 1]) into a concise
    natural-language sentence describing the predicted outcome.  Attributes
    not present in the input are omitted from the description; if no
    recognised attributes are provided, a default fallback sentence is
    returned.

    This is the MVP template path.  Production will delegate to a local LLM.
    """

    def decode(self, attributes: dict[str, float]) -> str:
        """Convert an attribute dict to a natural-language description.

        Args:
            attributes: Mapping of attribute name → float score in [0, 1].
                        Unknown keys are silently ignored.

        Returns:
            A descriptive sentence.  Falls back to a default string when no
            recognised attributes are present.
        """
        # Resolve descriptors for each slot that has a value in the input dict
        parts: dict[str, str] = {}
        for slot_name, attr_key in _SLOT_ORDER:
            value = attributes.get(attr_key)
            if value is None:
                continue
            label = _describe(attr_key, float(value))
            if label is not None:
                parts[slot_name] = label

        if not parts:
            return _DEFAULT_DESCRIPTION

        # Compose sentence from available slots
        return self._compose(parts)

    # ── Sentence composition ──────────────────────────────────────────────────

    def _compose(self, parts: dict[str, str]) -> str:
        """Assemble slot-resolved descriptors into a readable sentence.

        Fills the template clauses in order; omits clauses whose slots are
        not present.
        """
        clauses: list[str] = []

        # Clause 1 — Scope + magnitude
        scope = parts.get("scope")
        magnitude = parts.get("magnitude")
        if scope and magnitude:
            clauses.append(f"A {magnitude} {scope} outcome")
        elif scope:
            clauses.append(f"A {scope} outcome")
        elif magnitude:
            clauses.append(f"A {magnitude} outcome")

        # Clause 2 — Urgency + duration
        urgency = parts.get("urgency")
        duration = parts.get("duration")
        if urgency and duration:
            clauses.append(f"unfolding in a {urgency}, {duration} manner")
        elif urgency:
            clauses.append(f"unfolding {urgency}")
        elif duration:
            clauses.append(f"with {duration} effects")

        # Clause 3 — Chain + certainty
        chain = parts.get("chain")
        certainty = parts.get("certainty")
        if chain and certainty:
            clauses.append(f"with {certainty} {chain} consequences")
        elif chain:
            clauses.append(f"with {chain} consequences")
        elif certainty:
            clauses.append(f"with {certainty} cause")

        # Clause 4 — Stakes + tension
        stakes = parts.get("stakes")
        tension = parts.get("tension")
        if stakes and tension:
            clauses.append(f"carrying {stakes} under {tension} conditions")
        elif stakes:
            clauses.append(f"carrying {stakes}")
        elif tension:
            clauses.append(f"under {tension} conditions")

        # Clause 5 — Pivotality
        pivotality = parts.get("pivotality")
        if pivotality:
            clauses.append(f"representing a {pivotality} moment")

        if not clauses:
            return _DEFAULT_DESCRIPTION

        # Join clauses naturally
        sentence = ", ".join(clauses)
        # Capitalise first character and terminate with a period
        if sentence and not sentence[0].isupper():
            sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith("."):
            sentence += "."
        return sentence
