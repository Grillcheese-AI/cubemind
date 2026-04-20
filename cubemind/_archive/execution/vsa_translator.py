"""
VSATranslator — probe specialist world models to generate English descriptions.

The "Particle Accelerator": smash known semantic concepts into specialist vectors
and observe what comes out.  For each concept in the codebook, bind it with the
specialist vector and ask which codebook entry the result most closely resembles.

If the best-matching output concept equals the input concept (similarity above the
constant threshold), the specialist acts as an *identity* / *constant* rule for
that concept.  Otherwise it is a *transform* (the input concept maps to a different
concept).

Used after WorldManager training to turn discovered specialist vectors into
human-readable descriptions, e.g. "economic collapse pattern" or
"military escalation".
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes


class VSATranslator:
    """Translate algebraic specialist world-model vectors into English descriptions.

    Probes each concept in a shared codebook through a specialist vector to
    determine what transformation the specialist encodes.

    Args:
        bc:       BlockCodes instance (determines k, l).
        codebook: Mapping from concept name to block-code vector of shape (k, l).
    """

    # Similarity threshold above which input == output is considered "constant"
    _CONSTANT_THRESHOLD: float = 0.85

    def __init__(self, bc: BlockCodes, codebook: dict[str, np.ndarray]) -> None:
        self.bc = bc
        self.codebook = codebook

        # Pre-stack codebook into an array (n, k, l) for efficient batch lookup
        self._names: list[str] = list(codebook.keys())
        self._vectors: np.ndarray = np.stack(
            [codebook[n] for n in self._names], axis=0
        )  # (n, k, l)

    # ── Public interface ──────────────────────────────────────────────────────

    def probe(self, specialist: np.ndarray, input_concept: str) -> dict:
        """Bind an input concept with the specialist and find the best codebook match.

        The operation is:
            probed = bind(codebook[input_concept], specialist)
            best   = argmax similarity(probed, codebook[*])

        Args:
            specialist:    Block-code vector (k, l) representing a world rule.
            input_concept: Name of the concept to push through the specialist.
                           Must be a key in ``self.codebook``.

        Returns:
            Dict with keys:
              - "input":      str — the input_concept name.
              - "name":       str — best-matching codebook concept name.
              - "similarity": float — similarity of the probed vector to the match.

        Raises:
            KeyError: If input_concept is not in the codebook.
        """
        concept_vec = self.codebook[input_concept]
        probed = self.bc.bind(concept_vec, specialist)
        similarities = self.bc.similarity_batch(probed, self._vectors)  # (n,)
        best_idx = int(np.argmax(similarities))
        return {
            "input": input_concept,
            "name": self._names[best_idx],
            "similarity": float(similarities[best_idx]),
        }

    def translate(self, specialist: np.ndarray) -> dict:
        """Probe every codebook concept through the specialist and build a description.

        For each concept:
          - Bind it with the specialist.
          - Find the best-matching concept in the codebook.
          - Classify as "constant" if input == output (similarity >= threshold),
            or "transform" if the specialist maps it to a different concept.

        Args:
            specialist: Block-code vector (k, l) representing a world rule.

        Returns:
            Dict with keys:
              - "probes":  list[dict] — one probe result per codebook concept, each
                           containing "input", "name", "similarity", and "kind"
                           ("transform" | "constant").
              - "summary": str — human-readable English description of the specialist.
        """
        probes: list[dict] = []
        transforms: list[str] = []
        constants: list[str] = []

        for concept_name in self._names:
            probe_result = self.probe(specialist, concept_name)
            output_name = probe_result["name"]
            is_constant = (
                output_name == concept_name
                and probe_result["similarity"] >= self._CONSTANT_THRESHOLD
            )
            kind = "constant" if is_constant else "transform"
            probe_result["kind"] = kind
            probes.append(probe_result)

            if kind == "constant":
                constants.append(concept_name)
            else:
                transforms.append(f"{concept_name} → {output_name}")

        summary = self._build_summary(transforms, constants)
        return {"probes": probes, "summary": summary}

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(transforms: list[str], constants: list[str]) -> str:
        """Compose a human-readable one-line summary from probe results.

        Args:
            transforms: List of "input → output" strings for non-identity mappings.
            constants:  List of concept names that map to themselves.

        Returns:
            Non-empty English description string.
        """
        parts: list[str] = []

        if transforms:
            parts.append("transforms: " + ", ".join(transforms))
        if constants:
            parts.append("constants: " + ", ".join(constants))

        if not parts:
            return "no mappings detected"
        return "; ".join(parts)
