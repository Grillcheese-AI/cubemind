"""Canonical CubeMind VSA-VM opcode registry.

Mirrors the enum order in ``opcode-vsa-rs/src/ir.rs::CubeMindOpcode`` so
the multitask opcode head's class IDs index directly into the same
opcode set the Rust VM dispatches on.

Two indexing schemes coexist in the codebase:

  1. **Bytecode** (``ir.rs::bytecode()``): non-contiguous u8 in [0x00..0xFF],
     stable across versions for serialization. Sum=0x34, Skip=0xFF, etc.
  2. **Position-based class ID** (this module): contiguous u32 in [0..N).
     Used by the multitask opcode head where logits are a softmax over
     N classes — non-contiguous indices would waste rows.

We use **position-based** IDs throughout the training pipeline. A
separate map is provided for translating to/from bytecode if/when the
trained head's predictions are sent to the Rust VM.

Also exposes ``normalize_opcode`` which folds common variants
(``BINDROLE``, ``BindRole``, ``bind_role``) to the canonical
``BIND_ROLE`` form before lookup, so noisy data sources still validate.
"""

from __future__ import annotations

import re

# ── Canonical opcode list — ORDER MATTERS, matches ir.rs enum order ────
#
# Adding an opcode: append to this list AND add it to opcode-vsa-rs's
# CubeMindOpcode enum AND cubelang/src/vm.rs AND cubemind/reasoning/vm.py
# (the four-source opcode sync rule from CLAUDE.md). Never insert in the
# middle — that breaks every saved checkpoint's head weights.

CANONICAL_OPCODES: tuple[str, ...] = (
    # Register lifecycle
    "CREATE", "DESTROY",
    # Arithmetic
    "ASSIGN", "ADD", "SUB", "MUL", "DIV", "SUM",
    # Data movement
    "TRANSFER", "COPY", "PUSH", "POP",
    # Comparison / query
    "COMPARE", "QUERY",
    # Memory
    "STORE", "RECALL",
    # Role binding
    "BIND_ROLE", "UNBIND_ROLE",
    # Control flow
    "COND", "LOOP", "CALL", "JMP", "LABEL",
    # Sequence
    "SEQ", "UNSEQ",
    # Pattern discovery
    "DIFF", "DETECT_PATTERN", "PREDICT", "MATCH",
    # Reasoning
    "DEBATE", "ASK",
    # Rule discovery
    "DISCOVER", "DISCOVER_SEQUENCE",
    # Cleanup / memory
    "CLEANUP", "REMEMBER", "FORGET",
    # Decode / score
    "DECODE", "SCORE",
    # World / specialisation
    "SPECIALIZE",
    # Bandit exploration
    "EXPLORE", "REWARD",
    # JIT (MindForge)
    "FORGE", "FORGE_ALL",
    # Extended inference / temporal / parallel
    "INFER", "BROADCAST", "SYNC", "MERGE", "SPLIT", "FILTER",
    "MAP_ROLES", "REDUCE", "TEMPORAL_BIND", "ANALOGY",
    # Type inference (HM Algorithm J)
    "UNIFY", "INST", "GEN", "NEWVAR",
    # No-op
    "SKIP",
)

NUM_OPCODES: int = len(CANONICAL_OPCODES)  # 58 as of vm.md April 2026

# Forward + reverse maps for fast lookup.
OPCODE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_OPCODES)}
ID_TO_OPCODE: dict[int, str] = {i: name for i, name in enumerate(CANONICAL_OPCODES)}


# ── Name normalization ──────────────────────────────────────────────────

_CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")

# Compact / no-underscore variants seen in real data → canonical form.
# Order matters: longer keys first so e.g. ``UNBINDROLE`` doesn't get
# mangled by the ``BINDROLE`` rule.
_COMPACT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("DISCOVERSEQUENCE", "DISCOVER_SEQUENCE"),
    ("DETECTPATTERN",    "DETECT_PATTERN"),
    ("TEMPORALBIND",     "TEMPORAL_BIND"),
    ("UNBINDROLE",       "UNBIND_ROLE"),
    ("BINDROLE",         "BIND_ROLE"),
    ("MAPROLES",         "MAP_ROLES"),
    ("FORGEALL",         "FORGE_ALL"),
)


def normalize_opcode(name: str) -> str | None:
    """Fold a noisy opcode name into the canonical form, or return None
    if it doesn't map to any known opcode.

    Accepts ``BindRole``, ``bind_role``, ``BINDROLE``, ``BIND_ROLE``,
    ``BIND ROLE`` — all collapse to ``BIND_ROLE``.
    """
    if not name:
        return None
    s = str(name).strip()
    # Camel/Pascal → snake, but ONLY if the string has at least one
    # lowercase letter. Applying this to SCREAMING_CASE inputs mangles
    # them (BIND_ROLE → B_I_N_D___R_O_L_E).
    if any(c.islower() for c in s):
        s = _CAMEL_TO_SNAKE.sub("_", s)
    s = s.replace(" ", "_").upper()
    # Collapse runs of underscores and trim.
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    # Apply compact-variant table for things that lost their underscores.
    for compact, canonical in _COMPACT_VARIANTS:
        if s == compact:
            s = canonical
            break
    return s if s in OPCODE_TO_ID else None


def opcode_id(name: str) -> int | None:
    """Normalize ``name`` and return its position-based class ID, or
    None if the name isn't a recognised opcode."""
    canonical = normalize_opcode(name)
    if canonical is None:
        return None
    return OPCODE_TO_ID[canonical]


__all__ = [
    "CANONICAL_OPCODES",
    "NUM_OPCODES",
    "OPCODE_TO_ID",
    "ID_TO_OPCODE",
    "normalize_opcode",
    "opcode_id",
]
