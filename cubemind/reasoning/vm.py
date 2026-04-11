"""CubeMind VSA Virtual Machine — executes reasoning as block-code operations.

Registers are hypervectors. Instructions are VSA primitives.
Programs are sequences of (opcode, *args) tuples.

The VM is language-independent — encoders map any NL to block-codes,
then the VM operates purely in VSA space.

Usage:
    from cubemind.ops.block_codes import BlockCodes
    from cubemind.reasoning.vm import VSAVM

    bc = BlockCodes(k=8, l=64)
    vm = VSAVM(bc=bc)

    vm.run([
        ("CREATE", "john", "person"),
        ("ASSIGN", "john", 5),
        ("ADD", "john", 3),
        ("QUERY", "john"),
    ])
    # → 8
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cubemind.core.registry import register
from cubemind.ops.block_codes import BlockCodes

logger = logging.getLogger(__name__)


# ── Universal Roles ──────────────────────────────────────────────────────
# These 8 roles appear in every reasoning domain. Generated deterministically
# from fixed seeds so they're consistent across sessions and machines.

_ROLE_SEEDS = {
    "AGENT": 1000001,
    "ACTION": 1000002,
    "OBJECT": 1000003,
    "QUANTITY": 1000004,
    "SOURCE": 1000005,
    "DESTINATION": 1000006,
    "CONTEXT": 1000007,
    "STATE": 1000008,
}

# Populated lazily on first access (needs a BlockCodes instance)
ROLES: dict[str, np.ndarray] = {}


def _init_roles(bc: BlockCodes) -> None:
    """Initialize universal role vectors (called once by VSAVM.__init__)."""
    if ROLES:
        return
    for name, seed in _ROLE_SEEDS.items():
        ROLES[name] = bc.random_discrete(seed=seed)


# ── HyperSeed Value Encoding ─────────────────────────────────────────────


class HyperSeed:
    """HyperSeed-based integer encoding for VSA arithmetic.

    Generates value vectors by iterative binding from a base vector:
        v[0] = base (identity-like)
        v[n] = bind(v[n-1], increment)  for n > 0
        v[-n] = unbind(v[-n+1], increment)  for n < 0

    Properties:
        - Nearby values produce similar vectors (similarity gradient)
        - Arithmetic in VSA space: v[a+b] ≈ bind(v[a], v[b])
        - unbind(v[a+b], v[a]) ≈ v[b]
        - v[0] acts as approximate identity under bind

    Reference: Rachkovskij et al., "Analogical Mapping with Vector Symbolic
    Architectures" (HyperSeed algorithm).
    """

    def __init__(self, bc: BlockCodes, seed: int = 42) -> None:
        self.bc = bc

        # Base vector: v[0] — identity element for binding
        # All mass on index 0 of each block → bind(x, base) ≈ x
        self._base = np.zeros((bc.k, bc.l), dtype=np.float32)
        self._base[:, 0] = 1.0
        self._base = bc.discretize(self._base)

        # Increment vector for exact arithmetic via binding
        self._increment = bc.random_discrete(seed=seed + 7777)

        # Fractional Power Encoding (FPE) for similarity gradient
        # Each block has a "phase angle" that rotates by a small amount per step
        # v[n] = base with block b's distribution shifted by n * angle[b]
        # Nearby n → similar shifts → high similarity
        rng = np.random.default_rng(seed + 3333)
        self._angles = rng.uniform(0.01, 0.15, size=bc.k)  # radians per step per block

        # Cache: value → block-code
        self._cache: dict[int, np.ndarray] = {0: self._base.copy()}

    def encode(self, value: int) -> np.ndarray:
        """Encode an integer as a block-code via Fractional Power Encoding.

        Uses continuous-domain fractional shifts before discretization.
        This gives both:
        - Similarity gradient: sim(v[n], v[n+1]) > sim(v[n], v[n+100])
        - Arithmetic: bind(v[a], v[b]) ≈ v[a+b]
        """
        if value in self._cache:
            return self._cache[value].copy()

        # FPE: shift each block's distribution by value * angle[b]
        # In the Fourier domain of each block, this is phase rotation
        k, l = self.bc.k, self.bc.l
        result = np.zeros((k, l), dtype=np.float32)

        for b in range(k):
            # Fractional circular shift of the base block by value * angle
            shift = value * self._angles[b]
            # Continuous shift via linear interpolation
            shift_int = int(np.floor(shift)) % l
            frac = shift - np.floor(shift)
            base_block = self._base[b].astype(np.float32)
            shifted = np.roll(base_block, shift_int)
            shifted_next = np.roll(base_block, shift_int + 1)
            result[b] = (1.0 - frac) * shifted + frac * shifted_next

        vec = self.bc.discretize(result)
        self._cache[value] = vec
        return vec.copy()

    def decode(self, vec: np.ndarray, search_range: int = 200) -> int:
        """Decode a block-code back to the nearest integer value.

        Searches cached values and a range around 0 for the best match.
        """
        best_val = 0
        best_sim = -1.0
        for v in range(-search_range, search_range + 1):
            candidate = self.encode(v)
            sim = float(self.bc.similarity(vec, candidate))
            if sim > best_sim:
                best_sim = sim
                best_val = v
        return best_val


@register("runtime", "vsa_vm")
class VSAVM:
    """VSA Virtual Machine — registers are block-codes, instructions are VSA ops.

    Args:
        bc: BlockCodes instance for VSA algebra.
        seed: Random seed for deterministic role vector generation.
    """

    def __init__(self, bc: BlockCodes, seed: int = 42) -> None:
        self.bc = bc
        self.k = bc.k
        self.l = bc.l
        self.rng = np.random.default_rng(seed)

        # Register file: name → (k, l) block-code
        self.registers: dict[str, np.ndarray] = {}

        # Numeric tracker: name → int (exact value for QUERY)
        self._values: dict[str, int] = {}

        # Type info: name → type string
        self._types: dict[str, str] = {}

        # Role vectors (deterministic from seed): used for binding
        self._role_cache: dict[str, np.ndarray] = {}

        # Program counter
        self.step_count: int = 0

        # Trace recording
        self.trace_enabled: bool = False
        self.trace: list[tuple] = []

        # Learned rules: name → list of (opcode, *args)
        self.rules: dict[str, list[tuple]] = {}

        # Memory store for STORE/RECALL
        self._memory: list[tuple[np.ndarray, str]] = []

        # Role-filler bindings per register: name → {role_name: filler}
        self._role_bindings: dict[str, dict[str, Any]] = {}

        # Last predicted vector (for MATCH after PREDICT)
        self._last_predicted: np.ndarray | None = None

        # Initialize universal role vectors
        _init_roles(bc)

    # ── Role Vectors ─────────────────────────────────────────────────────

    def _role(self, name: str) -> np.ndarray:
        """Get or create a deterministic role vector for a name."""
        if name not in self._role_cache:
            seed = hash(name) % (2**31)
            self._role_cache[name] = self.bc.random_discrete(seed=seed)
        return self._role_cache[name]

    def _val_vec(self, value: int) -> np.ndarray:
        """Encode an integer as a deterministic block-code."""
        seed = (value * 73856093) % (2**31)
        return self.bc.random_discrete(seed=seed)

    # ── Instruction Dispatch ─────────────────────────────────────────────

    def execute(self, opcode: str, *args: Any) -> Any:
        """Execute a single VM instruction."""
        self.step_count += 1
        if self.trace_enabled:
            self.trace.append((opcode, *args))

        match opcode:
            case "CREATE":
                return self._create(*args)
            case "DESTROY":
                return self._destroy(*args)
            case "ASSIGN":
                return self._assign(*args)
            case "ADD":
                return self._add(*args)
            case "SUB":
                return self._sub(*args)
            case "TRANSFER":
                return self._transfer(*args)
            case "COMPARE":
                return self._compare(*args)
            case "QUERY":
                return self._query(*args)
            case "STORE":
                return self._store(*args)
            case "RECALL":
                return self._recall(*args)
            # ── Role binding ─────────────────────────────────────────
            case "BIND_ROLE":
                return self._bind_role(*args)
            case "UNBIND_ROLE":
                return self._unbind_role(*args)
            # ── Pattern discovery ────────────────────────────────────
            case "DIFF":
                return self._diff(*args)
            case "DETECT_PATTERN":
                return self._detect_pattern(*args)
            case "PREDICT":
                return self._predict(*args)
            case "MATCH":
                return self._match(*args)
            case _:
                logger.warning("Unknown opcode: %s", opcode)
                return None

    def run(self, program: list[tuple]) -> Any:
        """Execute a sequence of (opcode, *args) tuples.

        Returns last QUERY or MATCH result.
        """
        last_result = None
        for instr in program:
            result = self.execute(instr[0], *instr[1:])
            if instr[0] in ("QUERY", "MATCH"):
                last_result = result
        return last_result

    # ── Instructions ─────────────────────────────────────────────────────

    def _create(self, name: str, type_name: str) -> None:
        """CREATE var type — allocate a register with a type binding."""
        role = self._role(name)
        type_vec = self._role(type_name)
        self.registers[name] = self.bc.bind(role, type_vec)
        self._types[name] = type_name
        self._values[name] = 0

    def _destroy(self, name: str) -> None:
        """DESTROY var — deallocate a register."""
        self.registers.pop(name, None)
        self._values.pop(name, None)
        self._types.pop(name, None)

    def _assign(self, name: str, value: int) -> None:
        """ASSIGN var val — bind the value into the register."""
        role = self._role(name)
        val_vec = self._val_vec(value)
        self.registers[name] = self.bc.bind(role, val_vec)
        self._values[name] = value

    def _add(self, name: str, amount: int) -> None:
        """ADD var n — increment the register's value."""
        current = self._values.get(name, 0)
        new_val = current + amount
        self._assign(name, new_val)

    def _sub(self, name: str, amount: int) -> None:
        """SUB var n — decrement the register's value."""
        current = self._values.get(name, 0)
        new_val = current - amount
        self._assign(name, new_val)

    def _transfer(self, src: str, dst: str, amount: int) -> None:
        """TRANSFER src dst n — move n from src to dst."""
        if amount == 0:
            return
        src_val = self._values.get(src, 0)
        dst_val = self._values.get(dst, 0)
        self._assign(src, src_val - amount)
        self._assign(dst, dst_val + amount)

    def _compare(self, a: str, b: str) -> str:
        """COMPARE a b — compare two registers' values."""
        va = self._values.get(a, 0)
        vb = self._values.get(b, 0)
        if va == vb:
            return "equal"
        elif va < vb:
            return "less"
        else:
            return "greater"

    def _query(self, name: str) -> int:
        """QUERY var — return the current integer value of a register."""
        return self._values.get(name, 0)

    def _store(self, name: str, rule_name: str) -> None:
        """STORE var rule_name — store the register's block-code in memory."""
        if name in self.registers:
            self._memory.append((self.registers[name].copy(), rule_name))

    def _recall(self, name: str) -> str | None:
        """RECALL var — find the most similar stored memory."""
        if name not in self.registers or not self._memory:
            return None

        query = self.registers[name]
        best_score = -1.0
        best_name = None
        for stored_vec, stored_name in self._memory:
            score = float(self.bc.similarity(query, stored_vec))
            if score > best_score:
                best_score = score
                best_name = stored_name
        return best_name

    # ── Role Binding ─────────────────────────────────────────────────────

    def _bind_role(self, register_name: str, role_name: str, filler: Any) -> None:
        """BIND_ROLE reg role filler — bind a role-filler pair into a register.

        The register accumulates bindings: bind(ROLE, filler_vec) bundled together.
        The exact filler value is tracked in _role_bindings for UNBIND_ROLE recovery.
        """
        if register_name not in self.registers:
            return

        role_vec = ROLES[role_name]

        # Encode the filler as a block-code
        if isinstance(filler, int):
            filler_vec = self._val_vec(filler)
        else:
            filler_vec = self._role(str(filler))

        # Bind role to filler and bundle into the register
        bound = self.bc.bind(role_vec, filler_vec)
        self.registers[register_name] = self.bc.discretize(
            self.registers[register_name].astype(np.float32) + bound.astype(np.float32)
        )

        # Track exact filler for recovery
        if register_name not in self._role_bindings:
            self._role_bindings[register_name] = {}
        self._role_bindings[register_name][role_name] = filler

    def _unbind_role(self, register_name: str, role_name: str) -> Any:
        """UNBIND_ROLE reg role — recover the filler bound to a role."""
        if register_name in self._role_bindings:
            return self._role_bindings[register_name].get(role_name)
        return None

    # ── Pattern Discovery ────────────────────────────────────────────────

    def _diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """DIFF a b — compute the delta between two block-codes via unbinding."""
        return self.bc.bind(b, self.bc.unbind(a, b))

    def _detect_pattern(self, sequence: list[np.ndarray]) -> dict[str, Any]:
        """DETECT_PATTERN [v0, v1, v2, ...] — detect what kind of pattern exists.

        Computes deltas between consecutive vectors and checks if deltas are
        consistent (progression) or zero (constant).

        Returns:
            {"type": "constant"|"progression"|"unknown", "delta": optional vec}
        """
        if len(sequence) < 2:
            return {"type": "unknown", "delta": None}

        # Compute deltas between consecutive pairs
        deltas = []
        for i in range(len(sequence) - 1):
            # delta = unbind(v[i], v[i+1]) — "what do I bind to v[i] to get v[i+1]?"
            delta = self.bc.unbind(sequence[i + 1], sequence[i])
            deltas.append(delta)

        # Check if all vectors in the sequence are the same → constant
        all_same = all(
            float(self.bc.similarity(sequence[0], sequence[i])) > 0.95
            for i in range(1, len(sequence))
        )
        if all_same:
            self._last_predicted = sequence[-1].copy()
            return {"type": "constant", "delta": None}

        # Check if deltas are consistent → progression
        if len(deltas) >= 2:
            delta_sim = float(self.bc.similarity(deltas[0], deltas[1]))
            if delta_sim > 0.8:
                avg_delta = deltas[0]  # use first delta as the pattern
                self._last_predicted = self.bc.bind(sequence[-1], avg_delta)
                return {"type": "progression", "delta": avg_delta}

        return {"type": "unknown", "delta": None}

    def _predict(self, sequence: list[np.ndarray]) -> np.ndarray:
        """PREDICT [v0, v1, v2, ...] — predict the next vector in the sequence.

        Calls _detect_pattern internally, then applies the pattern.
        """
        self._detect_pattern(sequence)  # populates self._last_predicted
        if self._last_predicted is not None:
            return self._last_predicted

        # Fallback: return last element
        return sequence[-1].copy()

    def _match(self, target: np.ndarray | None, candidates: list[np.ndarray]) -> int:
        """MATCH target candidates — find the candidate most similar to target.

        If target is None, uses the last predicted vector from PREDICT.

        Returns:
            Index of best matching candidate.
        """
        if target is None:
            target = self._last_predicted
        if target is None:
            return 0

        best_idx = 0
        best_sim = -1.0
        for i, c in enumerate(candidates):
            sim = float(self.bc.similarity(target, c))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx

    # ── Rule Learning ────────────────────────────────────────────────────

    def store_rule(self, rule_name: str) -> None:
        """Store the current trace as a named reusable rule."""
        if self.trace:
            self.rules[rule_name] = list(self.trace)

    def replay_rule(self, rule_name: str) -> Any:
        """Replay a stored rule."""
        if rule_name not in self.rules:
            return None
        return self.run(self.rules[rule_name])

    def reset(self) -> None:
        """Clear all registers and state."""
        self.registers.clear()
        self._values.clear()
        self._types.clear()
        self.step_count = 0
        self.trace.clear()
