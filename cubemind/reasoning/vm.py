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
            case _:
                logger.warning("Unknown opcode: %s", opcode)
                return None

    def run(self, program: list[tuple]) -> Any:
        """Execute a sequence of (opcode, *args) tuples. Returns last QUERY result."""
        last_result = None
        for instr in program:
            result = self.execute(instr[0], *instr[1:])
            if instr[0] == "QUERY":
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
