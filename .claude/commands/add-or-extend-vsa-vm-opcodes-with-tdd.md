---
name: add-or-extend-vsa-vm-opcodes-with-tdd
description: Workflow command scaffold for add-or-extend-vsa-vm-opcodes-with-tdd in cubemind.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-extend-vsa-vm-opcodes-with-tdd

Use this workflow when working on **add-or-extend-vsa-vm-opcodes-with-tdd** in `cubemind`.

## Goal

Adds new opcodes to the VSA Virtual Machine, implements their logic, and adds/updates extensive TDD tests to validate functionality.

## Common Files

- `cubemind/reasoning/vm.py`
- `tests/test_vm.py`
- `tests/test_vm_discover.py`
- `tests/test_vm_raven.py`
- `cubemind/reasoning/vm.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or extend 'cubemind/reasoning/vm.py' to implement new opcode(s) and their logic.
- Update or add corresponding TDD tests in 'tests/test_vm.py' (and/or 'tests/test_vm_discover.py', 'tests/test_vm_raven.py' for specialized behaviors).
- Update opcode documentation and/or architecture docs (e.g., 'cubemind/reasoning/vm.md', '.claude/plan/architecture-map.md') if needed.
- Commit both implementation and tests together.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.