```markdown
# cubemind Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the core development patterns, coding conventions, and workflows used in the `cubemind` Python codebase. The repository focuses on reasoning systems, virtual machine opcodes, and neural training pipelines, with a strong emphasis on test-driven development (TDD) and maintainable architecture documentation. You'll learn how to extend the VSA Virtual Machine, iterate on training pipelines, wire CLI commands, and keep documentation in sync with the codebase.

---

## Coding Conventions

**File Naming**

- Use `snake_case` for Python files and modules.
    - Example: `vm_discover.py`, `harrier_pretrain.py`

**Import Style**

- Use absolute imports.
    - Example:
      ```python
      from cubemind.reasoning.vm import VSAVirtualMachine
      ```

**Export Style**

- Default Python export style (no explicit `__all__` unless needed).

**Commit Patterns**

- Freeform commit messages, sometimes prefixed (e.g., `fix`, `build_temporal_corpus`).
- Average commit message length: ~49 characters.
- Implementation and tests are often committed together.

---

## Workflows

### Add or Extend VSA-VM Opcodes with TDD
**Trigger:** When you want to extend the reasoning capabilities of the VSA Virtual Machine by adding new instructions.  
**Command:** `/add-vm-opcode`

1. Edit or extend `cubemind/reasoning/vm.py` to implement new opcode(s) and their logic.
2. Update or add corresponding TDD tests in:
    - `tests/test_vm.py`
    - `tests/test_vm_discover.py`
    - `tests/test_vm_raven.py` (for specialized behaviors)
3. Update opcode documentation and/or architecture docs if needed:
    - `cubemind/reasoning/vm.md`
    - `.claude/plan/architecture-map.md`
4. Commit both implementation and tests together.

**Example:**
```python
# In cubemind/reasoning/vm.py
def opcode_NEWOP(self, arg):
    # Implement new opcode logic here
    pass
```
```python
# In tests/test_vm.py
def test_newop_opcode():
    vm = VSAVirtualMachine()
    vm.execute('NEWOP', 42)
    assert vm.last_result == expected_value
```

---

### TDD Feature Development Cycle
**Trigger:** When adding a new core feature or module with robust test coverage.  
**Command:** `/tdd-feature`

1. Implement the new feature/module in the appropriate `cubemind/` subdirectory.
2. Write or update corresponding tests in `tests/` (e.g., `test_vm.py`, `test_mindforge_vm.py`).
3. Iterate until all tests pass.
4. Commit both implementation and tests in the same commit.

**Example:**
```python
# In cubemind/mindforge/new_module.py
def new_feature(x):
    return x * 2
```
```python
# In tests/test_mindforge_vm.py
def test_new_feature():
    assert new_feature(3) == 6
```

---

### Training Pipeline Iteration and Fix
**Trigger:** When improving training stability, performance, or adding new capabilities to a training pipeline.  
**Command:** `/fix-training`

1. Edit the relevant training script in `cubemind/training/` (e.g., `vsa_lm.py`, `harrier_pretrain.py`).
2. Adjust loss functions, optimizer logic, or data pipeline as needed.
3. Test the new training logic (with real runs or tests).
4. Commit the changes, often with a detailed message about the fix or improvement.

**Example:**
```python
# In cubemind/training/vsa_lm.py
loss = custom_loss_fn(outputs, targets)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

### CLI Wiring and Argument Passing
**Trigger:** When adding a new CLI command or ensuring CLI arguments are correctly passed to backend logic.  
**Command:** `/wire-cli`

1. Edit `cubemind/__main__.py` to add or update CLI commands or argument parsing.
2. Wire the CLI arguments to the appropriate training/inference scripts (e.g., `cubemind/training/vsa_lm.py`).
3. Test CLI usage to ensure new arguments are respected.
4. Commit changes.

**Example:**
```python
# In cubemind/__main__.py
import argparse
from cubemind.training import vsa_lm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

vsa_lm.train(epochs=args.epochs)
```

---

### Documentation and Architecture Update
**Trigger:** When documenting new system capabilities, opcode tables, or architectural changes.  
**Command:** `/update-docs`

1. Edit or create documentation files:
    - `cubemind/reasoning/vm.md`
    - `.claude/plan/architecture-map.md`
    - `CLAUDE.md`
2. Summarize new features, opcodes, integration maps, or design decisions.
3. Commit documentation updates, sometimes alongside code changes.

**Example:**
```markdown
# VSA Virtual Machine Opcodes

## NEWOP
- Description: Performs a new operation on the VM stack.
- Arguments: int
```

---

## Testing Patterns

- Tests are primarily written in Python, typically in the `tests/` directory.
- Test file naming: `test_*.py` (e.g., `test_vm.py`, `test_vm_discover.py`).
- TDD is the norm: implement tests alongside or before new features.
- Some legacy or auxiliary tests may use `.test.ts` (TypeScript), but Python is primary.
- Example test:
    ```python
    def test_opcode_add():
        vm = VSAVirtualMachine()
        vm.execute('ADD', 1, 2)
        assert vm.last_result == 3
    ```

---

## Commands

| Command        | Purpose                                                        |
|----------------|----------------------------------------------------------------|
| /add-vm-opcode | Add or extend VSA-VM opcodes with TDD and update documentation |
| /tdd-feature   | Implement new features/modules with test-driven development     |
| /fix-training  | Iterate and fix training pipelines or add new capabilities      |
| /wire-cli      | Wire new features or arguments into the CLI entrypoint          |
| /update-docs   | Update documentation and architecture maps                     |
```
