"""CubeMind Module Registry — discover and swap components by name.

Each component registers itself with a role and name:

    @register("encoder", "bio_vision")
    class BioVisionEncoder:
        ...

    @register("encoder", "cnn")
    class CNNEncoder:
        ...

Then instantiate by name:

    encoder_cls = registry.get("encoder", "bio_vision")
    encoder = encoder_cls(k=80, l=128)

Or list all available implementations for a role:

    registry.list("encoder")
    # ["bio_vision", "cnn", "harrier", "text"]

The registry is global and populated at import time via decorators.
The DI container (functional/registry.py) uses this to resolve string config
values like {"encoder": "bio_vision"} into actual classes.
"""

from __future__ import annotations

from typing import Any, Type


class _ModuleRegistry:
    """Global registry: role → name → class."""

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, Type]] = {}

    def add(self, role: str, name: str, cls: Type) -> None:
        """Register a class under a role and name."""
        if role not in self._registry:
            self._registry[role] = {}
        self._registry[role][name] = cls

    def get(self, role: str, name: str) -> Type:
        """Get a registered class by role and name.

        Raises:
            KeyError: If role or name not found.
        """
        if role not in self._registry:
            raise KeyError(f"Unknown role: {role!r}. Available: {list(self._registry.keys())}")
        if name not in self._registry[role]:
            raise KeyError(
                f"Unknown {role}: {name!r}. Available: {list(self._registry[role].keys())}"
            )
        return self._registry[role][name]

    def list(self, role: str) -> list[str]:
        """List all registered names for a role."""
        return list(self._registry.get(role, {}).keys())

    def roles(self) -> list[str]:
        """List all registered roles."""
        return list(self._registry.keys())

    def all(self) -> dict[str, list[str]]:
        """Return full registry map: {role: [names]}."""
        return {role: list(names.keys()) for role, names in self._registry.items()}

    def create(self, role: str, name: str, **kwargs: Any) -> Any:
        """Shortcut: get class and instantiate with kwargs."""
        cls = self.get(role, name)
        return cls(**kwargs)

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._registry.values())
        return f"<ModuleRegistry: {total} modules across {len(self._registry)} roles>"


# Global singleton
registry = _ModuleRegistry()


def register(role: str, name: str | None = None):
    """Decorator: register a class in the module registry.

    Args:
        role: Component role — "encoder", "memory", "router", "expert",
              "detector", "executor", "estimator", "loss", "optimizer", etc.
        name: Registration name. Defaults to the class name in snake_case.

    Usage:
        @register("encoder", "bio_vision")
        class BioVisionEncoder:
            ...

        @register("encoder")  # auto-name: "cnn_encoder"
        class CNNEncoder:
            ...
    """
    def decorator(cls: Type) -> Type:
        reg_name = name or _to_snake_case(cls.__name__)
        registry.add(role, reg_name, cls)
        cls._registry_role = role
        cls._registry_name = reg_name
        return cls
    return decorator


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            result.append("_")
        result.append(ch.lower())
    return "".join(result)
