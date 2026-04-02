"""CubeMind Component Registry — DI via python-dependency-injector.

Uses dependency-injector for the heavy lifting (containers, providers,
wiring). CubeMind adds:
- @component / @observer / @emits decorators for auto-registration
- Auto-ID assignment on all components
- Observer pattern wiring at boot time
- Lifecycle tracking

Usage:
    from cubemind.functional.registry import CubeMindContainer, component, observer

    # Define container
    class MySystem(CubeMindContainer):
        config = providers.Configuration()
        expert = providers.Factory(MyExpert, d_input=config.d_input)
        router = providers.Singleton(MyRouter, n_experts=config.n_experts)
        memory = providers.Singleton(HippocampalFormation, feature_dim=config.d_input)

    # Boot
    container = MySystem()
    container.config.from_dict({"d_input": 32, "n_experts": 4})
    container.boot()

    # Use
    expert = container.expert()
    router = container.router()
"""

from __future__ import annotations

import uuid
import functools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

from dependency_injector import containers, providers
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════


class CubeMindContainer(containers.DeclarativeContainer):
    """Base container for CubeMind systems.

    Extends dependency-injector's DeclarativeContainer with:
    - Auto-ID assignment
    - Observer wiring
    - Lifecycle management
    - Boot/shutdown hooks
    """

    config = providers.Configuration()

    def boot(self) -> Dict[str, Any]:
        """Boot all components: assign IDs, wire observers, pre-flight.

        Returns:
            Boot report dict.
        """
        report = {"components": 0, "observers_wired": 0, "warnings": []}

        # Walk all providers
        for name, provider in self.providers.items():
            if name == "config":
                continue
            report["components"] += 1

            # Try to resolve and assign ID
            try:
                instance = provider()
                if not hasattr(instance, "_component_id"):
                    instance._component_id = f"{name}_{uuid.uuid4().hex[:6]}"
                logger.debug("Booted {} as {}", type(instance).__name__,
                             instance._component_id)
            except Exception as e:
                report["warnings"].append(f"{name}: {e}")

        # Wire observers
        wired = self._wire_observers()
        report["observers_wired"] = wired

        logger.info("Container booted: {} components, {} observers",
                     report["components"], report["observers_wired"])
        return report

    def _wire_observers(self) -> int:
        """Wire @observer classes to @emits components."""
        wired = 0
        observers = []
        emitters = []

        for name, provider in self.providers.items():
            if name == "config":
                continue
            try:
                instance = provider()
            except Exception:
                continue

            if hasattr(instance, "_observes"):
                observers.append(instance)
            if hasattr(instance, "_emits") or hasattr(instance, "_observers"):
                emitters.append(instance)

        for obs in observers:
            for event in obs._observes:
                handler = getattr(obs, event, None) or getattr(obs, f"on_{event}", None)
                if handler is None:
                    continue
                for emitter in emitters:
                    if emitter is obs:
                        continue
                    if not hasattr(emitter, "_observers"):
                        emitter._observers = defaultdict(list)
                    emitter._observers[event].append(handler)
                    wired += 1

        return wired

    def shutdown(self) -> None:
        """Graceful shutdown."""
        for name, provider in self.providers.items():
            if name == "config":
                continue
            try:
                instance = provider()
                if hasattr(instance, "shutdown"):
                    instance.shutdown()
            except Exception:
                pass
        logger.info("Container shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATORS (thin wrappers — metadata only)
# ═══════════════════════════════════════════════════════════════════════════════


def component(role: str, singleton: bool = False) -> Callable:
    """Mark a class as a registerable component with a role.

    Args:
        role: Component role ("expert", "router", "memory", etc.)
        singleton: If True, DI container should use providers.Singleton.
    """
    def decorator(cls: Type) -> Type:
        cls._component_role = role
        cls._component_singleton = singleton
        cls._component_id = None
        cls._observers = defaultdict(list)

        def emit(self, event: str, **data):
            for callback in self._observers.get(event, []):
                try:
                    callback(source=self, event=event, **data)
                except Exception as e:
                    logger.warning("Observer error on {}: {}", event, e)

        cls.emit = emit
        return cls
    return decorator


def observer(*events: str) -> Callable:
    """Mark a class as an event observer.

    Args:
        events: Event names to subscribe to.
    """
    def decorator(cls: Type) -> Type:
        cls._observes = list(events)
        cls._component_role = getattr(cls, "_component_role", "observer")
        cls._component_id = None
        return cls
    return decorator


def emits(*events: str) -> Callable:
    """Declare which events a component emits (for documentation + wiring)."""
    def decorator(cls: Type) -> Type:
        cls._emits = list(events)
        if not hasattr(cls, "_observers"):
            cls._observers = defaultdict(list)
        return cls
    return decorator


def singleton(cls: Type) -> Type:
    """Class decorator: ensure only one instance exists."""
    instances = {}

    @functools.wraps(cls, updated=())
    class SingletonWrapper(cls):
        def __new__(klass, *args, **kwargs):
            if cls not in instances:
                instances[cls] = super().__new__(klass)
            return instances[cls]

    SingletonWrapper._singleton_cls = cls
    return SingletonWrapper


def dataset(name: str, source: str = "", split: str = "train",
            max_samples: int | None = None) -> Callable:
    """Mark a class as a dataset definition with metadata."""
    def decorator(cls: Type) -> Type:
        cls._dataset_name = name
        cls._source = source
        cls._split = split
        cls._max_samples = max_samples
        _datasets[name] = cls
        return cls
    return decorator


# Dataset registry
_datasets: dict[str, Type] = {}


def get_dataset(name: str, **kwargs) -> Any:
    """Instantiate a registered dataset by name."""
    if name not in _datasets:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(_datasets.keys())}")
    return _datasets[name](**kwargs)
