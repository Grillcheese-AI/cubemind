"""CubeMind decorators — method + class decorators, all self-aware.

Method decorators (handle self automatically):
    @gpu_fallback  — try grilly GPU, fall back to numpy
    @timed         — measure execution time, attach to result
    @logged        — log calls via loguru
    @cached        — cache result by input hash

Class decorators:
    @configurable  — auto-generate __init__ from Config dataclass
    @grilly_layer  — register class as a grilly-compatible layer
    @trackable     — add stats() and n_uses tracking

Factory:
    layer(type, **kwargs) — resolve from grilly.nn → cubemind modules
"""

from __future__ import annotations

import functools
import importlib
import inspect
import time
from typing import Any, Callable, Type

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD DECORATORS (handle self for bound methods)
# ═══════════════════════════════════════════════════════════════════════════════


def gpu_fallback(fn: Callable) -> Callable:
    """Try grilly GPU op matching fn.__name__, fall back to numpy.

    Works on both free functions and bound methods (handles self).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Separate self from args if this is a bound method
        is_method = args and hasattr(args[0], fn.__name__)
        gpu_args = args[1:] if is_method else args

        try:
            from grilly.backend import _bridge
            if _bridge is not None and _bridge.is_available():
                gpu_fn = getattr(_bridge, fn.__name__, None)
                if gpu_fn is not None:
                    try:
                        result = gpu_fn(*gpu_args, **kwargs)
                        if result is not None:
                            return result
                    except Exception:
                        pass
        except ImportError:
            pass
        return fn(*args, **kwargs)
    return wrapper


def timed(fn: Callable) -> Callable:
    """Measure execution time. Attaches elapsed_ms to dict results.

    Works on both free functions and methods.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if isinstance(result, dict):
            result["elapsed_ms"] = elapsed_ms
        elif hasattr(result, "__dict__"):
            result.elapsed_ms = elapsed_ms
        return result
    return wrapper


def logged(fn: Callable) -> Callable:
    """Log calls via loguru. Shows class.method name for bound methods."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            from loguru import logger
            # Detect class name from self
            if args and hasattr(args[0], "__class__"):
                cls_name = args[0].__class__.__name__
                name = f"{cls_name}.{fn.__name__}"
            else:
                name = fn.__name__
            kwarg_str = ", ".join(f"{k}={v}" for k, v in list(kwargs.items())[:3])
            logger.debug("{name}({kw})", name=name, kw=kwarg_str)
        except ImportError:
            pass
        return fn(*args, **kwargs)
    return wrapper


def cached(fn: Callable) -> Callable:
    """Cache result by hashing numpy array inputs. LRU style.

    Works on methods — caches per-instance via self._cache dict.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Build cache key from args
        key_parts = []
        for a in args:
            if isinstance(a, np.ndarray):
                key_parts.append(hash(a.data.tobytes()))
            elif hasattr(a, "__hash__"):
                try:
                    key_parts.append(hash(a))
                except TypeError:
                    key_parts.append(id(a))
            else:
                key_parts.append(id(a))
        for k, v in sorted(kwargs.items()):
            key_parts.append(hash((k, str(v))))
        cache_key = tuple(key_parts)

        # Get or create cache on self (for methods) or on function (for free fns)
        if args and hasattr(args[0], "_cache"):
            cache = args[0]._cache
        elif not hasattr(wrapper, "_fn_cache"):
            wrapper._fn_cache = {}
            cache = wrapper._fn_cache
        else:
            cache = wrapper._fn_cache

        if cache_key in cache:
            return cache[cache_key]

        result = fn(*args, **kwargs)
        cache[cache_key] = result
        return result
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════


def configurable(config_class: Type) -> Callable:
    """Class decorator: auto-generate __init__ from a Config dataclass.

    Usage:
        @dataclass
        class ExpertConfig:
            d_input: int = 32
            d_output: int = 32
            seed: int = 42

        @configurable(ExpertConfig)
        class MyExpert:
            def forward(self, x): ...

        # Creates: MyExpert(d_input=32, d_output=32, seed=42)
        # Stores: self.config = ExpertConfig(...)
    """
    def decorator(cls: Type) -> Type:
        original_init = cls.__init__ if hasattr(cls, "__init__") else None

        def new_init(self, **kwargs):
            # Filter kwargs to only config fields
            import dataclasses
            field_names = {f.name for f in dataclasses.fields(config_class)}
            config_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
            extra_kwargs = {k: v for k, v in kwargs.items() if k not in field_names}
            self.config = config_class(**config_kwargs)
            if original_init and original_init is not object.__init__:
                original_init(self, **extra_kwargs)

        cls.__init__ = new_init
        cls._config_class = config_class
        return cls
    return decorator


def trackable(cls: Type) -> Type:
    """Class decorator: add usage tracking and stats().

    Adds: self.n_uses, self.n_forward, self._track_use(), self.stats()
    Wraps forward() to auto-increment counters.
    """
    original_init = cls.__init__ if hasattr(cls, "__init__") else None
    original_forward = cls.forward if hasattr(cls, "forward") else None

    def new_init(self, *args, **kwargs):
        if original_init:
            original_init(self, *args, **kwargs)
        self.n_uses = 0
        self.n_forward = 0
        self._cache = {}

    def tracked_forward(self, *args, **kwargs):
        self.n_forward += 1
        return original_forward(self, *args, **kwargs)

    def stats(self) -> dict:
        base = {"n_uses": self.n_uses, "n_forward": self.n_forward,
                "class": self.__class__.__name__}
        if hasattr(self, "_extra_stats"):
            base.update(self._extra_stats())
        return base

    cls.__init__ = new_init
    if original_forward:
        cls.forward = tracked_forward
    cls.stats = stats
    return cls


def grilly_layer(layer_name: str | None = None) -> Callable:
    """Class decorator: register as a grilly-compatible layer.

    The class gets a _layer_name attribute and is registered in the
    layer() factory's search path.

    Usage:
        @grilly_layer("MyCustomLayer")
        class MyCustomLayer:
            def __init__(self, d_input, d_output): ...
            def forward(self, x): ...
    """
    def decorator(cls: Type) -> Type:
        cls._layer_name = layer_name or cls.__name__
        # Register in module-level registry
        _custom_layers[cls._layer_name] = cls
        return cls
    return decorator


# Custom layer registry (populated by @grilly_layer)
_custom_layers: dict[str, Type] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


def layer(layer_type: str, **kwargs) -> Any:
    """Resolve and instantiate a layer by type name.

    Search order:
    1. Custom registry (@grilly_layer decorated classes)
    2. grilly.nn
    3. cubemind submodules

    Usage:
        linear = layer("Linear", in_features=64, out_features=32)
        gif = layer("GIFNeuron", input_dim=64, hidden_dim=64, L=16)
    """
    # 1. Custom registry
    if layer_type in _custom_layers:
        return _custom_layers[layer_type](**kwargs)

    # 2. grilly.nn
    try:
        import grilly.nn as gnn
        if hasattr(gnn, layer_type):
            try:
                return getattr(gnn, layer_type)(**kwargs)
            except TypeError:
                pass
    except ImportError:
        pass

    # 3. cubemind submodules
    _modules = [
        "cubemind.brain.gif_neuron", "cubemind.brain.synapsis",
        "cubemind.brain.snn_ffn", "cubemind.brain.addition_linear",
        "cubemind.brain.neurogenesis", "cubemind.brain.spike_vsa_bridge",
        "cubemind.brain.identity", "cubemind.memory.formation",
        "cubemind.execution.mindforge", "cubemind.execution.moqe",
    ]
    for mod_path in _modules:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, layer_type):
                return getattr(mod, layer_type)(**kwargs)
        except (ImportError, Exception):
            continue

    raise ValueError(f"Layer '{layer_type}' not found. "
                     f"Custom: {list(_custom_layers.keys())}")
