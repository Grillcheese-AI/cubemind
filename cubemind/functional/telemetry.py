"""CubeMind Telemetry — structured logging + tracing via Pydantic Logfire.

Uses logfire for:
- Structured spans (forward pass, training step, memory retrieval)
- Metrics (loss, neuron count, expert usage, latency)
- Tracing (end-to-end request flow through brain components)
- OpenTelemetry export (Jaeger, Grafana, self-hosted)

Falls back to loguru if logfire is not configured.

Usage:
    from cubemind.functional.telemetry import telemetry, metered, traced

    # Configure once at startup
    telemetry.configure(service_name="cubemind", token="...")

    # Decorator: auto-trace a method with span + metrics
    class MyModel:
        @traced("forward")
        def forward(self, x):
            return self.w @ x

        @metered("train_step")
        def train_step(self, x, target):
            ...

    # Manual spans
    with telemetry.span("memory_retrieval", k=5):
        results = hippo.retrieve(query, k=5)

    # Manual metrics
    telemetry.metric("loss", 0.345)
    telemetry.metric("neuron_count", 64)
    telemetry.counter("expert_spawns")
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict

_logfire = None
_loguru = None

try:
    import logfire as _lf
    _logfire = _lf
except ImportError:
    pass

try:
    from loguru import logger as _lg
    _loguru = _lg
except ImportError:
    pass


class Telemetry:
    """Unified telemetry: logfire (preferred) → loguru (fallback)."""

    def __init__(self):
        self._configured = False

    def configure(self, service_name: str = "cubemind", token: str | None = None,
                  environment: str = "dev", **kwargs) -> None:
        """Configure logfire. Call once at startup."""
        if _logfire is not None:
            config_kwargs = {"service_name": service_name, "environment": environment}
            if token:
                config_kwargs["token"] = token
            config_kwargs.update(kwargs)
            try:
                _logfire.configure(**config_kwargs)
                self._configured = True
            except Exception:
                pass  # Logfire not available, use fallback

    def span(self, name: str, **attributes):
        """Create a tracing span."""
        if _logfire and self._configured:
            return _logfire.span(name, **attributes)
        # Fallback: context manager that just logs
        return _FallbackSpan(name, attributes)

    def metric(self, name: str, value: float, **labels) -> None:
        """Record a gauge metric."""
        if _logfire and self._configured:
            _logfire.info("{name}={value}", name=name, value=value, **labels)
        elif _loguru:
            _loguru.debug("{} = {:.4f}", name, value)

    def counter(self, name: str, delta: int = 1, **labels) -> None:
        """Increment a counter."""
        if _logfire and self._configured:
            _logfire.info("{name} += {delta}", name=name, delta=delta, **labels)
        elif _loguru:
            _loguru.debug("{} += {}", name, delta)

    def info(self, msg: str, **kwargs) -> None:
        if _logfire and self._configured:
            _logfire.info(msg, **kwargs)
        elif _loguru:
            _loguru.info(msg, **kwargs)

    def warn(self, msg: str, **kwargs) -> None:
        if _logfire and self._configured:
            _logfire.warn(msg, **kwargs)
        elif _loguru:
            _loguru.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        if _logfire and self._configured:
            _logfire.error(msg, **kwargs)
        elif _loguru:
            _loguru.error(msg, **kwargs)


class _FallbackSpan:
    """Context manager fallback when logfire is not available."""

    def __init__(self, name: str, attributes: dict):
        self.name = name
        self.attributes = attributes

    def __enter__(self):
        self._t0 = time.perf_counter()
        if _loguru:
            _loguru.debug("span start: {}", self.name)
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self._t0) * 1000
        if _loguru:
            _loguru.debug("span end: {} ({:.1f}ms)", self.name, elapsed)


# Global singleton
telemetry = Telemetry()


def traced(span_name: str) -> Callable:
    """Decorator: wrap method in a tracing span.

    Records: span duration, component class name, method name.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            cls_name = args[0].__class__.__name__ if args else ""
            full_name = f"{cls_name}.{span_name}" if cls_name else span_name
            with telemetry.span(full_name):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def metered(metric_name: str) -> Callable:
    """Decorator: auto-record call count + latency."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            telemetry.counter(f"{metric_name}_calls")
            telemetry.metric(f"{metric_name}_latency_ms", elapsed_ms)
            return result
        return wrapper
    return decorator
