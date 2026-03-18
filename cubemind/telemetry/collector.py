"""Metrics collector — thread-safe time-series storage for pipeline telemetry.

Stores metrics as named time-series with timestamps. Each metric can have
optional tags for filtering (e.g., stage, topic, expert).

Design: append-only ring buffer per metric name. No external dependencies.
Thread-safe via a simple lock. Zero overhead when no metrics are being read.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class MetricPoint:
    """Single metric observation."""
    timestamp: float
    value: float
    tags: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Global metrics store for CubeMind pipeline telemetry.

    Thread-safe. Each metric name maps to a ring buffer of MetricPoints.
    Metrics are cheap to record (~1μs) and zero-cost when not read.

    Args:
        max_points_per_metric: Max history per metric name (ring buffer).
    """

    def __init__(self, max_points_per_metric: int = 10_000) -> None:
        self._max = max_points_per_metric
        self._data: dict[str, list[MetricPoint]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._listeners: list[callable] = []
        self._enabled = True

    def record(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a metric value.

        Args:
            name: Dotted metric name (e.g., "perception.latency_ms")
            value: Numeric value
            tags: Optional key-value tags for filtering
        """
        if not self._enabled:
            return
        point = MetricPoint(
            timestamp=time.monotonic(),
            value=value,
            tags=tags or {},
        )
        with self._lock:
            buf = self._data[name]
            if len(buf) >= self._max:
                buf.pop(0)
            buf.append(point)
            self._counters[name] += 1

        # Notify listeners (non-blocking)
        for listener in self._listeners:
            try:
                listener(name, point)
            except Exception:
                pass

    def record_timing(self, name: str, tags: dict[str, str] | None = None):
        """Context manager to record elapsed time in milliseconds.

        Usage:
            with metrics.record_timing("perception.encode_ms"):
                result = encoder.encode(text)
        """
        return _TimingContext(self, name, tags)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += amount

    def get(self, name: str, last: int | None = None) -> list[MetricPoint]:
        """Get metric history.

        Args:
            name: Metric name
            last: Return only the last N points (None = all)

        Returns:
            List of MetricPoints, oldest first.
        """
        with self._lock:
            buf = self._data.get(name, [])
            if last is not None:
                return list(buf[-last:])
            return list(buf)

    def get_latest(self, name: str) -> MetricPoint | None:
        """Get the most recent value for a metric."""
        with self._lock:
            buf = self._data.get(name, [])
            return buf[-1] if buf else None

    def get_mean(self, name: str, last: int = 100) -> float | None:
        """Get the mean of the last N values."""
        points = self.get(name, last=last)
        if not points:
            return None
        return sum(p.value for p in points) / len(points)

    def get_count(self, name: str) -> int:
        """Total number of recordings for a metric (including evicted)."""
        return self._counters.get(name, 0)

    def metric_names(self) -> list[str]:
        """List all recorded metric names."""
        with self._lock:
            return sorted(self._data.keys())

    def summary(self) -> dict[str, dict]:
        """Get a summary of all metrics: latest value, mean, count."""
        with self._lock:
            result = {}
            for name, buf in self._data.items():
                if not buf:
                    continue
                values = [p.value for p in buf[-100:]]
                result[name] = {
                    "latest": buf[-1].value,
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": self._counters[name],
                }
            return result

    def add_listener(self, callback: callable) -> None:
        """Add a real-time listener called on every record().

        Args:
            callback: function(name: str, point: MetricPoint)
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: callable) -> None:
        """Remove a listener."""
        self._listeners = [l for l in self._listeners if l is not callback]

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        """Disable collection (zero overhead when disabled)."""
        self._enabled = False

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._data.clear()
            self._counters.clear()

    def export_json(self) -> dict:
        """Export all metrics as JSON-serializable dict."""
        with self._lock:
            result = {}
            for name, buf in self._data.items():
                result[name] = [
                    {"t": p.timestamp, "v": p.value, **p.tags}
                    for p in buf
                ]
            return result


class _TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, collector: MetricsCollector, name: str, tags: dict | None):
        self._collector = collector
        self._name = name
        self._tags = tags
        self._start = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.monotonic() - self._start) * 1000
        self._collector.record(self._name, elapsed_ms, self._tags)


# ── Global singleton ─────────────────────────────────────────────────────
metrics = MetricsCollector()
