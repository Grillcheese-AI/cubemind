"""CubeMind telemetry — metrics collection and live visualization.

Every pipeline stage emits metrics through a global MetricsCollector.
The collector stores time-series data that can be visualized live
in terminal or exported for analysis.

Usage:
    from cubemind.telemetry import metrics

    # Record a metric
    metrics.record("perception.latency_ms", 12.5)
    metrics.record("memory.surprise", 0.87)
    metrics.record("routing.top_score", 0.95, tags={"topic": "science"})

    # Get recent values
    metrics.get("perception.latency_ms", last=100)

    # Live dashboard
    from cubemind.telemetry.dashboard import LiveDashboard
    dash = LiveDashboard()
    dash.start()  # terminal-based live view
"""

from cubemind.telemetry.collector import MetricsCollector, metrics

__all__ = ["MetricsCollector", "metrics"]
