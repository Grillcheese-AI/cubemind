"""Tests for cubemind.telemetry — metrics collection and visualization."""

import time

import numpy as np
import pytest

from cubemind.telemetry.collector import MetricsCollector, MetricPoint


class TestMetricsCollector:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = MetricsCollector(max_points_per_metric=100)

    def test_record_and_get(self):
        self.m.record("test.metric", 42.0)
        points = self.m.get("test.metric")
        assert len(points) == 1
        assert points[0].value == 42.0

    def test_record_with_tags(self):
        self.m.record("test.tagged", 1.0, tags={"stage": "perception"})
        point = self.m.get_latest("test.tagged")
        assert point.tags["stage"] == "perception"

    def test_get_latest(self):
        self.m.record("x", 1.0)
        self.m.record("x", 2.0)
        self.m.record("x", 3.0)
        assert self.m.get_latest("x").value == 3.0

    def test_get_mean(self):
        for v in [10.0, 20.0, 30.0]:
            self.m.record("avg", v)
        assert self.m.get_mean("avg") == 20.0

    def test_get_mean_empty(self):
        assert self.m.get_mean("nonexistent") is None

    def test_ring_buffer_eviction(self):
        m = MetricsCollector(max_points_per_metric=5)
        for i in range(10):
            m.record("buf", float(i))
        points = m.get("buf")
        assert len(points) == 5
        assert points[0].value == 5.0  # oldest surviving
        assert points[-1].value == 9.0  # newest

    def test_get_count_includes_evicted(self):
        m = MetricsCollector(max_points_per_metric=3)
        for i in range(10):
            m.record("c", float(i))
        assert m.get_count("c") == 10  # total recordings, not buffer size
        assert len(m.get("c")) == 3   # buffer holds last 3

    def test_get_last_n(self):
        for i in range(20):
            self.m.record("series", float(i))
        last5 = self.m.get("series", last=5)
        assert len(last5) == 5
        assert last5[0].value == 15.0
        assert last5[-1].value == 19.0

    def test_metric_names(self):
        self.m.record("a.b", 1.0)
        self.m.record("c.d", 2.0)
        self.m.record("a.b", 3.0)
        names = self.m.metric_names()
        assert names == ["a.b", "c.d"]

    def test_summary(self):
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            self.m.record("s", v)
        summary = self.m.summary()
        assert "s" in summary
        assert summary["s"]["latest"] == 5.0
        assert summary["s"]["mean"] == 3.0
        assert summary["s"]["min"] == 1.0
        assert summary["s"]["max"] == 5.0
        assert summary["s"]["count"] == 5

    def test_disable_enable(self):
        self.m.record("before", 1.0)
        self.m.disable()
        self.m.record("during", 2.0)
        self.m.enable()
        self.m.record("after", 3.0)
        assert self.m.get_latest("before") is not None
        assert self.m.get_latest("during") is None  # not recorded
        assert self.m.get_latest("after") is not None

    def test_reset(self):
        self.m.record("x", 1.0)
        assert len(self.m.metric_names()) > 0
        self.m.reset()
        assert len(self.m.metric_names()) == 0

    def test_increment(self):
        self.m.increment("counter", 5)
        self.m.increment("counter", 3)
        assert self.m.get_count("counter") == 8

    def test_record_timing(self):
        with self.m.record_timing("test.timing"):
            time.sleep(0.01)
        point = self.m.get_latest("test.timing")
        assert point is not None
        assert point.value >= 5.0  # at least 5ms (sleep 10ms with overhead)
        assert point.value < 500.0  # sanity

    def test_listener(self):
        received = []
        self.m.add_listener(lambda name, point: received.append((name, point.value)))
        self.m.record("listen", 42.0)
        assert len(received) == 1
        assert received[0] == ("listen", 42.0)

    def test_remove_listener(self):
        received = []
        cb = lambda name, point: received.append(name)
        self.m.add_listener(cb)
        self.m.record("a", 1.0)
        self.m.remove_listener(cb)
        self.m.record("b", 2.0)
        assert received == ["a"]  # b not captured

    def test_export_json(self):
        self.m.record("j", 1.0, tags={"k": "v"})
        data = self.m.export_json()
        assert "j" in data
        assert data["j"][0]["v"] == 1.0
        assert data["j"][0]["k"] == "v"

    def test_thread_safety(self):
        """Concurrent writes from multiple threads shouldn't crash."""
        import threading
        errors = []

        def writer(tid):
            try:
                for i in range(100):
                    self.m.record(f"thread.{tid}", float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads recorded
        for tid in range(8):
            assert self.m.get_count(f"thread.{tid}") == 100


class TestGlobalMetrics:

    def test_global_singleton_works(self):
        from cubemind.telemetry import metrics
        metrics.record("global.test", 99.0)
        assert metrics.get_latest("global.test").value == 99.0
        metrics.reset()
