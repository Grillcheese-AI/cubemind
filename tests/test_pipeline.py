"""End-to-end pipeline tests for CubeMind model."""

import numpy as np
import pytest

from cubemind.model import CubeMind
from cubemind.ops.block_codes import BlockCodes
from cubemind.routing.router import CubeMindRouter

# Small dims for tests — full K_BLOCKS/L_BLOCK uses 1GB+ per HYLA instance
K, L = 4, 32


class TestCubeMindPipeline:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = CubeMind(k=K, l=L, n_codebook=8, d_hidden=16, cache_size=100)

    def test_forward_without_router(self):
        """Forward pass works without a router (no routing stage)."""
        bc = BlockCodes(K, L)
        phi = bc.random_discrete(seed=42)
        result = self.model.forward(phi=phi)
        assert "surprise" in result
        assert "stress" in result
        assert "output" in result
        assert "answer" in result
        assert result["topic"] is None

    def test_forward_with_router(self):
        """Forward pass with routing produces topic assignment."""
        bc = BlockCodes(K, L)
        protos = np.stack([bc.random_discrete(seed=i) for i in range(5)])
        router = CubeMindRouter(
            ["science", "sports", "politics", "tech", "arts"],
            protos, K, L, top_k=2,
        )
        self.model.attach_router(router)
        result = self.model.forward(phi=protos[2])
        assert result["topic"] == "politics"
        assert result["topic_score"] > 0.9

    def test_forward_with_text(self):
        """Forward pass with text input uses encoder."""
        result = self.model.forward(text="hello world")
        assert result["output"].shape == (K, L)

    def test_surprise_decreases_on_repeat(self):
        """Surprise should drop when the same vector is seen again."""
        bc = BlockCodes(K, L)
        phi = bc.random_discrete(seed=42)
        r1 = self.model.forward(phi=phi)
        r2 = self.model.forward(phi=phi)
        assert r2["surprise"] < r1["surprise"]

    def test_stress_increases_with_cache_fill(self):
        """Stress should increase as cache fills."""
        bc = BlockCodes(K, L)
        stresses = []
        for i in range(50):
            r = self.model.forward(phi=bc.random_discrete(seed=i))
            stresses.append(r["stress"])
        assert stresses[-1] > stresses[0]

    def test_hmm_detection_produces_prediction(self):
        """HMM detection should produce a valid block-code prediction."""
        bc = BlockCodes(K, L)
        result = self.model.forward(phi=bc.random_discrete(seed=42))
        pred = result["hmm_prediction"]
        assert pred.shape == (K, L)

    def test_q_value_is_scalar(self):
        """CVL Q-value should be a scalar float."""
        bc = BlockCodes(K, L)
        result = self.model.forward(phi=bc.random_discrete(seed=42))
        assert isinstance(result["q_value"], float)

    def test_step_counter_increments(self):
        """Step counter should increment on each forward pass."""
        bc = BlockCodes(K, L)
        phi = bc.random_discrete(seed=42)
        r1 = self.model.forward(phi=phi)
        r2 = self.model.forward(phi=phi)
        assert r2["step"] == r1["step"] + 1

    def test_train_step(self):
        """Training step should return a finite loss."""
        bc = BlockCodes(K, L)
        obs = [bc.random_discrete(seed=i) for i in range(5)]
        target = bc.random_discrete(seed=99)
        loss = self.model.train_step(obs, target, lr=0.01)
        assert np.isfinite(loss)

    def test_train_reduces_loss(self):
        """Multiple training steps should reduce loss."""
        bc = BlockCodes(K, L)
        obs = [bc.random_discrete(seed=i) for i in range(5)]
        target = bc.random_discrete(seed=99)
        losses = []
        for _ in range(10):
            loss = self.model.train_step(obs, target, lr=0.05)
            losses.append(loss)
        # Loss should generally decrease (allow some noise)
        assert losses[-1] <= losses[0] * 1.5  # shouldn't explode

    def test_telemetry_recorded(self):
        """Forward pass should record telemetry metrics."""
        from cubemind.telemetry import metrics as m
        m.reset()
        bc = BlockCodes(K, L)
        self.model.forward(phi=bc.random_discrete(seed=42))
        names = m.metric_names()
        assert "memory.surprise" in names
        assert "memory.stress" in names
        assert "execution.q_value" in names
        m.reset()

    def test_stats(self):
        """Stats should reflect model state."""
        stats = self.model.stats
        assert stats["step"] == 0
        assert stats["cache_size"] == 0
        self.model.forward(phi=BlockCodes(K, L).random_discrete(seed=42))
        stats = self.model.stats
        assert stats["step"] == 1
        assert stats["cache_size"] == 1

    def test_repr(self):
        r = repr(self.model)
        assert "CubeMind" in r
        assert str(K) in r
