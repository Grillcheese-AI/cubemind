"""Tests for the Cloud API endpoints (Decision Oracle demo)."""

from __future__ import annotations

import os

# Use small dims for tests to avoid OOM
os.environ["CUBEMIND_K"] = "4"
os.environ["CUBEMIND_L"] = "32"
os.environ["CUBEMIND_N_WORLDS"] = "16"
os.environ["CUBEMIND_D_HIDDEN"] = "32"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from cubemind.cloud.api import app  # noqa: E402

client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert isinstance(body["n_worlds"], int)
        assert body["n_worlds"] > 0


# ── /predict ──────────────────────────────────────────────────────────────────


class TestPredict:
    @pytest.fixture()
    def payload(self) -> dict:
        return {
            "question": "Should I change careers?",
            "context": {"job": "engineer", "years": "5"},
            "top_k": 3,
        }

    def test_predict_returns_200(self, payload):
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_response_structure(self, payload):
        body = client.post("/predict", json=payload).json()
        assert body["question"] == payload["question"]
        assert isinstance(body["n_worlds"], int)
        assert isinstance(body["elapsed_ms"], float)
        assert isinstance(body["futures"], list)
        assert len(body["futures"]) == payload["top_k"]

    def test_predict_future_fields(self, payload):
        body = client.post("/predict", json=payload).json()
        future = body["futures"][0]
        assert "world_id" in future
        assert "score" in future
        assert "plausibility" in future
        assert "q_value" in future

    def test_predict_default_top_k(self):
        """Omitting top_k should use the default (10)."""
        resp = client.post("/predict", json={
            "question": "test",
            "context": {"a": "b"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["futures"]) == 10

    def test_predict_missing_question(self):
        """Missing required field should return 422."""
        resp = client.post("/predict", json={"context": {"a": "b"}})
        assert resp.status_code == 422


# ── /book ─────────────────────────────────────────────────────────────────────


class TestBook:
    @pytest.fixture()
    def payload(self) -> dict:
        return {
            "passage": (
                "The old wizard stood at the edge of the cliff. "
                "Below, the sea churned with ancient power."
            ),
            "question": "What happens next?",
            "top_k": 5,
        }

    def test_book_returns_200(self, payload):
        resp = client.post("/book", json=payload)
        assert resp.status_code == 200

    def test_book_response_structure(self, payload):
        body = client.post("/book", json=payload).json()
        assert "passage_preview" in body
        assert isinstance(body["n_worlds"], int)
        assert isinstance(body["elapsed_ms"], float)
        assert isinstance(body["endings"], list)
        assert len(body["endings"]) == payload["top_k"]

    def test_book_ending_fields(self, payload):
        body = client.post("/book", json=payload).json()
        ending = body["endings"][0]
        assert "world_id" in ending
        assert "score" in ending
        assert "plausibility" in ending
        assert "q_value" in ending

    def test_book_passage_preview_truncated(self, payload):
        body = client.post("/book", json=payload).json()
        # Preview should not exceed 120 characters
        assert len(body["passage_preview"]) <= 120

    def test_book_default_top_k(self):
        """Omitting top_k should use the default (10)."""
        resp = client.post("/book", json={
            "passage": "Once upon a time.",
            "question": "Then what?",
        })
        assert resp.status_code == 200
        assert len(resp.json()["endings"]) == 10

    def test_book_missing_passage(self):
        """Missing required field should return 422."""
        resp = client.post("/book", json={"question": "What?"})
        assert resp.status_code == 422


# ── Schema validation ─────────────────────────────────────────────────────────


class TestSchemas:
    """Pydantic models should accept valid data and reject bad data."""

    def test_predict_request_rejects_bad_context_type(self):
        resp = client.post("/predict", json={
            "question": "test",
            "context": "not-a-dict",
        })
        assert resp.status_code == 422

    def test_book_request_rejects_bad_top_k(self):
        resp = client.post("/book", json={
            "passage": "text",
            "question": "q",
            "top_k": -1,
        })
        assert resp.status_code == 422
