"""Tests for cubemind.perception.categorizer.Categorizer."""

from __future__ import annotations

import time
import pytest

from cubemind.perception.categorizer import Categorizer


@pytest.fixture(scope="module")
def cat() -> Categorizer:
    return Categorizer()


def test_init(cat: Categorizer):
    assert cat.n_subdomains == 160
    assert cat.n_patterns > 15000
    assert cat.n_subwords > 15000


def test_categorize_physics(cat: Categorizer):
    r = cat.categorize("Einstein discovered the theory of relativity")
    assert r["subdomain"] == "physics"
    assert r["score"] > 0


def test_categorize_biology(cat: Categorizer):
    r = cat.categorize("DNA replication occurs during cell division in mitosis")
    assert r["subdomain"] in ("biology", "genetics", "anatomy")
    assert r["score"] > 0


def test_categorize_history(cat: Categorizer):
    r = cat.categorize("The Roman Empire conquered Gaul under Julius Caesar")
    assert r["subdomain"] in ("ancient_rome", "ancient", "history", "empire")


def test_categorize_art(cat: Categorizer):
    r = cat.categorize("Picasso painted Guernica using cubism techniques")
    assert r["subdomain"] == "art"


def test_categorize_finance(cat: Categorizer):
    r = cat.categorize("The stock market crashed and bond yields spiked")
    assert r["subdomain"] in ("finance", "markets", "investing", "economics")


def test_categorize_programming(cat: Categorizer):
    r = cat.categorize("Python function def with decorators and async await")
    assert r["subdomain"] in ("programming", "software", "computing", "computerscience")


def test_categorize_music(cat: Categorizer):
    r = cat.categorize("Beethoven composed the symphony in D minor")
    assert r["subdomain"] == "music"


def test_categorize_empty(cat: Categorizer):
    r = cat.categorize("")
    assert r["subdomain"] == "general"
    assert r["score"] == 0


def test_categorize_short(cat: Categorizer):
    r = cat.categorize("hi")
    assert r["subdomain"] == "general"


def test_return_all_scores(cat: Categorizer):
    r = cat.categorize("quantum physics experiments", return_all=True)
    assert "all_scores" in r
    assert len(r["all_scores"]) > 0


def test_confidence_range(cat: Categorizer):
    r = cat.categorize("The mitochondria is the powerhouse of the cell")
    assert 0 <= r["confidence"] <= 1.0


def test_batch(cat: Categorizer):
    texts = [
        "The planets orbit around the sun",
        "Shakespeare wrote Hamlet in London",
        "Machine learning uses neural networks",
    ]
    results = cat.categorize_batch(texts)
    assert len(results) == 3
    for r in results:
        assert "subdomain" in r


def test_speed(cat: Categorizer):
    """Should categorize 1000+ texts per second."""
    texts = [
        "The theory of relativity revolutionized physics",
        "DNA replication in eukaryotic cells",
        "Medieval castle architecture in Europe",
    ] * 100  # 300 texts

    t0 = time.perf_counter()
    cat.categorize_batch(texts)
    elapsed = time.perf_counter() - t0

    rate = len(texts) / elapsed
    assert rate > 500, f"Too slow: {rate:.0f} texts/sec (need 500+)"
    print(f"  Categorizer speed: {rate:.0f} texts/sec")
