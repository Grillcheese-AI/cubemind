"""Tests for cubemind.perception.encoder.Encoder."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.perception.encoder import Encoder

K, L = 4, 32


@pytest.fixture(scope="module")
def enc() -> Encoder:
    return Encoder(k=K, l=L)


def test_init(enc: Encoder):
    assert enc.k == K
    assert enc.l == L


def test_encode_text(enc: Encoder):
    hv = enc.encode("hello world")
    assert hv.shape == (K, L)


def test_encode_different_texts(enc: Encoder):
    a = enc.encode("hello")
    b = enc.encode("goodbye")
    assert not np.array_equal(a, b)


def test_encode_same_text_deterministic(enc: Encoder):
    a = enc.encode("test")
    b = enc.encode("test")
    np.testing.assert_array_equal(a, b)


def test_encode_empty(enc: Encoder):
    hv = enc.encode("")
    assert hv.shape == (K, L)


def test_encode_long_text(enc: Encoder):
    text = "word " * 1000
    hv = enc.encode(text)
    assert hv.shape == (K, L)
    assert np.all(np.isfinite(hv))


def test_encode_unicode(enc: Encoder):
    hv = enc.encode("こんにちは世界")
    assert hv.shape == (K, L)


def test_encode_numbers(enc: Encoder):
    hv = enc.encode("12345 67890")
    assert hv.shape == (K, L)
