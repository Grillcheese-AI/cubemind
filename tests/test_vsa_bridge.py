"""Tests for cubemind.ops.vsa_bridge — binary packing and LSH."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.vsa_bridge import binarize_and_pack, LSHProjector


def test_binarize_and_pack_shape():
    x = np.random.randn(128).astype(np.float32)
    packed = binarize_and_pack(x)
    # 128 bits = 16 bytes = 16 uint8 or 4 uint32 depending on impl
    assert packed is not None
    assert len(packed) > 0


def test_binarize_deterministic():
    x = np.random.randn(64).astype(np.float32)
    p1 = binarize_and_pack(x)
    p2 = binarize_and_pack(x)
    np.testing.assert_array_equal(p1, p2)


def test_lsh_projector_init():
    proj = LSHProjector(d_input=64, d_output=128, seed=42)
    assert proj is not None
    assert proj.d_input == 64
    assert proj.d_output == 128


def test_lsh_projector_project():
    proj = LSHProjector(d_input=64, d_output=128, seed=42)
    x = np.random.randn(64).astype(np.float32)
    h = proj.project(x)
    assert h is not None
    assert h.shape == (128,)


def test_lsh_projector_deterministic():
    proj = LSHProjector(d_input=64, d_output=128, seed=42)
    x = np.random.randn(64).astype(np.float32)
    h1 = proj.project(x)
    h2 = proj.project(x)
    np.testing.assert_array_equal(h1, h2)


def test_lsh_similar_inputs_similar_projection():
    proj = LSHProjector(d_input=64, d_output=128, seed=42)
    x = np.random.randn(64).astype(np.float32)
    y = x + np.random.randn(64).astype(np.float32) * 0.01

    hx = proj.project(x)
    hy = proj.project(y)

    # Cosine similarity should be high for similar inputs
    cos_sim = np.dot(hx, hy) / (np.linalg.norm(hx) * np.linalg.norm(hy) + 1e-8)
    assert cos_sim > 0.9
