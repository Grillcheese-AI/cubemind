"""Test biologically-grounded vision neurons.

Tests opponent-color, motion tuning, luminance, and developmental maturity.
"""

import numpy as np
import pytest

from cubemind.perception.bio_vision import (
    OpponentColorNeurons, MotionNeurons, LuminanceNeurons, BioVisionEncoder,
)


class TestOpponentColor:

    @pytest.fixture
    def color(self):
        return OpponentColorNeurons(grid_h=4, grid_w=4)

    def test_red_activates_rg_positive(self, color):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red in BGR
        feat = color.process(frame).reshape(4, 4, 3)
        rg_mean = feat[:, :, 0].mean()
        assert rg_mean > 0, f"Red should produce positive R-G: {rg_mean:.3f}"

    def test_green_activates_rg_negative(self, color):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 1] = 255  # Green
        feat = color.process(frame).reshape(4, 4, 3)
        rg_mean = feat[:, :, 0].mean()
        assert rg_mean < 0, f"Green should produce negative R-G: {rg_mean:.3f}"

    def test_blue_activates_by_positive(self, color):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue in BGR
        feat = color.process(frame).reshape(4, 4, 3)
        by_mean = feat[:, :, 1].mean()
        assert by_mean > 0, f"Blue should produce positive B-Y: {by_mean:.3f}"

    def test_output_shape(self, color):
        frame = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        feat = color.process(frame)
        assert feat.shape == (4 * 4 * 3,)

    def test_grey_near_zero_opponent(self, color):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        feat = color.process(frame).reshape(4, 4, 3)
        assert abs(feat[:, :, 0].mean()) < 0.05  # RG near zero
        assert abs(feat[:, :, 1].mean()) < 0.1   # BY near zero


class TestMotionNeurons:

    @pytest.fixture
    def motion(self):
        return MotionNeurons(grid_h=4, grid_w=4, n_directions=8)

    def test_no_motion_on_static(self, motion):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        _ = motion.process(frame)  # Init prev
        feat = motion.process(frame)  # Same frame
        assert np.max(np.abs(feat)) < 0.01

    def test_motion_detected_on_change(self, motion):
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        frame2[:, 50:] = 200  # Right half brightens
        _ = motion.process(frame1)
        feat = motion.process(frame2)
        assert np.max(feat) > 0.01, "Should detect brightness change as motion"

    def test_output_shape(self, motion):
        frame = np.random.default_rng(42).integers(0, 256, (100, 100), dtype=np.uint8)
        _ = motion.process(frame)
        feat = motion.process(frame)
        assert feat.shape == (4 * 4 * 8,)

    def test_maturity_narrows_tuning(self, motion):
        motion.set_maturity(0.2)
        sigma_baby = motion.sigma
        motion.set_maturity(1.0)
        sigma_adult = motion.sigma
        assert sigma_adult < sigma_baby


class TestLuminanceNeurons:

    @pytest.fixture
    def lum(self):
        return LuminanceNeurons(grid_h=4, grid_w=4)

    def test_bright_higher_than_dark(self, lum):
        bright = np.full((100, 100, 3), 200, dtype=np.uint8)
        dark = np.full((100, 100, 3), 20, dtype=np.uint8)
        feat_bright = lum.process(bright).reshape(4, 4, 2)
        lum.prev_luminance = None
        feat_dark = lum.process(dark).reshape(4, 4, 2)
        assert feat_bright[:, :, 0].mean() > feat_dark[:, :, 0].mean()

    def test_transient_on_change(self, lum):
        dark = np.full((100, 100, 3), 20, dtype=np.uint8)
        bright = np.full((100, 100, 3), 200, dtype=np.uint8)
        _ = lum.process(dark)
        feat = lum.process(bright).reshape(4, 4, 2)
        transient = feat[:, :, 1].mean()
        assert transient > 0.5, f"Sudden change should produce transient: {transient:.3f}"

    def test_no_transient_on_static(self, lum):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        _ = lum.process(frame)
        feat = lum.process(frame).reshape(4, 4, 2)
        assert feat[:, :, 1].mean() < 0.01

    def test_weber_fechner_log(self, lum):
        """Log compression: response ratio should be less than input ratio."""
        f1 = np.full((100, 100), 25, dtype=np.uint8)
        f2 = np.full((100, 100), 100, dtype=np.uint8)
        f3 = np.full((100, 100), 250, dtype=np.uint8)
        r1 = lum.process(f1).reshape(4, 4, 2)[:, :, 0].mean()
        lum.prev_luminance = None
        lum.process(f2).reshape(4, 4, 2)[:, :, 0].mean()
        lum.prev_luminance = None
        r3 = lum.process(f3).reshape(4, 4, 2)[:, :, 0].mean()
        # Input goes 25→100→250 (4x then 2.5x)
        # Log response should grow sublinearly
        input_ratio = 250 / 25  # 10x
        response_ratio = r3 / (r1 + 1e-8)
        assert response_ratio < input_ratio, (
            f"Weber-Fechner: response ratio {response_ratio:.1f} should < input ratio {input_ratio:.1f}")


class TestBioVisionEncoder:

    @pytest.fixture
    def encoder(self):
        return BioVisionEncoder(grid_h=4, grid_w=4, n_directions=4, maturity=0.5)

    def test_output_shape(self, encoder):
        frame = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        feat = encoder.process(frame)
        assert feat.shape == (encoder.d_features,)
        assert feat.dtype == np.float32

    def test_maturity_scales_output(self):
        baby = BioVisionEncoder(grid_h=4, grid_w=4, maturity=0.2)
        adult = BioVisionEncoder(grid_h=4, grid_w=4, maturity=1.0)
        frame = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        # Need two frames for motion
        _ = baby.process(frame)
        _ = adult.process(frame)
        feat_baby = baby.process(frame)
        feat_adult = adult.process(frame)
        assert np.linalg.norm(feat_adult) >= np.linalg.norm(feat_baby)

    def test_grow_increases_maturity(self, encoder):
        initial = encoder.maturity
        encoder.grow(delta=0.1)
        assert encoder.maturity > initial

    def test_grow_capped_at_one(self, encoder):
        for _ in range(1000):
            encoder.grow(delta=0.1)
        assert encoder.maturity == 1.0

    def test_different_frames_different_features(self, encoder):
        red = np.zeros((100, 100, 3), dtype=np.uint8)
        red[:, :, 2] = 255
        blue = np.zeros((100, 100, 3), dtype=np.uint8)
        blue[:, :, 0] = 255
        f_red = encoder.process(red)
        f_blue = encoder.process(blue)
        assert not np.allclose(f_red, f_blue)

    def test_feature_names_length(self, encoder):
        assert len(encoder.feature_names) == encoder.d_features
