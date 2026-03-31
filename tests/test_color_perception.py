"""Test Color Perception — color wavelength modulates neurochemistry.

Based on Roy et al. (Cognitive Neurodynamics, 2021): EEG multifractal
analysis shows different colors produce measurably different brain
complexity patterns. Blue = highest complexity, Red = highest arousal.

The color_response function extracts dominant color from a frame and
maps it to neurochemical drives — no hardcoding, just wavelength-based
modulation following the empirical data.

All tests self-contained.
"""

import numpy as np
import pytest

from cubemind.perception.snn import NeurochemicalState


# ── Color Perception Module ──────────────────────────────────────────

def extract_color_stats(frame: np.ndarray) -> dict:
    """Extract color statistics from a BGR frame.

    Returns dict with:
      - dominant_hue: 0-360 (red=0/360, green=120, blue=240)
      - saturation: 0-1 (grey=0, vivid=1)
      - brightness: 0-1
      - warmth: -1 (cool/blue) to +1 (warm/red)
      - red_ratio, green_ratio, blue_ratio: 0-1
    """
    if frame.ndim == 2:
        # Greyscale
        brightness = float(np.mean(frame) / 255.0)
        return {
            "dominant_hue": 0, "saturation": 0.0, "brightness": brightness,
            "warmth": 0.0, "red_ratio": 0.33, "green_ratio": 0.33,
            "blue_ratio": 0.33,
        }

    # BGR frame
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)

    total = r + g + b + 1e-8
    r_ratio = float(np.mean(r / total))
    g_ratio = float(np.mean(g / total))
    b_ratio = float(np.mean(b / total))

    brightness = float(np.mean(frame) / 255.0)

    # Saturation: how far from grey (equal RGB)
    mean_rgb = (r + g + b) / 3
    saturation = float(np.mean(np.abs(r - mean_rgb) + np.abs(g - mean_rgb)
                                + np.abs(b - mean_rgb)) / (mean_rgb.mean() + 1e-8))
    saturation = min(saturation / 2.0, 1.0)  # Normalize to [0, 1]

    # Warmth: red/yellow positive, blue/green negative
    warmth = float((r_ratio - b_ratio) * 2)
    warmth = max(-1.0, min(1.0, warmth))

    # Dominant hue (simplified — which channel dominates)
    means = {"red": np.mean(r), "green": np.mean(g), "blue": np.mean(b)}
    dominant = max(means, key=means.get)
    hue_map = {"red": 0, "green": 120, "blue": 240}
    dominant_hue = hue_map.get(dominant, 0)

    return {
        "dominant_hue": dominant_hue,
        "saturation": saturation,
        "brightness": brightness,
        "warmth": warmth,
        "red_ratio": r_ratio,
        "green_ratio": g_ratio,
        "blue_ratio": b_ratio,
    }


def color_to_neurochemistry(color_stats: dict) -> dict:
    """Map color statistics to neurochemical drive signals.

    Based on Roy et al. (2021) EEG findings:
      - Blue: highest brain complexity → dopamine (exploration, curiosity)
      - Red: highest arousal → cortisol (alertness, fight/flight)
      - Green: calmness → serotonin (relaxation, baseline)
      - Yellow/Orange: warmth → oxytocin (social warmth) + dopamine
      - High saturation: more intense response
      - Grey/desaturated: baseline (all drives low)

    Returns dict with novelty, threat, focus, valence drives
    suitable for NeurochemicalState.update().
    """
    sat = color_stats["saturation"]
    warmth = color_stats["warmth"]
    r = color_stats["red_ratio"]
    g = color_stats["green_ratio"]
    b = color_stats["blue_ratio"]
    brightness = color_stats["brightness"]

    # Base intensity from saturation (grey = no color drive)
    intensity = sat * 0.8

    # Novelty: blue drives exploration (highest complexity in EEG)
    novelty = intensity * (b * 2.0 + brightness * 0.3) + b * 0.2

    # Threat: red drives alertness/arousal
    threat = intensity * (r * 1.2 - g * 0.3)

    # Focus: determined by brightness contrast and saturation
    focus = intensity * (sat * 0.5 + abs(brightness - 0.5) * 0.5)

    # Valence: warm colors positive, cool colors neutral-to-positive
    #   Green = calming positive, Blue = engaged positive,
    #   Red = arousing (can be negative or positive)
    valence = warmth * 0.3 + g * 0.4 - r * 0.1

    return {
        "novelty": float(max(0, min(1, novelty))),
        "threat": float(max(0, min(1, threat))),
        "focus": float(max(0, min(1, focus))),
        "valence": float(max(-1, min(1, valence))),
    }


# ── Tests ─────────────────────────────────────────────────────────────

class TestColorExtraction:

    def test_red_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red channel in BGR
        stats = extract_color_stats(frame)
        assert stats["red_ratio"] > stats["blue_ratio"]
        assert stats["warmth"] > 0.3

    def test_blue_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel in BGR
        stats = extract_color_stats(frame)
        assert stats["blue_ratio"] > stats["red_ratio"]
        assert stats["warmth"] < -0.3

    def test_green_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 1] = 255  # Green channel
        stats = extract_color_stats(frame)
        assert stats["green_ratio"] > stats["red_ratio"]

    def test_grey_low_saturation(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        stats = extract_color_stats(frame)
        assert stats["saturation"] < 0.1

    def test_vivid_high_saturation(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Pure red
        stats = extract_color_stats(frame)
        assert stats["saturation"] > 0.3

    def test_greyscale_input(self):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        stats = extract_color_stats(frame)
        assert stats["saturation"] == 0.0


class TestColorNeurochemistry:

    def test_blue_drives_novelty(self):
        """Blue = highest EEG complexity → novelty/dopamine."""
        blue_stats = extract_color_stats(
            np.zeros((50, 50, 3), dtype=np.uint8) + np.array([255, 0, 0], dtype=np.uint8))
        grey_stats = extract_color_stats(
            np.full((50, 50, 3), 128, dtype=np.uint8))

        blue_drive = color_to_neurochemistry(blue_stats)
        grey_drive = color_to_neurochemistry(grey_stats)

        assert blue_drive["novelty"] > grey_drive["novelty"], (
            f"Blue should drive more novelty: {blue_drive['novelty']:.3f} vs {grey_drive['novelty']:.3f}")

    def test_red_drives_threat(self):
        """Red = highest arousal → threat/cortisol."""
        red_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        red_frame[:, :, 2] = 255
        green_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        green_frame[:, :, 1] = 255

        red_drive = color_to_neurochemistry(extract_color_stats(red_frame))
        green_drive = color_to_neurochemistry(extract_color_stats(green_frame))

        assert red_drive["threat"] > green_drive["threat"]

    def test_grey_baseline(self):
        """Grey = low saturation → minimal neurochemical drive."""
        grey = np.full((50, 50, 3), 128, dtype=np.uint8)
        drive = color_to_neurochemistry(extract_color_stats(grey))
        assert drive["novelty"] < 0.1
        assert drive["threat"] < 0.1

    def test_green_positive_valence(self):
        """Green = calmness → positive valence."""
        green = np.zeros((50, 50, 3), dtype=np.uint8)
        green[:, :, 1] = 255
        drive = color_to_neurochemistry(extract_color_stats(green))
        assert drive["valence"] > 0, f"Green should be positive valence: {drive['valence']:.3f}"

    def test_drives_in_range(self):
        """All drives should be in valid ranges."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            frame = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
            drive = color_to_neurochemistry(extract_color_stats(frame))
            assert 0 <= drive["novelty"] <= 1
            assert 0 <= drive["threat"] <= 1
            assert 0 <= drive["focus"] <= 1
            assert -1 <= drive["valence"] <= 1


class TestColorIntegrationWithNeurochemistry:

    def test_red_increases_cortisol(self):
        """Viewing red frame should increase cortisol over time."""
        nc = NeurochemicalState()
        initial_cortisol = nc.cortisol

        red = np.zeros((50, 50, 3), dtype=np.uint8)
        red[:, :, 2] = 255

        for _ in range(10):
            drive = color_to_neurochemistry(extract_color_stats(red))
            nc.update(**drive)

        assert nc.cortisol > initial_cortisol

    def test_blue_higher_novelty_than_grey(self):
        """Blue frame should produce higher novelty drive than grey."""
        blue = np.zeros((50, 50, 3), dtype=np.uint8)
        blue[:, :, 0] = 255
        grey = np.full((50, 50, 3), 128, dtype=np.uint8)

        blue_drive = color_to_neurochemistry(extract_color_stats(blue))
        grey_drive = color_to_neurochemistry(extract_color_stats(grey))

        assert blue_drive["novelty"] > grey_drive["novelty"]

    def test_different_videos_different_profiles(self):
        """Red-heavy vs blue-heavy frames produce different drive profiles."""
        red = np.zeros((50, 50, 3), dtype=np.uint8)
        red[:, :, 2] = 200; red[:, :, 1] = 50

        blue = np.zeros((50, 50, 3), dtype=np.uint8)
        blue[:, :, 0] = 200; blue[:, :, 1] = 50

        red_drive = color_to_neurochemistry(extract_color_stats(red))
        blue_drive = color_to_neurochemistry(extract_color_stats(blue))

        # Red should drive more threat, blue more novelty
        assert red_drive["threat"] > blue_drive["threat"], (
            f"Red threat {red_drive['threat']:.3f} should > blue {blue_drive['threat']:.3f}")
        assert blue_drive["novelty"] > red_drive["novelty"], (
            f"Blue novelty {blue_drive['novelty']:.3f} should > red {red_drive['novelty']:.3f}")
