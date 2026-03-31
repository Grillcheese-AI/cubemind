"""Color perception — wavelength modulates neurochemistry.

Based on Roy et al. (Cognitive Neurodynamics, 2021): EEG multifractal
analysis shows colors produce different brain complexity patterns.
Blue = highest complexity, Red = highest arousal.

Extracts color statistics from frames and maps them to neurochemical
drives without hardcoding — wavelength-based modulation following
empirical EEG data.
"""

from __future__ import annotations

import numpy as np


def extract_color_stats(frame: np.ndarray) -> dict:
    """Extract color statistics from a BGR frame.

    Returns dict with dominant_hue, saturation, brightness, warmth,
    and per-channel ratios.
    """
    if frame.ndim == 2:
        brightness = float(np.mean(frame) / 255.0)
        return {
            "dominant_hue": 0, "saturation": 0.0, "brightness": brightness,
            "warmth": 0.0, "red_ratio": 0.33, "green_ratio": 0.33,
            "blue_ratio": 0.33,
        }

    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)

    total = r + g + b + 1e-8
    r_ratio = float(np.mean(r / total))
    g_ratio = float(np.mean(g / total))
    b_ratio = float(np.mean(b / total))

    brightness = float(np.mean(frame) / 255.0)

    mean_rgb = (r + g + b) / 3
    saturation = float(np.mean(np.abs(r - mean_rgb) + np.abs(g - mean_rgb)
                                + np.abs(b - mean_rgb)) / (mean_rgb.mean() + 1e-8))
    saturation = min(saturation / 2.0, 1.0)

    warmth = float((r_ratio - b_ratio) * 2)
    warmth = max(-1.0, min(1.0, warmth))

    means = {"red": np.mean(r), "green": np.mean(g), "blue": np.mean(b)}
    dominant = max(means, key=means.get)
    hue_map = {"red": 0, "green": 120, "blue": 240}

    return {
        "dominant_hue": hue_map.get(dominant, 0),
        "saturation": saturation,
        "brightness": brightness,
        "warmth": warmth,
        "red_ratio": r_ratio,
        "green_ratio": g_ratio,
        "blue_ratio": b_ratio,
    }


def color_to_neurochemistry(color_stats: dict,
                            prev_stats: dict | None = None) -> dict:
    """Map color statistics to neurochemical drives.

    Based on Roy et al. (2021) EEG findings:
      Blue: highest brain complexity -> dopamine (exploration)
      Red: highest arousal -> cortisol (alertness)
      Green: calmness -> serotonin (relaxation)
      High saturation + warm: intense arousal (explosions, fire)

    Also detects sudden visual transients (flash/explosion) by comparing
    to previous frame's stats. A sudden brightness spike with high
    saturation triggers a multi-channel arousal burst.
    """
    sat = color_stats["saturation"]
    r = color_stats["red_ratio"]
    g = color_stats["green_ratio"]
    b = color_stats["blue_ratio"]
    brightness = color_stats["brightness"]
    warmth = color_stats["warmth"]

    intensity = sat * 0.8

    # Base drives from color channels
    novelty = intensity * (b * 2.0 + brightness * 0.3) + b * 0.2
    threat = intensity * (r * 1.2 - g * 0.3)
    focus = intensity * (sat * 0.5 + abs(brightness - 0.5) * 0.5)
    valence = warmth * 0.3 + g * 0.4 - r * 0.1

    # Transient detection: sudden brightness/saturation change = explosion/flash
    if prev_stats is not None:
        brightness_delta = abs(brightness - prev_stats.get("brightness", brightness))
        sat_delta = abs(sat - prev_stats.get("saturation", sat))
        warmth_delta = abs(warmth - prev_stats.get("warmth", warmth))

        # Flash/explosion: bright + saturated + warm + sudden
        transient = brightness_delta + sat_delta * 0.5 + warmth_delta * 0.3
        if transient > 0.15:
            # Multi-channel arousal burst
            novelty += transient * 1.5   # Surprising!
            threat += transient * 0.8    # Startling
            focus += transient * 1.0     # Grabs attention

    # Hot colors at high saturation = intense (fire, explosions, blood)
    hot_intensity = max(0, warmth) * sat
    if hot_intensity > 0.3:
        novelty += hot_intensity * 0.5
        threat += hot_intensity * 0.4

    return {
        "novelty": float(max(0, min(1, novelty))),
        "threat": float(max(0, min(1, threat))),
        "focus": float(max(0, min(1, focus))),
        "valence": float(max(-1, min(1, valence))),
    }
