"""RAVEN panel renderer — attribute dicts → 80x80 grayscale images.

Renders geometric shapes from I-RAVEN/I-RAVEN-X attribute dictionaries
into numpy arrays for the visual perception pipeline.

Shape types (Type attribute modulo 5):
  0: circle, 1: square, 2: triangle, 3: pentagon, 4: hexagon

Size attribute controls the shape radius (scaled to image).
Color attribute controls grayscale fill (0=black, maxval=white).

No external dependencies — pure numpy with basic rasterization.
"""

from __future__ import annotations

import math

import numpy as np


# ── Shape Rasterizers ─────────────────────────────────────────────────────

def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int, color: float) -> None:
    """Draw a filled circle."""
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[mask] = color


def _draw_polygon(
    img: np.ndarray, cx: int, cy: int, r: int, n_sides: int, color: float,
) -> None:
    """Draw a filled regular polygon."""
    h, w = img.shape
    # Compute vertices
    angles = [2 * math.pi * i / n_sides - math.pi / 2 for i in range(n_sides)]
    verts_x = [cx + r * math.cos(a) for a in angles]
    verts_y = [cy + r * math.sin(a) for a in angles]

    # Scanline fill via point-in-polygon (ray casting)
    yy, xx = np.mgrid[:h, :w]
    # Vectorized ray casting
    inside = np.zeros((h, w), dtype=bool)
    n = len(verts_x)
    j = n - 1
    for i in range(n):
        yi, xi = verts_y[i], verts_x[i]
        yj, xj = verts_y[j], verts_x[j]
        cond = ((yi <= yy) & (yy < yj)) | ((yj <= yy) & (yy < yi))
        slope = np.where(cond, xi + (yy - yi) / (yj - yi + 1e-10) * (xj - xi), 0.0)
        inside ^= cond & (xx < slope)
        j = i
    img[inside] = color


def _draw_square(img: np.ndarray, cx: int, cy: int, r: int, color: float) -> None:
    """Draw a filled axis-aligned square."""
    h, w = img.shape
    x0 = max(0, cx - r)
    x1 = min(w, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r)
    img[y0:y1, x0:x1] = color


# ── Main Renderer ─────────────────────────────────────────────────────────

SHAPE_NAMES = ["circle", "square", "triangle", "pentagon", "hexagon"]
SHAPE_SIDES = [0, 4, 3, 5, 6]  # 0 = circle (special case)


def render_panel(
    attrs: dict,
    size: int = 80,
    maxval: int = 10,
    bg_color: float = 0.0,
) -> np.ndarray:
    """Render a single RAVEN panel from attribute dict to grayscale image.

    Args:
        attrs:    Dict with 'Type', 'Size', 'Color' (integer values).
        size:     Output image size (size x size pixels).
        maxval:   Maximum attribute value (for normalization).
        bg_color: Background color [0, 1].

    Returns:
        (size, size) float32 array in [0, 1].
    """
    img = np.full((size, size), bg_color, dtype=np.float32)

    type_val = int(attrs.get("Type", 0))
    size_val = int(attrs.get("Size", maxval // 2))
    color_val = int(attrs.get("Color", maxval // 2))

    # Shape type: modulo 5 to map to our 5 shapes
    shape_idx = type_val % 5

    # Size: map to radius [size*0.1, size*0.4]
    min_r = int(size * 0.1)
    max_r = int(size * 0.4)
    r = min_r + int((size_val / max(maxval, 1)) * (max_r - min_r))
    r = max(3, min(r, max_r))

    # Color: map to grayscale [0.15, 0.95] (avoid pure black/white for visibility)
    fill = 0.15 + 0.8 * (color_val / max(maxval, 1))

    cx, cy = size // 2, size // 2

    if shape_idx == 0:
        _draw_circle(img, cx, cy, r, fill)
    elif shape_idx == 1:
        _draw_square(img, cx, cy, r, fill)
    else:
        n_sides = SHAPE_SIDES[shape_idx]
        _draw_polygon(img, cx, cy, r, n_sides, fill)

    return img


def render_problem(
    context: list[dict],
    candidates: list[dict],
    size: int = 80,
    maxval: int = 10,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render all panels of a RAVEN problem.

    Args:
        context:    List of context panel attribute dicts.
        candidates: List of 8 candidate attribute dicts.
        size:       Panel image size.
        maxval:     Maximum attribute value.

    Returns:
        (context_images, candidate_images): Lists of (size, size) float32 arrays.
    """
    ctx_imgs = [render_panel(a, size=size, maxval=maxval) for a in context]
    cand_imgs = [render_panel(a, size=size, maxval=maxval) for a in candidates]
    return ctx_imgs, cand_imgs


def render_rpm_grid(
    context: list[dict],
    n: int = 3,
    size: int = 80,
    maxval: int = 10,
    gap: int = 2,
) -> np.ndarray:
    """Render the context panels into an n x n grid image (last cell blank).

    Args:
        context: List of (n*n - 1) panel attribute dicts.
        n:       Grid size.
        size:    Per-panel pixel size.
        maxval:  Max attribute value.
        gap:     Pixel gap between panels.

    Returns:
        (n*size + (n-1)*gap, n*size + (n-1)*gap) float32 array.
    """
    grid_size = n * size + (n - 1) * gap
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    idx = 0
    for row in range(n):
        for col in range(n):
            y0 = row * (size + gap)
            x0 = col * (size + gap)
            if row == n - 1 and col == n - 1:
                # Last cell: question mark (draw a ? or leave blank)
                grid[y0:y0 + size, x0:x0 + size] = 0.1
            else:
                panel_img = render_panel(context[idx], size=size, maxval=maxval)
                grid[y0:y0 + size, x0:x0 + size] = panel_img
                idx += 1

    return grid
