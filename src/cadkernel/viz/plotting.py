"""Matplotlib helpers shared across scripts, so each script stays focused on the concept
it's demonstrating rather than on plot boilerplate.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_curve_2d(ax, points: np.ndarray, control_points: np.ndarray | None = None,
                   label: str | None = None, color: str = "C0"):
    """Plot a 2D curve, optionally with its control polygon overlaid."""
    points = np.asarray(points)
    ax.plot(points[:, 0], points[:, 1], color=color, lw=2, label=label)
    if control_points is not None:
        cp = np.asarray(control_points)
        ax.plot(cp[:, 0], cp[:, 1], "o--", color="gray", lw=1, ms=6, label="control polygon")
    ax.set_aspect("equal")
    if label:
        ax.legend()
    return ax


def new_2d_axes(title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def polygon_patch(ax, points: np.ndarray, **kwargs):
    """Shade a filled polygon (used to draw convex hulls under a curve)."""
    from matplotlib.patches import Polygon

    patch = Polygon(np.asarray(points), closed=True, **kwargs)
    ax.add_patch(patch)
    return patch


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Convex hull of a 2D point set, counter-clockwise (monotone chain / Andrew's algorithm).

    Written out rather than pulled from scipy because the convex hull is the object the Bezier
    containment proof is *about* -- see notes/01_Curves/Bernstein_Basis_Properties.md.
    """
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def build(sequence):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return np.array(build(pts) + build(pts[::-1]))