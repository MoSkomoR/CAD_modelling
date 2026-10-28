"""Drawing helpers for half-edge solids (module 03 and 04 scripts)."""
from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..ops.measure import halfedge_points
from ..topology.halfedge import HalfEdge, face_normal


def keyhole(loops: list[np.ndarray]) -> np.ndarray:
    """One fillable polygon for a face with rings: cut a two-way bridge from the outer loop to
    each ring (matplotlib cannot fill a polygon with holes directly)."""
    polygon = list(loops[0])
    for ring in loops[1:]:
        dist = np.linalg.norm(np.array(polygon)[:, None] - ring[None], axis=-1)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        bridge = list(ring[j:]) + list(ring[: j + 1])
        polygon = polygon[: i + 1] + bridge + polygon[i:]
    return np.array(polygon)


def face_collection(faces: list[list[np.ndarray]], colors, alpha=0.25) -> Poly3DCollection:
    """faces: per face, its loops as point arrays (outer first), as in halfedge.snapshot."""
    return Poly3DCollection([keyhole(loops) for loops in faces], facecolors=colors, alpha=alpha,
                            edgecolors="none")


def halfedge_arrow(he: HalfEdge, inset: float = 0.08, trim: float = 0.15):
    """Start point and vector of an arrow drawn for `he`, pushed into its own face.

    The face lies to the left of a half-edge (seen from outside), so the offset direction is
    normal x direction. Twins land on opposite sides of their edge, visibly separate.
    """
    a, b = he.origin.point, he.target.point
    d = b - a
    length = np.linalg.norm(d)
    left = np.cross(face_normal(he.face), d / length)
    start = a + trim * d + inset * left
    return start, (1 - 2 * trim) * d


def loop_polyline(loop, samples: int = 24) -> np.ndarray:
    """A loop drawn through its edge curves, each half-edge sampled in its own direction (the
    straight segment where an edge has no curve)."""
    t = np.linspace(0.0, 1.0, samples + 1)[:-1]
    return np.vstack([halfedge_points(he, t) for he in loop.halfedges()])
