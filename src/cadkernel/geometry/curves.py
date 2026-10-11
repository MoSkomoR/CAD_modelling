"""Parametric curves: lines, arcs, and Bezier curves via De Casteljau's algorithm.

Notes: notes/01_Curves/Lines_and_Arcs.md, notes/01_Curves/Bezier_Curves.md,
       notes/01_Curves/De_Casteljau_Derivation.md,
       notes/01_Curves/Bernstein_Basis_Properties.md
"""
from __future__ import annotations

import math

import numpy as np


def line(p0: np.ndarray, p1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Parametric line C(t) = (1-t) p0 + t p1, t in [0, 1]."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    return (1 - t) * p0 + t * p1


def arc(center: np.ndarray, radius: float, t: np.ndarray,
         theta0: float = 0.0, theta1: float = 2 * np.pi) -> np.ndarray:
    """Circular arc in the XY plane: C(t) = center + radius * (cos(theta), sin(theta)),
    with theta linearly interpolated between theta0 and theta1 as t goes 0 -> 1.

    This is the "analytic" way to represent a circle: exact, but not a polynomial curve,
    which is exactly why Bezier/B-spline curves (rational forms, see geometry/nurbs.py) exist -- CAD
    kernels need a *single* curve representation that can do lines, conics, and free-form
    shapes, and plain trig functions don't compose with that machinery.
    """
    center = np.asarray(center, dtype=float)
    t = np.asarray(t, dtype=float)
    theta = (1 - t) * theta0 + t * theta1
    return center + radius * np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def bezier_de_casteljau(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a single point on a Bezier curve at parameter t via De Casteljau's algorithm.

    De Casteljau's algorithm evaluates the curve by *repeated linear interpolation* between
    control points, rather than by expanding the Bernstein polynomial basis directly. Given
    control points P_0..P_n, it builds a triangle of intermediate points:

        P_i^0 = P_i                                    (the control points themselves)
        P_i^k = (1-t) P_i^(k-1) + t P_{i+1}^(k-1)       for k = 1..n, i = 0..n-k

    and C(t) = P_0^n. This is numerically stable and geometrically intuitive: each level of
    the triangle is just linear interpolation ("lerp") between the points of the level below.
    See notes/01_Curves/De_Casteljau_Derivation.md for the derivation and why this is
    equivalent to the Bernstein polynomial form.
    """
    points = np.asarray(control_points, dtype=float).copy()
    n = len(points) - 1
    for k in range(1, n + 1):
        points[: n - k + 1] = (1 - t) * points[: n - k + 1] + t * points[1 : n - k + 2]
    return points[0]


def bezier_curve(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a Bezier curve at an array of parameter values t in [0, 1]."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.array([bezier_de_casteljau(control_points, ti) for ti in t])


def bernstein_basis(n: int, t: np.ndarray) -> np.ndarray:
    """All n+1 Bernstein basis polynomials B_i^n(t) = C(n,i) (1-t)^(n-i) t^i, evaluated at t.

    Returns an array of shape (len(t), n+1): row j holds the basis values at t[j].

    The two facts that give Bezier curves nearly all their useful properties live here rather
    than in the curve itself: the basis is non-negative on [0, 1], and it sums to 1 for every t
    (partition of unity, by the binomial theorem). See
    notes/01_Curves/Bernstein_Basis_Properties.md.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
    i = np.arange(n + 1)
    coeff = np.array([math.comb(n, k) for k in range(n + 1)], dtype=float)
    # NumPy evaluates 0.0 ** 0 as 1.0, which is exactly the convention the endpoints need.
    return coeff * (1 - t) ** (n - i) * t**i


def bezier_subdivide(control_points: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a Bezier curve at parameter t into two Bezier curves of the same degree.

    This is the "subdivision for free" result: running De Casteljau's algorithm at t already
    computes both halves' control points as a side effect. The two outer edges of the triangle
    are exactly the control polygons of the two sub-curves:

        left  = [P_0^0, P_0^1, ..., P_0^n]      (the "first" edge of the triangle)
        right = [P_0^n, P_1^(n-1), ..., P_n^0]  (the "last" edge, walked back down)

    Nothing equivalent falls out of evaluating the closed Bernstein form, and this is what makes
    De Casteljau the algorithm real kernels build on: recursive subdivision plus the convex hull
    property gives robust curve/curve and curve/surface intersection, adaptive tessellation with
    guaranteed error bounds, ray casting and trimming.
    """
    levels = de_casteljau_triangle(control_points, t)
    n = len(np.asarray(control_points)) - 1
    left = np.array([levels[k][0] for k in range(n + 1)])
    right = np.array([levels[n - k][-1] for k in range(n + 1)])
    return left, right


def bezier_elevate(control_points: np.ndarray) -> np.ndarray:
    """The same curve, written as a Bezier curve of one degree higher (n+2 control points).

        Q_0 = P_0,   Q_{n+1} = P_n,   Q_i = (i/(n+1)) P_{i-1} + (1 - i/(n+1)) P_i

    Multiply C(t) by 1 = (1-t) + t and regroup: every B_i^n becomes a blend of B_i^{n+1} and
    B_{i+1}^{n+1}, and collecting coefficients gives exactly these corner-cutting weights. The
    shape does not change; only the representation gains a handle. Kernels need this to make two
    curves *compatible* (same degree) before lofting or skinning between them.
    """
    points = np.asarray(control_points, dtype=float)
    n = len(points) - 1
    i = np.arange(1, n + 1)[:, None] / (n + 1)
    interior = i * points[:-1] + (1 - i) * points[1:]
    return np.vstack([points[:1], interior, points[-1:]])


def de_casteljau_triangle(control_points: np.ndarray, t: float) -> list[np.ndarray]:
    """Like bezier_de_casteljau, but returns every intermediate level of the triangle
    (including the input control points as level 0). Used to visualize the construction:
    each level's points are connected to draw the successive lerp segments that collapse
    down to the final curve point.
    """
    points = np.asarray(control_points, dtype=float).copy()
    levels = [points.copy()]
    n = len(points) - 1
    for k in range(1, n + 1):
        points[: n - k + 1] = (1 - t) * points[: n - k + 1] + t * points[1 : n - k + 2]
        levels.append(points[: n - k + 1].copy())
    return levels