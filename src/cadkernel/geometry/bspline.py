"""B-spline curves: piecewise polynomials with local control, built on a knot vector.

Conventions used throughout (they match Piegl & Tiller, "The NURBS Book"):

    control points  P_0 .. P_n          (n+1 of them)
    degree          p
    knot vector     u_0 <= u_1 <= ... <= u_m,   m = n + p + 1
    valid domain    [u_p, u_{n+1}]

A clamped knot vector repeats each end knot p+1 times; that is what makes the curve start at P_0
and end at P_n. With no interior knots at all it *is* a Bezier curve (test_bezier_is_a_special_
case_of_bspline) -- a B-spline is a Bezier curve that has been allowed to break into pieces.

Notes: notes/01_Curves/B_Splines.md, notes/01_Curves/De_Boor_Algorithm.md,
       notes/01_Curves/B_Spline_Interpolation.md
"""
from __future__ import annotations

import numpy as np

from .curves import bezier_elevate


def clamped_knots(n_ctrl: int, p: int, interior=None) -> np.ndarray:
    """Clamped (open) knot vector on [0, 1] for n_ctrl control points of degree p.

    The end knots are repeated p+1 times so the curve interpolates its first and last control
    points. `interior` gives the n_ctrl - p - 1 interior knots explicitly; by default they are
    spaced uniformly ("open uniform").
    """
    count = n_ctrl - p - 1
    if count < 0:
        raise ValueError(f"degree {p} needs at least {p + 1} control points, got {n_ctrl}")
    if interior is None:
        interior = np.linspace(0.0, 1.0, count + 2)[1:-1]
    interior = np.sort(np.asarray(interior, dtype=float))
    if len(interior) != count:
        raise ValueError(f"expected {count} interior knots, got {len(interior)}")
    return np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])


def find_span(knots: np.ndarray, p: int, t):
    """Index k of the knot span [u_k, u_{k+1}) containing t, restricted to the valid domain.

    t equal to the right end u_{n+1} belongs to the last non-empty span, so the closed interval
    [u_p, u_{n+1}] is covered. Vectorised over t.
    """
    knots = np.asarray(knots, dtype=float)
    n = len(knots) - p - 2
    span = np.searchsorted(knots, t, side="right") - 1
    return np.clip(span, p, n)


def bspline_basis_levels(knots: np.ndarray, p: int, t) -> list[np.ndarray]:
    """Cox-de Boor, returning every degree from 0 to p: levels[d] has shape (len(t), m-d).

        N_{i,0}(t) = 1 if u_i <= t < u_{i+1} else 0
        N_{i,d}(t) = (t - u_i) / (u_{i+d} - u_i) * N_{i,d-1}(t)
                   + (u_{i+d+1} - t) / (u_{i+d+1} - u_{i+1}) * N_{i+1,d-1}(t)

    with the convention 0/0 := 0 (a repeated knot makes a span empty, and a function built on an
    empty span contributes nothing). Degree 0 is a set of box functions, one per span; each step
    blends two neighbours into a function one span wider and one degree smoother. That is the
    whole reason N_{i,p} is zero outside [u_i, u_{i+p+1}): it was built from p+1 boxes.

    At t = u_m (the right end), the half-open boxes would all be zero; the last non-empty box is
    switched on instead so the closed domain is covered.
    """
    knots = np.asarray(knots, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
    left, right = knots[:-1], knots[1:]
    boxes = ((left <= t) & (t < right)).astype(float)
    last = np.flatnonzero(left < right)[-1]
    boxes[(t[:, 0] == knots[-1]), last] = 1.0

    levels = [boxes]
    for d in range(1, p + 1):
        prev = levels[-1]
        u_i, u_id = knots[: -d - 1], knots[d:-1]
        u_i1, u_id1 = knots[1:-d], knots[d + 1 :]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(u_id > u_i, (t - u_i) / (u_id - u_i), 0.0)
            b = np.where(u_id1 > u_i1, (u_id1 - t) / (u_id1 - u_i1), 0.0)
        levels.append(a * prev[:, :-1] + b * prev[:, 1:])
    return levels


def bspline_basis(knots: np.ndarray, p: int, t) -> np.ndarray:
    """All n+1 B-spline basis functions N_{i,p}(t), shape (len(t), n+1).

    The piecewise analogue of bernstein_basis: still non-negative and a partition of unity on the
    valid domain, but each function is now non-zero on at most p+1 knot spans instead of on the
    whole interval. See notes/01_Curves/B_Splines.md for the proofs.
    """
    return bspline_basis_levels(knots, p, t)[-1]


def bspline_curve(control_points: np.ndarray, knots: np.ndarray, p: int, t) -> np.ndarray:
    """Evaluate the curve at many parameters at once: C(t) = sum_i N_{i,p}(t) P_i.

    Bulk evaluation through the basis matrix -- the same fast-path trade as bezier_surface; de_boor
    is the algorithm for one point and the one that generalizes (knot insertion, rationals).
    """
    points = np.asarray(control_points, dtype=float)
    return bspline_basis(knots, p, t) @ points


def de_boor_stages(control_points: np.ndarray, knots: np.ndarray, p: int, t: float) -> dict:
    """Every level of de Boor's algorithm at t, for drawing.

    Only the p+1 control points P_{k-p} .. P_k of the span k containing t are touched -- the
    other basis functions are zero there. Level r lerps neighbours with a weight that depends on
    the knots (not just on t, as in De Casteljau):

        alpha = (t - u_{i}) / (u_{i+p+1-r} - u_{i})       for i = k-p+r .. k
        P_i^r = (1 - alpha) P_{i-1}^{r-1} + alpha P_i^{r-1}

    and after p levels one point is left: C(t). With Bezier knots every alpha is exactly t, and
    this *is* De Casteljau (test_de_boor_is_de_casteljau_on_bezier_knots).
    """
    knots = np.asarray(knots, dtype=float)
    points = np.asarray(control_points, dtype=float)
    k = int(find_span(knots, p, t))
    level = points[k - p : k + 1].copy()
    levels = [level.copy()]
    for r in range(1, p + 1):
        i = np.arange(k - p + r, k + 1)
        alpha = ((t - knots[i]) / (knots[i + p + 1 - r] - knots[i]))[:, None]
        level = (1 - alpha) * level[:-1] + alpha * level[1:]
        levels.append(level.copy())
    return {"span": k, "levels": levels, "point": levels[-1][0]}


def de_boor(control_points: np.ndarray, knots: np.ndarray, p: int, t: float) -> np.ndarray:
    """One point on the curve via de Boor's algorithm (see de_boor_stages)."""
    return de_boor_stages(control_points, knots, p, t)["point"]


def bspline_derivative(
    control_points: np.ndarray, knots: np.ndarray, p: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """The derivative C'(t), as a B-spline of degree p-1 on the knot vector with its ends trimmed.

        Q_i = p (P_{i+1} - P_i) / (u_{i+p+1} - u_{i+1}),    i = 0 .. n-1

    The hodograph of a Bezier curve, n (P_{i+1} - P_i), is the special case with all knot gaps 1.
    Differentiating twice is just calling this twice, which is how the tests measure continuity
    at a knot. A zero-length gap (a knot of multiplicity > p) gives a zero control point.
    """
    points = np.asarray(control_points, dtype=float)
    knots = np.asarray(knots, dtype=float)
    n = len(points) - 1
    gaps = knots[p + 1 : p + 1 + n] - knots[1 : 1 + n]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(gaps > 0, p / gaps, 0.0)[:, None]
    return scale * np.diff(points, axis=0), knots[1:-1], p - 1


def insert_knot(
    control_points: np.ndarray, knots: np.ndarray, p: int, t: float
) -> tuple[np.ndarray, np.ndarray]:
    """Boehm's algorithm: insert the knot t once, without changing the curve.

    One more knot means one more control point. Only the p control points around t move, each
    replaced by a lerp of its neighbours at the knot-dependent ratio

        alpha_i = (t - u_i) / (u_{i+p} - u_i),    Q_i = (1 - alpha_i) P_{i-1} + alpha_i P_i

    for i = k-p+1 .. k (k the span containing t); the rest are copied. This is the B-spline
    counterpart of subdivision: the new polygon hugs the curve more tightly, and inserting a knot
    p times splits the curve into two independent pieces there (bspline_to_bezier).
    """
    points = np.asarray(control_points, dtype=float)
    knots = np.asarray(knots, dtype=float)
    n = len(points) - 1
    if not knots[p] < t < knots[n + 1]:
        raise ValueError("t must lie strictly inside the valid domain")
    multiplicity = int(np.sum(knots == t))
    if multiplicity >= p:
        raise ValueError(f"knot {t} already has multiplicity {multiplicity} >= degree {p}")
    k = int(find_span(knots, p, t))
    new = np.empty((n + 2, points.shape[1]))
    new[: k - p + 1] = points[: k - p + 1]
    new[k + 1 :] = points[k:]
    i = np.arange(k - p + 1, k + 1)
    alpha = ((t - knots[i]) / (knots[i + p] - knots[i]))[:, None]
    new[i] = (1 - alpha) * points[i - 1] + alpha * points[i]
    return new, np.insert(knots, k + 1, t)


def _require_clamped(knots: np.ndarray, p: int) -> None:
    if not (np.all(knots[: p + 1] == knots[0]) and np.all(knots[-p - 1 :] == knots[-1])):
        raise ValueError("expected a clamped knot vector (end knots repeated p+1 times)")


def bspline_to_bezier(
    control_points: np.ndarray, knots: np.ndarray, p: int
) -> list[np.ndarray]:
    """Split a clamped B-spline into its Bezier segments, one per non-empty knot span.

    Insert every interior knot until it has multiplicity p. The curve is unchanged, but now each
    span's p+1 control points are exactly that span's Bezier control polygon, and neighbouring
    segments share an end point. This is how a kernel reuses every Bezier tool -- hull bounds,
    De Casteljau subdivision, the flatness test -- on a B-spline: decompose, then work per span.
    """
    points = np.asarray(control_points, dtype=float)
    knots = np.asarray(knots, dtype=float)
    _require_clamped(knots, p)
    interior = knots[p + 1 : -p - 1]
    for value in np.unique(interior):
        for _ in range(p - int(np.sum(knots == value))):
            points, knots = insert_knot(points, knots, p, value)
    return [points[j : j + p + 1] for j in range(0, len(points) - 1, p)]


def elevate_degree(
    control_points: np.ndarray, knots: np.ndarray, p: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """The same curve as a B-spline of degree p+1.

    Decompose into Bezier segments, raise each with bezier_elevate, and stitch them back together
    with every interior breakpoint at multiplicity p+1. The shape and its true continuity are
    untouched, but the knot vector now only *promises* C^0 at the breakpoints: removing the
    redundant knots again needs knot removal, which this repo does not implement
    (notes/99_Not_Covered_In_Code.md).
    """
    knots = np.asarray(knots, dtype=float)
    segments = [bezier_elevate(segment) for segment in bspline_to_bezier(control_points, knots, p)]
    q = p + 1
    points = np.vstack([segments[0]] + [segment[1:] for segment in segments[1:]])
    breaks = np.unique(knots)
    new_knots = np.concatenate(
        [np.full(q + 1, breaks[0]), np.repeat(breaks[1:-1], q), np.full(q + 1, breaks[-1])]
    )
    return points, new_knots, q


def interpolation_parameters(points: np.ndarray, method: str = "centripetal") -> np.ndarray:
    """Parameter values t_0 = 0 < ... < t_n = 1 at which the curve should hit each data point.

    "uniform":     equal steps, ignoring geometry.
    "chord":       proportional to the distance between consecutive points.
    "centripetal": proportional to the square root of that distance (Lee, 1989).

    Centripetal is the default because it was the only one of the three that neither looped nor
    overshot on every data set in test_parametrization_comparison -- chord length, the usual
    textbook default, overshot worst on two of the three.
    """
    points = np.asarray(points, dtype=float)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if method == "uniform":
        steps = np.ones_like(steps)
    elif method == "centripetal":
        steps = np.sqrt(steps)
    elif method != "chord":
        raise ValueError(f"unknown parametrization {method!r}")
    params = np.concatenate([[0.0], np.cumsum(steps)])
    return params / params[-1]


def averaged_knots(params: np.ndarray, p: int) -> np.ndarray:
    """Knots by averaging (de Boor): each interior knot is the mean of p consecutive parameters.

    This places every knot span so that each basis function "sees" data at its own parameters,
    which is what keeps the interpolation matrix non-singular (Schoenberg-Whitney condition).
    """
    params = np.asarray(params, dtype=float)
    n = len(params) - 1
    interior = [params[j : j + p].mean() for j in range(1, n - p + 1)]
    return np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])


def interpolate_points(
    points: np.ndarray, p: int = 3, parametrization: str = "centripetal"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global B-spline interpolation: control points whose curve passes through every data point.

    Choose a parameter t_k for each data point Q_k, choose knots, and require C(t_k) = Q_k:

        sum_i N_{i,p}(t_k) P_i = Q_k,    k = 0 .. n

    -- n+1 linear equations in the n+1 unknown control points. The matrix is banded (each row has
    at most p+1 non-zeros, by local support), so a real kernel uses a banded solver; the dense
    np.linalg.solve here gives the same answer. Returns (control_points, knots, params).
    """
    points = np.asarray(points, dtype=float)
    if len(points) < p + 1:
        raise ValueError(f"degree {p} interpolation needs at least {p + 1} points")
    params = interpolation_parameters(points, parametrization)
    knots = averaged_knots(params, p)
    matrix = bspline_basis(knots, p, params)
    return np.linalg.solve(matrix, points), knots, params
