"""NURBS curves: B-splines with a weight per control point -- the R in NURBS.

Everything here rests on one idea. A rational curve in d dimensions is an ordinary polynomial
(B-spline or Bezier) curve in d+1 *homogeneous* dimensions, projected back down:

    lift      (P_i, w_i)  ->  (w_i P_i, w_i)
    evaluate  the polynomial curve upstairs, with the existing machinery
    project   divide by the last coordinate

so weights cost almost no new code. De Casteljau, de Boor, knot insertion and the hodograph are
all reused unchanged from curves.py and bspline.py; only the projection is new. Written out,

    C(t) = sum_i N_{i,p}(t) w_i P_i / sum_i N_{i,p}(t) w_i  =  sum_i R_{i,p}(t) P_i.

Conventions follow bspline.py (Piegl & Tiller): control points P_0 .. P_n, weights w_0 .. w_n,
degree p, knots u_0 .. u_m with m = n + p + 1.

Notes: notes/01_Curves/NURBS.md, notes/01_Curves/NURBS_Derivations.md,
       notes/01_Curves/Lines_and_Arcs.md
"""
from __future__ import annotations

import numpy as np

from .bspline import bspline_basis, bspline_curve, bspline_derivative, clamped_knots, insert_knot
from .curves import bezier_de_casteljau


def _lift(control_points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(P_i, w_i) -> (w_i P_i, w_i): the homogeneous control points."""
    points = np.asarray(control_points, dtype=float)
    w = np.asarray(weights, dtype=float)
    return np.concatenate([points * w[:, None], w[:, None]], axis=1)


def _project(homogeneous: np.ndarray) -> np.ndarray:
    """(x, w) -> x / w. The only step that is not an affine combination."""
    return homogeneous[..., :-1] / homogeneous[..., -1:]


def _unlift(homogeneous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of _lift, for control points (not curve points): back to (P_i, w_i)."""
    return _project(homogeneous), homogeneous[:, -1].copy()


# --- Rational Bezier: the special case on Bezier knots ------------------------------------------


def rational_bezier(control_points: np.ndarray, weights: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a *rational* Bezier curve -- a NURBS curve with no interior knots.

    Lift (P_i, w_i) -> (w_i P_i, w_i), run plain De Casteljau up there, divide by the last
    coordinate:

        C(t) = sum_i B_i^n(t) w_i P_i / sum_i B_i^n(t) w_i

    This is the smallest curve that does something polynomials provably cannot: an exact circular
    arc (see rational_arc_quadratic and the impossibility proof in
    notes/01_Curves/Lines_and_Arcs.md).
    """
    homogeneous = _lift(control_points, weights)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return _project(np.array([bezier_de_casteljau(homogeneous, ti) for ti in t]))


def rational_arc_quadratic(center: np.ndarray, radius: float, theta0: float,
                           theta1: float) -> tuple[np.ndarray, np.ndarray]:
    """Control points and weights of the *exact* rational quadratic Bezier for a circular arc.

    Requires a sweep of less than pi (a full circle is built from several such arcs, see
    nurbs_circle). The endpoints sit on the circle; the middle control point is where the two
    endpoint tangents meet, at distance radius / cos(delta/2) along the angle bisector; the
    weights are (1, cos(delta/2), 1) where delta = theta1 - theta0. Derived in
    notes/01_Curves/NURBS_Derivations.md.
    """
    center = np.asarray(center, dtype=float)
    delta = theta1 - theta0
    if not 0 < abs(delta) < np.pi:
        raise ValueError("rational quadratic arc requires 0 < |theta1 - theta0| < pi")
    mid_angle = 0.5 * (theta0 + theta1)
    on_circle = lambda a: center + radius * np.array([np.cos(a), np.sin(a)])
    p1 = center + (radius / np.cos(delta / 2)) * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    control_points = np.array([on_circle(theta0), p1, on_circle(theta1)])
    weights = np.array([1.0, np.cos(delta / 2), 1.0])
    return control_points, weights


# --- NURBS curves -------------------------------------------------------------------------------


def nurbs_basis(knots: np.ndarray, p: int, weights: np.ndarray, t) -> np.ndarray:
    """The rational basis R_{i,p}(t) = w_i N_{i,p}(t) / sum_j w_j N_{j,p}(t), shape (len(t), n+1).

    Still a partition of unity (the denominator is exactly the sum of the numerators), and still
    zero outside [u_i, u_{i+p+1}) -- so local support survives. Non-negativity survives only for
    positive weights; that is the one B-spline guarantee a weight can break.
    """
    w = np.asarray(weights, dtype=float)
    weighted = bspline_basis(knots, p, t) * w[None, :]
    return weighted / weighted.sum(axis=1, keepdims=True)


def nurbs_curve(control_points: np.ndarray, weights: np.ndarray, knots: np.ndarray, p: int,
                t) -> np.ndarray:
    """Evaluate C(t) = sum_i R_{i,p}(t) P_i, by lifting, evaluating the B-spline, projecting."""
    return _project(bspline_curve(_lift(control_points, weights), knots, p, t))


def nurbs_derivative(control_points: np.ndarray, weights: np.ndarray, knots: np.ndarray, p: int,
                     t) -> np.ndarray:
    """C'(t), by the quotient rule on the homogeneous curve.

    Upstairs the curve is a B-spline A(t) = (w(t) C(t), w(t)), and its derivative is again a
    B-spline (bspline_derivative). Differentiating w C = A_x gives w' C + w C' = A_x', so

        C'(t) = (A_x'(t) - w'(t) C(t)) / w(t).

    Unlike the polynomial case, C' is *not* a NURBS curve of degree p-1 with control points one
    can write down: it is a quotient whose denominator is w(t)^2 (notes/01_Curves/NURBS.md
    measures the degree growth).
    """
    lifted = _lift(control_points, weights)
    A = bspline_curve(lifted, knots, p, t)
    d_points, d_knots, d_p = bspline_derivative(lifted, knots, p)
    dA = bspline_curve(d_points, d_knots, d_p, t)
    C = _project(A)
    return (dA[:, :-1] - dA[:, -1:] * C) / A[:, -1:]


def nurbs_insert_knot(control_points: np.ndarray, weights: np.ndarray, knots: np.ndarray, p: int,
                      t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boehm insertion for NURBS: insert in homogeneous space, then unlift.

    Inserting into (P_i, w_i) directly would be wrong -- the lerps must act on the weighted
    points w_i P_i, so the new weights are lerps of the old ones and the new points are
    weight-biased lerps of the old points.
    """
    new_lifted, new_knots = insert_knot(_lift(control_points, weights), knots, p, t)
    new_points, new_weights = _unlift(new_lifted)
    return new_points, new_weights, new_knots


def nurbs_circle(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The exact full circle as one degree-2 NURBS: four quarter arcs, nine control points.

    The control polygon is the circumscribed square (corners and edge midpoints), weights
    1, s, 1, s, ... with s = cos(pi/4) = sqrt(2)/2 on the corners, and the knot vector

        0 0 0  1/4 1/4  1/2 1/2  3/4 3/4  1 1 1

    -- each interior knot doubled, so every quarter is its own rational_arc_quadratic. Returns
    (control_points, weights, knots, p).
    """
    center = np.asarray(center, dtype=float)
    s = np.sqrt(0.5)
    square = np.array(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], float
    )
    weights = np.array([1, s, 1, s, 1, s, 1, s, 1], float)
    knots = clamped_knots(9, 2, [0.25, 0.25, 0.5, 0.5, 0.75, 0.75])
    return center + radius * square, weights, knots, 2


def nurbs_arc(center: np.ndarray, radius: float,
              sweep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The exact arc from angle 0 to angle `sweep` (0 < sweep <= 2 pi) as one degree-2 NURBS.

    nurbs_circle generalized: the sweep is cut into ceil(sweep / (pi/2)) equal pieces, each a
    rational_arc_quadratic (so each piece turns by at most pi/2, well inside the < pi that one
    quadratic piece allows), joined end to end with every interior knot doubled. A full turn
    gives the four quarter arcs of nurbs_circle again (test_arc_of_a_full_turn_is_the_circle).
    Returns (control_points, weights, knots, p).
    """
    if not 0 < sweep <= 2 * np.pi + 1e-12:
        raise ValueError("the sweep must lie in (0, 2 pi]")
    pieces = int(np.ceil(sweep / (np.pi / 2) - 1e-9))
    angles = np.linspace(0.0, sweep, pieces + 1)
    points, weights = [], []
    for k, (a0, a1) in enumerate(zip(angles[:-1], angles[1:])):
        p, w = rational_arc_quadratic(center, radius, a0, a1)
        points.append(p if k == 0 else p[1:])
        weights.append(w if k == 0 else w[1:])
    breaks = np.repeat(np.arange(1, pieces) / pieces, 2)
    return np.vstack(points), np.concatenate(weights), clamped_knots(2 * pieces + 1, 2, breaks), 2


def projective_map(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 projective map (homography) to 2D points: lift to (x, y, 1), multiply, project."""
    points = np.asarray(points, dtype=float)
    image = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, dtype=float).T
    return _project(image)


def nurbs_projective_map(matrix: np.ndarray, control_points: np.ndarray,
                         weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a 2D NURBS curve by a projective map, exactly, by mapping its control data.

    The homogeneous control points (w_i P_i, w_i) are mapped linearly by the 3x3 matrix; projecting
    gives the new control points and the last coordinate gives the new weights. Note the weights
    *change*: a projective map is not just "move the control points" (it is only for affine maps,
    which leave the last row (0, 0, 1) and so every weight alone).
    """
    image = _lift(control_points, weights) @ np.asarray(matrix, dtype=float).T
    return _unlift(image)
