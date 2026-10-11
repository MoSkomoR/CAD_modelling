"""B-spline and NURBS surfaces: the curve machinery of module 01, run once in u and once in v.

A tensor-product surface is a rectangular net of control points P_ij, i along u, j along v, and

    S(u, v) = sum_i sum_j N_{i,p}(u) N_{j,q}(v) P_ij.

Read the double sum from the inside out: for a fixed u, each column net[:, j] is the control
polygon of an ordinary B-spline curve in u, and evaluating every column at u leaves one point per
column -- the control polygon of the v-isocurve through S(u, .). A surface is a *curve of curves*
(bspline_isocurve), exactly as a Bezier surface was De Casteljau run twice. Weights change nothing
in that picture: lift (P_ij, w_ij) -> (w_ij P_ij, w_ij), evaluate the polynomial surface upstairs,
divide by the last coordinate -- the lift/project of nurbs.py, one dimension of net wider.

The second half builds the analytic surfaces of mechanical CAD -- cylinder, cone, sphere, torus --
as surfaces of revolution: a NURBS profile in the xz-plane times the exact NURBS circle, with the
weights multiplying. None of them is reachable without weights (no polynomial is a circle).

Conventions follow Piegl & Tiller: degree p in u on knots_u, degree q in v on knots_v; nets are
(n+1, m+1, d) arrays as in surfaces.py.

Notes: notes/02_Surfaces/NURBS_Surfaces.md, notes/02_Surfaces/Surfaces_of_Revolution.md
"""
from __future__ import annotations

import numpy as np

from .bspline import bspline_basis, bspline_curve, bspline_derivative, clamped_knots, insert_knot
from .nurbs import _project, nurbs_circle
from .surfaces import _as_net


def _lift_net(control_net: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(P_ij, w_ij) -> (w_ij P_ij, w_ij), shape (n+1, m+1, d+1)."""
    net = _as_net(control_net)
    w = np.asarray(weights, dtype=float)
    return np.concatenate([net * w[..., None], w[..., None]], axis=2)


def _unlift_net(homogeneous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _project(homogeneous), homogeneous[..., -1].copy()


def _along(net: np.ndarray, direction: str, curve_op):
    """Apply a curve operation to every column (direction 'u') or row ('v') of the net at once.

    Curve operations here (derivative, knot insertion) are linear in the control points, so a
    whole family of curves can go through in one call: flatten the other two axes into the
    coordinate axis, run the curve code, fold back. This is the code version of "the curve
    algorithm, once per column".
    """
    if direction == "v":
        result = _along(np.swapaxes(net, 0, 1), "u", curve_op)
        return np.swapaxes(result[0], 0, 1), *result[1:]
    rows, cols, dim = net.shape
    points, *rest = curve_op(net.reshape(rows, cols * dim))
    return (points.reshape(len(points), cols, dim), *rest)


# --- B-spline surfaces -------------------------------------------------------------------------


def bspline_isocurve(control_net: np.ndarray, knots: np.ndarray, degree: int, *,
                     u: float | None = None, v: float | None = None) -> np.ndarray:
    """Control points of an isoparametric curve: evaluate every column at u (or every row at v).

    Fixing u leaves a B-spline curve in v on the *v* knots and degree, whose m+1 control points
    are the columns' curve points at u. Pass the knots and degree of the direction being fixed.
    """
    net = _as_net(control_net)
    if (u is None) == (v is None):
        raise ValueError("fix exactly one of u or v")
    if u is not None:
        return np.array([bspline_curve(net[:, j], knots, degree, [u])[0]
                         for j in range(net.shape[1])])
    return np.array([bspline_curve(net[i, :], knots, degree, [v])[0]
                     for i in range(net.shape[0])])


def bspline_surface_point(control_net, knots_u, knots_v, p: int, q: int, u: float,
                          v: float) -> np.ndarray:
    """One point by the two passes: columns at u give the v-isocurve's polygon, evaluate it at v."""
    polygon = bspline_isocurve(control_net, knots_u, p, u=u)
    return bspline_curve(polygon, knots_v, q, [v])[0]


def bspline_surface(control_net, knots_u, knots_v, p: int, q: int, u, v) -> np.ndarray:
    """Evaluate on the grid u x v, shape (len(u), len(v), d): S = N(u)^T P N(v).

    The closed tensor form, for bulk evaluation -- the same trade as bezier_surface; the two-pass
    bspline_surface_point is the algorithm, and agrees with it (test_two_passes_match_the_grid).
    """
    net = _as_net(control_net)
    basis_u = bspline_basis(knots_u, p, np.atleast_1d(u))
    basis_v = bspline_basis(knots_v, q, np.atleast_1d(v))
    return np.einsum("ai,ijd,bj->abd", basis_u, net, basis_v)


def bspline_surface_partials(control_net, knots_u, knots_v, p: int, q: int, u, v):
    """S_u and S_v on the grid: differentiate every column (or row) with bspline_derivative.

    The derivative of a tensor product in u is the tensor product of the u-derivative curves with
    the untouched v-basis -- a B-spline surface of degree (p-1, q), on trimmed u-knots.
    """
    net = _as_net(control_net)
    d_u, ku, pu = _along(net, "u", lambda pts: bspline_derivative(pts, knots_u, p))
    d_v, kv, qv = _along(net, "v", lambda pts: bspline_derivative(pts, knots_v, q))
    return (bspline_surface(d_u, ku, knots_v, pu, q, u, v),
            bspline_surface(d_v, knots_u, kv, p, qv, u, v))


# --- NURBS surfaces ----------------------------------------------------------------------------


def nurbs_surface(control_net, weights, knots_u, knots_v, p: int, q: int, u, v) -> np.ndarray:
    """Lift, evaluate the B-spline surface one dimension up, project."""
    return _project(bspline_surface(_lift_net(control_net, weights), knots_u, knots_v, p, q, u, v))


def nurbs_surface_basis(weights, knots_u, knots_v, p: int, q: int, u: float, v: float) -> np.ndarray:
    """The rational tensor basis R_ij(u, v) = w_ij N_i(u) N_j(v) / sum_kl w_kl N_k(u) N_l(v).

    Note it does *not* factor into a u-part times a v-part unless the weights do (w_ij = a_i b_j).
    """
    w = np.asarray(weights, dtype=float)
    product = np.outer(bspline_basis(knots_u, p, [u])[0], bspline_basis(knots_v, q, [v])[0]) * w
    return product / product.sum()


def nurbs_surface_partials(control_net, weights, knots_u, knots_v, p: int, q: int, u, v):
    """S_u, S_v by the quotient rule upstairs, as nurbs_derivative does for curves:

        S_u = (A_u - w_u S) / w,   S_v = (A_v - w_v S) / w,   where A = (w S, w).
    """
    lifted = _lift_net(control_net, weights)
    A = bspline_surface(lifted, knots_u, knots_v, p, q, u, v)
    A_u, A_v = bspline_surface_partials(lifted, knots_u, knots_v, p, q, u, v)
    S = _project(A)
    w = A[..., -1:]
    return (A_u[..., :-1] - A_u[..., -1:] * S) / w, (A_v[..., :-1] - A_v[..., -1:] * S) / w


def nurbs_surface_normal(control_net, weights, knots_u, knots_v, p: int, q: int, u, v):
    """Unit normal S_u x S_v / |S_u x S_v| on the grid. Zero where the surface is singular
    (a collapsed row of the net, e.g. the poles of nurbs_sphere): there the normal exists only as
    a limit."""
    s_u, s_v = nurbs_surface_partials(control_net, weights, knots_u, knots_v, p, q, u, v)
    normal = np.cross(s_u, s_v)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(length > 0, normal / length, 0.0)


def nurbs_surface_insert_knot(control_net, weights, knots, degree: int, t: float, direction: str):
    """Insert t once in the u (or v) direction: Boehm on every column (or row) of the *lifted* net.

    Returns (net, weights, knots of that direction). Upstairs, for the same reason as for curves
    (notes/01_Curves/NURBS.md, "operations must happen upstairs").
    """
    lifted = _lift_net(control_net, weights)
    new_lifted, new_knots = _along(lifted, direction, lambda pts: insert_knot(pts, knots, degree, t))
    new_net, new_weights = _unlift_net(new_lifted)
    return new_net, new_weights, new_knots


# --- Surfaces of revolution ----------------------------------------------------------------------


def surface_of_revolution(profile_points, profile_weights, profile_knots, p: int):
    """Revolve a NURBS profile in the xz-plane about the z-axis, exactly.

    The profile point (r_i, z_i) is carried round on its own circle: it is copied onto the nine
    control points of nurbs_circle (the circumscribed square), scaled by its distance r_i from
    the axis, at height z_i. The weights multiply, w_ij = w_i * c_j, and that product is what
    makes the result exact: the homogeneous sum factors into (profile in u) x (circle in v), so

        S(u, v) = (r(u) cos theta(v), r(u) sin theta(v), z(u)),

    with (r(u), z(u)) the profile and (cos theta, sin theta) the NURBS circle
    (notes/02_Surfaces/Surfaces_of_Revolution.md). Profile points on the axis (r_i = 0) collapse
    a whole row of the net to one point: a pole.

    Returns (net, weights, knots_u, knots_v, p, q) with u along the profile, v around the axis.
    """
    profile = np.asarray(profile_points, dtype=float)
    pw = np.asarray(profile_weights, dtype=float)
    square, circle_w, knots_v, q = nurbs_circle(np.zeros(2), 1.0)
    r, z = profile[:, 0], profile[:, 1]
    net = np.empty((len(profile), len(square), 3))
    net[..., 0] = r[:, None] * square[None, :, 0]
    net[..., 1] = r[:, None] * square[None, :, 1]
    net[..., 2] = z[:, None]
    weights = pw[:, None] * circle_w[None, :]
    return net, weights, np.asarray(profile_knots, dtype=float), knots_v, p, q


_S = np.sqrt(0.5)


def cylinder_profile(radius: float, height: float):
    """A vertical segment at distance `radius` from the axis: (points, weights, knots, p)."""
    return np.array([[radius, 0.0], [radius, height]]), np.ones(2), clamped_knots(2, 1), 1


def cone_profile(radius: float, height: float):
    """A slanted segment from the base rim to the apex on the axis."""
    return np.array([[radius, 0.0], [0.0, height]]), np.ones(2), clamped_knots(2, 1), 1


def sphere_profile(radius: float):
    """A semicircle from the south pole to the north pole: two exact quarter arcs."""
    points = radius * np.array([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1]], float)
    weights = np.array([1, _S, 1, _S, 1])
    return points, weights, clamped_knots(5, 2, [0.5, 0.5]), 2


def torus_profile(major: float, minor: float):
    """A full circle of radius `minor` centred at distance `major` from the axis."""
    points, weights, knots, p = nurbs_circle(np.array([major, 0.0]), minor)
    return points, weights, knots, p


def nurbs_cylinder(radius: float, height: float):
    return surface_of_revolution(*cylinder_profile(radius, height))


def nurbs_cone(radius: float, height: float):
    return surface_of_revolution(*cone_profile(radius, height))


def nurbs_sphere(radius: float):
    return surface_of_revolution(*sphere_profile(radius))


def nurbs_torus(major: float, minor: float):
    return surface_of_revolution(*torus_profile(major, minor))
