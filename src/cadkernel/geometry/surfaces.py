"""Tensor-product Bezier surfaces, evaluated by running De Casteljau's algorithm twice.

The whole module is one idea: a Bezier *curve* takes a row of control points and collapses it
to a point by repeated lerp; a Bezier *surface* takes a rectangular net of control points and
collapses it to a point by doing that twice -- once down every column, then once along the row
of results. No new algorithm is needed, only the observation that the intermediate results of
De Casteljau are points, so they can be fed straight back into De Casteljau.

Notes: notes/02_Surfaces/Bezier_Surfaces.md
"""
from __future__ import annotations

import numpy as np

from .curves import bernstein_basis, bezier_de_casteljau, bezier_subdivide


def _as_net(control_net: np.ndarray) -> np.ndarray:
    """Control net as an (n+1, m+1, d) array: index [i, j] is P_ij, i along u, j along v."""
    net = np.asarray(control_net, dtype=float)
    if net.ndim != 3:
        raise ValueError(f"control net must be (n+1, m+1, dim), got shape {net.shape}")
    return net


def bezier_surface_point(control_net: np.ndarray, u: float, v: float) -> np.ndarray:
    """Evaluate S(u, v) on a tensor-product Bezier patch by two passes of De Casteljau.

    Pass 1 collapses the u direction: each *column* net[:, j] is the control polygon of a
    degree-n Bezier curve, so De Casteljau at u turns it into a single point Q_j.
    Pass 2 collapses the v direction: Q_0..Q_m is itself the control polygon of a degree-m
    Bezier curve -- the isoparametric curve v -> S(u, v) -- so De Casteljau at v finishes it.

    Doing v first and u second gives the same point (test_surface_evaluation_order_does_not_
    matter): the double sum S = sum_i sum_j B_i^n(u) B_j^m(v) P_ij is symmetric in how the two
    sums are nested. That is the definition of a *tensor-product* surface, and it is why the
    curve machinery transfers without a single new derivation.
    """
    net = _as_net(control_net)
    column_points = np.array([bezier_de_casteljau(net[:, j], u) for j in range(net.shape[1])])
    return bezier_de_casteljau(column_points, v)


def bezier_surface(control_net: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate the patch on the grid u x v, returning shape (len(u), len(v), dim).

    Uses the closed tensor-product Bernstein form -- S = B(u)^T P B(v) -- because for a whole
    grid that is one pair of matrix contractions instead of len(u)*len(v) separate De Casteljau
    runs. Identical results (test_surface_bernstein_matches_de_casteljau); the algorithm is the
    right tool for *one* point and for subdivision, the closed form for bulk tessellation.
    """
    net = _as_net(control_net)
    n, m = net.shape[0] - 1, net.shape[1] - 1
    basis_u = bernstein_basis(n, np.atleast_1d(u))          # (len(u), n+1)
    basis_v = bernstein_basis(m, np.atleast_1d(v))          # (len(v), m+1)
    return np.einsum("ai,ijd,bj->abd", basis_u, net, basis_v)


def de_casteljau_surface_stages(control_net: np.ndarray, u: float, v: float) -> dict:
    """Every intermediate object of the two-pass evaluation, for the interactive script.

    Returns the m+1 points Q_j left by the u pass (which are the control polygon of the
    v-isocurve through S(u, v)), the mirrored n+1 points R_i from the v pass, and the point
    itself. Drawing all of it at once is what makes "a surface is De Casteljau, twice" visible
    rather than merely stated.
    """
    net = _as_net(control_net)
    n, m = net.shape[0] - 1, net.shape[1] - 1
    q = np.array([bezier_de_casteljau(net[:, j], u) for j in range(m + 1)])
    r = np.array([bezier_de_casteljau(net[i, :], v) for i in range(n + 1)])
    return {
        "u_pass": q,                                  # control polygon of v -> S(u, v)
        "v_pass": r,                                  # control polygon of u -> S(u, v)
        "point": bezier_de_casteljau(q, v),
        "point_other_order": bezier_de_casteljau(r, u),
    }


def bezier_isocurve(control_net: np.ndarray, *, u: float | None = None,
                    v: float | None = None) -> np.ndarray:
    """Control points of an isoparametric curve of the patch.

    Fixing one parameter leaves an honest Bezier curve of the *other* direction's degree, and
    its control points are exactly the points the first De Casteljau pass already produced.
    So every curve routine written in module 01 -- subdivision, hull bounds, the flatness test,
    ray/curve intersection -- applies to a surface's isocurves for free.
    """
    net = _as_net(control_net)
    if (u is None) == (v is None):
        raise ValueError("fix exactly one of u or v")
    if u is not None:
        return np.array([bezier_de_casteljau(net[:, j], u) for j in range(net.shape[1])])
    return np.array([bezier_de_casteljau(net[i, :], v) for i in range(net.shape[0])])


def bezier_surface_partials(control_net: np.ndarray, u: float,
                            v: float) -> tuple[np.ndarray, np.ndarray]:
    """The two tangent vectors S_u(u, v) and S_v(u, v).

    Differentiating the Bernstein form in u drops the degree by one and replaces the control
    net by its forward differences, scaled by n -- the surface analogue of the curve hodograph
    C'(t) = n (P_1^{n-1} - P_0^{n-1}) from notes/01_Curves/Bezier_Curves.md. The cross product
    of the two is the (unnormalized) surface normal.
    """
    net = _as_net(control_net)
    n, m = net.shape[0] - 1, net.shape[1] - 1
    s_u = n * bezier_surface_point(np.diff(net, axis=0), u, v) if n > 0 else np.zeros(net.shape[2])
    s_v = m * bezier_surface_point(np.diff(net, axis=1), u, v) if m > 0 else np.zeros(net.shape[2])
    return s_u, s_v


def bezier_surface_normal(control_net: np.ndarray, u: float, v: float) -> np.ndarray:
    """Unit surface normal, S_u x S_v normalized. Degenerate (zero-length) at singular points."""
    s_u, s_v = bezier_surface_partials(control_net, u, v)
    normal = np.cross(s_u, s_v)
    length = np.linalg.norm(normal)
    return normal / length if length > 0 else normal


def bezier_surface_subdivide(control_net: np.ndarray, u: float,
                             v: float) -> list[list[np.ndarray]]:
    """Split the patch at (u, v) into four sub-patches, returned as [[SW, SE], [NW, NE]].

    Subdivision -- the property that makes De Casteljau worth its O(n^2) on curves -- survives
    the move to surfaces unchanged, because the tensor-product structure lets it be applied one
    direction at a time: split every column at u, then split every row of both halves at v.
    Each piece is a Bezier patch of the same bidegree covering one quarter of the parameter
    domain, so subdivide-and-cull against control-net bounding boxes works on surfaces exactly
    as it does on curves. This is the backbone of real surface/surface intersection.
    """
    net = _as_net(control_net)
    lower, upper = (np.stack(halves, axis=1) for halves in zip(
        *(bezier_subdivide(net[:, j], u) for j in range(net.shape[1]))))
    quadrants = []
    for half in (lower, upper):
        left, right = (np.stack(pieces, axis=0) for pieces in zip(
            *(bezier_subdivide(half[i, :], v) for i in range(half.shape[0]))))
        quadrants.append([left, right])
    return quadrants
