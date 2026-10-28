"""Skinning: a surface through a sequence of section curves -- the geometry of a loft.

Given curves C^0, ..., C^K (the sections), find a surface S(u, v) and parameters
v_0 < ... < v_K with S(u, v_s) = C^s(u) for every s. Two steps (Piegl & Tiller, ch. 10.3):

  1. COMPATIBILITY. Line the sections' control points up: raise every curve to the highest degree
     (elevate_degree) and give every curve the union of all knot vectors (insert_knot). Neither
     step changes a curve's shape, only its representation, and afterwards curve s has control
     points P_0^s .. P_n^s on one common knot vector.
  2. INTERPOLATION ACROSS. For each i, the column P_i^0 .. P_i^K is a set of data points; one
     B-spline in v through all of them (bspline.py's interpolation, at shared parameters v_s)
     gives column i of the net. Then S(u, v_s) = sum_i N_i(u) P_i^s = C^s(u).

Both steps act on the lifted points (w P, w), as every NURBS operation must
(notes/01_Curves/NURBS.md). Notes: notes/04_Modeling_Operations/Loft.md
"""
from __future__ import annotations

import numpy as np

from .bspline import averaged_knots, bspline_basis, elevate_degree, insert_knot, interpolation_parameters
from .entities import NurbsCurve, NurbsSurface
from .nurbs import _lift, _unlift


def compatible(curves: list[NurbsCurve]) -> list[NurbsCurve]:
    """The same curves on one degree and one knot vector (shapes unchanged).

    Degree: elevate_degree on the lifted points until every curve has the highest degree.
    Knots: every knot value gets, in every curve, the highest multiplicity it has in any curve
    (Boehm insertion, again on the lifted points). Knots are compared exactly, so a knot that
    differs by rounding is inserted as a separate, very short span.
    """
    p = max(c.p for c in curves)
    lifted = []
    for c in curves:
        points, knots, d = _lift(c.points, c.weights), c.knots, c.p
        while d < p:
            points, knots, d = elevate_degree(points, knots, d)
        lifted.append((points, knots))
    values = np.unique(np.concatenate([k[p + 1:-p - 1] for _, k in lifted]))
    target = {v: max(int(np.sum(k == v)) for _, k in lifted) for v in values}
    out = []
    for points, knots in lifted:
        for v, m in target.items():
            for _ in range(m - int(np.sum(knots == v))):
                points, knots = insert_knot(points, knots, p, v)
        out.append(NurbsCurve(*_unlift(points), knots, p))
    return out


def skin_parameters(columns: list[np.ndarray], method: str = "centripetal") -> np.ndarray:
    """One parameter v_s per section, shared by every column of every face.

    Each column (the (K+1, 3) path of one control point through the sections) gets its own
    parameters by interpolation_parameters, and they are averaged (Piegl & Tiller's choice).
    Shared, because a loft has several faces side by side: two faces meet along a curve only if
    they interpolate the common boundary column at the same v_s with the same knots.
    Columns that do not move (zero length) carry no information and are skipped.
    """
    usable = [c for c in columns if np.linalg.norm(np.diff(c, axis=0), axis=1).sum() > 0]
    if not usable:
        raise ValueError("the sections coincide")
    return np.mean([interpolation_parameters(c, method) for c in usable], axis=0)


def skinned_surface(sections: list[NurbsCurve], q: int, params: np.ndarray) -> NurbsSurface:
    """The surface of degree q in v through compatible sections at the parameters `params`.

    One linear system M X = D solves for every column at once: M_{sj} = N_{j,q}(v_s) on the
    averaged knots, D_s the lifted control points of section s. Requires K + 1 >= q + 1 sections.
    """
    if len(sections) < q + 1:
        raise ValueError(f"degree {q} across needs at least {q + 1} sections")
    knots_v = averaged_knots(params, q)
    data = np.stack([_lift(c.points, c.weights) for c in sections])  # (K+1, n+1, 4)
    matrix = bspline_basis(knots_v, q, params)
    solved = np.linalg.solve(matrix, data.reshape(len(sections), -1)).reshape(data.shape)
    net_h = solved.transpose(1, 0, 2)  # (n+1, K+1, 4): u along the sections, v across
    weights = net_h[..., -1]
    if np.any(weights <= 0):
        raise ValueError("interpolating the weights across the sections made one non-positive")
    return NurbsSurface(net_h[..., :-1] / weights[..., None], weights, sections[0].knots,
                        knots_v, sections[0].p, q)
