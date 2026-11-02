"""Moving frames along a space curve: how a sweep decides which way the profile faces.

A sweep places a planar profile at every point C(t) of a path, in the plane normal to the
tangent T(t). The tangent fixes that plane but not the rotation of the profile inside it: a
second unit vector r(t), normal to T, is needed, and (T, r, T x r) is the frame. Two choices:

  * FRENET. r = N, the principal normal, from C' and C''. Defined only where the curvature is
    non-zero: undefined on a straight stretch, and on a planar curve it flips to the other side
    at an inflection. Where it is defined it turns about T at the rate of the torsion.
  * ROTATION-MINIMIZING (RMF). r' parallel to T: r never turns about the tangent. Defined
    wherever T is, by a linear ODE from any start r_0. Computed here by double reflection
    (Wang, Juttler, Zheng & Liu 2008, cited), with the obvious alternative, projecting r_i onto
    the next normal plane, kept for comparison.

Notes: notes/04_Modeling_Operations/05_Sweep.md
"""
from __future__ import annotations

import numpy as np


def frenet_frames(d1: np.ndarray, d2: np.ndarray, tol: float = 1e-9):
    """(T, N, B) from the first two derivatives C', C'' (rows): T = C'/|C'|,
    B = C' x C'' / |C' x C''|, N = B x T. Rows are NaN where the curvature
    |C' x C''| / |C'|^3 is below tol: there N and B do not exist."""
    d1, d2 = np.atleast_2d(d1), np.atleast_2d(d2)
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    tangent = d1 / speed
    cross = np.cross(d1, d2)
    size = np.linalg.norm(cross, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        binormal = np.where(size > tol * speed**3, cross / size, np.nan)
    return tangent, np.cross(binormal, tangent), binormal


def curvature_torsion(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray):
    """kappa = |C' x C''| / |C'|^3 and tau = (C' x C'') . C''' / |C' x C''|^2, for any
    parametrization (do Carmo, sec. 1-5, cited)."""
    cross = np.cross(d1, d2)
    size2 = np.einsum("ij,ij->i", cross, cross)
    kappa = np.sqrt(size2) / np.linalg.norm(d1, axis=1) ** 3
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.einsum("ij,ij->i", cross, d3) / size2
    return kappa, tau


def _reflect(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """x reflected in the plane through the origin normal to v."""
    return x - (2.0 * (v @ x) / (v @ v)) * v


def rotation_minimizing_frames(points: np.ndarray, tangents: np.ndarray, r0,
                               method: str = "double_reflection"):
    """(T, R, S) at the sample points: R rotation-minimizing, started at r0, and S = T x R.

    points, tangents: C(t_i) and the unit tangents T(t_i), for increasing t_i.
    r0:               the first R; its component along T_0 is removed and it is normalized.
    method:
      "double_reflection"  R_1 reflects the frame in the plane bisecting C(t_i) and C(t_{i+1}),
                           R_2 in the plane bisecting the reflected tangent and T_{i+1}. R_2 R_1
                           is a rotation taking T_i to T_{i+1}, so R stays a unit normal exactly;
                           it approximates the RMF to fourth order in the step (Wang et al.
                           2008, cited; measured in test_frames.py).
      "projection"         R_{i+1} = R_i minus its T_{i+1} component, normalized. The obvious
                           scheme, kept to be measured against.
    """
    points, tangents = np.asarray(points, float), np.asarray(tangents, float)
    r = np.asarray(r0, dtype=float)
    r = r - (r @ tangents[0]) * tangents[0]
    rs = [r / np.linalg.norm(r)]
    for i in range(len(points) - 1):
        r, t_i, t_next = rs[-1], tangents[i], tangents[i + 1]
        if method == "double_reflection":
            v1 = points[i + 1] - points[i]
            if v1 @ v1 == 0.0:
                rs.append(r)
                continue
            r_l, t_l = _reflect(r, v1), _reflect(t_i, v1)
            v2 = t_next - t_l
            rs.append(_reflect(r_l, v2) if v2 @ v2 > 0.0 else r_l)
        elif method == "projection":
            r = r - (r @ t_next) * t_next
            rs.append(r / np.linalg.norm(r))
        else:
            raise ValueError(f"unknown method {method!r}")
    rs = np.array(rs)
    return tangents, rs, np.cross(tangents, rs)


def angle_about_tangent(r_from: np.ndarray, s_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """The angle of r_to measured in the normal plane from r_from towards s_from = T x r_from,
    row by row, in (-pi, pi]."""
    return np.arctan2(np.einsum("ij,ij->i", r_to, s_from), np.einsum("ij,ij->i", r_to, r_from))


def default_normal(tangent: np.ndarray) -> np.ndarray:
    """A unit vector normal to `tangent`: the coordinate axis least aligned with it, with its
    tangent component removed. Used to start a frame where the Frenet normal does not exist."""
    axis = np.eye(3)[np.argmin(np.abs(tangent))]
    r = axis - (axis @ tangent) * tangent
    return r / np.linalg.norm(r)
