"""Sweep: a planar profile carried along a space curve (the path), turned by a moving frame.

The exact swept surface is X(u, t) = C(t) + x(u) r(t) + y(u) s(t): profile point (x(u), y(u))
placed in the normal plane of the path at C(t), with (T, r, s) the frame there (geometry/frames.py).
For a general path and frame that is not a NURBS surface -- the frame alone involves |C'|, a
square root. So the sweep here is SIMPLIFIED: the profile is placed exactly at a number of
sections along the path (a rigid motion of a NURBS curve is exact), and a skin is interpolated
through them (geometry/skinning.py, as loft does). The surface is exact at the sections and
an approximation between them; 05_Sweep.md measures how good.

Topologically a sweep is an extrusion (sweep_lamina): the profile at t = 0, its copy at t = 1,
the skins' boundary curves as rails.

Notes: notes/04_Modeling_Operations/05_Sweep.md
"""
from __future__ import annotations

import numpy as np

from ..geometry.entities import NurbsCurve, Plane
from ..geometry.frames import default_normal, frenet_frames, rotation_minimizing_frames
from ..geometry.skinning import skin_parameters
from ..topology.halfedge import Solid
from .extrude import _check_profile, _oriented, sweep_lamina
from .loft import skin_faces
from .measure import GAUSS_POINTS, _gauss_on_spans

SUBSTEPS = 32


def path_frames(path: NurbsCurve, t, frame: str = "rmf", substeps: int = SUBSTEPS):
    """(C, T, r, s) at the increasing parameters t: points, unit tangents, and the frame's two
    normals, s = T x r.

    frame "frenet": r = N, the principal normal; an error where the curvature is zero.
    frame "rmf":    r rotation-minimizing, started at N(0) (or, if the curvature is zero at
                    t = 0, at default_normal) and carried by double reflection from t = 0 over
                    `substeps` steps per gap between consecutive t (and from 0 to t[0]).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if frame == "frenet":
        C, d1, d2 = path.derivatives(t, 2)
        _check_regular(d1)
        tangent, normal, binormal = frenet_frames(d1, d2)
        undefined = np.isnan(normal[:, 0])
        if undefined.any():
            raise ValueError(f"the Frenet frame is undefined at t = {t[undefined][0]:.4g}: the "
                             "path's curvature is zero there")
        return C, tangent, normal, binormal
    if frame != "rmf":
        raise ValueError(f"unknown frame {frame!r}")
    stops = np.unique(np.concatenate([[0.0], t]))
    grid = np.concatenate([np.linspace(a, b, substeps + 1)[:-1]
                           for a, b in zip(stops[:-1], stops[1:])] + [stops[-1:]])
    C, d1, d2 = path.derivatives(grid, 2)
    _check_regular(d1)
    tangent, normal, _ = frenet_frames(d1, d2)
    r0 = normal[0] if not np.isnan(normal[0, 0]) else default_normal(tangent[0])
    tangent, r, s = rotation_minimizing_frames(C, tangent, r0)
    pick = np.searchsorted(grid, t)
    return C[pick], tangent[pick], r[pick], s[pick]


def _check_regular(d1: np.ndarray) -> None:
    if np.linalg.norm(d1, axis=1).min() == 0.0:
        raise ValueError("the path must be regular (C' != 0): its tangent is undefined")


def arc_length(curve: NurbsCurve, n: int = GAUSS_POINTS) -> float:
    """L = \\int |C'(t)| dt, by Gauss-Legendre on every knot span."""
    t, w = _gauss_on_spans(curve.spans(), n)
    return float(w @ np.linalg.norm(curve.derivative(t), axis=1))


def place(profile: list[NurbsCurve], C, T, r, s) -> list[NurbsCurve]:
    """The profile, drawn in the xy-plane, moved to the normal plane at C: x along r, y along s,
    and the plane's normal z along T. A rotation plus a translation, so exact."""
    return [c.transformed(C, np.array([r, s, T])) for c in profile]


def sweep(profile: list[NurbsCurve], path: NurbsCurve, frame: str = "rmf", sections: int = 9,
          degree: int = 3, parametrization: str = "path", steps: list | None = None) -> Solid:
    """A solid swept by a closed profile along an open path.

    profile:  closed loop of curves in the xy-plane (z = 0); the origin is the point that runs
              along the path. Any orientation: it is made counter-clockwise seen from +z, which
              becomes the path's direction.
    path:     a regular curve (C' != 0), traversed from t = 0 to 1.
    frame:    "rmf" (rotation-minimizing, the default) or "frenet" (05_Sweep.md compares them).
    sections: how many copies of the profile are placed, at equally spaced t. More sections:
              closer to the exact sweep (measured in 05_Sweep.md).
    degree:   degree across the sections, lowered to sections - 1 if needed.
    parametrization: "path" skins with v_s = t_s, the path parameters of the sections; "chord"
              or "centripetal" space them as loft does (skin_parameters). "path" by default: it
              was the most accurate in 7 of the 8 cases of test_sweep_spacing_comparison.

    Not checked: that the sections do not cut each other -- a profile wider than the radius of
    curvature 1/kappa folds the sweep (05_Sweep.md).
    """
    _check_profile(list(profile))
    if np.abs(np.vstack([c.points for c in profile])[:, 2]).max() > 1e-9:
        raise ValueError("the profile must lie in the xy-plane (z = 0)")
    if sections < 2:
        raise ValueError("a sweep needs at least two sections")
    outer = _oriented(list(profile), np.array([0.0, 0.0, 1.0]))
    t = np.linspace(0.0, 1.0, sections)
    C, T, r, s = path_frames(path, t, frame)
    placed = [place(outer, *f) for f in zip(C, T, r, s)]
    columns = [[sec[k] for sec in placed] for k in range(len(outer))]
    if parametrization == "path":
        params = t
    else:
        params = skin_parameters([np.array([c.points[i] for c in col])
                                  for col in columns for i in range(len(col[0].points))],
                                 parametrization)
    sides, rails = skin_faces(columns, min(degree, sections - 1), params)
    return sweep_lamina(
        placed[0],
        far=placed[-1],
        rails=rails,
        sides=sides,
        start=Plane(C[0], -T[0]),
        end=Plane(C[-1], T[-1]),
        steps=steps,
    )[0]
