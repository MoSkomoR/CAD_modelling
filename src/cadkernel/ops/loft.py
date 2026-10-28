"""Loft: a solid through a sequence of closed sections -- skinning, with the extrude topology.

Each section is a closed profile of n edges, and edge k of every section belongs to side face k:
the face's surface is the skin through those curves (geometry/skinning.py). Vertex k
of every section lies on one longitudinal edge, the skin's u = 0 boundary curve. The first and
last sections are planar and become the caps. Topologically this is an extrusion
(sweep_lamina): only the curves and surfaces differ.

Notes: notes/04_Modeling_Operations/Loft.md
"""
from __future__ import annotations

import numpy as np

from ..geometry.entities import NurbsCurve, Plane
from ..geometry.skinning import compatible, skin_parameters, skinned_surface
from ..topology.halfedge import Solid
from .extrude import _check_profile, profile_vector_area, sweep_lamina


def _unit_normal(section: list[NurbsCurve], name: str) -> np.ndarray:
    """Unit normal of a planar section, from its vector area; rejects a non-planar one."""
    a = profile_vector_area(section)
    n = a / np.linalg.norm(a)
    controls = np.vstack([c.points for c in section])
    if np.abs((controls - section[0].start) @ n).max() > 1e-9:
        raise ValueError(f"the {name} section must be planar: it becomes a cap")
    return n


def loft(sections: list[list[NurbsCurve]], degree: int = 3,
         parametrization: str = "centripetal", steps: list | None = None) -> Solid:
    """A solid through closed sections, first to last.

    sections:        K + 1 >= 2 closed profiles with the same number n of edges; edge k of each
                     section is joined to edge k of the next, vertex k to vertex k. All must run
                     the same way round; first and last must be planar.
    degree:          degree across the sections, lowered to K if there are too few sections.
    parametrization: how the sections are spaced in v ("uniform", "chord", "centripetal"),
                     as in curve interpolation (skin_parameters). Centripetal by default: in
                     the two measurements of Loft.md it was never the best rule and never the
                     worst, while chord length (the textbook choice for skinning) was best on
                     one and pinched a waist into the solid on the other
                     (test_loft_parametrization_comparison, test_loft_overshoot_comparison).

    Every side face has sense +1: u runs along the sections counter-clockwise seen from the last
    section, v from the first section to the last (Loft.md).
    """
    if len(sections) < 2:
        raise ValueError("a loft needs at least two sections")
    n = len(sections[0])
    if any(len(s) != n for s in sections):
        raise ValueError("every section needs the same number of edges")
    for s in sections:
        _check_profile(list(s))
        if any(not np.isclose(c.weights[0], 1) or not np.isclose(c.weights[-1], 1) for c in s):
            raise ValueError("section curves must have end weights 1 (the longitudinal edges "
                             "are then polynomial and shared exactly by neighbouring faces)")
    centroid = [np.mean([c.start for c in s], axis=0) for s in sections]
    axis = centroid[-1] - centroid[0]
    turns = [np.sign(profile_vector_area(s) @ axis) for s in sections]
    if len(set(turns)) != 1 or turns[0] == 0:
        raise ValueError("the sections must all run the same way round")
    if turns[0] < 0:
        sections = [[c.reversed() for c in reversed(s)] for s in sections]

    columns = [compatible([s[k] for s in sections]) for k in range(n)]  # columns[k][s]
    params = skin_parameters([np.array([c.points[i] for c in col])
                              for col in columns for i in range(len(col[0].points))],
                             parametrization)
    q = min(degree, len(sections) - 1)
    sides = [skinned_surface(col, q, params) for col in columns]
    rails = [NurbsCurve(side.net[0], side.weights[0], side.knots_v, q) for side in sides]
    first = [col[0] for col in columns]
    last = [col[-1] for col in columns]
    return sweep_lamina(
        first,
        far=last,
        rails=rails,
        sides=sides,
        start=Plane(first[0].start, -_unit_normal(first, "first")),
        end=Plane(last[0].start, _unit_normal(last, "last")),
        steps=steps,
    )[0]
