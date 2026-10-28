"""Extrude: a closed planar profile, translated along a vector, becomes a solid.

The profile is a loop of edges, each carrying a NURBS curve -- what a CAD sketch is. Extrusion
is the box construction of module 03 (topology/halfedge.py, _box) with geometry attached as each
Euler operator runs (sweep_lamina, which revolve and loft reuse with other curves and surfaces):

    mvfs, mev x (n-1), mef     the profile as a lamina: two faces back to back, one per side
    mev x n                    a straight edge from every profile vertex, along d
    mef x n                    a side face between consecutive straight edges

Each operator creates an edge (solid.edges[-1]) or a face (mef's return value), and that is the
moment its curve or surface is known: a profile curve, its copy moved by d, a straight segment,
or the extruded surface S(u, v) = C(u) + v d. A profile with holes continues the way
block_with_hole does: each hole is drawn into the top face, cut free as a ring by kemr, swept
down, and opened through the bottom by kfmrh (genus + 1 per hole).

Notes: notes/04_Modeling_Operations/Extrude.md
"""
from __future__ import annotations

import numpy as np

from ..geometry.entities import NurbsCurve, Plane, extruded_surface
from ..topology.halfedge import (
    Solid,
    _record,
    attach_curve,
    kemr,
    lkfmrh,
    mef,
    mev,
    mvfs,
)
from .measure import GAUSS_POINTS, _gauss_on_spans


def profile_vector_area(profile: list[NurbsCurve], n: int = GAUSS_POINTS) -> np.ndarray:
    """1/2 \\oint x \\times dx around a closed profile: unit normal times enclosed area."""
    total = np.zeros(3)
    for curve in profile:
        t, w = _gauss_on_spans(curve.spans(), n)
        total += 0.5 * (w[:, None] * np.cross(curve(t), curve.derivative(t))).sum(axis=0)
    return total


def _check_profile(profile: list[NurbsCurve], tol: float = 1e-9) -> None:
    """Closed, and at least two edges. Not checked: that the profile does not cross itself, and
    that holes lie inside it without touching -- a real kernel must (Extrude.md)."""
    if len(profile) < 2:
        raise ValueError("a profile needs at least two edges (each edge has two vertices)")
    for a, b in zip(profile, profile[1:] + profile[:1]):
        if not np.allclose(a.end, b.start, atol=tol):
            raise ValueError("profile curves must join end to start, and close")


def _oriented(profile: list[NurbsCurve], d: np.ndarray) -> list[NurbsCurve]:
    """The profile counter-clockwise seen from the tip of d (vector area . d > 0)."""
    if profile_vector_area(profile) @ d > 0:
        return list(profile)
    return [c.reversed() for c in reversed(profile)]


def extrude(profile: list[NurbsCurve], direction, holes=(), steps: list | None = None) -> Solid:
    """Sweep a closed planar profile along the vector `direction` into a solid.

    profile:  closed loop of curves, each ending where the next starts (any orientation).
    holes:    closed loops inside the profile; each becomes a through-hole.
    steps:    if a list, a snapshot is appended after every operator (as in module 03).

    The bottom face is the profile's plane, the top face the plane moved by d, and edge k of the
    profile sweeps side face k. Every side face's surface is S(u, v) = C_k(u) + v d with u
    running along the profile counter-clockwise (seen from the tip of d); S_u x S_v then points
    out of the solid on the outer boundary and into the hole on a hole's wall, which is what
    `sense` records (+1 and -1). Proved in Extrude.md.
    """
    d = np.asarray(direction, dtype=float)
    _check_profile(profile)
    outer = _oriented(profile, d)
    normal = profile_vector_area(outer)
    normal /= np.linalg.norm(normal)
    for loop in [outer, *holes]:
        controls = np.vstack([c.points for c in loop])
        if np.abs((controls - outer[0].start) @ normal).max() > 1e-9:
            raise ValueError("profile and holes must lie in one plane")
    if abs(normal @ d) < 1e-12:
        raise ValueError("the extrusion direction lies in the profile's plane")

    solid, cap, bottom, up = sweep_lamina(
        outer,
        far=[c.translated(d) for c in outer],
        rails=[NurbsCurve.line(c.start, c.start + d) for c in outer],
        sides=[extruded_surface(c, d) for c in outer],
        start=Plane(outer[0].start, -normal),
        end=Plane(outer[0].start + d, normal),
        steps=steps,
    )
    for hole in holes:
        _check_profile(list(hole))
        _drill(solid, cap, bottom, up[0], _oriented(list(hole), d), d, steps)
    return solid


def lamina(profile: list[NurbsCurve], steps: list | None = None):
    """mvfs, mev x (n-1), mef: the profile as a lamina, two faces back to back. Returns
    (solid, cap, bottom, vertices): `cap` holds the profile's half-edges running along the
    profile (vertex k -> k + 1), `bottom` those running against it."""
    solid, cap, v0 = mvfs(profile[0].start)
    _record(steps, solid, "mvfs: the first profile vertex")
    verts = [v0]
    for curve in profile[:-1]:
        verts.append(mev(cap, verts[-1], curve.end))
        attach_curve(solid.edges[-1], curve)
    _record(steps, solid, f"{len(profile) - 1} x mev: the profile as an open wire")
    bottom = mef(cap, verts[-1], verts[0])
    attach_curve(solid.edges[-1], profile[-1])
    _record(steps, solid, "mef: close the profile -- a lamina, two faces back to back")
    return solid, cap, bottom, verts


def sweep_lamina(profile: list[NurbsCurve], far, rails, sides, start: Plane, end: Plane,
                 steps: list | None = None):
    """The construction extrude, a partial revolve and loft share: a profile swept to a far copy.

    profile:  the oriented profile, n curves; vertex k is profile[k].start.
    far:      far[k] is the far copy of profile edge k (the top of an extrusion).
    rails:    rails[k] is the curve vertex k travels along to its far copy -- or None, where the
              vertex does not move (on the axis of a revolve: a pole). No edge is made there.
    sides:    sides[k] is the surface profile edge k sweeps, with u along the edge and v from
              the profile to the far copy (sense +1) -- or None for an edge that sweeps no face
              (lying on the axis of a revolve). Its far copy is then the edge itself.
    start, end: the planes of the start cap (the lamina's bottom) and the end cap.

    Topology as in module 03's box: the lamina, mev along every rail, mef for every side.
    Returns (solid, end cap, start cap, far vertices).
    """
    solid, cap, bottom, verts = lamina(profile, steps)
    bottom.surface = start
    up = []
    for v, rail in zip(verts, rails):
        if rail is None:
            up.append(v)
            continue
        up.append(mev(cap, v, rail.end))
        attach_curve(solid.edges[-1], rail)
    _record(steps, solid, f"{sum(r is not None for r in rails)} x mev: a rail from every vertex")
    for k, side in enumerate(sides):
        if side is None:
            continue
        face = mef(cap, up[k], up[(k + 1) % len(up)])
        attach_curve(solid.edges[-1], far[k])
        face.surface, face.sense = side, 1
        _record(steps, solid, f"mef: side face {k}, swept by profile edge {k}")
    cap.surface = end
    return solid, cap, bottom, up


def _drill(solid: Solid, top, bottom, corner, hole: list[NurbsCurve], d: np.ndarray,
           steps: list | None) -> None:
    """A through-hole, exactly as block_with_hole makes one: a spike from a top corner and the
    hole's outline drawn on the top face (mev), closed into a face (mef), the spike cut so the
    outline is a ring of the top (kemr), that face swept down to the bottom (mev, mef), and
    merged into the bottom as a second ring (kfmrh) -- the step that raises the genus."""
    first = mev(top, corner, hole[0].start + d)  # the spike; kemr removes it below
    inner = [first]
    for curve in hole[:-1]:
        inner.append(mev(top, inner[-1], curve.end + d))
        attach_curve(solid.edges[-1], curve.translated(d))
    _record(steps, solid, "mev: a spike and the hole's outline on the top face")
    plug = mef(top, inner[0], inner[-1], v1_to=inner[1])
    attach_curve(solid.edges[-1], hole[-1].translated(d))
    _record(steps, solid, "mef: close the outline -- a new face inside the top")
    kemr(top, corner, first)
    _record(steps, solid, "kemr: cut the spike -- the outline is now a ring of the top")
    low = []
    for v in inner:
        low.append(mev(plug, v, v.point - d))
        attach_curve(solid.edges[-1], NurbsCurve.line(v.point - d, v.point))
    _record(steps, solid, f"{len(inner)} x mev: sweep the hole's face down")
    for k, curve in enumerate(hole):
        wall = mef(plug, low[k], low[(k + 1) % len(low)])
        attach_curve(solid.edges[-1], curve)
        wall.surface, wall.sense = extruded_surface(curve, d), -1
        _record(steps, solid, f"mef: wall {k} of the hole")
    lkfmrh(bottom, plug)
    _record(steps, solid, "kfmrh: the hole's face becomes a ring of the bottom -- genus + 1")
