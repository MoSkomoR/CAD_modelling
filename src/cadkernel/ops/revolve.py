"""Revolve: a closed profile in the xz-plane, turned about the z-axis, becomes a solid.

Every point of the profile moves on a circle about the axis, so every profile edge sweeps a
surface of revolution (module 02, exact) and every profile vertex sweeps an arc. What the
topology must be depends on two things, and each case is built from Euler operators differently:

  * a PARTIAL turn (angle < 2 pi): the extrude construction (sweep_lamina) with arcs for rails,
    surfaces of revolution for sides, and two planar caps. Genus 0.
  * a FULL turn, profile off the axis: the far copy of the profile *is* the profile, so there are
    no caps. Each profile edge becomes a SEAM -- one edge whose two half-edges both lie in the
    same face, where the face's surface meets itself -- and each vertex sweeps a full circle, a
    CLOSED edge whose two ends are the same vertex. Built as Mantyla's rotational sweep does it:
    the lamina, its two faces made one by kfmrh (genus 1), a closed edge by mekr, then one closed
    edge and face per remaining vertex by mef.
  * a FULL turn, profile with one edge on the axis: that edge sweeps nothing and is dropped, its
    two ends become POLES (a vertex with no circle). Built from the profile as an open wire, one
    closed edge and face per vertex off the axis by mef. Genus 0.

Proofs of the counts and the orientation, and Pappus's theorem for the volume, are in
notes/04_Modeling_Operations/Revolve.md.
"""
from __future__ import annotations

import numpy as np

from ..geometry.entities import NurbsCurve, Plane, revolved_surface, turning_arc
from ..topology.halfedge import (
    Solid,
    _record,
    attach_curve,
    lkfmrh,
    lmef,
    lmekr,
    mev,
    mvfs,
)
from .extrude import _check_profile, _oriented, lamina, sweep_lamina
from .measure import GAUSS_POINTS, _gauss_on_spans

AXIS_TOL = 1e-9
Y = np.array([0.0, 1.0, 0.0])  # the direction a point of the xz-plane (x > 0) starts to turn in


def _prepare(profile: list[NurbsCurve]) -> tuple[list[NurbsCurve], bool]:
    """Check the profile and orient it: counter-clockwise seen from the tip of +y, the direction
    the profile starts to move in (the same rule as extrude's). If one edge lies on the axis it
    is moved to the end of the list, so its vertices are the first and the last."""
    _check_profile(profile)
    controls = np.vstack([c.points for c in profile])
    if np.abs(controls[:, 1]).max() > AXIS_TOL:
        raise ValueError("the profile must lie in the xz-plane")
    if controls[:, 0].min() < -AXIS_TOL:
        raise ValueError("the profile crosses the axis")
    outer = _oriented(profile, Y)
    on_axis = [bool(np.all(np.abs(c.points[:, 0]) < AXIS_TOL)) for c in outer]
    if sum(on_axis) > 1:
        raise ValueError("at most one profile edge may lie on the axis")
    if any(on_axis):
        j = on_axis.index(True)
        outer = outer[j + 1:] + outer[: j + 1]
    axis = any(on_axis)
    n = len(outer)
    for k, c in enumerate(outer):
        if axis and k == n - 1:
            continue
        if np.any(c.points[1:-1, 0] < AXIS_TOL):
            raise ValueError("a profile edge touches the axis between its vertices")
        for j, x in ((k, c.start[0]), ((k + 1) % n, c.end[0])):
            if x < AXIS_TOL and not (axis and j in (0, n - 1)):
                raise ValueError("the profile touches the axis at a single vertex: the solid "
                                 "would be non-manifold there")
    return outer, axis


def revolve(profile: list[NurbsCurve], angle: float = 2 * np.pi,
            steps: list | None = None) -> Solid:
    """Turn a closed profile in the xz-plane (x >= 0) about the z-axis by `angle` (radians).

    profile: closed loop of curves in the xz-plane (see entities.xz), any orientation. At most
             one edge may lie on the axis; no other point of the profile may touch it.
    angle:   in (0, 2 pi]; 2 pi is a full turn.

    Edge k of the oriented profile sweeps a face whose surface is revolved_surface(curve k), with
    u along the profile and v around the axis; S_u x S_v then points out of the solid, so every
    face has sense +1 (Revolve.md).
    """
    if not 0 < angle <= 2 * np.pi + 1e-12:
        raise ValueError("the angle must lie in (0, 2 pi]")
    outer, axis = _prepare(profile)
    if np.isclose(angle, 2 * np.pi):
        return _full_turn_with_poles(outer, steps) if axis else _full_turn(outer, steps)
    n = len(outer)
    pole = [axis and k in (0, n - 1) for k in range(n)]
    return sweep_lamina(
        outer,
        far=[c.rotated(angle) for c in outer],
        rails=[None if pole[k] else turning_arc(c.start, angle) for k, c in enumerate(outer)],
        sides=[None if axis and k == n - 1 else revolved_surface(c, angle)
               for k, c in enumerate(outer)],
        start=Plane(np.zeros(3), -Y),
        end=Plane(np.zeros(3), [-np.sin(angle), np.cos(angle), 0.0]),
        steps=steps,
    )[0]


def _attach_circle(circle_edge, vertex, before_face) -> None:
    """The circle a vertex sweeps runs with increasing angle along its half-edge in the face of
    the profile edge *before* the vertex (and so against it in the face after). With u along the
    profile and v around the axis, that makes every face's loop counter-clockwise in (u, v):
    outward for sense +1 (Revolve.md, "Orientation of the closed edges")."""
    he = circle_edge.he1 if circle_edge.he1.face is before_face else circle_edge.he2
    attach_curve(circle_edge, turning_arc(vertex.point, 2 * np.pi), along=he)


def _full_turn(outer: list[NurbsCurve], steps: list | None) -> Solid:
    """No vertex on the axis: n seams, n closed edges, n faces, genus 1.

    lamina                  cap: e_0+ .. e_{n-1}+ (with the profile), bottom: the same backwards
    kfmrh(cap, bottom)      the bottom's loop becomes a ring of the cap        F-1, R+1, G+1
    mekr at vertex 0        a closed edge c_0 joins the two loops into one     E+1, R-1
    mef at vertex k         a closed edge c_k splits off the face of edge k-1  E+1, F+1 (k = 1..n-1)
    """
    solid, cap, bottom, verts = lamina(outer, steps)
    n = len(outer)
    forward = [e.he1 if e.he1.origin is verts[k] else e.he2 for k, e in enumerate(solid.edges)]
    backward = [he.twin for he in forward]
    lkfmrh(cap, bottom)
    _record(steps, solid, "kfmrh: the back of the lamina becomes a ring of its front -- genus 1")
    lmekr(forward[0], backward[n - 1])
    circles = [solid.edges[-1]]
    _record(steps, solid, "mekr: the circle of vertex 0 joins the two loops")
    faces = []
    for k in range(1, n):
        faces.append(lmef(backward[k - 1], forward[k]))
        circles.append(solid.edges[-1])
        _record(steps, solid, f"mef: the circle of vertex {k} closes the face of edge {k - 1}")
    faces.append(cap)
    for k, face in enumerate(faces):
        face.surface, face.sense = revolved_surface(outer[k]), 1
    for k, circle in enumerate(circles):
        _attach_circle(circle, verts[k], faces[k - 1])
    return solid


def _full_turn_with_poles(outer: list[NurbsCurve], steps: list | None) -> Solid:
    """Edge n-1 on the axis, vertices 0 and n-1 poles: n-1 seams, n-2 closed edges, n-1 faces.

    mvfs, mev x (n-1)    the profile without its axis edge: an open wire, one face whose loop
                         runs out along the wire and back
    mef at vertex k      a closed edge c_k, from the vertex to itself, cuts the loop there:
                         the part before it is the face of edge k-1    E+1, F+1 (k = 1..n-2)
    """
    solid, face, v0 = mvfs(outer[0].start)
    verts = [v0]
    for curve in outer[:-1]:
        verts.append(mev(face, verts[-1], curve.end))
        attach_curve(solid.edges[-1], curve)
    _record(steps, solid, f"mvfs, {len(outer) - 1} x mev: the profile without its axis edge")
    forward = [e.he1 if e.he1.origin is verts[k] else e.he2 for k, e in enumerate(solid.edges)]
    faces, circles = [], []
    for k in range(1, len(outer) - 1):
        rest = lmef(forward[k], forward[k - 1].twin)
        faces.append(face)
        circles.append(solid.edges[-1])
        face = rest
        _record(steps, solid, f"mef: the circle of vertex {k} closes the face of edge {k - 1}")
    faces.append(face)
    for k, f in enumerate(faces):
        f.surface, f.sense = revolved_surface(outer[k]), 1
    for k, circle in enumerate(circles, start=1):
        _attach_circle(circle, verts[k], faces[k - 1])
    return solid


def profile_area_and_centroid(profile: list[NurbsCurve], n: int = GAUSS_POINTS):
    """Area A of the region a closed xz-profile encloses, and the distance x_bar of its centroid
    from the axis, both from Green's theorem on the boundary (independent of any 3D surface):

        A = 1/2 \\oint (x dz - z dx),      A x_bar = \\iint x dA = 1/2 \\oint x^2 dz,

    signs taken for a counter-clockwise boundary in the (x, z) coordinates, then made positive.
    Pappus: the volume of the revolve is angle * x_bar * A (Revolve.md)."""
    area = moment = 0.0
    for c in profile:
        t, w = _gauss_on_spans(c.spans(), n)
        p, dp = c(t), c.derivative(t)
        x, z, dx, dz = p[:, 0], p[:, 2], dp[:, 0], dp[:, 2]
        area += 0.5 * w @ (x * dz - z * dx)
        moment += 0.5 * w @ (x * x * dz)
    return abs(area), moment / area
