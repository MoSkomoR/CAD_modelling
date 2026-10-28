"""Revolve: a closed profile, turned about an axis, sweeps out a solid -- and a full turn needs
seams, closed edges and poles.

WHAT YOU SEE
------------
Left, the PROFILE in the half-plane of a sketch: the horizontal coordinate r is the distance from
the axis of revolution (the z-axis, the thick vertical line at r = 0), the vertical one is the
height z. The profile is a closed loop of cubic Bezier curves; the square markers are its
VERTICES, the round ones the other control points. Drag any of them. Points cannot cross the
axis: r is clamped at 0.05, and in the "touching the axis" profile the two end vertices are held
on it (r = 0).

Right, the SOLID that revolve() builds by turning the profile about the z-axis by the angle a:
  * profile edge k sweeps the FACE in colour k. Its surface is the surface of revolution
        S(u, v) = (x(u) cos phi(v), x(u) sin phi(v), z(u)),    u, v in [0, 1],
    where (x(u), z(u)) is the curve of profile edge k and phi(v) runs from 0 to a. It is an exact
    NURBS surface (notes/02_Surfaces/Surfaces_of_Revolution.md).
  * every vertex at distance r > 0 from the axis sweeps an ARC of radius r: an edge.
  * for a < 360 deg the solid has two PLANAR CAPS (grey): the profile where it starts and the
    profile turned by a.

At a = 360 deg the far copy of the profile coincides with the profile, the caps disappear, and
three things appear that a partial turn never needs (Revolve.md):
  * SEAM (red): a profile edge becomes an edge whose two half-edges both belong to one face --
    the line where that face's surface meets itself (v = 0 and v = 1 are the same points).
  * CLOSED EDGE (blue): each vertex off the axis sweeps a full circle, an edge that starts and
    ends at the same vertex.
  * POLE (orange): with the "touching the axis" profile, the edge on the axis sweeps nothing and
    is dropped; its two ends sweep nothing either, and stay single vertices without a circle.

The title gives the counts V, E, F, R (rings), G (genus: through-holes) and both sides of the
Euler-Poincare formula V - E + F - R = 2 (S - G), S = 1 shell. A full turn of the profile "off
the axis" makes a ring-shaped solid, G = 1; one "touching the axis" makes a ball-like one, G = 0.

The last line compares two volumes: the solid's, from the divergence theorem over its faces,
and PAPPUS's theorem, V = a * x_bar * A, with A the area the profile encloses and x_bar the
distance of the profile's centroid from the axis -- computed from the profile alone. Revolve.md
proves Pappus; the two agree to rounding at every angle and every drag.

Controls: the "profile" radio buttons switch between the two profiles, "angle a" turns, "nets"
shows each face's control net (the profile's control points carried round on the circle's).

Notes: notes/04_Modeling_Operations/Revolve.md
Run:   python scripts/04_modeling_ops/02_revolve.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.entities import NurbsCurve, Plane, xz
from cadkernel.ops.measure import volume
from cadkernel.ops.revolve import profile_area_and_centroid, revolve
from cadkernel.topology.halfedge import check, euler_poincare
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_radiobuttons,
    add_slider,
    run,
)
from cadkernel.viz.topology import face_collection, loop_polyline

PROFILES = {
    "off the axis": np.array([
        [2.6, -0.1], [2.7, 0.6], [2.2, 0.9],
        [1.7, 0.9], [1.1, 0.9], [0.9, 0.5],
        [0.9, 0.0], [0.9, -0.6], [1.2, -0.9],
        [1.8, -0.9], [2.4, -0.9], [2.5, -0.6],
    ]),
    "touching the axis": np.array([
        [0.0, -1.4], [0.9, -1.5], [1.6, -0.8],
        [1.5, 0.0], [1.4, 0.8], [0.8, 1.4],
        [0.0, 1.4],
    ]),
}
T = np.linspace(0.0, 1.0, 40)


def profile_curves(points, mode):
    """Cubic Beziers on points 3k .. 3k + 3. Off the axis the loop closes back to point 0; on the
    axis it closes with a straight edge along the axis from the last point to the first."""
    if mode == "off the axis":
        closed = np.vstack([points, points[:1]])
        curves = [NurbsCurve.bezier(closed[3 * k : 3 * k + 4]) for k in range(4)]
    else:
        curves = [NurbsCurve.bezier(points[3 * k : 3 * k + 4]) for k in range(2)]
        curves.append(NurbsCurve.line(points[-1], points[0]))
    return xz(curves)


def main():
    fig = plt.figure(figsize=(15, 8))
    ax2 = fig.add_axes([0.04, 0.3, 0.28, 0.6])
    ax3 = fig.add_axes([0.34, 0.12, 0.64, 0.8], projection="3d")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.4, 3.2)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_xlabel("r (distance from the axis)")
    ax2.set_ylabel("z")
    ax2.axvline(0.0, color="k", lw=3)
    ax2.set_title("profile (drag the markers)", fontsize=10)
    ax3.view_init(elev=22, azim=-55)
    state = {"mode": "off the axis"}
    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        clear()
        (show_nets,) = check_boxes.get_status()
        points = widget.points
        points[:, 0] = np.maximum(points[:, 0], 0.05)  # keep the profile off the far side
        if state["mode"] == "touching the axis":
            points[[0, -1], 0] = 0.0
        widget.artist.set_data(points[:, 0], points[:, 1])
        profile = profile_curves(points, state["mode"])
        angle = np.radians(s_angle.val)
        solid = revolve(profile, angle)
        state["solid"] = solid

        # -- left: the profile ---------------------------------------------------------------
        for k, curve in enumerate(profile):
            artists.extend(ax2.plot(curve(T)[:, 0], curve(T)[:, 2], color=f"C{k}", lw=2.4))
        artists.extend(ax2.plot(*points[::3].T, "s", color="k", ms=9, zorder=6))

        # -- right: faces, edges, vertices ----------------------------------------------------
        for face in solid.faces:
            if isinstance(face.surface, Plane):
                loops = [loop_polyline(loop) for loop in face.loops]
                artists.append(ax3.add_collection3d(face_collection([loops], ["0.7"], 0.4)))
                continue
            first = face.surface.net[:, 0]  # the profile curve, reversed if revolve reoriented it
            k = next(i for i, c in enumerate(profile)
                     if np.allclose(first, c.points) or np.allclose(first[::-1], c.points))
            grid = face.surface(T, np.linspace(0, 1, 48))
            artists.append(ax3.plot_surface(*grid.transpose(2, 0, 1), color=f"C{k}", alpha=0.4,
                                            lw=0))
            if show_nets:
                net = face.surface.net
                for line in (*net, *net.transpose(1, 0, 2)):
                    artists.extend(ax3.plot(*line.T, ":", color=f"C{k}", lw=0.8))
        seams = [e for e in solid.edges if e.he1.face is e.he2.face]
        closed = [e for e in solid.edges if e.he1.origin is e.he2.origin]
        for edge in solid.edges:
            style = (dict(color="r", lw=3.0) if edge in seams
                     else dict(color="b", lw=2.2) if edge in closed else dict(color="k", lw=1.3))
            artists.extend(ax3.plot(*edge.curve(np.linspace(0, 1, 97)).T, **style))
        vertices = np.array([v.point for v in solid.vertices])
        poles = np.hypot(vertices[:, 0], vertices[:, 1]) < 1e-9
        artists.extend(ax3.plot(*vertices[~poles].T, "o", color="k", ms=4))
        artists.extend(ax3.plot(*vertices[poles].T, "o", color="orange", ms=11, mec="k"))
        artists.extend(ax3.plot([0, 0], [0, 0], [-2.0, 2.0], "k--", lw=1))
        for setter in (ax3.set_xlim, ax3.set_ylim):
            setter(-3.0, 3.0)
        ax3.set_zlim(-3.0, 3.0)
        ax3.set_box_aspect((1, 1, 1))

        c = solid.counts()
        lhs, rhs = euler_poincare(c)
        area, x_bar = profile_area_and_centroid(profile)
        state.update(volume=volume(solid), pappus=angle * x_bar * area, sides=(lhs, rhs),
                     seams=len(seams), closed=len(closed), poles=int(poles.sum()))
        fig.suptitle(
            f"V={c['V']}  E={c['E']}  F={c['F']}  R={c['R']}  G={c['G']}:   "
            f"V - E + F - R = {lhs},  2(S - G) = {rhs}      "
            f"seams {len(seams)} (red),  closed edges {len(closed)} (blue),  "
            f"poles {int(poles.sum())} (orange)\n"
            f"volume = {state['volume']:.6f}      Pappus: a x_bar A = {angle:.4f} x "
            f"{x_bar:.4f} x {area:.4f} = {state['pappus']:.6f}",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def choose(label):
        state["mode"] = label
        widget.set_points(PROFILES[label])

    widget = DraggableControlPoints(ax2, PROFILES["off the axis"], on_change=redraw,
                                    show_polygon=True)
    s_angle = add_slider(fig, [0.45, 0.04, 0.35, 0.03], "angle a (deg)", 10, 360, 360, redraw)
    check_boxes = add_checkbuttons(fig, [0.06, 0.04, 0.12, 0.08], ["nets"], [False], redraw)
    radio = add_radiobuttons(fig, [0.19, 0.04, 0.13, 0.12], list(PROFILES), 0, choose)
    redraw()

    def smoke():
        def agrees():
            assert check(state["solid"]) == []
            assert abs(state["volume"] - state["pappus"]) < 1e-11
            lhs, rhs = state["sides"]
            assert lhs == rhs

        agrees()
        assert state["solid"].genus == 1 and (state["seams"], state["closed"]) == (4, 4)
        s_angle.set_val(120)
        agrees()
        assert state["solid"].genus == 0 and state["seams"] == state["closed"] == 0
        widget.simulate_drag(4, (1.4, 1.3))
        agrees()
        radio.set_active(1)
        s_angle.set_val(360)
        agrees()
        assert state["solid"].genus == 0 and state["poles"] == 2
        assert (state["seams"], state["closed"]) == (2, 1)
        widget.simulate_drag(3, (2.0, 0.3))
        check_boxes.set_active(0)
        agrees()
        print(f"  volume {state['volume']:.6f} = Pappus {state['pappus']:.6f}")

    run(fig, smoke)


if __name__ == "__main__":
    main()
