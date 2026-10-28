"""Extrude: a closed profile, moved along a vector d, sweeps out a solid with exact curved faces.

WHAT YOU SEE
------------
Left, the PROFILE: a closed curve in the xy-plane made of four cubic Bezier curves joined end to
end. The four square markers are where they join: the profile's VERTICES. The round markers are
the control points of the four curves; drag any marker. The profile is a loop of four EDGES,
each carrying its own curve -- exactly what a sketch in a CAD program is.

Right, the SOLID that extrude() builds from it:
  * the BOTTOM face is the region of the plane z = 0 the profile encloses; the TOP face is the
    same region moved by d. Both are PLANAR faces whose boundary is curved: a plane says nothing
    about where the face ends, its edges do.
  * profile edge k sweeps SIDE FACE k (one colour each). Its surface is the extruded surface
        S_k(u, v) = C_k(u) + v d,     u, v in [0, 1],
    where C_k is the curve of profile edge k and d the extrusion vector. Every point of the curve
    moves along the same straight line segment, so the side face is RULED: through every point
    there is a straight line on the surface (tick "rulings": the lines u = const).
  * black curves are the EDGES: profile edges, their copies moved by d, and one straight edge
    along d from every vertex. Black dots are the vertices.

Controls:
  * "height" is the component of d along the z-axis, h; "tilt" leans d sideways by an angle a,
    so d = (h tan a, 0, h). Tilting shears the solid, and the volume printed in the title does
    not change: the volume is (profile area) x h for every tilt (Cavalieri's principle, proved in
    Extrude.md as a shear with determinant 1).
  * "hole" adds a circle of radius 0.6 around the origin as a hole: the solid then has a
    through-hole, its genus G goes from 0 to 1, and the hole's walls are four quarter-cylinders.
  * "rulings" draws the straight lines u = const on each side face.
  * "side-face nets" draws each side face's control net: two columns, the curve's control points
    P_i and the same points moved by d. A degree-(3, 1) NURBS surface, with exactly that net,
    is the extruded surface exactly (Extrude.md derives it).

The title gives the counts V (vertices), E (edges), F (faces), R (rings: inner boundary loops of
faces), G (genus: through-holes), and both sides of the Euler-Poincare formula
V - E + F - R = 2 (S - G) with S = 1 shell (notes/03_Topology/Euler_Operators.md).

Notes: notes/04_Modeling_Operations/Extrude.md
Run:   python scripts/04_modeling_ops/01_extrude.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.entities import NurbsCurve, Plane, circle_arcs
from cadkernel.ops.extrude import extrude, profile_vector_area
from cadkernel.ops.measure import volume
from cadkernel.topology.halfedge import check, euler_poincare
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_slider,
    run,
)
from cadkernel.viz.topology import face_collection, loop_polyline

PROFILE = np.array(
    [
        [2.2, -0.2], [2.4, 1.0], [1.4, 2.0],
        [0.0, 2.0], [-1.2, 2.0], [-2.2, 1.4],
        [-2.2, 0.0], [-2.2, -1.2], [-1.4, -2.2],
        [0.2, -2.0], [1.4, -1.9], [2.0, -1.4],
    ]
)
HOLE = circle_arcs([0.0, 0.0], 0.6)
T = np.linspace(0.0, 1.0, 40)


def profile_curves(points):
    """Four cubic Beziers; curve k uses control points 3k .. 3k + 3 (the last wraps to 0)."""
    closed = np.vstack([points, points[:1]])
    return [NurbsCurve.bezier(closed[3 * k : 3 * k + 4]) for k in range(4)]


def source_edge(face, profile):
    """Which profile edge swept this side face: its net's first column is that curve's control
    polygon (reversed if extrude had to reorient the profile)."""
    first = face.surface.net[:, 0, :2]
    return next(k for k, c in enumerate(profile)
                if np.allclose(first, c.points[:, :2]) or np.allclose(first[::-1], c.points[:, :2]))


def main():
    fig = plt.figure(figsize=(15, 8))
    ax2 = fig.add_axes([0.04, 0.3, 0.3, 0.6])
    ax3 = fig.add_axes([0.36, 0.14, 0.62, 0.8], projection="3d")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_title("profile (drag the markers)", fontsize=10)
    ax3.view_init(elev=24, azim=-60)
    state = {}
    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        clear()
        show_hole, show_rulings, show_nets = check_boxes.get_status()
        profile = profile_curves(widget.points)
        holes = [HOLE] if show_hole else []
        h, tilt = s_height.val, np.radians(s_tilt.val)
        d = np.array([h * np.tan(tilt), 0.0, h])
        solid = extrude(profile, d, holes)
        state["solid"] = solid

        # -- left: the profile, its vertices, the hole ---------------------------------------
        for k, curve in enumerate(profile):
            artists.extend(ax2.plot(*curve(T)[:, :2].T, color=f"C{k}", lw=2.4))
        artists.extend(ax2.plot(*widget.points[::3].T, "s", color="k", ms=9, zorder=6))
        if show_hole:
            for arc in HOLE:
                artists.extend(ax2.plot(*arc(T)[:, :2].T, "--", color="0.3", lw=1.5))

        # -- right: faces, edges, vertices ----------------------------------------------------
        for face in solid.faces:
            if isinstance(face.surface, Plane):
                loops = [loop_polyline(loop) for loop in face.loops]
                artists.append(
                    ax3.add_collection3d(face_collection([loops], ["0.75"], alpha=0.35))
                )
                continue
            color = f"C{source_edge(face, profile)}" if face.sense == 1 else "0.5"
            grid = face.surface(T, np.linspace(0, 1, 6))
            artists.append(
                ax3.plot_surface(*grid.transpose(2, 0, 1), color=color, alpha=0.45, lw=0)
            )
            if show_rulings:
                for line in face.surface(np.linspace(0, 1, 7), [0.0, 1.0]):
                    artists.extend(ax3.plot(*line.T, color=color, lw=0.8))
            if show_nets:
                net = face.surface.net
                for column in net.transpose(1, 0, 2):
                    artists.extend(ax3.plot(*column.T, "o--", color=color, ms=3, lw=0.8))
                for row in net:
                    artists.extend(ax3.plot(*row.T, ":", color=color, lw=0.8))
        for edge in solid.edges:
            artists.extend(ax3.plot(*edge.curve(T).T, color="k", lw=1.4))
        points = np.array([v.point for v in solid.vertices])
        artists.extend(ax3.plot(*points.T, "o", color="k", ms=4))

        all_points = np.vstack([points, np.c_[widget.points, np.zeros(len(widget.points))]])
        lo, hi = all_points.min(axis=0) - 0.3, all_points.max(axis=0) + 0.3
        span = (hi - lo).max()
        mid = 0.5 * (lo + hi)
        for setter, m in zip((ax3.set_xlim, ax3.set_ylim, ax3.set_zlim), mid):
            setter(m - span / 2, m + span / 2)
        ax3.set_box_aspect((1, 1, 1))

        c = solid.counts()
        lhs, rhs = euler_poincare(c)
        area = profile_vector_area(profile)[2]
        hole_area = np.pi * 0.36 if show_hole else 0.0
        state["volume"] = volume(solid)
        state["area"] = abs(area) - hole_area
        state["sides"] = (lhs, rhs)
        fig.suptitle(
            f"V={c['V']}  E={c['E']}  F={c['F']}  R={c['R']}  G={c['G']}:   "
            f"V - E + F - R = {lhs},  2(S - G) = {rhs}\n"
            f"volume = {state['volume']:.4f}   =   area {state['area']:.4f}  x  height {h:.2f}",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    widget = DraggableControlPoints(ax2, PROFILE, on_change=redraw, show_polygon=True)
    s_height = add_slider(fig, [0.45, 0.07, 0.3, 0.03], "height h", 0.2, 3.0, 1.5, redraw)
    s_tilt = add_slider(fig, [0.45, 0.03, 0.3, 0.03], "tilt a (deg)", -50, 50, 0, redraw)
    check_boxes = add_checkbuttons(
        fig, [0.06, 0.05, 0.18, 0.15], ["hole", "rulings", "side-face nets"],
        [False, True, False], redraw,
    )
    redraw()

    def smoke():
        assert check(state["solid"]) == []
        assert np.isclose(state["volume"], state["area"] * 1.5)
        upright = state["volume"]
        s_tilt.set_val(35)
        assert np.isclose(state["volume"], upright)  # Cavalieri: tilting keeps the volume
        check_boxes.set_active(0)  # the hole
        assert state["sides"] == (0, 0) and state["solid"].genus == 1
        assert check(state["solid"]) == []
        assert np.isclose(state["volume"], state["area"] * 1.5)
        widget.simulate_drag(1, (2.8, 1.6))
        check_boxes.set_active(2)
        assert check(state["solid"]) == []
        assert np.isclose(state["volume"], state["area"] * 1.5)
        print(f"  volume {state['volume']:.4f} = area x height, at every tilt")

    run(fig, smoke)


if __name__ == "__main__":
    main()
