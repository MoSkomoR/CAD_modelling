"""Loft: a solid through a stack of cross-sections, square to circle to square -- and how the
spacing of the sections in the parameter v shapes everything between them.

WHAT YOU SEE
------------
Right, four SECTIONS (closed profiles in horizontal planes, drawn in black) and the solid that
loft() builds through them:
    section 0   a square,           at height z_0 = 0
    section 1   a circle,           at height z_1
    section 2   a circle,           at height z_2
    section 3   a smaller square,   at height z_3
Every section is a loop of four edges with four VERTICES; vertex k of each section is joined to
vertex k of the next by a LONGITUDINAL EDGE (thick dark curves), and edge k of every section
belongs to SIDE FACE k (one colour each). The bottom and top sections are planar CAPS.

Left, the same solid seen in one number per height: the horizontal axis is the distance rho of
a vertex from the z-axis, the vertical axis the height z. Each marker is one section's (rho, z):
drag it to resize the section or move it up and down. The curve is a longitudinal edge, drawn
the same way: distance from the axis against height.

How the surface is made (Loft.md):
  1. COMPATIBILITY. Edge k of the four sections -- two straight lines and two circular arcs --
     is rewritten on one degree and one knot vector, without changing its shape (degree
     elevation, knot insertion; module 01). Their control points now line up: P_i^s, control
     point i of section s.
  2. SKINNING. For each i, one B-spline in v interpolates P_i^0, ..., P_i^3 at parameters
     v_0 = 0 < v_1 < v_2 < v_3 = 1. That is the curve interpolation of
     scripts/01_curves/07_bspline_interpolation.py, done once per column of control points.
     The surface then passes through every section exactly: S(u, v_s) = section s.

Controls:
  * "spacing" chooses how the v_s are spaced, as in curve interpolation: uniform (equal steps),
    chord (steps proportional to the distance between consecutive sections), centripetal
    (proportional to its square root). Pull the two circles close together (z_1 = 1.0,
    z_2 = 1.25) and switch: with chord spacing the longitudinal edges swing into a waist far
    thinner than any section (rho down to 0.23, against 0.6 for the thinnest section);
    centripetal swings less, uniform hardly at all. On
    other data chord is the best of the three (Loft.md measures both); no rule wins everywhere,
    and loft() defaults to centripetal, which was never the worst.
  * "degree across" is the degree of the B-splines in v: 1 gives straight longitudinal edges
    (a ruled surface between consecutive sections), 3 a smooth (C^2) one.
  * "twist top" turns the top square by 30 deg: vertex k still joins vertex k, so every side
    face twists.
  * "skin nets" draws the control nets of the side faces.

The title gives V, E, F and the volume (divergence theorem over the faces, as in Extrude.md).

Notes: notes/04_Modeling_Operations/Loft.md
Run:   python scripts/04_modeling_ops/03_loft.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.entities import NurbsCurve, Plane, circle_arcs, polygon_segments
from cadkernel.ops.loft import loft
from cadkernel.ops.measure import volume
from cadkernel.topology.halfedge import check
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_radiobuttons,
    add_slider,
    run,
)
from cadkernel.viz.topology import face_collection, loop_polyline

SECTIONS = np.array([[1.4, 0.0], [1.0, 1.0], [0.6, 2.6], [0.6, 3.0]])  # (rho, z) per section
SHAPES = ["square", "circle", "circle", "square"]
T = np.linspace(0.0, 1.0, 40)


def section(shape, rho, z, turn=0.0):
    """A closed loop of four edges whose vertices are at distance rho from the z-axis, at
    angles -45, 45, 135, 225 deg (+ turn): the square's corners, or the circle cut into four arcs
    at the same angles, so vertex k sits at the same angle in every section."""
    if shape == "square":
        a = np.radians(-45 + 90 * np.arange(4)) + turn
        curves = polygon_segments(np.c_[rho * np.cos(a), rho * np.sin(a)])
    else:
        curves = [c.rotated(-np.pi / 4 + turn) for c in circle_arcs([0, 0], rho)]
    return [NurbsCurve(c.points + [0, 0, z], c.weights, c.knots, c.p) for c in curves]


def main():
    fig = plt.figure(figsize=(15, 8))
    ax2 = fig.add_axes([0.05, 0.3, 0.25, 0.6])
    ax3 = fig.add_axes([0.33, 0.1, 0.65, 0.82], projection="3d")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2.2)
    ax2.set_ylim(-0.2, 3.4)
    ax2.set_xlabel("rho (distance of the vertices from the axis)")
    ax2.set_ylabel("z")
    ax2.set_title("sections (drag the markers)", fontsize=10)
    ax3.view_init(elev=18, azim=-60)
    state = {}
    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        clear()
        show_twist, show_nets = check_boxes.get_status()
        points = widget.points
        points[:, 0] = np.maximum(points[:, 0], 0.1)
        points[0, 1] = 0.0
        points[1:, 1] = np.maximum(points[1:, 1], points[:-1, 1] + 0.05)  # keep the order
        widget.artist.set_data(points[:, 0], points[:, 1])
        turns = [0.0, 0.0, 0.0, np.radians(30) if show_twist else 0.0]
        sections = [section(shape, rho, z, turn)
                    for shape, (rho, z), turn in zip(SHAPES, points, turns)]
        degree = int(radio_degree.value_selected)
        solid = loft(sections, degree=degree, parametrization=radio_spacing.value_selected)
        state["solid"], state["sections"] = solid, sections

        # -- right: faces, longitudinal edges, sections ---------------------------------------
        sides = [f for f in solid.faces if not isinstance(f.surface, Plane)]
        for k, face in enumerate(sides):
            grid = face.surface(T, np.linspace(0, 1, 60))
            artists.append(ax3.plot_surface(*grid.transpose(2, 0, 1), color=f"C{k}", alpha=0.4,
                                            lw=0))
            if show_nets:
                net = face.surface.net
                for line in (*net, *net.transpose(1, 0, 2)):
                    artists.extend(ax3.plot(*line.T, "o:", color=f"C{k}", lw=0.8, ms=2))
        for face in solid.faces:
            if isinstance(face.surface, Plane):
                loops = [loop_polyline(loop) for loop in face.loops]
                artists.append(ax3.add_collection3d(face_collection([loops], ["0.7"], 0.4)))
        rails = [e for e in solid.edges if e.curve.p == min(degree, 3) and
                 abs(e.curve.start[2] - e.curve.end[2]) > 1e-9]
        for edge in rails:
            artists.extend(ax3.plot(*edge.curve(np.linspace(0, 1, 120)).T, color="0.15", lw=2.2))
        for s in sections:
            for curve in s:
                artists.extend(ax3.plot(*curve(T).T, color="k", lw=1.2))
        lim = 1.8
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_zlim(-0.2, 3.4)
        ax3.set_box_aspect((1, 1, 1))

        # -- left: (rho, z) of the sections and of one longitudinal edge ----------------------
        rail = next(e.curve for e in rails if np.isclose(e.curve.start[2], 0.0)
                    or np.isclose(e.curve.end[2], 0.0))
        p = rail(np.linspace(0, 1, 200))
        artists.extend(ax2.plot(np.hypot(p[:, 0], p[:, 1]), p[:, 2], color="0.15", lw=2))
        for (rho, z), shape in zip(points, SHAPES):
            artists.append(ax2.text(rho + 0.06, z, shape, fontsize=9, va="center"))

        c = solid.counts()
        state["volume"] = volume(solid)
        state["rho"] = np.hypot(p[:, 0], p[:, 1])
        fig.suptitle(
            f"V={c['V']}  E={c['E']}  F={c['F']}   (an extrusion's counts: V = 2n, E = 3n, "
            f"F = n + 2, n = 4)      volume = {state['volume']:.4f}\n"
            f"spacing {radio_spacing.value_selected}, degree {degree} across the sections",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    widget = DraggableControlPoints(ax2, SECTIONS, on_change=redraw, show_polygon=False)
    radio_spacing = add_radiobuttons(fig, [0.05, 0.04, 0.11, 0.13],
                                     ["uniform", "chord", "centripetal"], 2, redraw)
    radio_degree = add_radiobuttons(fig, [0.17, 0.04, 0.06, 0.13], ["1", "2", "3"], 2, redraw)
    check_boxes = add_checkbuttons(fig, [0.24, 0.04, 0.09, 0.13], ["twist top", "skin nets"],
                                   [False, False], redraw)
    ax2.text(0.05, -0.17, "spacing          degree across", transform=ax2.transAxes,
             fontsize=9)
    redraw()

    def smoke():
        def valid():
            solid = state["solid"]
            assert check(solid) == []
            sides = [f.surface for f in solid.faces if not isinstance(f.surface, Plane)]
            for s in state["sections"]:  # every section lies on the skin
                for curve in s:
                    assert min(f.distance(curve(np.linspace(0, 1, 5))).max() for f in sides) < 1e-9
            assert state["volume"] > 0

        valid()
        widget.simulate_drag(2, (0.7, 1.25))  # the two circles close together
        waist = {}
        for i, name in enumerate(("uniform", "chord", "centripetal")):
            radio_spacing.set_active(i)
            valid()
            waist[name] = state["rho"].min()
        assert waist["chord"] < waist["centripetal"] < waist["uniform"]  # Loft.md
        radio_degree.set_active(0)
        valid()
        check_boxes.set_active(0)
        check_boxes.set_active(1)
        valid()
        print("  thinnest rho: " + ", ".join(f"{k} {v:.3f}" for k, v in waist.items()))

    run(fig, smoke)


if __name__ == "__main__":
    main()
