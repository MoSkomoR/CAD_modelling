"""Sweep: a rectangle carried along a space curve -- and the frame that decides which way it
faces, Frenet's or the rotation-minimizing one.

WHAT YOU SEE
------------
Top left, the PATH seen from above: a cubic Bezier curve C(t), t in [0, 1], whose four control
points you can drag in x and y. The "lift" slider raises the control points to the heights
lift * (0, 2, 1, 2); at lift = 0 the path is a plane curve (in the xy-plane), otherwise it twists
out of every plane.

Right, the SOLID that sweep() builds: the rectangular PROFILE (0.5 by 0.16) placed in the plane
normal to the path at a number of SECTIONS (black), and a skin interpolated through them
(05_Sweep.md), one colour per side face. At each section the red arrow is the frame's normal r:
the direction the profile's long side points.

The path's tangent T(t) fixes the plane the profile lies in, but not its rotation in that plane:
r(t), a unit vector normal to T, does. Two choices ("frame"):
  * FRENET: r = N, the principal normal, the direction in which the path turns (N = B x T,
    B = C' x C'' / |C' x C''|). It needs the curvature kappa = |C' x C''| / |C'|^3 to be non-zero.
    Set lift = 0: the path is planar, has an INFLECTION (kappa = 0, where it switches from turning
    left to turning right) and N jumps to -N there. The profile turns half a turn between two
    sections, and the skin twists through itself. Raise lift a little: kappa is small but no
    longer zero, and N still turns by nearly half a turn, now over a short stretch of t. With
    9 sections at lift = 0, one section sits exactly on the inflection and has no frame.
  * ROTATION-MINIMIZING (RMF): r' is parallel to T, so r never turns about the tangent; computed by
    double reflection (05_Sweep.md). It is defined wherever T is, and on a plane curve it stays in
    the plane: no flip.

Bottom left, the angle by which Frenet's N is turned away from the RMF's r, about T, against t
(solid line). 05_Sweep.md proves that it equals the integral of the TORSION, \\int tau ds, where
tau measures how fast the path leaves its osculating plane (the plane of T and N) and s is arc
length. The dashed line is that integral computed from C', C'', C''' -- it lies on the solid
line wherever kappa > 0. At lift = 0, tau = 0: the angle is 0, then pi after the inflection.

Controls: "frame" switches between the two, "sections" sets how many copies of the profile are
skinned (more: closer to the exact sweep; 05_Sweep.md measures how close, and Frenet's fast turn
needs far more sections than the RMF), "frames" and "sections" toggle the arrows and the black
profiles.

The title gives V, E, F and the volume against A * L (profile area times path length), which
05_Sweep.md proves is the exact sweep's volume whatever the frame, as long as the profile's
centroid runs along the path and the solid does not fold.

Notes: notes/04_Modeling_Operations/05_Sweep.md
Run:   python scripts/04_modeling_ops/04_sweep.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.entities import NurbsCurve, Plane, polygon_segments
from cadkernel.geometry.frames import angle_about_tangent, curvature_torsion, frenet_frames
from cadkernel.ops.measure import volume
from cadkernel.ops.sweep import arc_length, path_frames, place, sweep
from cadkernel.topology.halfedge import check
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_radiobuttons,
    add_slider,
    run,
)
from cadkernel.viz.topology import face_collection, loop_polyline

HEIGHTS = np.array([0.0, 2.0, 1.0, 2.0])  # times lift: z of the control points
PATH_XY = np.array([[0.0, 0.0], [1.4, 1.8], [2.2, -1.8], [3.6, 0.0]])
PROFILE = polygon_segments([[0.25, -0.08], [0.25, 0.08], [-0.25, 0.08], [-0.25, -0.08]])
AREA = 0.5 * 0.16
FRAMES = {"rotation-minimizing": "rmf", "Frenet": "frenet"}
T_FINE = np.linspace(0.0, 1.0, 401)
U = np.linspace(0.0, 1.0, 12)


def main():
    fig = plt.figure(figsize=(15, 8.5))
    ax_top = fig.add_axes([0.05, 0.47, 0.27, 0.45])
    ax_angle = fig.add_axes([0.05, 0.25, 0.27, 0.15])
    ax3 = fig.add_axes([0.36, 0.08, 0.62, 0.84], projection="3d")
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.3)
    ax_top.set_xlim(-0.6, 4.2)
    ax_top.set_ylim(-2.4, 2.4)
    ax_top.set_title("path, seen from above (drag the markers)", fontsize=10)
    ax_angle.set_xlim(0, 1)
    ax_angle.set_xlabel("t")
    ax_angle.set_ylabel("angle (rad)")
    ax_angle.grid(True, alpha=0.3)
    ax3.view_init(elev=30, azim=-60)
    state = {}
    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        clear()
        show_frames, show_sections = check_boxes.get_status()
        lift = slider_lift.val
        count = int(round(slider_sections.val))
        frame = FRAMES[radio_frame.value_selected]
        path = NurbsCurve.bezier(np.c_[widget.points, lift * HEIGHTS])
        state["path"] = path
        curve = path(T_FINE)
        artists.extend(ax_top.plot(curve[:, 0], curve[:, 1], color="0.15", lw=2))
        artists.extend(ax3.plot(*curve.T, color="0.15", lw=1.2, ls="--"))

        # -- bottom left: Frenet's N measured from the RMF's r, and the integrated torsion ------
        _, d1, d2, d3 = path.derivatives(T_FINE, 3)
        _, normal, _ = frenet_frames(d1, d2)
        _, _, r, s = path_frames(path, T_FINE, "rmf")
        angle = angle_about_tangent(r, s, normal)
        kappa, tau = curvature_torsion(d1, d2, d3)
        integrand = np.nan_to_num(tau) * np.linalg.norm(d1, axis=1)
        torsion = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                                   * np.diff(T_FINE))])
        artists.extend(ax_angle.plot(T_FINE, angle, color="C3", lw=2,
                                     label="angle from r (RMF) to N (Frenet)"))
        artists.extend(ax_angle.plot(T_FINE, angle[0] + torsion, "k--", lw=1,
                                     label="integral of tau ds"))
        ax_angle.set_ylim(-3.5, 3.5)
        state["angle"], state["torsion"], state["kappa"] = angle, torsion, kappa
        artists.append(ax_angle.legend(loc="lower left", fontsize=8))

        # -- right: the solid -------------------------------------------------------------------
        try:
            solid = sweep(PROFILE, path, frame=frame, sections=count)
        except ValueError as error:
            state["solid"] = None
            fig.suptitle(f"{radio_frame.value_selected}: {error}", fontsize=11, color="C3")
            fig.canvas.draw_idle()
            return
        state["solid"] = solid
        sides = [f for f in solid.faces if not isinstance(f.surface, Plane)]
        for k, face in enumerate(sides):
            grid = face.surface(U, np.linspace(0, 1, 80))
            artists.append(ax3.plot_surface(*grid.transpose(2, 0, 1), color=f"C{k}", alpha=0.45,
                                            lw=0))
        for face in solid.faces:
            if isinstance(face.surface, Plane):
                loops = [loop_polyline(loop) for loop in face.loops]
                artists.append(ax3.add_collection3d(face_collection([loops], ["0.7"], 0.5)))
        C, T, r, s = path_frames(path, np.linspace(0, 1, count), frame)
        if show_sections:
            for f in zip(C, T, r, s):
                for edge in place(PROFILE, *f):
                    artists.extend(ax3.plot(*edge(U).T, color="k", lw=1.2))
        if show_frames:
            artists.append(ax3.quiver(*C.T, *(0.45 * r).T, color="C3", lw=1.5))
        ax3.set_xlim(-0.5, 4.1)
        ax3.set_ylim(-2.3, 2.3)
        ax3.set_zlim(-0.5, 2.5)
        ax3.set_box_aspect((4.6, 4.6, 3.0))  # equal scales on all three axes

        c = solid.counts()
        state["volume"], state["AL"] = volume(solid), AREA * arc_length(path)
        fig.suptitle(
            f"{radio_frame.value_selected} frame, {count} sections      V={c['V']}  E={c['E']}  "
            f"F={c['F']}\nvolume = {state['volume']:.5f}      A L = {state['AL']:.5f}  "
            f"(exact sweep, 05_Sweep.md)",
            fontsize=11, color="k",
        )
        fig.canvas.draw_idle()

    widget = DraggableControlPoints(ax_top, PATH_XY, on_change=redraw)
    radio_frame = add_radiobuttons(fig, [0.05, 0.03, 0.12, 0.11], list(FRAMES), 0, redraw)
    slider_lift = add_slider(fig, [0.22, 0.12, 0.1, 0.03], "lift", 0.0, 1.5, 0.5, redraw)
    slider_sections = add_slider(fig, [0.22, 0.07, 0.1, 0.03], "sections", 2, 48, 12, redraw,
                                 valfmt="%d")
    check_boxes = add_checkbuttons(fig, [0.05, 0.145, 0.12, 0.07], ["frames", "sections"],
                                   [True, True], redraw)
    redraw()

    def smoke():
        def valid():
            solid = state["solid"]
            assert check(solid) == []
            if radio_frame.value_selected == "rotation-minimizing":
                assert np.isclose(state["volume"], state["AL"], rtol=1e-3)
            gap = np.angle(np.exp(1j * (state["angle"] - state["angle"][0] - state["torsion"])))
            assert np.abs(gap).max() < 1e-2  # Frenet's angle is the integrated torsion

        valid()
        radio_frame.set_active(1)
        valid()
        slider_lift.set_val(0.0)  # planar: Frenet flips at the inflection
        angle = state["angle"]  # 0 before the inflection, pi after, NaN at t = 1/2 itself
        jump = abs(angle[-1] - angle[0])
        assert np.isnan(angle[200]) and np.isclose(jump, np.pi)
        assert jump > 3.0 and check(state["solid"]) == []
        slider_sections.set_val(9)  # a section exactly at the inflection t = 1/2: no Frenet frame
        assert state["solid"] is None
        radio_frame.set_active(0)
        slider_sections.set_val(12)
        assert np.nanmax(np.abs(state["angle"])) > 3.0 and np.abs(state["torsion"]).max() < 1e-12
        widget.simulate_drag(1, (1.0, 2.0))
        slider_lift.set_val(1.0)
        valid()
        check_boxes.set_active(0)
        check_boxes.set_active(1)
        print(f"  Frenet's jump at the inflection: {jump:.3f} rad;  volume {state['volume']:.5f}"
              f" vs A L {state['AL']:.5f}")

    run(fig, smoke)


if __name__ == "__main__":
    main()
