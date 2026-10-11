"""Surfaces of revolution: a NURBS profile times the exact NURBS circle.

Left: the profile, a NURBS curve in the xz-plane (horizontal = distance r from the z-axis,
vertical = height z). Drag its control points, or load a preset: cylinder (a vertical segment),
cone (a segment ending on the axis), sphere (a semicircle of two quarter arcs), torus (a full
circle off the axis). The numbers are the profile weights w_i.

Right: the same profile revolved about the z-axis.

  * **Every profile control point carries its own circle.** Profile point i is copied onto the
    nine control points of the NURBS circle -- the circumscribed square -- scaled by its distance
    r_i from the axis and lifted to its height z_i (the dashed squares). Stacked, those squares
    are the control net.
  * **The weights multiply.** Pick a net point with the two "pick" sliders: its weight is the
    profile weight times the circle weight, w_ij = w_i * c_j. That product is what makes the
    revolution exact: upstairs the surface factors into (profile in u) x (circle in v).
  * **"how far round"** grows the surface in v from 0 to 1. It is not exactly an angle: the
    NURBS circle is not traversed at constant speed.
  * **Poles.** A profile point on the axis (r_i = 0) has a square of size zero: all nine control
    points of its row sit on top of each other. Tick "poles" to see the rows that collapse and
    the parallels shrinking into them -- the surface is singular there (no tangent plane from
    S_u x S_v), although it looks perfectly smooth.

Notes: notes/02_Surfaces/Surfaces_of_Revolution.md
Run:   python scripts/02_surfaces/04_surface_of_revolution.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.nurbs import nurbs_curve
from cadkernel.geometry.nurbs_surfaces import (
    cone_profile,
    cylinder_profile,
    nurbs_surface,
    sphere_profile,
    surface_of_revolution,
    torus_profile,
)
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_radiobuttons,
    add_slider,
    run,
)

PRESETS = {
    "cylinder": cylinder_profile(1.0, 2.0),
    "cone": cone_profile(1.3, 2.0),
    "sphere": sphere_profile(1.2),
    "torus": torus_profile(1.8, 0.6),
}
T = np.linspace(0.0, 1.0, 200)
GRID_U = np.linspace(0.0, 1.0, 40)
NEAR_POLE = np.array([0.015, 0.04, 0.08, 0.13])


def main():
    fig = plt.figure(figsize=(15, 8))
    ax2 = fig.add_axes([0.05, 0.3, 0.28, 0.6])
    ax3 = fig.add_axes([0.36, 0.18, 0.62, 0.8], projection="3d")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color="k", lw=1.5)
    ax2.text(0.05, 2.6, "axis", fontsize=9)
    ax2.set_xlim(-0.5, 3.0)
    ax2.set_ylim(-1.6, 2.9)
    ax2.set_xlabel("r (distance from axis)")
    ax2.set_ylabel("z")
    ax2.set_title("profile", fontsize=10)
    ax3.view_init(elev=22, azim=-55)
    ax3.set_box_aspect((1, 1, 1))
    for setter in (ax3.set_xlim, ax3.set_ylim):
        setter(-2.6, 2.6)
    ax3.set_zlim(-1.6, 2.9)

    state = {"preset": "sphere"}
    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        clear()
        profile_points, weights, knots, p = (widget.points, *state["profile"][1:])
        net, w, knots_u, knots_v, p, q = surface_of_revolution(
            profile_points, weights, knots, p
        )
        state["surface"] = (net, w, knots_u, knots_v, p, q)
        show_squares, show_meridians, show_poles = check.get_status()
        i = min(int(round(s_i.val)), len(net) - 1)
        j = int(round(s_j.val))

        # -- left: the profile and its weights ---------------------------------------------------
        artists.extend(
            ax2.plot(
                *nurbs_curve(profile_points, weights, knots, p, T).T, color="C3", lw=2.4
            )
        )
        for k, (point, weight) in enumerate(zip(profile_points, weights)):
            artists.append(
                ax2.annotate(
                    f"{weight:.3g}",
                    point,
                    textcoords="offset points",
                    xytext=(7, 5),
                    fontsize=8,
                    color="C3" if k == i else "0.3",
                )
            )
        artists.extend(
            ax2.plot(*profile_points[i], "o", ms=13, mfc="none", mec="C3", mew=2)
        )

        # -- right: the revolution ---------------------------------------------------------------
        sweep = np.linspace(0.0, s_round.val, max(3, int(80 * s_round.val)))
        surface = nurbs_surface(net, w, knots_u, knots_v, p, q, GRID_U, sweep)
        artists.append(
            ax3.plot_surface(
                *surface.transpose(2, 0, 1), color="C3", alpha=0.3, linewidth=0
            )
        )
        artists.extend(
            ax3.plot(
                *nurbs_surface(net, w, knots_u, knots_v, p, q, T, [0.0])[:, 0].T,
                color="C3",
                lw=2.4,
            )
        )
        if show_squares:
            for row in net:
                artists.extend(ax3.plot(*row.T, "o--", color="0.4", ms=3, lw=0.8))
        if show_meridians:
            for column in net.transpose(1, 0, 2):
                artists.extend(ax3.plot(*column.T, "-", color="0.55", lw=0.7))
        artists.extend(ax3.plot(*net[i, j][:, None], "o", color="C0", ms=11))

        if show_poles:
            on_axis = [k for k in (0, len(net) - 1) if np.allclose(net[k], net[k, 0])]
            for k in on_axis:
                artists.extend(ax3.plot(*net[k, 0][:, None], "*", color="k", ms=18))
                artists.append(
                    ax3.text(*net[k, 0], "  9 control points here", fontsize=9)
                )
                for u in NEAR_POLE if k == 0 else 1 - NEAR_POLE:
                    ring = nurbs_surface(net, w, knots_u, knots_v, p, q, [u], T)[0]
                    artists.extend(ax3.plot(*ring.T, color="k", lw=0.9))
            state["poles"] = on_axis

        c = w[i, j] / weights[i]
        fig.suptitle(
            f"{state['preset']}: {net.shape[0]} x {net.shape[1]} control net    "
            f"picked P_{i},{j}: weight = w_{i} * c_{j} = {weights[i]:.3g} * {c:.3g} = {w[i, j]:.3g}",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def on_preset(label):
        state["preset"] = label
        state["profile"] = PRESETS[label]
        s_i.valmax = len(PRESETS[label][0]) - 1
        s_i.ax.set_xlim(s_i.valmin, s_i.valmax)
        widget.set_points(PRESETS[label][0])

    state["profile"] = PRESETS["sphere"]
    widget = DraggableControlPoints(ax2, PRESETS["sphere"][0], on_change=redraw)
    s_round = add_slider(
        fig, [0.45, 0.10, 0.3, 0.03], "how far round (v)", 0.02, 1.0, 0.75, redraw
    )
    s_i = add_slider(
        fig, [0.45, 0.06, 0.3, 0.03], "pick i (profile)", 0, 4, 1, redraw, valfmt="%d"
    )
    s_j = add_slider(
        fig, [0.45, 0.02, 0.3, 0.03], "pick j (circle)", 0, 8, 1, redraw, valfmt="%d"
    )
    radio = add_radiobuttons(
        fig, [0.05, 0.03, 0.12, 0.18], list(PRESETS), active=2, on_change=on_preset
    )
    check = add_checkbuttons(
        fig,
        [0.80, 0.02, 0.17, 0.13],
        ["squares (rows)", "meridians (columns)", "poles"],
        [True, False, False],
        redraw,
    )
    redraw()

    def smoke():
        check.set_active(2)
        assert state["poles"] == [0, 4]  # sphere: both rows on the axis collapse
        for label in PRESETS:
            radio.set_active(list(PRESETS).index(label))
            net, w, *_ = state["surface"]
            assert net.shape[1] == 9
            # Every net weight is a profile weight times a circle weight.
            assert np.allclose(w, np.outer(PRESETS[label][1], w[0] / w[0, 0]))
        assert state["poles"] == []  # torus: nothing on the axis
        radio.set_active(1)
        assert state["poles"] == [len(PRESETS["cone"][0]) - 1]  # cone: the apex
        s_round.set_val(0.3)
        s_i.set_val(1)
        s_j.set_val(3)
        widget.simulate_drag(
            0, (1.6, -0.2)
        )  # drag the cone's rim: still a surface of revolution
        net, *_ = state["surface"]
        assert np.allclose(np.hypot(net[0, ::2, 0], net[0, ::2, 1]), 1.6, atol=1e-2)
        print(
            f"  presets {list(PRESETS)}: poles found where the profile meets the axis"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
