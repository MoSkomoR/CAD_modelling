"""Bezier curves and De Casteljau's algorithm, live.

Drag any control point and watch the whole curve respond (that "whole" is global support --
script 03 makes it precise). Move the t slider to walk the De Casteljau construction: each
level of the triangle is nothing but repeated linear interpolation between the level below,
collapsing to the single point on the curve.

Tick "subdivision" to see the payoff that makes this algorithm, rather than the closed
Bernstein formula, the one real kernels build on: the two outer edges of the triangle you are
already computing *are* the control polygons of the two halves of the curve, split at t. Free
subdivision is what turns into adaptive tessellation, intersection by subdivide-and-cull, ray
casting and trimming.

Notes: notes/01_Curves/Bezier_Curves.md, notes/01_Curves/De_Casteljau_Derivation.md
Run:   python scripts/01_curves/02_bezier_de_casteljau.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.curves import (
    bezier_curve,
    bezier_subdivide,
    de_casteljau_triangle,
)
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_slider,
    run,
)


def main():
    control_points = np.array(
        [[0.0, 0.0], [1.0, 2.5], [3.0, 2.5], [4.0, 0.0], [5.0, -1.5]]
    )
    t_dense = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.20, right=0.78)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 6.5)
    ax.set_ylim(-3.5, 4.0)

    (curve_artist,) = ax.plot([], [], color="C0", lw=2.5, zorder=3, label="curve")
    (point_artist,) = ax.plot([], [], "*", color="crimson", ms=18, zorder=6)
    level_artists = []  # the De Casteljau triangle, one polyline per level
    subdivision_artists = []  # the two half-curve control polygons

    def draw_triangle(points, t):
        for artist in level_artists:
            artist.remove()
        level_artists.clear()

        levels = de_casteljau_triangle(points, t)
        for k, level in enumerate(levels[1:], start=1):
            colour = plt.cm.viridis(k / max(len(levels) - 1, 1))
            (artist,) = ax.plot(
                level[:, 0],
                level[:, 1],
                "o-",
                ms=4,
                lw=1.2,
                color=colour,
                alpha=0.85,
                zorder=4,
            )
            level_artists.append(artist)
        point_artist.set_data([levels[-1][0, 0]], [levels[-1][0, 1]])

    def draw_subdivision(points, t):
        for artist in subdivision_artists:
            artist.remove()
        subdivision_artists.clear()
        if not check.get_status()[0]:
            return
        left, right = bezier_subdivide(points, t)
        for polygon, colour, name in (
            (left, "tab:orange", "left half"),
            (right, "tab:purple", "right half"),
        ):
            (artist,) = ax.plot(
                polygon[:, 0],
                polygon[:, 1],
                "s:",
                ms=5,
                lw=1.4,
                color=colour,
                alpha=0.9,
                zorder=2,
                label=name,
            )
            subdivision_artists.append(artist)

    def redraw(points=None, _=None):
        points = control_points_widget.points
        t = s_t.val
        curve = bezier_curve(points, t_dense)
        curve_artist.set_data(curve[:, 0], curve[:, 1])
        draw_triangle(points, t)
        draw_subdivision(points, t)
        ax.set_title(
            f"Bezier degree {len(points) - 1}   |   De Casteljau at t = {t:.3f}"
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
        fig.canvas.draw_idle()

    control_points_widget = DraggableControlPoints(ax, control_points, on_change=redraw)
    s_t = add_slider(fig, [0.15, 0.08, 0.6, 0.03], "t", 0.0, 1.0, 0.35, redraw)
    check = add_checkbuttons(
        fig, [0.80, 0.03, 0.16, 0.09], ["subdivision"], [False], redraw
    )

    redraw()

    def smoke():
        control_points_widget.simulate_drag(2, (3.5, 3.4))
        s_t.set_val(0.7)
        check.set_active(0)  # turn subdivision on
        redraw()
        assert len(subdivision_artists) == 2, "subdivision overlay did not draw"
        left, right = bezier_subdivide(control_points_widget.points, s_t.val)
        assert np.allclose(left[-1], right[0])

    run(fig, smoke)


if __name__ == "__main__":
    main()
