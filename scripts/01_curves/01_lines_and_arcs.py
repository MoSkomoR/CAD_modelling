"""Lines and arcs -- the exact analytic curve types, made draggable.

Drag the two square endpoints to reshape the line; use the sliders to sweep the arc's radius
and angular extent. The point of playing with this one is mostly to feel how little these
representations let you say: two endpoints, or a centre plus two angles. Anything that isn't a
straight line or a piece of a circle has no home here at all -- which is the motivation for
everything that follows.

Notes: notes/01_Curves/Lines_and_Arcs.md
Run:   python scripts/01_curves/01_lines_and_arcs.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.curves import arc, line
from cadkernel.viz.interactive import DraggableControlPoints, add_slider, run


def main():
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(bottom=0.28)
    ax.set_title("Line (drag endpoints) and arc (sliders)")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    t = np.linspace(0, 1, 200)

    (line_artist,) = ax.plot([], [], color="C0", lw=2.5, label="line")
    (arc_artist,) = ax.plot([], [], color="C1", lw=2.5, label="arc")
    (centre_artist,) = ax.plot([0], [0], "+", color="C1", ms=12)

    def redraw_line(points):
        pts = line(points[0], points[1], t)
        line_artist.set_data(pts[:, 0], pts[:, 1])

    endpoints = DraggableControlPoints(
        ax, np.array([[-3.0, -2.5], [2.5, 3.0]]), on_change=redraw_line, color="C0"
    )

    def redraw_arc(_=None):
        pts = arc(
            np.array([0.0, 0.0]),
            s_radius.val,
            t,
            theta0=np.deg2rad(s_theta0.val),
            theta1=np.deg2rad(s_theta1.val),
        )
        arc_artist.set_data(pts[:, 0], pts[:, 1])
        fig.canvas.draw_idle()

    s_radius = add_slider(
        fig, [0.15, 0.15, 0.7, 0.03], "radius", 0.2, 3.5, 2.0, redraw_arc
    )
    s_theta0 = add_slider(
        fig, [0.15, 0.09, 0.7, 0.03], "theta0 (deg)", -360, 360, 0.0, redraw_arc
    )
    s_theta1 = add_slider(
        fig, [0.15, 0.03, 0.7, 0.03], "theta1 (deg)", -360, 360, 270.0, redraw_arc
    )

    redraw_line(endpoints.points)
    redraw_arc()
    ax.legend(loc="upper left")

    def smoke():
        endpoints.simulate_drag(0, (-1.0, 1.0))
        s_radius.set_val(3.0)
        s_theta1.set_val(180.0)
        assert line_artist.get_xdata().size == len(t)
        assert arc_artist.get_xdata().size == len(t)

    run(fig, smoke)


if __name__ == "__main__":
    main()
