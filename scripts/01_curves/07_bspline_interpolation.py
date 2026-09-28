"""Interpolation: drag the *data points*, and a cubic B-spline is solved to pass through them.

Everywhere else in module 01 you drag control points and the curve follows them loosely. Here
the dragged points are data the curve must hit exactly, and the control points are an *output*:
one linear solve, sum_i N_{i,p}(t_k) P_i = Q_k, per redraw. Tick "control polygon" to see the
solved control points (for the centripetal curve) -- they are nowhere near the data.

Before the solve you must choose which parameter t_k each data point is hit at, and the choice
matters far more than it looks. The starting data set is the "uneven" one from
test_parametrization_comparison:
  * uniform      ignores spacing -- here it loops,
  * chord length (the usual default) -- here it overshoots by ~5 units,
  * centripetal  (square-root chord) -- the only one clean on all three test sets.
The title reports, for each shown curve, the worst interpolation residual (always ~1e-16 -- every
curve does hit every point) and whether it self-intersects.

Tick "single polynomial" for the Runge contrast: one polynomial through all the points (solved in
the Bernstein basis, centripetal parameters). On the starting data it strays 4.1 units from the
data polygon against the centripetal spline's 0.72 -- same points, same parameters, one piece
instead of four.

Notes: notes/01_Curves/B_Spline_Interpolation.md
Run:   python scripts/01_curves/07_bspline_interpolation.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.bspline import bspline_curve, interpolate_points
from cadkernel.geometry.curves import bernstein_basis, bezier_curve
from cadkernel.viz.interactive import DraggableControlPoints, add_checkbuttons, run

DATA = np.array([[0, 0], [0.1, 0.3], [0.2, 0.1], [3, 1], [3.2, 0.8], [3.1, 1.2], [6, 0]], float)
T = np.linspace(0.0, 1.0, 1501)
METHODS = {"uniform": "tab:blue", "chord": "tab:red", "centripetal": "tab:green"}


def self_intersects(polyline):
    """Does a dense polyline cross itself? (Segment-pair test; skips neighbouring segments.)"""
    a, b = polyline[:-1], polyline[1:]
    for i in range(len(a) - 2):
        r, s = b[i] - a[i], b[i + 2 :] - a[i + 2 :]
        q = a[i + 2 :] - a[i]
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / denom
            v = (q[:, 0] * r[1] - q[:, 1] * r[0]) / denom
        if np.any((denom != 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)):
            return True
    return False


def main():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(left=0.06, right=0.78, bottom=0.08, top=0.90)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 8.5)
    ax.set_ylim(-3.5, 7.0)

    artists = []
    state = {}

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        data = widget.points
        status = check.get_status()
        shown = [m for m, on in zip(METHODS, status[:3]) if on]
        show_polygon, show_polynomial = status[3], status[4]
        clear()

        report = []
        for method in METHODS:
            control, knots, params = interpolate_points(data, 3, method)
            curve = bspline_curve(control, knots, 3, T)
            residual = np.abs(bspline_curve(control, knots, 3, params) - data).max()
            state[method] = (curve, residual)
            if method in shown:
                artists.extend(ax.plot(*curve.T, color=METHODS[method], lw=2.2, label=method))
                loop = "LOOPS" if self_intersects(curve) else "no loop"
                report.append(f"{method}: resid {residual:.0e}, {loop}")
            if method == "centripetal" and show_polygon:
                artists.extend(ax.plot(*control.T, "s--", color="tab:green", ms=6, lw=1, alpha=0.7))

        if show_polynomial:
            _, _, params = interpolate_points(data, 3, "centripetal")
            bezier = np.linalg.solve(bernstein_basis(len(data) - 1, params), data)
            curve = bezier_curve(bezier, T)
            state["polynomial"] = curve
            artists.extend(ax.plot(*curve.T, color="0.25", lw=1.6, ls="--", label="single polynomial"))

        artists.append(ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.45), fontsize=9))
        ax.set_title("cubic B-spline interpolation\n" + "   |   ".join(report), fontsize=10)
        fig.canvas.draw_idle()

    widget = DraggableControlPoints(ax, DATA.copy(), on_change=redraw, show_polygon=False, color="black")
    widget.artist.set_label("data points")
    check = add_checkbuttons(
        fig,
        [0.80, 0.55, 0.18, 0.30],
        ["uniform", "chord", "centripetal", "control polygon", "single polynomial"],
        [True, True, True, False, False],
        redraw,
    )

    redraw()

    def smoke():
        for method in METHODS:
            assert state[method][1] < 1e-13, f"{method} curve misses a data point"
        assert self_intersects(state["uniform"][0]), "uniform loops on this data (as measured)"
        assert not self_intersects(state["centripetal"][0])

        widget.simulate_drag(3, (3.0, 2.0))
        check.set_active(3)
        check.set_active(4)
        for method in METHODS:
            assert state[method][1] < 1e-13
        print(
            f"  every curve hits every point (worst residual "
            f"{max(state[m][1] for m in METHODS):.1e})"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
