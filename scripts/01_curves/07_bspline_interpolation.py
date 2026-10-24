"""Interpolation: drag the *data points*, and a cubic B-spline is solved to pass through them.

WHAT YOU SEE
------------
Seven black dots are the data points Q_0 .. Q_6. Up to three coloured curves pass exactly
through all of them -- blue, red and green. They differ only in one choice (below). Drag any
dot and all three curves are re-solved at once.

Checkboxes on the right:
  * uniform / chord / centripetal -- show or hide each of the three curves.
  * control polygon  -- the control points that were *solved for* to make the green curve hit
                        the dots. Note how far from the data they sit.
  * single polynomial -- the dashed grey curve: one polynomial through all seven dots, instead
                        of a spline, for comparison (see "Runge's phenomenon" below).
The title prints, per curve, the worst miss at a data point (always about 1e-16, i.e. it hits
every dot) and whether the curve crosses itself ("LOOPS").

WHAT IS DIFFERENT FROM THE OTHER SCRIPTS
----------------------------------------
Everywhere else in module 01 you drag *control points* and the curve follows them loosely. Here
the curve must pass through the dragged points, so the control points P_0 .. P_6 become the
unknowns. A cubic B-spline is C(t) = sum_i N_{i,3}(t) P_i, and requiring it to pass through
Q_k at parameter t_k gives one linear equation per data point:

    sum_i N_{i,3}(t_k) P_i = Q_k,      k = 0 .. 6

Seven equations, seven unknown control points: one linear solve per redraw.

THE CHOICE THAT MATTERS: WHICH t_k BELONGS TO WHICH POINT
---------------------------------------------------------
Before solving, each data point needs a parameter value t_k: "the curve is at Q_k when t = t_k".
Think of t as time running from 0 to 1. The rule picks how much time the curve gets to travel
from one point to the next. With d_k = |Q_k - Q_{k-1}| the distance between neighbouring points,
t_0 = 0, and each rule normalised so that t_6 = 1:

  * uniform      (blue)  every step gets the same time:        t_k - t_{k-1} proportional to 1
  * chord length (red)   time proportional to distance:         ... proportional to d_k
  * centripetal  (green) time proportional to sqrt(distance):   ... proportional to sqrt(d_k)

("Chord" is the straight segment between two consecutive points, so "chord length" is just
d_k. "Centripetal" is Lee's 1989 name; the square root sits between the other two.)

On the starting data the gaps are short, short, long, short, short, long
(d = 0.32, 0.22, 2.94, 0.28, 0.41, 3.14), and the rules give (test_parameters_on_the_uneven_data):

    uniform      0  0.17  0.33  0.50  0.67  0.83  1
    chord        0  0.04  0.07  0.48  0.51  0.57  1
    centripetal  0  0.10  0.18  0.48  0.58  0.69  1

Why this changes the shape:
  * uniform gives a 0.22 gap and a 2.94 gap the same time, so the curve crawls through the
    cluster and must sprint across the gap. Arriving fast, it swings round and loops.
  * chord length makes the speed roughly constant, so the curve enters the tight cluster at full
    speed and has to swing wide to turn: on this data it strays ~5 units from the dots.
  * centripetal compresses the long/short ratio (sqrt(2.94/0.22) ~ 3.7 instead of ~13) and
    behaves on all three test data sets.
Measured in test_parametrization_comparison; that is why interpolate_points defaults to
centripetal, even though chord length is the usual textbook default.

RUNGE'S PHENOMENON (the "single polynomial" checkbox)
------------------------------------------------------
Through n+1 points there is exactly one polynomial of degree n. Interpolating with it seems
natural, but as more points are added that polynomial tends to oscillate more and more
violently between them, worst near the ends. This is Runge's phenomenon, named after Carl Runge
(1901), who showed it for the function f(x) = 1 / (1 + 25 x^2) sampled at equally spaced x on
[-1, 1]: the error of the degree-n interpolant grows without bound as n grows (cited, not proved
here; measured in test_piecewise_interpolation_escapes_runge -- 0.62, 3.66, 59.8 at 7, 13, 21
points, against 0.13, 0.0069, 0.0032 for the cubic spline).
A spline avoids it by staying degree 3 however many points there are: each piece only answers
to a few neighbouring points. The dashed curve here is the degree-6 polynomial through the same
seven dots with the same centripetal t_k: it strays 4.1 units from the data polygon against the
green spline's 0.72 (test_single_polynomial_strays_further_than_the_spline).

Notes: notes/01_Curves/B_Spline_Interpolation.md
Run:   python scripts/01_curves/07_bspline_interpolation.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.bspline import bspline_curve, interpolate_points
from cadkernel.geometry.curves import bernstein_basis, bezier_curve
from cadkernel.viz.interactive import DraggableControlPoints, add_checkbuttons, run

DATA = np.array(
    [[0, 0], [0.1, 0.3], [0.2, 0.1], [3, 1], [3.2, 0.8], [3.1, 1.2], [6, 0]], float
)
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
                artists.extend(
                    ax.plot(*curve.T, color=METHODS[method], lw=2.2, label=method)
                )
                loop = "LOOPS" if self_intersects(curve) else "no loop"
                report.append(f"{method}: resid {residual:.0e}, {loop}")
            if method == "centripetal" and show_polygon:
                artists.extend(
                    ax.plot(*control.T, "s--", color="tab:green", ms=6, lw=1, alpha=0.7)
                )

        if show_polynomial:
            _, _, params = interpolate_points(data, 3, "centripetal")
            bezier = np.linalg.solve(bernstein_basis(len(data) - 1, params), data)
            curve = bezier_curve(bezier, T)
            state["polynomial"] = curve
            artists.extend(
                ax.plot(
                    *curve.T, color="0.25", lw=1.6, ls="--", label="single polynomial"
                )
            )

        artists.append(
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.45), fontsize=9)
        )
        ax.set_title(
            "cubic B-spline interpolation\n" + "   |   ".join(report), fontsize=10
        )
        fig.canvas.draw_idle()

    widget = DraggableControlPoints(
        ax, DATA.copy(), on_change=redraw, show_polygon=False, color="black"
    )
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
        assert self_intersects(state["uniform"][0]), (
            "uniform loops on this data (as measured)"
        )
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
