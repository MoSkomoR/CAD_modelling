"""Weights: the R in NURBS -- what one extra number per control point buys, and what it costs.

Three panels:

  * **A weight as a shape handle.** A cubic NURBS on 7 points. Pick a point with "point i" and
    move "log10 w_i": the curve is pulled toward that point as w_i grows (it would pass through it
    in the limit) and pushed away as it shrinks -- but only over that point's knot support, the
    same locality the B-spline had. The dashed ghost is the plain B-spline (all w = 1). Tick
    "perspective" to apply a projective map: mapping the homogeneous control points reproduces the
    mapped curve exactly, while moving the points and keeping the weights does not.

  * **One weight, three conics.** A quadratic with weights (1, w, 1). The type is read off the
    weight: w < 1 ellipse, w = 1 parabola, w > 1 hyperbola. The faint curve is the *rest* of the
    same conic -- the middle weight negated. At w = cos 45 deg with the square-corner control point,
    the arc is an exact quarter circle.

  * **The exact full circle.** Nine control points on the circumscribed square, weights alternating
    1 and sqrt(2)/2, interior knots doubled. Radial error at machine precision -- but the ticks
    at equal steps of t are *not* equally spaced: exact geometry, inexact speed.

Notes: notes/01_Curves/NURBS.md, notes/01_Curves/NURBS_Derivations.md
Run:   python scripts/01_curves/08_nurbs_weights.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.bspline import bspline_curve, clamped_knots
from cadkernel.geometry.nurbs import (
    nurbs_circle,
    nurbs_curve,
    nurbs_projective_map,
    projective_map,
    rational_bezier,
)
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_slider,
    run,
)

P = 3
CONTROL = np.array(
    [[0, 0], [1, 2], [2.5, 2.2], [3.5, 0], [5, -0.5], [6, 1.5], [7, 0.5]], float
)
KNOTS = clamped_knots(len(CONTROL), P)
CONIC = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
PERSPECTIVE = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.12, 0.05, 1.0]])
T = np.linspace(0.0, 1.0, 801)
TICKS = np.linspace(0.0, 1.0, 33)[:-1]


def conic_type(w: float) -> str:
    if np.isclose(w, 1.0):
        return "parabola"
    return "ellipse" if w < 1 else "hyperbola"


def main():
    fig = plt.figure(figsize=(16, 7))
    ax_shape = fig.add_axes([0.04, 0.32, 0.40, 0.6])
    ax_conic = fig.add_axes([0.48, 0.32, 0.24, 0.6])
    ax_circle = fig.add_axes([0.75, 0.32, 0.23, 0.6])
    for ax in (ax_shape, ax_conic, ax_circle):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    ax_shape.set_xlim(-1, 8.5)
    ax_shape.set_ylim(-2, 3.5)
    ax_conic.set_xlim(-1.5, 3.5)
    ax_conic.set_ylim(-2.5, 2.5)

    artists = []
    state = {}

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        clear()
        points = widget.points
        i = int(round(s_i.val))
        weights = np.ones(len(points))
        weights[i] = 10.0**s_w.val
        show_ghost, show_perspective = check.get_status()

        # -- panel 1: a weight as a shape handle -----------------------------------------------
        curve = nurbs_curve(points, weights, KNOTS, P, T)
        state["curve"], state["weights"] = curve, weights
        support = (T >= KNOTS[i]) & (T <= KNOTS[i + P + 1])
        artists.extend(ax_shape.plot(*curve[support].T, color="gold", lw=9, alpha=0.35))
        artists.extend(ax_shape.plot(*curve.T, color="C3", lw=2.4, label="NURBS"))
        artists.extend(
            ax_shape.plot(
                *points[i], "o", ms=12 + 4 * s_w.val, mfc="none", mec="C3", mew=2.5
            )
        )
        if show_ghost:
            ghost = bspline_curve(points, KNOTS, P, T)
            artists.extend(
                ax_shape.plot(*ghost.T, "--", color="0.4", lw=1.3, label="B-spline")
            )
        message = ""
        if show_perspective:
            image = projective_map(PERSPECTIVE, curve)
            new_points, new_weights = nurbs_projective_map(PERSPECTIVE, points, weights)
            exact = nurbs_curve(new_points, new_weights, KNOTS, P, T)
            naive = nurbs_curve(
                projective_map(PERSPECTIVE, points), weights, KNOTS, P, T
            )
            state["perspective"] = (
                np.abs(exact - image).max(),
                np.abs(naive - image).max(),
            )
            artists.extend(ax_shape.plot(*image.T, color="0.85", lw=7, zorder=1))
            artists.extend(
                ax_shape.plot(*exact.T, color="C2", lw=2, label="mapped data")
            )
            artists.extend(
                ax_shape.plot(*naive.T, ":", color="C1", lw=2, label="points only")
            )
            artists.extend(ax_shape.plot(*new_points.T, "s:", color="C2", ms=5, lw=0.8))
            message = (
                f"\nperspective: mapped data {state['perspective'][0]:.0e},"
                f" points only {state['perspective'][1]:.2f}"
            )
        artists.append(ax_shape.legend(loc="upper right", fontsize=8))
        ax_shape.set_title(
            f"weight on P_{i}: w = {weights[i]:.3g}   (support highlighted)" + message,
            fontsize=10,
        )

        # -- panel 2: one weight, three conics -------------------------------------------------
        w = s_c.val
        arc = rational_bezier(CONIC, [1, w, 1], T)
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # w = 1 has a pole at t = 1/2
            rest = rational_bezier(CONIC, [1, -w, 1], T)
        rest[np.linalg.norm(rest, axis=1) > 50] = np.nan  # break the line at a pole
        state["conic"] = conic_type(w)
        artists.extend(ax_conic.plot(*rest.T, color="C0", lw=1, alpha=0.35))
        artists.extend(ax_conic.plot(*arc.T, color="C0", lw=2.4))
        artists.extend(
            ax_conic.plot(*CONIC.T, "o--", color="0.35", ms=6, lw=1, mfc="white")
        )
        ax_conic.set_title(f"weights (1, {w:.3f}, 1): {state['conic']}", fontsize=10)

        fig.canvas.draw_idle()

    # -- panel 3: the exact full circle (static) -------------------------------------------------
    c_points, c_weights, c_knots, c_p = nurbs_circle(np.zeros(2), 1.0)
    circle = nurbs_curve(c_points, c_weights, c_knots, c_p, T)
    ticks = nurbs_curve(c_points, c_weights, c_knots, c_p, TICKS)
    radial = np.abs(np.linalg.norm(circle, axis=1) - 1).max()
    ax_circle.plot(*c_points.T, "s--", color="0.35", ms=6, lw=1, mfc="white")
    ax_circle.plot(*circle.T, color="C3", lw=2.4)
    ax_circle.plot(*ticks.T, "o", color="k", ms=3)
    for label, (x, y) in zip(np.round(c_weights, 3), c_points[:-1]):
        ax_circle.annotate(
            f"{label:g}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8
        )
    ax_circle.set_title(
        f"full circle: radial error {radial:.0e}\nticks at equal steps of t",
        fontsize=10,
    )

    widget = DraggableControlPoints(ax_shape, CONTROL.copy(), on_change=redraw)
    s_i = add_slider(
        fig,
        [0.08, 0.20, 0.32, 0.03],
        "point i",
        0,
        len(CONTROL) - 1,
        2,
        redraw,
        valfmt="%d",
    )
    s_w = add_slider(fig, [0.08, 0.15, 0.32, 0.03], "log10 w_i", -1.5, 1.5, 0.0, redraw)
    s_c = add_slider(
        fig, [0.52, 0.20, 0.18, 0.03], "middle w", 0.05, 3.0, np.sqrt(0.5), redraw
    )
    check = add_checkbuttons(
        fig,
        [0.08, 0.01, 0.2, 0.11],
        ["B-spline ghost (all w = 1)", "perspective"],
        [True, False],
        redraw,
    )
    redraw()

    def smoke():
        assert radial < 1e-15, radial

        # Pull: a heavy weight brings the curve close to its point; a light one pushes it away.
        i = int(round(s_i.val))
        distance = lambda: np.linalg.norm(
            state["curve"] - widget.points[i], axis=1
        ).min()
        s_w.set_val(1.5)
        near = distance()
        s_w.set_val(-1.5)
        far = distance()
        assert near < 0.1 < far, (near, far)

        # Locality: changing w_i leaves the curve outside [u_i, u_{i+p+1}) untouched.
        s_w.set_val(0.0)
        before = state["curve"].copy()
        s_w.set_val(1.0)
        outside = (T < KNOTS[i]) | (T > KNOTS[i + P + 1])
        assert np.abs(state["curve"] - before)[outside].max() == 0.0

        # Projective invariance, through the real checkbox callback.
        check.set_active(1)
        exact, naive = state["perspective"]
        assert exact < 1e-12 and naive > 0.05, (exact, naive)

        # Conic types from the one middle weight.
        for w, kind in ((0.5, "ellipse"), (1.0, "parabola"), (2.0, "hyperbola")):
            s_c.set_val(w)
            assert state["conic"] == kind

        widget.simulate_drag(4, (5.0, -1.5))
        assert np.allclose(widget.points[4], (5.0, -1.5), atol=1e-6)
        print(
            f"  circle radial error {radial:.1e}; weight pull: {far:.2f} -> {near:.3f}"
        )
        print(
            f"  perspective: mapped control data {exact:.1e}, points only {naive:.2f}"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
