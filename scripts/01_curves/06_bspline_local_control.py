"""Local control: drag a control point of a B-spline and only p+1 spans move.

The curve is a cubic B-spline on 10 control points, coloured by knot span. Drag any point: the
part of the curve that responds is exactly the highlighted support [u_i, u_{i+p+1}) of the point
you picked with the "point i" slider -- everything else stays put to the last bit (the smoke
check measures the change outside the support: exactly 0.0). Tick "Bezier, same points" to see the
degree-9 Bezier curve on the same polygon: every one of its points moves, and by less.

The t slider walks de Boor's algorithm: only the p+1 control points of the span containing t
take part, lerped at knot-dependent ratios, collapsing to the point on the curve.

The remaining toggles are the tools a kernel builds on top, and none of them changes the curve:
  * "insert knot at t": Boehm insertion -- one more knot, one more control point, a tighter
    polygon. The B-spline counterpart of subdivision.
  * "Bezier segments": insert every interior knot p times and the polygon falls apart into one
    Bezier control polygon per span, sharing end points. This is how every Bezier tool is reused.
  * "elevate degree": the same curve as a degree-4 B-spline.

Notes: notes/01_Curves/B_Splines.md, notes/01_Curves/De_Boor_Algorithm.md
Run:   python scripts/01_curves/06_bspline_local_control.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.bspline import (
    bspline_basis,
    bspline_curve,
    bspline_to_bezier,
    clamped_knots,
    de_boor_stages,
    elevate_degree,
    insert_knot,
)
from cadkernel.geometry.curves import bezier_curve
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_slider,
    run,
)

P = 3
CONTROL = np.c_[np.arange(10.0), np.tile([0.0, 2.0, -1.0, 2.0, -1.0], 2)]
KNOTS = clamped_knots(len(CONTROL), P)
T = np.linspace(0.0, 1.0, 1201)
SPAN_COLOURS = plt.cm.tab10(np.arange(10))


def main():
    fig = plt.figure(figsize=(12, 8.5))
    ax = fig.add_axes([0.06, 0.40, 0.70, 0.54])
    ax_basis = fig.add_axes([0.06, 0.20, 0.70, 0.14])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.0, 10.0)
    ax.set_ylim(-3.0, 4.0)
    ax_basis.set_xlim(0, 1)
    ax_basis.set_ylim(-0.05, 1.05)
    ax_basis.set_xlabel("t")
    ax_basis.grid(True, alpha=0.3)

    artists = []
    state = {}

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        points = widget.points
        t = s_t.val
        i = int(round(s_i.val))
        show_stages, show_bezier, show_insert, show_segments, show_elevate = (
            check.get_status()
        )
        clear()

        curve = bspline_curve(points, KNOTS, P, T)
        span_index = np.searchsorted(np.unique(KNOTS), T, side="right") - 1
        for s in range(len(np.unique(KNOTS)) - 1):
            idx = np.flatnonzero(span_index == s)
            idx = np.append(idx, min(idx[-1] + 1, len(T) - 1))  # join to the next span
            artists.extend(
                ax.plot(*curve[idx].T, color=SPAN_COLOURS[s], lw=2.5, zorder=3)
            )

        support = (T >= KNOTS[i]) & (T <= KNOTS[i + P + 1])
        artists.extend(
            ax.plot(*curve[support].T, color="gold", lw=9, alpha=0.35, zorder=2)
        )
        artists.extend(
            ax.plot(*points[i], "o", ms=16, mfc="none", mec="gold", mew=3, zorder=6)
        )

        if show_bezier:
            artists.extend(
                ax.plot(
                    *bezier_curve(points, T).T, color="0.2", lw=1.5, ls="--", zorder=3
                )
            )

        if show_stages:
            stages = de_boor_stages(points, KNOTS, P, t)
            for r, level in enumerate(stages["levels"]):
                artists.extend(
                    ax.plot(
                        *level.T,
                        "o-",
                        ms=4,
                        lw=1.2,
                        color=plt.cm.viridis(r / P),
                        zorder=4,
                    )
                )
            artists.extend(
                ax.plot(*stages["point"], "*", color="crimson", ms=18, zorder=7)
            )

        messages = []
        if show_insert:
            new_points, new_knots = insert_knot(
                points, KNOTS, P, float(np.clip(t, 1e-6, 1 - 1e-6))
            )
            artists.extend(
                ax.plot(*new_points.T, "s:", color="tab:orange", ms=6, lw=1.4, zorder=5)
            )
            gap = np.abs(bspline_curve(new_points, new_knots, P, T) - curve).max()
            messages.append(f"insert: {gap:.0e}")
        if show_segments:
            for s, segment in enumerate(bspline_to_bezier(points, KNOTS, P)):
                artists.extend(
                    ax.plot(
                        *segment.T,
                        "D-",
                        color=SPAN_COLOURS[s],
                        ms=5,
                        lw=1,
                        alpha=0.8,
                        zorder=5,
                    )
                )
        if show_elevate:
            new_points, new_knots, q = elevate_degree(points, KNOTS, P)
            artists.extend(
                ax.plot(*new_points.T, "^:", color="tab:purple", ms=5, lw=1, zorder=5)
            )
            gap = np.abs(bspline_curve(new_points, new_knots, q, T) - curve).max()
            messages.append(f"elevate: {gap:.0e}")

        basis = bspline_basis(KNOTS, P, T)
        for j in range(basis.shape[1]):
            lw, alpha = (3, 1.0) if j == i else (1, 0.4)
            artists.extend(
                ax_basis.plot(
                    T,
                    basis[:, j],
                    lw=lw,
                    alpha=alpha,
                    color="goldenrod" if j == i else "0.4",
                )
            )
        artists.append(
            ax_basis.axvspan(KNOTS[i], KNOTS[i + P + 1], color="gold", alpha=0.2)
        )
        artists.append(ax_basis.axvline(t, color="crimson", lw=1))

        state["curve"] = curve
        ax.set_title(
            f"cubic B-spline, 10 control points  |  P_{i} influences t in [{KNOTS[i]:.2f}, {KNOTS[i + P + 1]:.2f})"
            + ("   |   curve change vs " + ", ".join(messages) if messages else ""),
            fontsize=10,
        )
        fig.canvas.draw_idle()

    widget = DraggableControlPoints(ax, CONTROL.copy(), on_change=redraw)
    s_t = add_slider(fig, [0.10, 0.10, 0.55, 0.025], "t", 0.0, 1.0, 0.45, redraw)
    s_i = add_slider(
        fig,
        [0.10, 0.06, 0.55, 0.025],
        "point i",
        0,
        len(CONTROL) - 1,
        5,
        redraw,
        valfmt="%d",
    )
    check = add_checkbuttons(
        fig,
        [0.79, 0.45, 0.19, 0.25],
        [
            "de Boor stages",
            "Bezier, same points",
            "insert knot at t",
            "Bezier segments",
            "elevate degree",
        ],
        [True, False, False, False, False],
        redraw,
    )

    redraw()

    def smoke():
        before = bspline_curve(widget.points, KNOTS, P, T)
        bezier_before = bezier_curve(widget.points, T)
        s_i.set_val(5)
        widget.simulate_drag(5, widget.points[5] + np.array([0.0, 1.0]))
        after = state["curve"]
        change = np.linalg.norm(after - before, axis=1)
        outside = (T < KNOTS[5]) | (T >= KNOTS[9])
        assert change.max() > 0.1, "the drag moved the curve"
        assert np.all(change[outside] == 0.0), "nothing outside the support moved"
        bezier_change = np.linalg.norm(
            bezier_curve(widget.points, T) - bezier_before, axis=1
        )
        assert np.all(bezier_change[1:-1] > 0.0), (
            "the Bezier on the same points moved everywhere"
        )

        for index in range(5):
            check.set_active(index)
        s_t.set_val(0.62)
        points = widget.points
        new_points, new_knots = insert_knot(points, KNOTS, P, 0.62)
        assert np.allclose(bspline_curve(new_points, new_knots, P, T), after)
        new_points, new_knots, q = elevate_degree(points, KNOTS, P)
        assert np.allclose(bspline_curve(new_points, new_knots, q, T), after)
        print(
            f"  moving P_5 changed {np.mean(change > 0):.0%} of the curve (outside: exactly 0); "
            f"the Bezier moved {np.mean(bezier_change > 0):.0%}"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
