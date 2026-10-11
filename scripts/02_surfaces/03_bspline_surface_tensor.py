"""A B-spline surface is a curve of curves. Watch the isocurve sweep the surface out.

The net has 11 x 9 control points: degree 3 along u (the columns), degree 2 along v (the rows),
each direction on its own knot vector.

  * **Curve of curves.** Every column of the net is the control polygon of an ordinary B-spline
    curve in u (thin blue). Evaluate all of them at the current u: one point per column (blue
    squares). Those points are a control polygon in their own right -- of the curve v -> S(u, v),
    the thick dark-blue isocurve. Drag the u slider and that curve sweeps across the patch; the shaded
    part of the surface is what it has traced so far. Nothing here is new: it is the B-spline
    curve of module 01, used twice.

  * **Local control, in two directions.** Pick a control point and raise it. Only the red
    rectangle of the surface moves: P_ij acts on [u_i, u_{i+p+1}) in u *and* [v_j, v_{j+q+1}) in
    v. The small square on the right is the (u, v) parameter domain with the knot lines drawn,
    so you can see the same rectangle where it is defined -- a product of two curve supports.

Notes: notes/02_Surfaces/NURBS_Surfaces.md
Run:   python scripts/02_surfaces/03_bspline_surface_tensor.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.bspline import bspline_curve, clamped_knots
from cadkernel.geometry.nurbs_surfaces import bspline_isocurve, bspline_surface
from cadkernel.viz.interactive import add_checkbuttons, add_slider, run

P, Q = 3, 2
X, Y = np.meshgrid(np.arange(11.0), np.arange(9.0), indexing="ij")
NET = np.stack([X, Y, 0.8 * np.sin(0.6 * X) * np.cos(0.5 * Y) + 0.3], axis=-1)
KNOTS_U = clamped_knots(11, P)
KNOTS_V = clamped_knots(9, Q)
FINE = np.linspace(0.0, 1.0, 80)
GRID = np.linspace(0.0, 1.0, 40)


def support(knots, degree, index, t):
    """Mask of parameters inside [u_index, u_{index+degree+1}); the right end closes the last span."""
    lo, hi = knots[index], knots[index + degree + 1]
    return (t >= lo) & ((t < hi) | ((hi == knots[-1]) & (t == hi)))


def main():
    net = NET.copy()
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.0, 0.2, 0.68, 0.78], projection="3d")
    ax_uv = fig.add_axes([0.72, 0.42, 0.25, 0.45])
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim(-1, 3)
    ax_uv.set_aspect("equal")
    ax_uv.set_xlim(0, 1)
    ax_uv.set_ylim(0, 1)
    ax_uv.set_xlabel("u")
    ax_uv.set_ylabel("v")
    ax_uv.set_title("parameter domain", fontsize=10)
    for k in np.unique(KNOTS_U):
        ax_uv.axvline(k, color="0.6", lw=0.8)
    for k in np.unique(KNOTS_V):
        ax_uv.axhline(k, color="0.6", lw=0.8)

    artists = []
    state = {}

    def clear():
        while artists:
            artists.pop().remove()

    def selected():
        return divmod(int(round(s_index.val)), net.shape[1])

    def redraw(_=None):
        clear()
        u = s_u.val
        i, j = selected()
        show_columns, show_net, show_support = check.get_status()

        surface = bspline_surface(net, KNOTS_U, KNOTS_V, P, Q, GRID, GRID)
        artists.append(
            ax.plot_surface(
                *surface.transpose(2, 0, 1), color="0.7", alpha=0.18, linewidth=0
            )
        )
        traced = GRID <= u
        if traced.sum() > 1:
            part = surface[traced]
            artists.append(
                ax.plot_surface(
                    *part.transpose(2, 0, 1), color="C0", alpha=0.35, linewidth=0
                )
            )

        # Curve of curves: every column is a u-curve; its points at u are the isocurve's polygon.
        polygon = bspline_isocurve(net, KNOTS_U, P, u=u)
        state["polygon"] = polygon
        if show_columns:
            for col in range(net.shape[1]):
                artists.extend(
                    ax.plot(
                        *bspline_curve(net[:, col], KNOTS_U, P, FINE).T,
                        color="C0",
                        lw=1,
                        alpha=0.6,
                    )
                )
        artists.extend(ax.plot(*polygon.T, "s--", color="C0", ms=7, lw=1.3))
        artists.extend(
            ax.plot(*bspline_curve(polygon, KNOTS_V, Q, FINE).T, color="navy", lw=3.5)
        )

        if show_net:
            for row in net:
                artists.extend(ax.plot(*row.T, "o-", color="0.35", ms=3, lw=0.7))
            for col in net.transpose(1, 0, 2):
                artists.extend(ax.plot(*col.T, "-", color="0.35", lw=0.7))
        artists.extend(ax.plot(*net[i, j][:, None], "o", color="C3", ms=11))

        # Local support: the rectangle [u_i, u_{i+p+1}) x [v_j, v_{j+q+1}).
        in_u = support(KNOTS_U, P, i, GRID)
        in_v = support(KNOTS_V, Q, j, GRID)
        state["support"] = (in_u, in_v)
        if show_support:
            patch = surface[np.ix_(in_u, in_v)]
            artists.append(
                ax.plot_surface(
                    *patch.transpose(2, 0, 1), color="C3", alpha=0.45, linewidth=0
                )
            )
        u0, u1 = KNOTS_U[i], KNOTS_U[i + P + 1]
        v0, v1 = KNOTS_V[j], KNOTS_V[j + Q + 1]
        artists.append(
            ax_uv.add_patch(
                plt.Rectangle(
                    (u0, v0),
                    u1 - u0,
                    v1 - v0,
                    color="C3",
                    alpha=0.3 if show_support else 0.0,
                )
            )
        )
        artists.append(ax_uv.axvline(u, color="C0", lw=3))
        artists.append(ax_uv.axvspan(0, u, color="C0", alpha=0.1))

        fig.suptitle(
            f"u = {u:.2f}: the blue squares are the {net.shape[1]} column curves evaluated at u, "
            f"and the control polygon of the isocurve v -> S(u, v)\n"
            f"P_{i},{j} acts on [u_{i}, u_{i + P + 1}) x [v_{j}, v_{j + Q + 1}) "
            f"= [{u0:.2f}, {u1:.2f}) x [{v0:.2f}, {v1:.2f})",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def on_index(_=None):
        i, j = selected()
        s_height.eventson = False
        s_height.set_val(net[i, j, 2])
        s_height.eventson = True
        redraw()

    def on_height(_=None):
        i, j = selected()
        net[i, j, 2] = s_height.val
        redraw()

    s_u = add_slider(fig, [0.08, 0.13, 0.4, 0.03], "u", 0.0, 1.0, 0.4, redraw)
    s_index = add_slider(
        fig,
        [0.08, 0.08, 0.4, 0.03],
        "control point",
        0,
        net.shape[0] * net.shape[1] - 1,
        49,
        None,
        valfmt="%d",
    )
    s_height = add_slider(
        fig, [0.08, 0.03, 0.4, 0.03], "height (z)", -1.5, 3.0, net[5, 4, 2]
    )
    s_index.on_changed(on_index)
    s_height.on_changed(on_height)
    check = add_checkbuttons(
        fig,
        [0.72, 0.05, 0.2, 0.18],
        ["column curves (in u)", "control net", "support of P_ij"],
        [True, False, True],
        redraw,
    )
    redraw()

    def smoke():
        # The polygon drawn is the columns' points at u, and its curve lies on the surface.
        for u in (0.0, 0.55, 1.0):
            s_u.set_val(u)
            iso = bspline_curve(state["polygon"], KNOTS_V, Q, GRID)
            exact = bspline_surface(net, KNOTS_U, KNOTS_V, P, Q, [u], GRID)[0]
            assert np.abs(iso - exact).max() < 1e-12
        # Raising the picked point moves the surface inside its rectangle only.
        s_index.set_val(49)
        before = bspline_surface(net, KNOTS_U, KNOTS_V, P, Q, GRID, GRID)
        s_height.set_val(net[5, 4, 2] + 1.0)
        after = bspline_surface(net, KNOTS_U, KNOTS_V, P, Q, GRID, GRID)
        in_u, in_v = state["support"]
        moved = np.abs(after - before).max(axis=-1)
        assert moved[~np.outer(in_u, in_v)].max() == 0.0 and moved.max() > 0.1
        for index in range(3):
            check.set_active(index)
        print(
            f"  P_5,4 moved {np.outer(in_u, in_v).mean():.0%} of the (u, v) grid, nothing outside"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
