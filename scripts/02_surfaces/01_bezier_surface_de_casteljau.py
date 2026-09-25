"""A Bezier surface is De Casteljau's algorithm run twice. Watch it happen.

The patch is a *tensor product*: a rectangular net of control points, degree n in u and degree
m in v. To evaluate S(u, v):

  Pass 1 (u).  Every column of the net is the control polygon of an ordinary Bezier curve.
               Run De Casteljau at u down each one. What is left is m+1 points Q_0..Q_m.
  Pass 2 (v).  Those m+1 points are themselves a control polygon -- of the isoparametric curve
               v -> S(u, v). Run De Casteljau on them at v, and you are done.

Nothing was invented for the second dimension. De Casteljau's intermediate results are *points*,
so they can be fed straight back into De Casteljau. Tick "other order" to run the passes the
other way round (v first, then u) and watch the second construction land on the same point --
the title prints the discrepancy, which is 0.

Drag the u and v sliders to move the evaluation point; pick a control point with "index" and
raise or lower it with "height" to feel how the patch responds. Tick "subdivide" to split the
patch at (u, v) into four sub-patches -- the surface version of subdivision-for-free, and the
reason kernels can intersect surfaces by recursive culling.

Notes: notes/02_Surfaces/Bezier_Surfaces.md
Run:   python scripts/02_surfaces/01_bezier_surface_de_casteljau.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.curves import bezier_curve
from cadkernel.geometry.surfaces import (
    bezier_isocurve,
    bezier_surface,
    bezier_surface_point,
    bezier_surface_subdivide,
    de_casteljau_surface_stages,
)
from cadkernel.viz.interactive import add_checkbuttons, add_slider, run

# A bicubic patch with a saddle in it, so the two directions are visibly different.
NET = np.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.2], [0.0, 2.0, 1.0], [0.0, 3.0, 0.0]],
        [[1.0, 0.0, 1.0], [1.0, 1.0, 2.2], [1.0, 2.0, 0.4], [1.0, 3.0, 0.6]],
        [[2.0, 0.0, 0.8], [2.0, 1.0, -0.6], [2.0, 2.0, 1.4], [2.0, 3.0, 1.2]],
        [[3.0, 0.0, 0.0], [3.0, 1.0, 0.8], [3.0, 2.0, 1.0], [3.0, 3.0, 0.2]],
    ]
)
GRID = np.linspace(0, 1, 30)
FINE = np.linspace(0, 1, 60)
QUADRANT_COLOURS = [["C1", "C2"], ["C4", "C5"]]


def main():
    net = NET.copy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.30, top=0.94)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=28, azim=-58)

    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        u, v = s_u.val, s_v.val
        show_net, show_stages, show_iso, show_other, show_split = check.get_status()
        clear()

        stages = de_casteljau_surface_stages(net, u, v)
        point = stages["point"]

        if show_split:
            # Four sub-patches, each drawn from its own control net -- together they are the
            # original surface, exactly (test_surface_subdivide_reproduces_original).
            quadrants = bezier_surface_subdivide(net, u, v)
            for a in range(2):
                for b in range(2):
                    piece = bezier_surface(quadrants[a][b], GRID, GRID)
                    artists.append(
                        ax.plot_surface(
                            piece[..., 0],
                            piece[..., 1],
                            piece[..., 2],
                            color=QUADRANT_COLOURS[a][b],
                            alpha=0.55,
                            linewidth=0,
                            antialiased=True,
                        )
                    )
        else:
            surface = bezier_surface(net, GRID, GRID)
            artists.append(
                ax.plot_surface(
                    surface[..., 0],
                    surface[..., 1],
                    surface[..., 2],
                    color="0.6",
                    alpha=0.35,
                    linewidth=0,
                    antialiased=True,
                )
            )

        if show_net:
            for i in range(net.shape[0]):
                artists.extend(
                    ax.plot(*net[i, :].T, "o-", color="0.35", ms=4, lw=0.8, alpha=0.8)
                )
            for j in range(net.shape[1]):
                artists.extend(
                    ax.plot(*net[:, j].T, "-", color="0.35", lw=0.8, alpha=0.8)
                )
            artists.extend(
                ax.plot(*net[selected_index()].T, "o", color="C3", ms=11, alpha=0.9)
            )

        if show_iso:
            iso_v = bezier_curve(bezier_isocurve(net, u=u), FINE)
            iso_u = bezier_curve(bezier_isocurve(net, v=v), FINE)
            artists.extend(ax.plot(*iso_v.T, color="C0", lw=2.5))
            artists.extend(ax.plot(*iso_u.T, color="C3", lw=2.5))

        if show_stages:
            # Pass 1 left these m+1 points; they are the control polygon of the v-isocurve.
            artists.extend(
                ax.plot(*stages["u_pass"].T, "s--", color="C0", ms=8, lw=1.6)
            )
            if show_other:
                artists.extend(
                    ax.plot(*stages["v_pass"].T, "^--", color="C3", ms=8, lw=1.6)
                )

        artists.extend(ax.plot(*point[:, None], "*", color="crimson", ms=20, zorder=10))

        discrepancy = np.abs(stages["point"] - stages["point_other_order"]).max()
        fig.suptitle(
            f"S({u:.3f}, {v:.3f}) = ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})      "
            f"|u-then-v  −  v-then-u| = {discrepancy:.1e}",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def selected_index():
        flat = int(round(s_index.val))
        return divmod(flat, net.shape[1])

    def on_index(_=None):
        i, j = selected_index()
        s_height.eventson = False
        s_height.set_val(net[i, j, 2])
        s_height.eventson = True
        redraw()

    def on_height(_=None):
        i, j = selected_index()
        net[i, j, 2] = s_height.val
        redraw()

    s_u = add_slider(fig, [0.12, 0.22, 0.34, 0.03], "u", 0.0, 1.0, 0.35, redraw)
    s_v = add_slider(fig, [0.12, 0.17, 0.34, 0.03], "v", 0.0, 1.0, 0.65, redraw)
    s_index = add_slider(
        fig,
        [0.12, 0.10, 0.34, 0.03],
        "control point",
        0,
        net.size // 3 - 1,
        5,
        None,
        valfmt="%d",
    )
    s_height = add_slider(
        fig, [0.12, 0.05, 0.34, 0.03], "height (z)", -2.0, 3.0, net[1, 1, 2]
    )
    s_index.on_changed(on_index)
    s_height.on_changed(on_height)

    check = add_checkbuttons(
        fig,
        [0.60, 0.03, 0.24, 0.22],
        [
            "control net",
            "De Casteljau stages",
            "isocurves",
            "other order (v first)",
            "subdivide",
        ],
        [True, True, True, False, False],
        redraw,
    )

    redraw()

    def smoke():
        # The corners of the patch are the corners of the net, and nothing else is interpolated.
        assert np.allclose(bezier_surface_point(net, 0.0, 0.0), net[0, 0])
        assert np.allclose(bezier_surface_point(net, 1.0, 1.0), net[-1, -1])

        # Both orders of the two passes land on the same point, at several (u, v).
        worst_order_gap = 0.0
        for u, v in ((0.0, 1.0), (0.25, 0.4), (0.8, 0.15)):
            s_u.set_val(u)
            s_v.set_val(v)
            stages = de_casteljau_surface_stages(net, u, v)
            gap = np.abs(stages["point"] - stages["point_other_order"]).max()
            worst_order_gap = max(worst_order_gap, gap)
            assert gap == 0.0 or gap < 1e-15
            # ...and the u pass really did leave the v-isocurve's control polygon behind.
            assert np.allclose(stages["u_pass"], bezier_isocurve(net, u=u))

        # Every checkbutton path draws without error.
        for index in range(5):
            check.set_active(index)
            redraw()

        # Subdivision at the current (u, v) reproduces the patch it came from.
        u, v = s_u.val, s_v.val
        quadrants = bezier_surface_subdivide(net, u, v)
        assert np.allclose(
            bezier_surface_point(quadrants[0][0], 1.0, 1.0),
            bezier_surface_point(net, u, v),
        )
        assert np.allclose(
            bezier_surface_point(quadrants[1][1], 0.5, 0.5),
            bezier_surface_point(net, u + 0.5 * (1 - u), v + 0.5 * (1 - v)),
        )

        # Dragging a control point moves the surface (global support, in two dimensions now).
        before = bezier_surface(net, GRID, GRID).copy()
        s_index.set_val(5)
        s_height.set_val(net[1, 1, 2] + 1.5)
        after = bezier_surface(net, GRID, GRID)
        moved = np.linalg.norm(after - before, axis=-1)
        assert moved.max() > 0.1
        assert moved[1:-1, 1:-1].min() > 0.0, "no interior point is left alone"

        print(
            f"  corners interpolated; u-then-v vs v-then-u agree to {worst_order_gap:.1e}"
        )
        print(
            f"  raising P_1,1 by 1.5 moved every interior point (min {moved[1:-1, 1:-1].min():.2e}, "
            f"max {moved.max():.2f})"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
