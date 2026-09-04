"""The Bezier properties, made visible rather than asserted.

Four panels, all live while you drag the control points:

  top-left      convex hull -- the shaded hull and its bounding box. Drag as violently as you
                like: the curve never escapes. That containment is what lets a kernel reject
                a curve-curve intersection with a box test, before any root-finding happens.

  top-right     the Bernstein basis functions B_i^n(t), with the selected one highlighted.
                Everything else on this figure is a consequence of two facts visible here:
                they are non-negative, and they sum to 1 at every t.

  bottom-left   affine invariance -- rotate/shear the control points with the sliders. The
                title prints max |transform(evaluate(P)) - evaluate(transform(P))|, which stays
                at machine epsilon: transforming the control points and transforming the curve
                are the same operation.

  bottom-right  global support -- nudge the selected control point and plot how far each point
                of the curve moved. The displacement is non-zero *everywhere* except the two
                endpoints, because B_i^n(t) > 0 on all of (0,1). No local control: this is the
                limitation B-splines exist to fix.

Notes: notes/01_Curves/Bernstein_Basis_Properties.md, notes/01_Curves/Bezier_Curves.md
Run:   python scripts/01_curves/03_bezier_properties.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.curves import bernstein_basis, bezier_curve
from cadkernel.viz.interactive import DraggableControlPoints, add_slider, run
from cadkernel.viz.plotting import convex_hull_2d, polygon_patch


def main():
    control_points = np.array(
        [[0.0, 0.0], [1.0, 2.5], [3.0, 2.5], [4.0, 0.0], [5.0, -1.5]]
    )
    n = len(control_points) - 1
    t = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(bottom=0.20, hspace=0.3, wspace=0.25)
    ax_hull, ax_basis, ax_affine, ax_support = (
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    )

    # -- top-left: convex hull ----------------------------------------------------------
    ax_hull.set_title("Convex hull containment")
    ax_hull.set_aspect("equal")
    ax_hull.grid(True, alpha=0.3)
    ax_hull.set_xlim(-1.5, 6.5)
    ax_hull.set_ylim(-3.5, 4.0)
    hull_patch = polygon_patch(
        ax_hull,
        convex_hull_2d(control_points),
        facecolor="C0",
        alpha=0.12,
        edgecolor="C0",
        lw=1.5,
    )
    (bbox_artist,) = ax_hull.plot(
        [], [], ":", color="gray", lw=1.2, label="bounding box"
    )
    (curve_artist,) = ax_hull.plot([], [], color="C0", lw=2.5, label="curve")

    # -- top-right: basis functions -----------------------------------------------------
    ax_basis.set_title("Bernstein basis $B_i^n(t)$")
    ax_basis.grid(True, alpha=0.3)
    ax_basis.set_xlabel("t")
    basis_artists = [
        ax_basis.plot(t, bernstein_basis(n, t)[:, i], lw=1.5)[0] for i in range(n + 1)
    ]
    (sum_artist,) = ax_basis.plot(
        t,
        bernstein_basis(n, t).sum(axis=1),
        "k--",
        lw=1.2,
        label=r"$\sum_i B_i^n(t) = 1$",
    )
    ax_basis.legend(fontsize=9)

    # -- bottom-left: affine invariance -------------------------------------------------
    ax_affine.set_title("Affine invariance")
    ax_affine.set_aspect("equal")
    ax_affine.grid(True, alpha=0.3)
    (affine_curve_artist,) = ax_affine.plot(
        [], [], color="C3", lw=4, alpha=0.35, label="transform(evaluate(P))"
    )
    (affine_points_artist,) = ax_affine.plot(
        [], [], color="C2", lw=1.6, ls="--", label="evaluate(transform(P))"
    )
    (affine_polygon_artist,) = ax_affine.plot([], [], "o:", color="0.6", ms=4, lw=1)
    ax_affine.legend(fontsize=9, loc="upper left")

    # -- bottom-right: global support ---------------------------------------------------
    ax_support.set_title("Global support: displacement along the curve")
    ax_support.grid(True, alpha=0.3)
    ax_support.set_xlabel("t")
    ax_support.set_ylabel(r"$\|C_{nudged}(t) - C(t)\|$")
    (support_artist,) = ax_support.plot([], [], color="C4", lw=2)

    def affine_matrix():
        angle = np.deg2rad(s_rotate.val)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        shear = np.array([[1.0, s_shear.val], [0.0, 1.0]])
        return rotation @ shear

    def redraw(_=None):
        points = widget.points
        index = int(round(s_index.val))
        curve = bezier_curve(points, t)

        # hull panel
        curve_artist.set_data(curve[:, 0], curve[:, 1])
        hull_patch.set_xy(convex_hull_2d(points))
        lo, hi = points.min(axis=0), points.max(axis=0)
        bbox_artist.set_data(
            [lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]]
        )
        inside = (curve >= lo - 1e-9).all() and (curve <= hi + 1e-9).all()
        ax_hull.set_title(f"Convex hull containment  (curve inside box: {inside})")

        # basis panel: highlight the selected control point's basis function
        for i, artist in enumerate(basis_artists):
            selected = i == index
            artist.set_linewidth(3.0 if selected else 1.0)
            artist.set_alpha(1.0 if selected else 0.35)
            artist.set_label(f"$B_{{{i}}}^{{{n}}}$" if selected else None)

        # affine panel
        A = affine_matrix()
        b = np.array([0.0, 0.0])
        evaluate_then_transform = curve @ A.T + b
        transform_then_evaluate = bezier_curve(points @ A.T + b, t)
        residual = np.abs(evaluate_then_transform - transform_then_evaluate).max()
        affine_curve_artist.set_data(
            evaluate_then_transform[:, 0], evaluate_then_transform[:, 1]
        )
        affine_points_artist.set_data(
            transform_then_evaluate[:, 0], transform_then_evaluate[:, 1]
        )
        transformed_polygon = points @ A.T + b
        affine_polygon_artist.set_data(
            transformed_polygon[:, 0], transformed_polygon[:, 1]
        )
        ax_affine.set_title(f"Affine invariance   max difference = {residual:.2e}")
        ax_affine.relim()
        ax_affine.autoscale_view()

        # support panel
        nudged = points.copy()
        nudged[index] = nudged[index] + np.array([0.0, s_nudge.val])
        displacement = np.linalg.norm(bezier_curve(nudged, t) - curve, axis=1)
        support_artist.set_data(t, displacement)
        interior = displacement[1:-1]
        ax_support.set_title(
            f"Global support: moving $P_{{{index}}}$ moves every interior point "
            f"(min interior shift = {interior.min():.3e})"
        )
        ax_support.relim()
        ax_support.autoscale_view()

        fig.canvas.draw_idle()

    widget = DraggableControlPoints(ax_hull, control_points, on_change=redraw)
    s_index = add_slider(
        fig, [0.15, 0.11, 0.7, 0.025], "control point i", 0, n, 2, redraw, valfmt="%d"
    )
    s_nudge = add_slider(
        fig, [0.15, 0.075, 0.7, 0.025], "nudge $P_i$ by", 0.05, 2.0, 0.8, redraw
    )
    s_rotate = add_slider(
        fig, [0.15, 0.04, 0.7, 0.025], "rotate (deg)", -180, 180, 35.0, redraw
    )
    s_shear = add_slider(
        fig, [0.15, 0.005, 0.7, 0.025], "shear", -1.5, 1.5, 0.4, redraw
    )

    redraw()

    def smoke():
        widget.simulate_drag(1, (0.5, 3.8))
        s_index.set_val(3)
        s_rotate.set_val(-70.0)
        s_shear.set_val(-0.9)
        redraw()

        points = widget.points
        curve = bezier_curve(points, t)
        lo, hi = points.min(axis=0), points.max(axis=0)
        assert (curve >= lo - 1e-9).all() and (curve <= hi + 1e-9).all(), (
            "curve escaped the box"
        )

        A = affine_matrix()
        residual = np.abs(
            bezier_curve(points, t) @ A.T - bezier_curve(points @ A.T, t)
        ).max()
        assert residual < 1e-12, f"affine invariance violated: {residual:.2e}"

        displacement = np.linalg.norm(support_artist.get_ydata())
        assert displacement > 0, "nudging a control point moved nothing"
        assert np.all(support_artist.get_ydata()[1:-1] > 0), "support was not global"

    run(fig, smoke)


if __name__ == "__main__":
    main()
