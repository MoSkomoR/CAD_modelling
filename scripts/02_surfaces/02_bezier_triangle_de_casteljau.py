"""A Bezier triangle is De Casteljau's algorithm run over three neighbours instead of two.

The tensor-product patch (script 01) needs a rectangular net because it lerps along u and v
independently. A Bezier *triangle* lerps over a triangular control lattice instead, in
barycentric coordinates (l0, l1, l2) with l0+l1+l2=1:

    P^s_(i,j,k) = l0 P^(s-1)_(i+1,j,k) + l1 P^(s-1)_(i,j+1,k) + l2 P^(s-1)_(i,j,k+1)

Each level blends *three* neighbours at the same fixed weights, the lattice loses one row per
level, and after n levels a single point remains: C(l0, l1, l2). Nothing about De Casteljau's
recursion changes -- only the shape of the lattice it collapses.

Drag the point in the left-hand parameter triangle to move (l0, l1, l2); the right-hand 3D plot
shows where that lands on the patch. Dragging outside the triangle is allowed -- the patch is
still a well-defined polynomial there, just no longer a convex combination (compare the rational
circle/De Casteljau notes, which do the same off-[0,1] check for curves). Pick a control point
with "index" and raise or lower it with "height" to feel global support on the triangular
lattice. Tick "De Casteljau stages" to draw every collapsing level of the lattice at the current
parameter point.

Notes: notes/02_Surfaces/Bezier_Triangles.md
Run:   python scripts/02_surfaces/02_bezier_triangle_de_casteljau.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.surfaces import (
    bezier_triangle,
    bezier_triangle_point,
    de_casteljau_triangle_stages,
)
from cadkernel.viz.interactive import (
    DraggableControlPoints,
    add_checkbuttons,
    add_slider,
    run,
)

DEGREE = 3

# Barycentric corners of the parameter domain, laid out as an equilateral triangle so it can
# double as the 2D plot of (l0, l1, l2) itself: A = l0-axis corner, B = l1-axis, C = l2-axis.
A, B, C = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])


def bary_to_xy(i, j, k, n=DEGREE):
    """Lay a control point's multi-index out in the same 2D triangle as the parameter domain."""
    return (i / n) * A + (j / n) * B + (k / n) * C


def xy_to_bary(point):
    """Barycentric coordinates of a 2D point relative to A, B, C -- defined for any point in
    the plane, not just inside the triangle, which is what lets the demo run off-domain."""
    det = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    l0 = ((B[1] - C[1]) * (point[0] - C[0]) + (C[0] - B[0]) * (point[1] - C[1])) / det
    l1 = ((C[1] - A[1]) * (point[0] - C[0]) + (A[0] - C[0]) * (point[1] - C[1])) / det
    return l0, l1, 1.0 - l0 - l1


def lattice_edges(indices):
    """Pairs of multi-indices one lattice step apart -- the triangulated-triangle edges drawn
    for the control net and for every intermediate De Casteljau level."""
    indices = set(indices)
    edges = set()
    for idx in indices:
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if idx[p] > 0:
                neighbor = list(idx)
                neighbor[p] -= 1
                neighbor[q] += 1
                neighbor = tuple(neighbor)
                if neighbor in indices:
                    edges.add(frozenset((idx, neighbor)))
    return [tuple(e) for e in edges]


# A cubic (n=3) patch: flat corners, a raised interior -- non-planar and asymmetric enough that
# the three barycentric directions are visibly different, the triangle analogue of the tensor
# script's saddle net.
_HEIGHTS = {
    (3, 0, 0): 0.0,
    (0, 3, 0): 0.0,
    (0, 0, 3): 0.0,
    (2, 1, 0): 1.0,
    (1, 2, 0): 1.4,
    (2, 0, 1): 0.6,
    (1, 0, 2): 1.2,
    (0, 1, 2): 0.8,
    (0, 2, 1): 1.6,
    (1, 1, 1): 2.0,
}
NET = {index: np.array([*bary_to_xy(*index), z]) for index, z in _HEIGHTS.items()}
INDICES = list(NET)

# A fine barycentric grid covering the whole simplex, for bulk surface evaluation.
RES = 24
_grid_l0, _grid_l1 = [], []
for _i in range(RES + 1):
    for _j in range(RES + 1 - _i):
        _grid_l0.append(_i / RES)
        _grid_l1.append(_j / RES)
GRID_L0, GRID_L1 = np.array(_grid_l0), np.array(_grid_l1)


def main():
    net = {k: v.copy() for k, v in NET.items()}

    fig = plt.figure(figsize=(12, 7))
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.set_title("parameter domain (l0, l1, l2)")
    ax2d.set_aspect("equal")
    ax2d.axis("off")
    ax2d.plot(*np.array([A, B, C, A]).T, "-", color="0.3", lw=1.2)
    for corner, label in ((A, "l0=1"), (B, "l1=1"), (C, "l2=1")):
        ax2d.annotate(
            label,
            corner,
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color="0.3",
        )

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.30, top=0.90, wspace=0.25)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=25, azim=-58)

    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def redraw(_=None):
        l0, l1, l2 = xy_to_bary(picker.points[0])
        clear()
        show_net, show_surface, show_stages = check.get_status()

        if show_surface:
            grid = bezier_triangle(net, GRID_L0, GRID_L1)
            artists.append(
                ax3d.plot_trisurf(
                    grid[:, 0],
                    grid[:, 1],
                    grid[:, 2],
                    color="0.6",
                    alpha=0.5,
                    linewidth=0.1,
                )
            )

        if show_net:
            for a, b in lattice_edges(INDICES):
                artists.extend(
                    ax3d.plot(
                        *np.array([net[a], net[b]]).T,
                        "-",
                        color="0.35",
                        lw=0.8,
                        alpha=0.8,
                    )
                )
            artists.extend(
                ax3d.plot(
                    *np.array(list(net.values())).T, "o", color="0.35", ms=4, alpha=0.8
                )
            )
            artists.extend(
                ax3d.plot(*net[selected_index()], "o", color="C3", ms=11, alpha=0.9)
            )

        if show_stages:
            levels = de_casteljau_triangle_stages(net, l0, l1, l2)
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(levels)))
            for level, color in zip(levels, colors):
                for a, b in lattice_edges(level):
                    artists.extend(
                        ax3d.plot(
                            *np.array([level[a], level[b]]).T, "-", color=color, lw=1.4
                        )
                    )

        valid = l0 >= 0 and l1 >= 0 and l2 >= 0
        point = bezier_triangle_point(net, l0, l1, l2)
        artists.extend(
            ax3d.plot(
                *point, "*", color="crimson" if valid else "0.4", ms=20, zorder=10
            )
        )

        ax2d.set_title(
            f"(l0, l1, l2) = ({l0:.2f}, {l1:.2f}, {l2:.2f})"
            + ("" if valid else "  -- outside the triangle")
        )
        fig.suptitle(
            f"S(l0, l1, l2) = ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
        )
        fig.canvas.draw_idle()

    def selected_index():
        return INDICES[int(round(s_index.val))]

    def on_index(_=None):
        s_height.eventson = False
        s_height.set_val(net[selected_index()][2])
        s_height.eventson = True
        redraw()

    def on_height(_=None):
        net[selected_index()][2] = s_height.val
        redraw()

    picker = DraggableControlPoints(
        ax2d,
        [bary_to_xy(1, 1, 1)],
        on_change=redraw,
        show_polygon=False,
        color="crimson",
    )
    ax2d.set_xlim(-0.5, 1.5)
    ax2d.set_ylim(-0.5, 1.3)

    s_index = add_slider(
        fig,
        [0.58, 0.16, 0.34, 0.03],
        "control point",
        0,
        len(INDICES) - 1,
        9,
        None,
        valfmt="%d",
    )
    s_height = add_slider(
        fig, [0.58, 0.10, 0.34, 0.03], "height (z)", -1.0, 3.0, net[(1, 1, 1)][2]
    )
    s_index.on_changed(on_index)
    s_height.on_changed(on_height)

    check = add_checkbuttons(
        fig,
        [0.06, 0.05, 0.30, 0.17],
        ["control net", "surface", "De Casteljau stages"],
        [True, True, True],
        redraw,
    )

    redraw()

    def smoke():
        # The three barycentric corners are interpolated, and nothing else is.
        assert np.allclose(bezier_triangle_point(net, 1.0, 0.0, 0.0), net[(3, 0, 0)])
        assert np.allclose(bezier_triangle_point(net, 0.0, 1.0, 0.0), net[(0, 3, 0)])
        assert np.allclose(bezier_triangle_point(net, 0.0, 0.0, 1.0), net[(0, 0, 3)])

        # Dragging the parameter point around the domain moves the smoke-tested point too.
        for xy in (bary_to_xy(2, 1, 0), bary_to_xy(0, 1, 2), bary_to_xy(1, 1, 1)):
            picker.simulate_drag(0, xy)
            l0, l1, l2 = xy_to_bary(picker.points[0])
            assert np.isclose(l0 + l1 + l2, 1.0)
            assert np.allclose(
                bezier_triangle_point(net, l0, l1, l2),
                bezier_triangle(net, [l0], [l1])[0],
            )

        # Every checkbutton path draws without error, including dragging off-domain.
        picker.simulate_drag(0, np.array([-0.3, -0.3]))
        for index in range(3):
            check.set_active(index)
            redraw()

        # Raising the centre control point moves the interior of the patch (global support).
        before = bezier_triangle(net, GRID_L0, GRID_L1).copy()
        s_index.set_val(INDICES.index((1, 1, 1)))
        s_height.set_val(net[(1, 1, 1)][2] + 1.5)
        after = bezier_triangle(net, GRID_L0, GRID_L1)
        moved = np.linalg.norm(after - before, axis=-1)
        interior = (GRID_L0 > 0.05) & (GRID_L1 > 0.05) & (GRID_L0 + GRID_L1 < 0.95)
        assert moved.max() > 0.1
        assert moved[interior].min() > 0.0, "no interior point is left alone"

        print(
            f"  corners interpolated; raising the centre point moved every interior point "
            f"(min {moved[interior].min():.2e}, max {moved.max():.2f})"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
