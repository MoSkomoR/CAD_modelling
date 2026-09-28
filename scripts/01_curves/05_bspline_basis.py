"""The B-spline basis, built by Cox-de Boor. Watch box functions grow into bumps.

Degree 0 is one box per knot span: 1 on [u_i, u_{i+1}), 0 elsewhere. Each Cox-de Boor step
blends two neighbouring functions of the level below into one that is a span wider and one
degree smoother:

    N_{i,d} = (t - u_i)/(u_{i+d} - u_i) N_{i,d-1}  +  (u_{i+d+1} - t)/(u_{i+d+1} - u_{i+1}) N_{i+1,d-1}

So N_{i,p} was built from p+1 boxes and cannot reach outside [u_i, u_{i+p+1}) -- that is local
support, and it is the whole reason B-splines exist (Bezier's Bernstein functions are non-zero
on the entire interval).

Top: the degree-p basis. The dashed line is their sum -- always exactly 1 (partition of unity).
The shaded band is the support of the highlighted function. Bottom: one Cox-de Boor level at a
time; slide "level" from 0 to p to watch the boxes become bumps.

Move the five interior knots and raise the middle knot's multiplicity: a repeated knot narrows
the functions around it and makes one of them peak at 1 there -- the curve is pulled onto a
control point and loses a degree of smoothness (script 06, and the continuity test).

Notes: notes/01_Curves/B_Splines.md
Run:   python scripts/01_curves/05_bspline_basis.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.bspline import bspline_basis_levels, clamped_knots
from cadkernel.viz.interactive import add_slider, run

T = np.linspace(0.0, 1.0, 801)


def main():
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(bottom=0.34, top=0.93, hspace=0.25)

    artists = []
    state = {}

    def clear():
        while artists:
            artists.pop().remove()

    def current_knots():
        p = int(round(s_degree.val))
        a, b, c, d, e = sorted(slider.val for slider in knot_sliders)
        multiplicity = min(int(round(s_mult.val)), p)
        interior = [a, b] + [c] * multiplicity + [d, e]
        n_ctrl = p + 1 + len(interior)
        return clamped_knots(n_ctrl, p, interior), p

    def redraw(_=None):
        knots, p = current_knots()
        levels = bspline_basis_levels(knots, p, T)
        basis = levels[-1]
        n_funcs = basis.shape[1]
        highlight = min(int(round(s_index.val)), n_funcs - 1)
        level = min(int(round(s_level.val)), p)
        state.update(knots=knots, p=p, levels=levels, highlight=highlight, level=level)
        clear()

        colours = plt.cm.tab10(np.arange(n_funcs) % 10)
        for i in range(n_funcs):
            lw, alpha = (3.0, 1.0) if i == highlight else (1.4, 0.55)
            artists.extend(
                ax_top.plot(T, basis[:, i], color=colours[i], lw=lw, alpha=alpha)
            )
        artists.extend(ax_top.plot(T, basis.sum(axis=1), "k--", lw=1.2, label="sum"))
        lo, hi = knots[highlight], knots[highlight + p + 1]
        artists.append(ax_top.axvspan(lo, hi, color=colours[highlight], alpha=0.12))

        shown = levels[level]
        for i in range(shown.shape[1]):
            artists.extend(
                ax_bottom.plot(
                    T,
                    shown[:, i],
                    color=plt.cm.viridis(i / max(shown.shape[1] - 1, 1)),
                    lw=1.6,
                )
            )

        for ax in (ax_top, ax_bottom):
            for value in np.unique(knots):
                count = int(np.sum(knots == value))
                artists.append(ax.axvline(value, color="0.6", lw=0.8, ls=":"))
                if 0 < value < 1 and count > 1:
                    artists.append(
                        ax.text(
                            value,
                            1.08,
                            f"x{count}",
                            ha="center",
                            fontsize=9,
                            color="0.3",
                        )
                    )

        ax_top.set_title(
            f"degree p = {p}: {n_funcs} basis functions N_(i,{p});  "
            f"N_{highlight},{p} is non-zero only on [{lo:.2f}, {hi:.2f})   "
            f"|  max |sum - 1| = {np.abs(basis.sum(axis=1) - 1).max():.1e}",
            fontsize=10,
        )
        ax_bottom.set_title(
            f"Cox-de Boor level d = {level}: N_(i,{level})", fontsize=10
        )
        fig.canvas.draw_idle()

    for ax in (ax_top, ax_bottom):
        ax.set_ylim(-0.05, 1.18)
        ax.grid(True, alpha=0.3)
    ax_bottom.set_xlabel("t")

    s_degree = add_slider(
        fig, [0.15, 0.25, 0.3, 0.025], "degree p", 1, 4, 3, redraw, valfmt="%d"
    )
    s_level = add_slider(
        fig, [0.15, 0.21, 0.3, 0.025], "level d", 0, 4, 1, redraw, valfmt="%d"
    )
    s_index = add_slider(
        fig, [0.15, 0.17, 0.3, 0.025], "highlight i", 0, 11, 4, redraw, valfmt="%d"
    )
    s_mult = add_slider(
        fig, [0.15, 0.13, 0.3, 0.025], "middle mult.", 1, 4, 1, redraw, valfmt="%d"
    )
    knot_sliders = [
        add_slider(
            fig,
            [0.62, 0.25 - 0.04 * j, 0.3, 0.025],
            f"knot {name}",
            0.01,
            0.99,
            (j + 1) / 6,
            redraw,
        )
        for j, name in enumerate("abcde")
    ]

    redraw()

    def smoke():
        cases = (
            (1, 1, (0.2, 0.3, 0.4, 0.6, 0.9)),
            (3, 2, (0.1, 0.3, 0.5, 0.55, 0.6)),
            (4, 3, (0.1, 0.2, 0.5, 0.7, 0.8)),
        )
        for p, mult, values in cases:
            s_degree.set_val(p)
            s_mult.set_val(mult)
            for slider, value in zip(knot_sliders, values):
                slider.set_val(value)
            for level in range(p + 1):
                s_level.set_val(level)
                levels = state["levels"]
                assert np.allclose(levels[level].sum(axis=1), 1.0), (
                    "every level sums to 1"
                )
            basis, knots = state["levels"][-1], state["knots"]
            assert np.all(basis >= 0)
            for i in range(basis.shape[1]):
                outside = (T < knots[i]) | (T > knots[i + p + 1])
                assert np.all(basis[outside, i] == 0.0), "local support"
        s_index.set_val(11)  # beyond the last function: clamps instead of crashing
        print(
            "  partition of unity and local support held for degrees 1, 3, 4 with repeated knots"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
