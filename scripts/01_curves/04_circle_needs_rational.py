"""Why NURBS are a necessity rather than a refinement: the circle.

Raise the degree slider and watch the best-fitting polynomial Bezier chase a quarter circle.
The honest result is probably not the one you expect: convergence is *geometric*, roughly an
order of magnitude per degree, and by degree ~13 the error has fallen to ~2e-15 and stops
improving -- not because it reached the circle, but because it hit the floor of double-precision
arithmetic. Polynomials approximate circles extremely well.

So the case for rational curves is not "polynomials aren't accurate enough". It is:

  * **Never exact, even in principle.** notes/01_Curves/Lines_and_Arcs.md proves no real
    polynomial parametrization of a circle exists. Whatever degree you spend, you are carrying
    an approximation whose error must be tracked in tolerances forever after.
  * **Cost.** Machine-precision agreement costs 14 control points. The rational quadratic costs
    3 control points and 3 weights -- and the gap squares on tensor-product surfaces: a sphere
    patch as ~14x14 = 196 control points versus 3x3 = 9.
  * **Semantics and closure.** A hole should *be* a cylinder: exact tangency, exact symmetry,
    an offset that is still exactly a circle, and a truthful answer to "is this face
    cylindrical?" for feature recognition and manufacturing. Fitted polynomials degrade under
    every one of those operations; exact conics do not.

Tick "rational quadratic" to see 3 control points land on the circle at machine epsilon.

(The polynomial fit is least-squares in the natural uniform-angle parametrization. A cleverer
parametrization moves the constant, never the conclusion.)

Notes: notes/01_Curves/Lines_and_Arcs.md, notes/01_Curves/Bezier_Curves.md
Run:   python scripts/01_curves/04_circle_needs_rational.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.geometry.curves import (
    bernstein_basis,
    bezier_curve,
    rational_arc_quadratic,
    rational_bezier,
)
from cadkernel.viz.interactive import add_checkbuttons, add_slider, run

CENTRE = np.array([0.0, 0.0])
RADIUS = 1.0
THETA0, THETA1 = 0.0, np.pi / 2
DEGREES = np.arange(2, 21)


def true_arc(t):
    theta = THETA0 + np.asarray(t) * (THETA1 - THETA0)
    return CENTRE + RADIUS * np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def best_polynomial_fit(degree, n_samples=400):
    """Least-squares fit of a degree-`degree` Bezier to the arc, at uniform-angle parameters."""
    t_fit = np.linspace(0, 1, n_samples)
    basis = bernstein_basis(degree, t_fit)
    control_points, *_ = np.linalg.lstsq(basis, true_arc(t_fit), rcond=None)
    return control_points


def radial_error(points):
    return np.abs(np.linalg.norm(points - CENTRE, axis=1) - RADIUS)


def main():
    t = np.linspace(0, 1, 400)
    arc_points = true_arc(t)

    rat_cps, rat_weights = rational_arc_quadratic(CENTRE, RADIUS, THETA0, THETA1)
    rat_points = rational_bezier(rat_cps, rat_weights, t)
    rat_err = radial_error(rat_points)

    convergence = np.array(
        [radial_error(bezier_curve(best_polynomial_fit(d), t)).max() for d in DEGREES]
    )

    fig, (ax_geom, ax_err, ax_conv) = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(bottom=0.26, wspace=0.28)

    ax_geom.set_title("Quarter circle")
    ax_geom.set_aspect("equal")
    ax_geom.grid(True, alpha=0.3)
    ax_geom.plot(
        arc_points[:, 0], arc_points[:, 1], lw=7, color="0.85", label="true circle"
    )
    (poly_artist,) = ax_geom.plot([], [], color="C0", lw=2, label="polynomial Bezier")
    (poly_polygon,) = ax_geom.plot([], [], "o--", color="C0", ms=4, lw=0.8, alpha=0.45)
    (rat_artist,) = ax_geom.plot([], [], color="C3", lw=2, label="rational quadratic")
    (rat_polygon,) = ax_geom.plot([], [], "s--", color="C3", ms=7, lw=0.9, alpha=0.7)
    ax_geom.legend(loc="lower left", fontsize=9)

    ax_err.set_title("Radial error along the arc")
    ax_err.set_yscale("log")
    ax_err.set_xlabel("t")
    ax_err.set_ylabel(r"$|\ \|C(t) - c\| - r\ |$")
    ax_err.grid(True, alpha=0.3, which="both")
    (poly_err_artist,) = ax_err.plot([], [], color="C0", lw=2, label="polynomial")
    (rat_err_artist,) = ax_err.plot([], [], color="C3", lw=2, label="rational")
    ax_err.legend(fontsize=9)

    ax_conv.set_title("Convergence, and the double-precision floor")
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("polynomial degree")
    ax_conv.set_ylabel("max radial error")
    ax_conv.grid(True, alpha=0.3, which="both")
    ax_conv.plot(DEGREES, convergence, "o-", color="C0", ms=4, label="polynomial fit")
    ax_conv.axhline(
        np.finfo(float).eps,
        color="0.5",
        ls=":",
        lw=1.5,
        label=f"machine epsilon ({np.finfo(float).eps:.1e})",
    )
    ax_conv.axhline(
        max(rat_err.max(), 1e-18),
        color="C3",
        ls="--",
        lw=1.5,
        label="rational quadratic (3 points)",
    )
    (marker_artist,) = ax_conv.plot([], [], "*", color="crimson", ms=18, zorder=5)
    ax_conv.legend(fontsize=8, loc="lower left")

    def redraw(_=None):
        degree = int(round(s_degree.val))
        control_points = best_polynomial_fit(degree)
        poly_points = bezier_curve(control_points, t)
        poly_err = radial_error(poly_points)

        poly_artist.set_data(poly_points[:, 0], poly_points[:, 1])
        poly_polygon.set_data(control_points[:, 0], control_points[:, 1])
        poly_err_artist.set_data(t, np.maximum(poly_err, 1e-18))
        marker_artist.set_data([degree], [max(poly_err.max(), 1e-18)])

        show_rational = check.get_status()[0]
        for artist in (rat_artist, rat_polygon, rat_err_artist):
            artist.set_visible(show_rational)
        if show_rational:
            rat_artist.set_data(rat_points[:, 0], rat_points[:, 1])
            rat_polygon.set_data(rat_cps[:, 0], rat_cps[:, 1])
            rat_err_artist.set_data(t, np.maximum(rat_err, 1e-18))

        headline = (
            f"degree {degree}: {degree + 1} control points, "
            f"max error {poly_err.max():.2e}"
        )
        if show_rational:
            headline += (
                f"      vs.  rational: 3 control points + 3 weights, "
                f"max error {rat_err.max():.2e}"
            )
        fig.suptitle(headline, fontsize=11)
        ax_err.relim()
        ax_err.autoscale_view()
        fig.canvas.draw_idle()

    s_degree = add_slider(
        fig, [0.15, 0.11, 0.7, 0.03], "polynomial degree", 2, 20, 2, redraw, valfmt="%d"
    )
    check = add_checkbuttons(
        fig, [0.15, 0.005, 0.22, 0.08], ["rational quadratic"], [False], redraw
    )

    redraw()

    def smoke():
        by_degree = dict(zip(DEGREES.tolist(), convergence.tolist()))

        # Geometric convergence while the maths, not the arithmetic, is the limit.
        assert by_degree[9] < by_degree[6] < by_degree[3] < by_degree[2], by_degree
        assert by_degree[9] < 1e-9, by_degree[9]

        # ... then a plateau at the double-precision floor: it stops improving, and never
        # becomes exact. This is the point the note makes -- accuracy is not the argument.
        floor = min(by_degree[d] for d in (15, 18, 20))
        assert floor > 0.0, "a polynomial cannot represent the circle exactly"
        assert by_degree[20] > 0.3 * by_degree[13], (
            "expected a plateau, not further convergence"
        )

        # The rational quadratic reaches the same floor with 3 control points instead of 14+.
        assert rat_err.max() < 1e-13, f"rational arc not exact: {rat_err.max():.2e}"

        s_degree.set_val(13)
        check.set_active(0)
        redraw()
        assert rat_artist.get_visible()
        print(
            f"  polynomial: deg 2 {by_degree[2]:.1e} -> deg 9 {by_degree[9]:.1e} "
            f"-> deg 13 {by_degree[13]:.1e} -> deg 20 {by_degree[20]:.1e} (plateau)"
        )
        print(f"  rational quadratic (3 control points): {rat_err.max():.1e}")

    run(fig, smoke)


if __name__ == "__main__":
    main()
