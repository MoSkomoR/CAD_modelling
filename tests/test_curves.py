from fractions import Fraction
from math import comb

import numpy as np
import pytest

from cadkernel.geometry.curves import (
    arc,
    bernstein_basis,
    bezier_de_casteljau,
    bezier_curve,
    bezier_subdivide,
    de_casteljau_triangle,
    line,
    rational_arc_quadratic,
    rational_bezier,
)


def test_line_endpoints():
    p0, p1 = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    pts = line(p0, p1, np.array([0.0, 1.0]))
    assert np.allclose(pts[0], p0)
    assert np.allclose(pts[1], p1)


def test_arc_radius_is_constant():
    center = np.array([1.0, -2.0])
    radius = 2.5
    t = np.linspace(0, 1, 50)
    pts = arc(center, radius, t)
    dist = np.linalg.norm(pts - center, axis=1)
    assert np.allclose(dist, radius)


def test_bezier_endpoint_interpolation():
    """A Bezier curve always passes through its first and last control points."""
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [4.0, 2.0]])
    pts = bezier_curve(control_points, np.array([0.0, 1.0]))
    assert np.allclose(pts[0], control_points[0])
    assert np.allclose(pts[1], control_points[-1])


def test_bezier_convex_hull_property():
    """Every point on a Bezier curve lies within the convex hull of its control points --
    here checked via bounding box, a necessary (not sufficient) but easy consequence."""
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [4.0, 2.0]])
    t = np.linspace(0, 1, 100)
    pts = bezier_curve(control_points, t)
    assert np.all(pts[:, 0] >= control_points[:, 0].min() - 1e-9)
    assert np.all(pts[:, 0] <= control_points[:, 0].max() + 1e-9)
    assert np.all(pts[:, 1] >= control_points[:, 1].min() - 1e-9)
    assert np.all(pts[:, 1] <= control_points[:, 1].max() + 1e-9)


# --- Bernstein basis: the two facts the property proofs rest on ---------------------------


def test_bernstein_partition_of_unity():
    """sum_i B_i^n(t) = 1 for every t -- this is what makes the curve an affine (indeed convex)
    combination of its control points, and hence what gives affine invariance."""
    for n in (1, 2, 5, 12):
        basis = bernstein_basis(n, np.linspace(0, 1, 51))
        assert np.allclose(basis.sum(axis=1), 1.0)


def test_bernstein_non_negative():
    for n in (1, 2, 5, 12):
        basis = bernstein_basis(n, np.linspace(0, 1, 51))
        assert np.all(basis >= 0.0)


def test_bernstein_matches_de_casteljau():
    """The closed Bernstein form and De Casteljau's algorithm evaluate the same curve --
    the equivalence proved by induction in notes/01_Curves/De_Casteljau_Derivation.md."""
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [4.0, 2.0], [5.0, 1.0]])
    t = np.linspace(0, 1, 37)
    basis = bernstein_basis(len(control_points) - 1, t)
    assert np.allclose(basis @ control_points, bezier_curve(control_points, t))


def test_bernstein_global_support():
    """Every basis function is strictly positive on the open interval (0, 1): that is exactly
    why moving any one control point perturbs the whole curve (no local control)."""
    basis = bernstein_basis(4, np.linspace(0.01, 0.99, 50))
    assert np.all(basis > 0.0)


def test_bernstein_expanded_power_form():
    """The explicit expansions written out in notes/01_Curves/Bezier_Curves.md.

    Coefficients are listed low-order first, so [3, -6, 3] means 3t - 6t^2 + 3t^3. These are
    the numbers the note contrasts with the (non-negative, sum-to-one) factored form.
    """
    expected = {
        1: [[1, -1], [0, 1]],
        2: [[1, -2, 1], [0, 2, -2], [0, 0, 1]],
        3: [[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]],
        4: [
            [1, -4, 6, -4, 1],
            [0, 4, -12, 12, -4],
            [0, 0, 6, -12, 6],
            [0, 0, 0, 4, -4],
            [0, 0, 0, 0, 1],
        ],
    }
    for n, rows in expected.items():
        for i, coeffs in enumerate(rows):
            # comb(n, i) * (1 - t)^(n - i) * t^i, expanded.
            poly = (
                np.polynomial.Polynomial([1.0, -1.0]) ** (n - i)
                * np.polynomial.Polynomial([0.0, 1.0]) ** i
                * comb(n, i)
            )
            assert np.allclose(poly.coef, coeffs), (n, i, poly.coef)


def test_bernstein_peaks_at_i_over_n():
    """Each basis function is a single bump peaking at t = i/n -- the note quotes
    B_1^3(1/3) = 4/9 as the concrete case."""
    for n in (2, 3, 4, 7):
        t = np.linspace(0, 1, 20001)
        basis = bernstein_basis(n, t)
        for i in range(n + 1):
            assert abs(t[basis[:, i].argmax()] - i / n) < 1e-3
    assert bernstein_basis(3, 1 / 3)[0, 1] == pytest.approx(4 / 9)


def test_bernstein_symmetry():
    """B_i^n(t) = B_{n-i}^n(1-t): reversing the control points reverses the curve, nothing else."""
    t = np.linspace(0, 1, 101)
    for n in (2, 3, 4, 7):
        assert np.allclose(bernstein_basis(n, t), bernstein_basis(n, 1 - t)[:, ::-1])


def test_worked_cubic_example_from_the_notes():
    """The side-by-side evaluation in notes/01_Curves/Bezier_Curves.md, pinned exactly.

    Both routes must land on C(1/3) = (7/9, 2), and the note's intermediate numbers -- the
    basis values 8/27, 4/9, 2/9, 1/27 and each De Casteljau level -- must be the real ones.
    """
    control_points = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]])
    t = 1 / 3
    expected_point = np.array([7 / 9, 2.0])

    # Route 1: the closed Bernstein form.
    basis = bernstein_basis(3, t)[0]
    assert np.allclose(basis, [8 / 27, 4 / 9, 2 / 9, 1 / 27])
    assert np.allclose(basis @ control_points, expected_point)

    # Route 2: De Casteljau, including every intermediate level the note tabulates.
    levels = de_casteljau_triangle(control_points, t)
    assert np.allclose(levels[1], [[0.0, 1.0], [1.0, 3.0], [3.0, 2.0]])
    assert np.allclose(levels[2], [[1 / 3, 5 / 3], [5 / 3, 8 / 3]])
    assert np.allclose(levels[3], [expected_point])
    assert np.allclose(bezier_de_casteljau(control_points, t), expected_point)

    # And exactly, in rational arithmetic, so "7/9" is not a floating-point coincidence.
    q, s = Fraction(1, 3), Fraction(2, 3)
    exact = [tuple(Fraction(int(v)) for v in p) for p in control_points]
    while len(exact) > 1:
        exact = [tuple(s * a[k] + q * b[k] for k in range(2)) for a, b in zip(exact, exact[1:])]
    assert exact[0] == (Fraction(7, 9), Fraction(2))


def test_last_lerp_gives_the_tangent():
    """The by-product claimed in notes/01_Curves/Bezier_Curves.md: the direction of De
    Casteljau's final interpolation is the curve tangent, C'(t) = n * (P_1^{n-1} - P_0^{n-1}).
    Checked against a central difference."""
    rng = np.random.default_rng(7)
    for n in (2, 3, 5, 8):
        control_points = rng.uniform(-1.0, 1.0, size=(n + 1, 2))
        for t in (0.13, 0.5, 0.87):
            penultimate = np.asarray(de_casteljau_triangle(control_points, t)[n - 1])
            tangent = n * (penultimate[1] - penultimate[0])
            h = 1e-6
            finite_difference = (
                bezier_de_casteljau(control_points, t + h)
                - bezier_de_casteljau(control_points, t - h)
            ) / (2 * h)
            assert np.allclose(tangent, finite_difference, rtol=1e-6)


def test_closed_form_and_de_casteljau_agree_at_high_degree():
    """The measurement behind the corrected stability claim in the notes.

    The usual story is that direct Bernstein evaluation is numerically inferior because it
    forms huge binomials. Measured against exact rational arithmetic, it is not: on [0, 1]
    the basis is non-negative, so nothing cancels subtractively between terms and both routes
    stay at ~1e-15. The closed form's real failure mode is overflow, far past any usable degree.
    """
    rng = np.random.default_rng(0)
    for n in (10, 30, 50):
        control_points = rng.uniform(-1.0, 1.0, size=(n + 1, 2))
        for t, tq in ((0.3, Fraction(3, 10)), (0.7, Fraction(7, 10))):
            reference = np.array(
                [
                    float(
                        sum(
                            comb(n, i)
                            * (1 - tq) ** (n - i)
                            * tq**i
                            * Fraction(float(control_points[i][k]))
                            for i in range(n + 1)
                        )
                    )
                    for k in range(2)
                ]
            )
            de_casteljau_err = np.abs(bezier_de_casteljau(control_points, t) - reference).max()
            closed_form_err = np.abs(bernstein_basis(n, t) @ control_points - reference).max()
            assert de_casteljau_err < 1e-14
            assert closed_form_err < 1e-14

    # Where the closed form does break: comb(n, n // 2) leaves the float64 range at n = 1030.
    bernstein_basis(1029, 0.3)
    with pytest.raises(OverflowError):
        bernstein_basis(1030, 0.3)


# --- Subdivision -------------------------------------------------------------------------


def test_bezier_subdivide_reproduces_original():
    """Splitting at s yields two curves that, reparametrized, trace the original exactly."""
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [4.0, 2.0], [5.0, 1.0]])
    s = 0.35
    left, right = bezier_subdivide(control_points, s)
    u = np.linspace(0, 1, 50)
    assert np.allclose(bezier_curve(left, u), bezier_curve(control_points, u * s))
    assert np.allclose(bezier_curve(right, u), bezier_curve(control_points, s + u * (1 - s)))


def test_bezier_subdivide_halves_meet_on_curve():
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [4.0, 2.0]])
    s = 0.6
    left, right = bezier_subdivide(control_points, s)
    assert np.allclose(left[-1], right[0])
    assert np.allclose(left[-1], bezier_curve(control_points, s)[0])


# --- Rational Bezier: what polynomials provably cannot do ---------------------------------


def test_rational_quadratic_is_an_exact_circular_arc():
    """A polynomial Bezier can only approximate a circle; the rational quadratic *is* one,
    to machine precision. This is the claim script 04 demonstrates visually."""
    center, radius = np.array([0.3, -0.7]), 2.4
    cps, weights = rational_arc_quadratic(center, radius, theta0=0.2, theta1=0.2 + np.pi / 2)
    pts = rational_bezier(cps, weights, np.linspace(0, 1, 101))
    err = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
    assert err.max() < 1e-13, f"max radial error {err.max():.2e}"


def test_rational_bezier_reduces_to_polynomial_when_weights_equal():
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [4.0, 2.0]])
    t = np.linspace(0, 1, 25)
    unit = np.ones(len(control_points))
    assert np.allclose(rational_bezier(control_points, unit, t), bezier_curve(control_points, t))
    # ... and weights are scale-invariant: only their ratios matter.
    assert np.allclose(rational_bezier(control_points, 3.7 * unit, t),
                       bezier_curve(control_points, t))


# --- Affine invariance ---------------------------------------------------------------------


def test_bezier_affine_invariance():
    """Transform-then-evaluate == evaluate-then-transform, for any affine map. The proof turns
    on partition of unity; see notes/01_Curves/Bernstein_Basis_Properties.md."""
    control_points = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [4.0, 2.0]])
    theta = 0.7
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ \
        np.array([[1.0, 0.6], [0.0, 1.3]])  # rotation composed with a shear+scale
    b = np.array([-2.0, 5.0])
    t = np.linspace(0, 1, 50)

    evaluate_then_transform = bezier_curve(control_points, t) @ A.T + b
    transform_then_evaluate = bezier_curve(control_points @ A.T + b, t)
    assert np.allclose(evaluate_then_transform, transform_then_evaluate)