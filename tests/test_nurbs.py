import numpy as np
import pytest

from cadkernel.geometry.bspline import bspline_curve, clamped_knots, insert_knot
from cadkernel.geometry.curves import bezier_curve
from cadkernel.geometry.nurbs import (
    nurbs_arc,
    nurbs_basis,
    nurbs_circle,
    nurbs_curve,
    nurbs_derivative,
    nurbs_insert_knot,
    nurbs_projective_map,
    projective_map,
    rational_arc_quadratic,
    rational_bezier,
)

T = np.linspace(0.0, 1.0, 2001)


def sample_nurbs(n_ctrl=7, p=3, interior=(0.2, 0.5, 0.55), seed=3):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n_ctrl, 2))
    weights = rng.uniform(0.3, 3.0, size=n_ctrl)
    return points, weights, clamped_knots(n_ctrl, p, list(interior)), p


def sample_rational_bezier(n_ctrl=4, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n_ctrl, 2)), rng.uniform(0.3, 3.0, size=n_ctrl)


# --- Rational Bezier: what polynomials provably cannot do ---------------------------------------


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


# --- NURBS reduces to what came before ----------------------------------------------------------


def test_nurbs_with_unit_weights_is_the_bspline():
    points, _, knots, p = sample_nurbs()
    gap = np.abs(nurbs_curve(points, np.ones(len(points)), knots, p, T)
                 - bspline_curve(points, knots, p, T)).max()
    assert gap < 1e-15


def test_rational_bezier_is_nurbs_on_bezier_knots():
    points, weights = sample_rational_bezier()
    gap = np.abs(nurbs_curve(points, weights, clamped_knots(4, 3), 3, T)
                 - rational_bezier(points, weights, T)).max()
    assert gap < 1e-15


def test_rational_basis_is_a_partition_of_unity_with_local_support():
    """sum_i R_{i,p} = 1 for any weights; non-negative for positive weights; and R_{i,p} is zero
    outside [u_i, u_{i+p+1}), exactly like N_{i,p}."""
    _, weights, knots, p = sample_nurbs()
    R = nurbs_basis(knots, p, weights, T)
    assert np.abs(R.sum(axis=1) - 1).max() < 1e-15
    assert R.min() >= 0.0
    for i in range(R.shape[1]):
        outside = (T < knots[i]) | (T > knots[i + p + 1])
        assert np.all(R[outside, i] == 0.0)


def test_changing_one_weight_is_local():
    points, weights, knots, p = sample_nurbs()
    i = 2
    heavier = weights.copy()
    heavier[i] *= 5.0
    moved = np.abs(nurbs_curve(points, heavier, knots, p, T)
                   - nurbs_curve(points, weights, knots, p, T)).max(axis=1)
    outside = (T < knots[i]) | (T > knots[i + p + 1])
    assert np.all(moved[outside] == 0.0)
    assert moved[~outside].max() > 0.1


# --- Exact conics -------------------------------------------------------------------------------


def test_full_circle_is_exact():
    center, radius = np.array([0.3, -0.2]), 1.7
    points, weights, knots, p = nurbs_circle(center, radius)
    curve = nurbs_curve(points, weights, knots, p, T)
    err = np.abs(np.linalg.norm(curve - center, axis=1) - radius).max()
    assert err < 1e-15, err
    assert np.allclose(curve[0], curve[-1])


def test_circle_is_c1_at_its_double_knots():
    """The doubled knots only *promise* C^0 (multiplicity 2 on degree 2). Measured: the
    derivative is continuous across them anyway -- the symmetry of the four quarters matches
    the end tangents in direction and length."""
    points, weights, knots, p = nurbs_circle(np.zeros(2), 1.0)
    for k in (0.25, 0.5, 0.75):
        left, right = nurbs_derivative(points, weights, knots, p, [k - 1e-9, k + 1e-9])
        assert np.abs(left - right).max() < 1e-7


def test_circle_is_not_arc_length_parametrized():
    """Exact geometry, inexact speed: |C'(t)| varies by a factor 4 - 2 sqrt(2) ~ 1.17 around the
    circle, and the angle drifts from 2 pi t by almost a degree."""
    points, weights, knots, p = nurbs_circle(np.zeros(2), 1.0)
    speed = np.linalg.norm(nurbs_derivative(points, weights, knots, p, T[:-1]), axis=1)
    assert np.isclose(speed.max() / speed.min(), 4 - 2 * np.sqrt(2), rtol=1e-6)
    curve = nurbs_curve(points, weights, knots, p, T)
    angle = np.unwrap(np.arctan2(curve[:, 1], curve[:, 0]))
    drift = np.degrees(np.abs(angle - 2 * np.pi * T).max())
    assert 0.9 < drift < 0.95, drift


def implicit_conic_discriminant(points):
    """Fit A x^2 + B xy + C y^2 + D x + E y + F = 0 through the points (null space of the design
    matrix) and return B^2 - 4AC: < 0 ellipse, = 0 parabola, > 0 hyperbola."""
    x, y = points.T
    design = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]
    _, singular, vt = np.linalg.svd(design)
    assert singular[-1] / singular[0] < 1e-14  # the points really do lie on one conic
    A, B, C = vt[-1][:3] / np.abs(vt[-1]).max()
    return B * B - 4 * A * C


def test_middle_weight_selects_the_conic_type():
    """Weights (1, w, 1) on a quadratic: w < 1 ellipse, w = 1 parabola, w > 1 hyperbola."""
    control = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    disc = {w: implicit_conic_discriminant(rational_bezier(control, [1, w, 1], T[::5]))
            for w in (0.3, 0.7, 1.0, 1.5, 4.0)}
    assert disc[0.3] < -0.1 and disc[0.7] < -0.1
    assert abs(disc[1.0]) < 1e-12
    assert disc[1.5] > 0.1 and disc[4.0] > 0.1


def test_large_weight_pulls_the_curve_to_its_point():
    """At t = 1/2 the quadratic sits at (M + w P_1) / (1 + w), M the chord midpoint: the distance
    to P_1 is |P_1 - M| / (1 + w) -- here 2 / (1 + w)."""
    control = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    for w in (1.0, 10.0, 100.0, 1000.0):
        distance = np.linalg.norm(rational_bezier(control, [1, w, 1], T) - control[1], axis=1)
        assert np.isclose(distance.min(), 2 / (1 + w), rtol=1e-12)


# --- Projective invariance: the genuinely new property ------------------------------------------

PERSPECTIVE = np.array([[1.0, 0.2, 0.1], [0.1, 0.9, -0.2], [0.3, 0.4, 1.0]])


def test_nurbs_is_projectively_invariant():
    """Map the homogeneous control points by the 3x3 matrix and the curve is mapped exactly.
    Moving only the control points (keeping the weights) does not work -- for NURBS, nor for a
    plain B-spline, which is why projective invariance is new with weights."""
    points, weights, knots, p = sample_nurbs()
    image = projective_map(PERSPECTIVE, nurbs_curve(points, weights, knots, p, T))
    new_points, new_weights = nurbs_projective_map(PERSPECTIVE, points, weights)
    exact = np.abs(nurbs_curve(new_points, new_weights, knots, p, T) - image).max()
    assert exact < 1e-14
    assert not np.allclose(new_weights, weights)

    naive = np.abs(nurbs_curve(projective_map(PERSPECTIVE, points), weights, knots, p, T)
                   - image).max()
    bspline_image = projective_map(PERSPECTIVE, bspline_curve(points, knots, p, T))
    polynomial = np.abs(bspline_curve(projective_map(PERSPECTIVE, points), knots, p, T)
                        - bspline_image).max()
    assert naive > 0.4 and polynomial > 0.4


def test_affine_maps_leave_the_weights_alone():
    points, weights, _, _ = sample_nurbs()
    affine = np.array([[0.8, -0.6, 2.0], [0.6, 0.8, -1.0], [0.0, 0.0, 1.0]])
    new_points, new_weights = nurbs_projective_map(affine, points, weights)
    assert np.allclose(new_weights, weights)
    assert np.allclose(new_points, projective_map(affine, points))


def test_perspective_image_of_a_circle_is_an_exact_ellipse():
    points, weights, knots, p = nurbs_circle(np.zeros(2), 1.0)
    new_points, new_weights = nurbs_projective_map(PERSPECTIVE, points, weights)
    image = nurbs_curve(new_points, new_weights, knots, p, T[::7])
    assert implicit_conic_discriminant(image) < -0.1


# --- Weights are not free: reparametrization ----------------------------------------------------


def test_scaling_weights_geometrically_only_reparametrizes():
    """w_i -> lambda^i w_i gives the same curve, traversed at a different speed: substitute
    t = lambda s / ((1 - s) + lambda s). One degree of freedom in the weights is pure
    parametrization."""
    points, weights = sample_rational_bezier()
    for lam in (0.5, 3.0):
        scaled = weights * lam ** np.arange(4)
        moved = rational_bezier(points, scaled, T)
        assert np.abs(moved - rational_bezier(points, weights, T)).max() > 0.1
        t = lam * T / ((1 - T) + lam * T)
        assert np.abs(moved - rational_bezier(points, weights, t)).max() < 1e-14


# --- The machinery that carries over ------------------------------------------------------------


def test_knot_insertion_must_happen_in_homogeneous_space():
    points, weights, knots, p = sample_nurbs()
    before = nurbs_curve(points, weights, knots, p, T)
    new_points, new_weights, new_knots = nurbs_insert_knot(points, weights, knots, p, 0.33)
    assert np.abs(nurbs_curve(new_points, new_weights, new_knots, p, T) - before).max() < 1e-14
    # Inserting into the points and the weights separately is wrong.
    naive_points, naive_knots = insert_knot(points, knots, p, 0.33)
    naive_weights, _ = insert_knot(weights[:, None], knots, p, 0.33)
    naive = nurbs_curve(naive_points, naive_weights[:, 0], naive_knots, p, T)
    assert np.abs(naive - before).max() > 0.1


def test_nurbs_derivative_matches_finite_differences():
    points, weights, knots, p = sample_nurbs()
    t = np.array([0.1, 0.37, 0.52, 0.8])
    h = 1e-6
    fd = (nurbs_curve(points, weights, knots, p, t + h)
          - nurbs_curve(points, weights, knots, p, t - h)) / (2 * h)
    assert np.abs(nurbs_derivative(points, weights, knots, p, t) - fd).max() < 1e-8


def test_derivative_degree_inflates():
    """The derivative of a degree-n rational Bezier is a quotient of polynomials of degree
    2n - 2 over 2n -- against n - 1 for a polynomial curve. Measured by fitting the numerator
    w(t)^2 C'(t) with polynomials of rising degree."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 60)
    for n in (2, 3, 4, 5):
        points = rng.uniform(-1, 1, size=(n + 1, 2))
        weights = rng.uniform(0.5, 2.0, size=n + 1)
        knots = clamped_knots(n + 1, n)
        w = bspline_curve(weights[:, None], knots, n, t)[:, 0]
        numerator = nurbs_derivative(points, weights, knots, n, t) * w[:, None] ** 2

        def fits(degree):
            coeffs = np.polynomial.polynomial.polyfit(t, numerator, degree)
            return np.abs(np.polynomial.polynomial.polyval(t, coeffs).T - numerator).max() < 1e-9

        assert fits(2 * n - 2) and not fits(2 * n - 3)


# --- What non-positive weights break ------------------------------------------------------------


def test_negative_middle_weight_gives_the_complementary_arc():
    """Flip the sign of the middle weight of a quarter-circle arc: still exactly on the circle,
    but now it traces the other three quarters -- far outside the control triangle. Its
    midpoint is the antipode of the arc's midpoint (45 degrees -> 225 degrees)."""
    control, weights = rational_arc_quadratic(np.zeros(2), 1.0, 0.0, np.pi / 2)
    flipped = rational_bezier(control, weights * [1, -1, 1], T)
    assert np.abs(np.linalg.norm(flipped, axis=1) - 1).max() < 1e-15
    angle = np.degrees(np.arctan2(flipped[1:-1, 1], flipped[1:-1, 0])) % 360
    assert angle.min() > 90.0
    assert np.allclose(rational_bezier(control, weights * [1, -1, 1], [0.5])[0], -np.sqrt(0.5) * np.ones(2))


def test_weight_minus_one_puts_a_pole_on_the_curve():
    """Weights (1, -1, 1): the denominator (1 - 2t)^2 vanishes at t = 1/2 and the curve runs off
    to infinity; any weight in (-1, 1) keeps it positive."""
    control = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    near_pole = rational_bezier(control, [1, -1, 1], [0.5 - 1e-4])[0]
    assert np.linalg.norm(near_pole) > 1e4
    denominator = (1 - T) ** 2 - 1.8 * T * (1 - T) + T**2  # w = -0.9
    assert denominator.min() > 0.04


def test_negative_weight_leaves_the_convex_hull():
    control = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    curve = rational_bezier(control, [1, -0.5, 1], T)
    assert curve[:, 1].min() == pytest.approx(-2.0)  # the triangle has y >= 0
    # Zero weight: the middle point is switched off and the curve is the chord.
    assert np.abs(rational_bezier(control, [1, 0, 1], T)[:, 1]).max() == 0.0


def test_arc_of_a_full_turn_is_the_circle():
    """nurbs_arc cuts the sweep into pieces of at most pi/2; a full turn gives nurbs_circle's four
    quarters, up to rounding of cos and sin."""
    arc = nurbs_arc(np.zeros(2), 1.0, 2 * np.pi)
    circle = nurbs_circle(np.zeros(2), 1.0)
    for a, b in zip(arc, circle):
        assert np.allclose(a, b, atol=1e-15)


def test_arcs_of_any_sweep_are_exact():
    """Every point is at the radius, the end is at the sweep angle, and a sweep of up to k
    quarter turns uses k pieces (2k + 1 control points)."""
    t = np.linspace(0, 1, 201)
    for sweep, pieces in [(0.3, 1), (np.pi / 2, 1), (2.0, 2), (np.pi, 2), (5.0, 4)]:
        points, weights, knots, p = nurbs_arc(np.array([0.5, -1.0]), 2.0, sweep)
        assert len(points) == 2 * pieces + 1
        c = nurbs_curve(points, weights, knots, p, t)
        assert np.allclose(np.linalg.norm(c - [0.5, -1.0], axis=1), 2.0, atol=1e-14)
        assert np.allclose(c[-1], [0.5 + 2 * np.cos(sweep), -1.0 + 2 * np.sin(sweep)])
