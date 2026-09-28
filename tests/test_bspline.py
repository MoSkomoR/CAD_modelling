import numpy as np
import pytest

from cadkernel.geometry.bspline import (
    bspline_basis,
    bspline_basis_levels,
    bspline_curve,
    bspline_derivative,
    bspline_to_bezier,
    clamped_knots,
    de_boor,
    de_boor_stages,
    elevate_degree,
    find_span,
    insert_knot,
    interpolate_points,
)
from cadkernel.geometry.curves import (
    bernstein_basis,
    bezier_curve,
    bezier_de_casteljau,
    bezier_elevate,
)
from cadkernel.viz.plotting import convex_hull_2d

T = np.linspace(0.0, 1.0, 1001)


def sample_curve(n_ctrl=8, p=3, interior=None, seed=0):
    points = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n_ctrl, 2))
    return points, clamped_knots(n_ctrl, p, interior), p


# --- The basis --------------------------------------------------------------------------------


def test_bspline_basis_is_non_negative_and_a_partition_of_unity():
    """The two facts every Bezier property rested on survive the move to piecewise: with uneven
    and repeated knots, the N_{i,p} are still convex weights at every t in the domain."""
    for p, interior in ((1, [0.3, 0.6]), (2, [0.2, 0.2, 0.7]), (3, [0.1, 0.5, 0.55, 0.9])):
        knots = clamped_knots(p + 1 + len(interior), p, interior)
        basis = bspline_basis(knots, p, T)
        assert np.all(basis >= 0.0)
        assert np.allclose(basis.sum(axis=1), 1.0)


def test_bspline_basis_has_local_support():
    """N_{i,p} is exactly zero outside [u_i, u_{i+p+1}) -- it is built from p+1 box functions and
    cannot reach further. This is the property Bezier lacked."""
    knots = clamped_knots(9, 3, [0.15, 0.3, 0.5, 0.6, 0.8])
    basis = bspline_basis(knots, 3, T)
    for i in range(basis.shape[1]):
        outside = (T < knots[i]) | (T > knots[i + 4])
        assert np.all(basis[outside, i] == 0.0)
        inside = (T > knots[i]) & (T < knots[i + 4])
        assert np.all(basis[inside, i] > 0.0)


def test_cox_de_boor_levels_start_as_boxes():
    """Degree 0 is one indicator function per knot span; each level is one span wider."""
    knots = clamped_knots(6, 2, [0.25, 0.5, 0.75])
    levels = bspline_basis_levels(knots, 2, T)
    assert [level.shape[1] for level in levels] == [len(knots) - 1, len(knots) - 2, len(knots) - 3]
    assert set(np.unique(levels[0])) <= {0.0, 1.0}
    assert np.allclose(levels[0].sum(axis=1), 1.0)


def test_find_span_covers_the_closed_domain():
    knots = clamped_knots(6, 3, [0.4, 0.4])
    assert find_span(knots, 3, 0.0) == 3
    assert find_span(knots, 3, 0.4) == 5  # the empty span [0.4, 0.4) is skipped
    assert find_span(knots, 3, 1.0) == 5  # right end belongs to the last non-empty span


# --- Evaluation, and the reduction to Bezier ---------------------------------------------------


def test_de_boor_matches_the_basis_sum():
    points, knots, p = sample_curve(interior=[0.2, 0.5, 0.55, 0.8], n_ctrl=8)
    via_basis = bspline_curve(points, knots, p, T)
    via_de_boor = np.array([de_boor(points, knots, p, t) for t in T])
    assert np.allclose(via_basis, via_de_boor, atol=1e-14)


def test_bezier_is_a_special_case_of_bspline():
    """No interior knots: the B-spline basis *is* the Bernstein basis, and de Boor *is*
    De Casteljau. A B-spline is a Bezier curve that has been allowed to break into pieces."""
    for p in (1, 2, 3, 5):
        knots = clamped_knots(p + 1, p)
        assert np.allclose(bspline_basis(knots, p, T), bernstein_basis(p, T), atol=1e-14)
        points = np.random.default_rng(p).uniform(-1, 1, size=(p + 1, 2))
        for t in (0.0, 0.3, 0.77, 1.0):
            assert np.allclose(de_boor(points, knots, p, t), bezier_de_casteljau(points, t))


def test_de_boor_touches_only_p_plus_one_points():
    points, knots, p = sample_curve(n_ctrl=10)
    stages = de_boor_stages(points, knots, p, 0.52)
    assert [len(level) for level in stages["levels"]] == [4, 3, 2, 1]
    k = stages["span"]
    assert np.array_equal(stages["levels"][0], points[k - p : k + 1])


def test_clamped_bspline_interpolates_its_end_points():
    points, knots, p = sample_curve()
    assert np.allclose(bspline_curve(points, knots, p, [0.0])[0], points[0])
    assert np.allclose(bspline_curve(points, knots, p, [1.0])[0], points[-1])


def test_bspline_affine_invariance():
    points, knots, p = sample_curve(seed=3)
    rotate = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    shear = np.array([[1.0, 0.4], [0.0, 1.0]])
    matrix, offset = shear @ rotate, np.array([2.0, -1.0])
    assert np.allclose(
        bspline_curve(points @ matrix.T + offset, knots, p, T),
        bspline_curve(points, knots, p, T) @ matrix.T + offset,
    )


def test_strong_convex_hull():
    """On span k the curve is a convex combination of only P_{k-p}..P_k, so it lies in *their*
    hull -- a much tighter bound than the hull of the whole polygon."""
    points, knots, p = sample_curve(n_ctrl=9, seed=4)
    for t in np.linspace(0, 1, 97):
        k = int(find_span(knots, p, t))
        hull = convex_hull_2d(points[k - p : k + 1])
        point = bspline_curve(points, knots, p, [t])[0]
        edges = np.roll(hull, -1, axis=0) - hull
        to_point = point - hull
        cross = edges[:, 0] * to_point[:, 1] - edges[:, 1] * to_point[:, 0]
        assert np.all(cross >= -1e-12)


# --- Local control, measured against Bezier ----------------------------------------------------


def test_moving_a_control_point_changes_only_p_plus_one_spans():
    """The measured version of local support, on the zig-zag curve used in the notes: moving P_5
    of a 10-point cubic changes the curve on exactly [u_5, u_9) = 4 of 7 spans, and by *exactly*
    0.0 elsewhere. The degree-9 Bezier on the same points moves everywhere."""
    points = np.c_[np.arange(10.0), np.tile([0.0, 2.0, -1.0, 2.0, -1.0], 2)]
    knots, p = clamped_knots(10, 3), 3
    t = np.linspace(0, 1, 2001)
    moved = points.copy()
    moved[5, 1] += 1.0

    change = np.linalg.norm(bspline_curve(moved, knots, p, t) - bspline_curve(points, knots, p, t), axis=1)
    outside = (t < knots[5]) | (t >= knots[9])
    assert np.all(change[outside] == 0.0)
    assert np.all(change[~outside & (t > knots[5])] > 0.0)
    assert np.isclose(change.max(), 0.667, atol=1e-3)

    bezier_change = np.linalg.norm(bezier_curve(moved, t) - bezier_curve(points, t), axis=1)
    assert np.all(bezier_change[1:-1] > 0.0)
    assert np.isclose(bezier_change.max(), 0.260, atol=1e-3)  # global *and* weak


# --- Continuity is set by knot multiplicity -----------------------------------------------------


def derivative_jumps(points, knots, p, at, orders, eps=1e-9):
    jumps = []
    for _ in range(orders):
        left = bspline_curve(points, knots, p, [at - eps])[0]
        right = bspline_curve(points, knots, p, [at + eps])[0]
        jumps.append(np.abs(left - right).max())
        points, knots, p = bspline_derivative(points, knots, p)
    return jumps


def test_continuity_is_p_minus_multiplicity():
    """A cubic is C^2 across a simple knot, C^1 across a double one, C^0 across a triple one.
    Continuity is not a constraint to maintain -- it is read off the knot vector."""
    for multiplicity in (1, 2, 3):
        interior = [0.25] + [0.5] * multiplicity + [0.75]
        points, knots, p = sample_curve(n_ctrl=4 + len(interior), interior=interior, seed=1)
        jumps = derivative_jumps(points, knots, p, 0.5, orders=4)
        smooth_orders = 3 - multiplicity  # C^(p-k): derivatives 0..p-k are continuous
        assert all(jump < 1e-6 for jump in jumps[: smooth_orders + 1])
        assert jumps[smooth_orders + 1] > 1.0


def test_bspline_derivative_matches_finite_differences():
    points, knots, p = sample_curve(interior=[0.3, 0.3, 0.7], n_ctrl=7, seed=5)
    d_points, d_knots, d_p = bspline_derivative(points, knots, p)
    h = 1e-6
    for t in (0.1, 0.45, 0.8):
        fd = (bspline_curve(points, knots, p, [t + h]) - bspline_curve(points, knots, p, [t - h])) / (2 * h)
        assert np.allclose(bspline_curve(d_points, d_knots, d_p, [t]), fd, atol=1e-7)


# --- Knot insertion, Bezier decomposition, degree elevation --------------------------------------


def test_knot_insertion_leaves_the_curve_unchanged():
    points, knots, p = sample_curve(interior=[0.2, 0.5, 0.8], n_ctrl=7, seed=6)
    before = bspline_curve(points, knots, p, T)
    for value in (0.37, 0.5, 0.5):  # a new knot, then raising an existing one twice
        points, knots = insert_knot(points, knots, p, value)
        assert np.allclose(bspline_curve(points, knots, p, T), before, atol=1e-14)
    with pytest.raises(ValueError):
        insert_knot(points, knots, p, 0.5)  # multiplicity already p
    with pytest.raises(ValueError):
        insert_knot(points, knots, p, 1.0)


def test_bspline_to_bezier_segments_reproduce_each_span():
    points, knots, p = sample_curve(interior=[0.2, 0.5, 0.55, 0.8], n_ctrl=8, seed=7)
    segments = bspline_to_bezier(points, knots, p)
    breaks = np.unique(knots)
    assert len(segments) == len(breaks) - 1
    for segment, a, b in zip(segments, breaks[:-1], breaks[1:]):
        t = np.linspace(a, b, 21)
        assert np.allclose(bezier_curve(segment, (t - a) / (b - a)), bspline_curve(points, knots, p, t))
    for left, right in zip(segments[:-1], segments[1:]):
        assert np.allclose(left[-1], right[0])


def test_bezier_degree_elevation_keeps_the_curve():
    points = np.random.default_rng(8).uniform(-1, 1, size=(4, 2))
    elevated = bezier_elevate(points)
    assert elevated.shape == (5, 2)
    assert np.allclose(bezier_curve(elevated, T), bezier_curve(points, T))


def test_bspline_degree_elevation_keeps_the_curve():
    points, knots, p = sample_curve(interior=[0.3, 0.6], n_ctrl=6, seed=9)
    new_points, new_knots, q = elevate_degree(points, knots, p)
    assert q == p + 1
    assert np.allclose(bspline_curve(new_points, new_knots, q, T), bspline_curve(points, knots, p, T))
    # Without knot removal the breakpoints end up with multiplicity p+1 (redundant, but correct).
    assert np.sum(new_knots == 0.3) == q


# --- Interpolation -----------------------------------------------------------------------------


def test_interpolation_passes_through_every_point():
    data = np.random.default_rng(10).uniform(0, 1, size=(9, 2))
    for method in ("uniform", "chord", "centripetal"):
        for p in (2, 3):
            control, knots, params = interpolate_points(data, p, method)
            assert np.allclose(bspline_curve(control, knots, p, params), data, atol=1e-13)


def self_intersects(polyline):
    a, b = polyline[:-1], polyline[1:]
    for i in range(len(a) - 2):
        r, s = b[i] - a[i], b[i + 2 :] - a[i + 2 :]
        q = a[i + 2 :] - a[i]
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / denom
            v = (q[:, 0] * r[1] - q[:, 1] * r[0]) / denom
        if np.any((denom != 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)):
            return True
    return False


def overshoot(curve, data):
    """Largest distance from a curve point to the data polyline."""
    best = np.full(len(curve), np.inf)
    for a, b in zip(data[:-1], data[1:]):
        ab = b - a
        w = np.clip(((curve - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(curve - (a + w[:, None] * ab), axis=1))
    return best.max()


PARAMETRIZATION_DATA = {
    "sharp turn": [[0, 0], [1, 0], [2, 0], [2.1, 0.1], [2.1, 1], [2.1, 2], [2.1, 3]],
    "uneven": [[0, 0], [0.1, 0.3], [0.2, 0.1], [3, 1], [3.2, 0.8], [3.1, 1.2], [6, 0]],
    "cluster": [[0, 0], [4, 0], [4.1, 0.05], [4.15, 0.2], [4.1, 0.35], [4, 0.4], [0, 0.4]],
}


def test_parametrization_comparison():
    """Measured on three awkward data sets (cubic, 4001 samples). Only centripetal is clean on
    all three; chord length, the usual default, overshoots worst on two; uniform loops once."""
    t = np.linspace(0, 1, 4001)
    results = {}
    for name, data in PARAMETRIZATION_DATA.items():
        data = np.asarray(data, dtype=float)
        for method in ("uniform", "chord", "centripetal"):
            control, knots, _ = interpolate_points(data, 3, method)
            curve = bspline_curve(control, knots, 3, t)
            results[name, method] = (self_intersects(curve), overshoot(curve, data))

    for name in PARAMETRIZATION_DATA:
        loops, dist = results[name, "centripetal"]
        assert not loops and dist < 0.75
    assert results["uneven", "uniform"][0]
    assert results["cluster", "chord"][0]
    assert results["uneven", "chord"][1] > 5.0 and results["cluster", "chord"][1] > 5.0


def test_piecewise_interpolation_escapes_runge():
    """Runge's function sampled uniformly on [-1, 1]: one interpolating polynomial through all
    the points gets *worse* as points are added; the cubic B-spline interpolant gets better."""
    runge = lambda x: 1.0 / (1.0 + 25.0 * x**2)
    t = np.linspace(0, 1, 4001)
    spline_errors, polynomial_errors = [], []
    for count in (7, 13, 21):
        x = np.linspace(-1, 1, count)
        data = np.c_[x, runge(x)]
        control, knots, params = interpolate_points(data, 3, "uniform")
        curve = bspline_curve(control, knots, 3, t)
        spline_errors.append(np.abs(curve[:, 1] - runge(curve[:, 0])).max())
        bezier = np.linalg.solve(bernstein_basis(count - 1, params), data)
        curve = bezier_curve(bezier, t)
        polynomial_errors.append(np.abs(curve[:, 1] - runge(curve[:, 0])).max())
    assert spline_errors[0] > spline_errors[1] > spline_errors[2]
    assert polynomial_errors[0] < polynomial_errors[1] < polynomial_errors[2]
    assert spline_errors[1] < 0.01 and polynomial_errors[1] > 3.0  # 13 points: 0.0069 vs 3.66


def test_single_polynomial_strays_further_than_the_spline():
    """Same data ("uneven"), same centripetal parameters: one degree-6 polynomial strays 4.1
    units from the data polygon, the cubic B-spline 0.72. Quoted in script 07."""
    data = np.asarray(PARAMETRIZATION_DATA["uneven"], dtype=float)
    t = np.linspace(0, 1, 1501)
    control, knots, params = interpolate_points(data, 3, "centripetal")
    spline = overshoot(bspline_curve(control, knots, 3, t), data)
    bezier = np.linalg.solve(bernstein_basis(len(data) - 1, params), data)
    polynomial = overshoot(bezier_curve(bezier, t), data)
    assert np.isclose(spline, 0.72, atol=0.01)
    assert np.isclose(polynomial, 4.14, atol=0.01)
