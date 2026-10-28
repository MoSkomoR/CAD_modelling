import numpy as np

from cadkernel.geometry.bspline import bspline_curve, clamped_knots
from cadkernel.geometry.nurbs_surfaces import (
    bspline_isocurve,
    bspline_surface,
    bspline_surface_point,
    nurbs_cone,
    nurbs_cylinder,
    nurbs_sphere,
    nurbs_surface,
    nurbs_surface_basis,
    nurbs_surface_insert_knot,
    nurbs_surface_normal,
    nurbs_surface_partials,
    nurbs_torus,
    sphere_profile,
    surface_of_revolution,
)
from cadkernel.geometry.surfaces import bezier_surface

U = np.linspace(0.0, 1.0, 101)
V = np.linspace(0.0, 1.0, 101)


def sample_surface(seed=0):
    """A bicubic-by-quadratic NURBS on a 6 x 5 net with interior knots in both directions."""
    rng = np.random.default_rng(seed)
    net = rng.uniform(-1.0, 1.0, size=(6, 5, 3))
    weights = rng.uniform(0.5, 2.0, size=(6, 5))
    knots_u = clamped_knots(6, 3, [0.3, 0.6])
    knots_v = clamped_knots(5, 2, [0.4, 0.5])
    return net, weights, knots_u, knots_v, 3, 2


def evaluate(surface, u=U, v=V):
    net, weights, knots_u, knots_v, p, q = surface
    return nurbs_surface(net, weights, knots_u, knots_v, p, q, u, v)


# --- The tensor product: a curve of curves ------------------------------------------------------


def test_two_passes_match_the_grid_in_either_order():
    """Columns at u give the v-isocurve's polygon; rows at v give the u-isocurve's. Both routes,
    and the closed tensor form, give the same point."""
    net, _, knots_u, knots_v, p, q = sample_surface()
    grid = bspline_surface(net, knots_u, knots_v, p, q, U, V)
    worst = 0.0
    for a in range(0, len(U), 10):
        for b in range(0, len(V), 10):
            u_first = bspline_surface_point(net, knots_u, knots_v, p, q, U[a], V[b])
            v_polygon = bspline_isocurve(net, knots_v, q, v=V[b])
            v_first = bspline_curve(v_polygon, knots_u, p, [U[a]])[0]
            worst = max(worst, np.abs(u_first - grid[a, b]).max(), np.abs(v_first - grid[a, b]).max())
    assert worst < 1e-14


def test_bspline_surface_on_bezier_knots_is_the_bezier_surface():
    net = np.random.default_rng(1).uniform(-1, 1, size=(4, 3, 3))
    surface = bspline_surface(net, clamped_knots(4, 3), clamped_knots(3, 2), 3, 2, U, V)
    assert np.abs(surface - bezier_surface(net, U, V)).max() < 1e-14


def test_unit_weights_give_the_bspline_surface():
    net, _, knots_u, knots_v, p, q = sample_surface()
    rational = nurbs_surface(net, np.ones(net.shape[:2]), knots_u, knots_v, p, q, U, V)
    assert np.abs(rational - bspline_surface(net, knots_u, knots_v, p, q, U, V)).max() < 1e-14


def test_rational_tensor_basis_is_a_partition_of_unity():
    _, weights, knots_u, knots_v, p, q = sample_surface()
    sums = [nurbs_surface_basis(weights, knots_u, knots_v, p, q, u, v).sum()
            for u in U[::7] for v in V[::7]]
    assert np.abs(np.array(sums) - 1).max() < 1e-14


def test_moving_one_control_point_is_local_in_both_directions():
    """P_ij only acts on [u_i, u_{i+p+1}) x [v_j, v_{j+q+1}): outside that rectangle the
    surface does not move at all (exactly 0), inside it does."""
    net, weights, knots_u, knots_v, p, q = sample_surface()
    i, j = 2, 1
    moved = net.copy()
    moved[i, j] += [0.0, 0.0, 1.0]
    change = np.abs(evaluate((moved, weights, knots_u, knots_v, p, q))
                    - evaluate((net, weights, knots_u, knots_v, p, q))).max(axis=-1)
    inside = (((U >= knots_u[i]) & (U < knots_u[i + p + 1]))[:, None]
              & ((V >= knots_v[j]) & (V < knots_v[j + q + 1]))[None, :])
    assert change[~inside].max() == 0.0
    assert change[inside].max() > 0.1


def test_knot_insertion_in_either_direction_keeps_the_surface():
    net, weights, knots_u, knots_v, p, q = sample_surface()
    before = evaluate((net, weights, knots_u, knots_v, p, q))
    new_net, new_w, new_u = nurbs_surface_insert_knot(net, weights, knots_u, p, 0.37, "u")
    assert new_net.shape == (7, 5, 3)
    assert np.abs(evaluate((new_net, new_w, new_u, knots_v, p, q)) - before).max() < 1e-14
    new_net, new_w, new_v = nurbs_surface_insert_knot(net, weights, knots_v, q, 0.37, "v")
    assert new_net.shape == (6, 6, 3)
    assert np.abs(evaluate((new_net, new_w, knots_u, new_v, p, q)) - before).max() < 1e-14


def test_partials_match_finite_differences():
    surface = sample_surface()
    u = np.array([0.13, 0.52, 0.77])
    v = np.array([0.2, 0.45, 0.9])
    h = 1e-6
    s_u, s_v = nurbs_surface_partials(*surface, u, v)
    fd_u = (evaluate(surface, u + h, v) - evaluate(surface, u - h, v)) / (2 * h)
    fd_v = (evaluate(surface, u, v + h) - evaluate(surface, u, v - h)) / (2 * h)
    assert np.abs(s_u - fd_u).max() < 1e-8
    assert np.abs(s_v - fd_v).max() < 1e-8


# --- Surfaces of revolution -----------------------------------------------------------------------


def test_the_four_primitives_satisfy_their_implicit_equations():
    sphere = evaluate(nurbs_sphere(2.0))
    assert np.abs(np.linalg.norm(sphere, axis=-1) - 2.0).max() < 1e-14

    cylinder = evaluate(nurbs_cylinder(1.5, 3.0))
    assert np.abs(np.hypot(cylinder[..., 0], cylinder[..., 1]) - 1.5).max() < 1e-14

    cone = evaluate(nurbs_cone(1.5, 3.0))
    rho = np.hypot(cone[..., 0], cone[..., 1])
    assert np.abs(rho - 1.5 * (1 - cone[..., 2] / 3.0)).max() < 1e-14

    torus = evaluate(nurbs_torus(3.0, 1.0))
    rho = np.hypot(torus[..., 0], torus[..., 1])
    assert np.abs((rho - 3.0) ** 2 + torus[..., 2] ** 2 - 1.0).max() < 1e-13


def test_revolution_factors_into_profile_times_circle():
    """S(u, v) = (r(u) cos theta(v), r(u) sin theta(v), z(u)): a u = const curve (a parallel)
    has constant height and constant distance from the axis; a v = const curve (a meridian)
    stays in one plane through the axis."""
    sphere = evaluate(nurbs_sphere(2.0))
    parallel = sphere[30]
    assert np.ptp(parallel[:, 2]) < 1e-14
    assert np.ptp(np.hypot(parallel[:, 0], parallel[:, 1])) < 1e-14
    meridian = sphere[1:-1, 37]
    assert np.ptp(np.arctan2(meridian[:, 1], meridian[:, 0])) < 1e-14


def test_partial_revolution_turns_by_the_angle():
    """With an angle, the circle becomes nurbs_arc: still a surface of revolution (distance to
    the axis and height depend on u only), and the v = 1 meridian sits at the angle."""
    for angle in (0.4, np.pi / 2, 2.5, 4.0):
        surface = evaluate(surface_of_revolution(*sphere_profile(1.5), angle))
        assert np.allclose(np.linalg.norm(surface, axis=-1), 1.5, atol=1e-14)
        end = surface[1:-1, -1]
        assert np.allclose(np.arctan2(end[:, 1], end[:, 0]) % (2 * np.pi), angle % (2 * np.pi))


def test_sphere_net_is_five_by_nine_with_product_weights():
    net, weights, *_ = nurbs_sphere(1.0)
    s = np.sqrt(0.5)
    assert net.shape == (5, 9, 3)
    profile = np.array([1, s, 1, s, 1])
    circle = np.array([1, s, 1, s, 1, s, 1, s, 1])
    assert np.array_equal(weights, np.outer(profile, circle))
    assert np.isclose(weights.min(), 0.5)  # corner of a corner: s * s = 1/2


def test_sphere_normal_is_radial_inward():
    """S_u x S_v points toward the centre with this orientation (u south to north, v
    counter-clockwise); away from the poles it is exactly -S / r."""
    sphere = nurbs_sphere(2.0)
    points = evaluate(sphere)
    normals = nurbs_surface_normal(*sphere, U, V)
    assert np.abs(normals[1:-1] + points[1:-1] / 2.0).max() < 1e-14


def test_sphere_pole_is_a_singular_point():
    """A whole row of the net collapses to the pole, so S_v vanishes there and S_u x S_v = 0: the
    normal is undefined *at* the pole and exists only as a limit, approached linearly in u."""
    sphere = nurbs_sphere(2.0)
    net = sphere[0]
    assert np.ptp(net[0], axis=0).max() == 0.0 and np.ptp(net[-1], axis=0).max() == 0.0
    s_u, s_v = nurbs_surface_partials(*sphere, [0.0, 1.0], V)
    assert np.abs(s_v).max() == 0.0
    assert np.linalg.norm(s_u, axis=-1).min() > 1.0  # the meridians still have speed
    assert np.abs(nurbs_surface_normal(*sphere, [0.0], V)).max() == 0.0
    errors = [np.abs(nurbs_surface_normal(*sphere, [e], V)[0] - [0, 0, 1]).max()
              for e in (1e-2, 1e-4, 1e-6)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-5


def test_sphere_speed_is_not_uniform_in_either_direction():
    """Both the parallels and the meridians inherit the NURBS circle's speed variation
    (4 - 2 sqrt 2 ~ 1.17, test_circle_is_not_arc_length_parametrized)."""
    t = np.linspace(0.0, 1.0, 2001)
    sphere = nurbs_sphere(1.0)
    s_u, _ = nurbs_surface_partials(*sphere, t, [0.0])
    _, s_v = nurbs_surface_partials(*sphere, [0.5], t)
    meridian_speed = np.linalg.norm(s_u[:, 0], axis=-1)
    equator_speed = np.linalg.norm(s_v[0], axis=-1)
    for speed in (meridian_speed, equator_speed):
        assert abs(speed.max() / speed.min() - (4 - 2 * np.sqrt(2))) < 1e-6
