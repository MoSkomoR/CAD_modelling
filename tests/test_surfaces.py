import numpy as np
import pytest

from cadkernel.geometry.curves import bernstein_basis, bezier_curve
from cadkernel.geometry.surfaces import (
    bernstein_triangle_basis,
    bezier_isocurve,
    bezier_surface,
    bezier_surface_normal,
    bezier_surface_partials,
    bezier_surface_point,
    bezier_surface_subdivide,
    bezier_triangle,
    bezier_triangle_point,
    de_casteljau_surface_stages,
    de_casteljau_triangle_stages,
)


def sample_net(n=3, m=3, dim=3, seed=0):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n + 1, m + 1, dim))


# --- The two-pass construction ------------------------------------------------------------


def test_surface_evaluation_order_does_not_matter():
    """The claim that makes 'De Casteljau, twice' well defined: collapsing u first and v first
    give the same point. Without this, a tensor-product surface would not be a single object."""
    net = sample_net(4, 2, seed=1)
    for u, v in ((0.0, 0.0), (0.3, 0.7), (0.5, 0.5), (1.0, 0.25), (1.0, 1.0)):
        stages = de_casteljau_surface_stages(net, u, v)
        assert np.allclose(stages["point"], stages["point_other_order"], atol=1e-15)
        assert np.allclose(stages["point"], bezier_surface_point(net, u, v))


def test_surface_bernstein_matches_de_casteljau():
    """The tensor-product closed form S = B(u)^T P B(v) and the two De Casteljau passes agree,
    just as they do for curves -- so bulk tessellation may use the fast path safely."""
    net = sample_net(3, 4, seed=2)
    u, v = np.linspace(0, 1, 11), np.linspace(0, 1, 9)
    grid = bezier_surface(net, u, v)
    for a, uu in enumerate(u):
        for b, vv in enumerate(v):
            assert np.allclose(grid[a, b], bezier_surface_point(net, uu, vv))


def test_tensor_basis_is_a_partition_of_unity_and_non_negative():
    """The two facts every surface property rests on, inherited by multiplication: a product of
    non-negative numbers is non-negative, and (sum_i B_i)(sum_j B_j) = 1 * 1 = 1."""
    u, v = np.linspace(0, 1, 17), np.linspace(0, 1, 13)
    weights = np.einsum("ai,bj->abij", bernstein_basis(3, u), bernstein_basis(4, v))
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=(2, 3)), 1.0)


# --- Properties inherited from the curve case ---------------------------------------------


def test_surface_corner_interpolation():
    """Endpoint interpolation squared: the patch passes through the four *corner* control
    points and no others. The other net points only pull."""
    net = sample_net(3, 2, seed=3)
    assert np.allclose(bezier_surface_point(net, 0.0, 0.0), net[0, 0])
    assert np.allclose(bezier_surface_point(net, 1.0, 0.0), net[-1, 0])
    assert np.allclose(bezier_surface_point(net, 0.0, 1.0), net[0, -1])
    assert np.allclose(bezier_surface_point(net, 1.0, 1.0), net[-1, -1])


def test_surface_boundary_curves_are_the_net_boundary_polygons():
    """The four edges of the patch are Bezier curves whose control points are the four edges
    of the net -- which is what makes patches joinable edge-to-edge into a B-rep face."""
    net = sample_net(3, 4, seed=4)
    t = np.linspace(0, 1, 13)
    assert np.allclose(bezier_surface(net, t, [0.0])[:, 0], bezier_curve(net[:, 0], t))
    assert np.allclose(bezier_surface(net, t, [1.0])[:, 0], bezier_curve(net[:, -1], t))
    assert np.allclose(bezier_surface(net, [0.0], t)[0], bezier_curve(net[0, :], t))
    assert np.allclose(bezier_surface(net, [1.0], t)[0], bezier_curve(net[-1, :], t))


def test_isocurve_is_a_bezier_curve_of_the_other_direction_degree():
    """Fixing u leaves a degree-m Bezier curve whose control points are exactly the points the
    first De Casteljau pass produced -- so every module 01 curve routine applies to isocurves."""
    net = sample_net(4, 2, seed=5)
    t = np.linspace(0, 1, 11)
    control_points = bezier_isocurve(net, u=0.37)
    assert control_points.shape == (3, 3)  # degree m = 2
    assert np.allclose(bezier_curve(control_points, t), bezier_surface(net, [0.37], t)[0])

    control_points = bezier_isocurve(net, v=0.62)
    assert control_points.shape == (5, 3)  # degree n = 4
    assert np.allclose(bezier_curve(control_points, t), bezier_surface(net, t, [0.62])[:, 0])

    with pytest.raises(ValueError):
        bezier_isocurve(net, u=0.5, v=0.5)


def test_surface_stays_in_the_bounding_box_of_its_control_net():
    """Convex hull containment, checked through its most-used consequence: the net's
    axis-aligned box bounds the patch, so intersection code can cull before evaluating."""
    net = sample_net(3, 3, seed=6)
    grid = bezier_surface(net, np.linspace(0, 1, 40), np.linspace(0, 1, 40)).reshape(-1, 3)
    assert np.all(grid >= net.reshape(-1, 3).min(axis=0) - 1e-12)
    assert np.all(grid <= net.reshape(-1, 3).max(axis=0) + 1e-12)


def test_surface_affine_invariance():
    """Transform the net, get the transformed patch -- inherited from the curve case, and for
    the same reason: the tensor basis is still a partition of unity."""
    net = sample_net(3, 2, seed=7)
    angle = 0.7
    rotate = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                       [np.sin(angle), np.cos(angle), 0.0],
                       [0.0, 0.0, 1.0]])
    offset = np.array([1.5, -2.0, 0.25])
    u, v = np.linspace(0, 1, 9), np.linspace(0, 1, 9)
    transformed_then_evaluated = bezier_surface(net @ rotate.T + offset, u, v)
    evaluated_then_transformed = bezier_surface(net, u, v) @ rotate.T + offset
    assert np.allclose(transformed_then_evaluated, evaluated_then_transformed)


def test_surface_subdivide_reproduces_original():
    """Subdivision survives the move to surfaces: split every column at u, then every row at v,
    and the four sub-patches reproduce their quarters of the original exactly."""
    net = sample_net(4, 3, seed=8)
    us, vs = 0.37, 0.62
    quadrants = bezier_surface_subdivide(net, us, vs)
    spans = [[((0.0, us), (0.0, vs)), ((0.0, us), (vs, 1.0))],
             [((us, 1.0), (0.0, vs)), ((us, 1.0), (vs, 1.0))]]
    for a in range(2):
        for b in range(2):
            (u0, u1), (v0, v1) = spans[a][b]
            for local_u in np.linspace(0, 1, 7):
                for local_v in np.linspace(0, 1, 7):
                    assert np.allclose(
                        bezier_surface_point(quadrants[a][b], local_u, local_v),
                        bezier_surface_point(net, u0 + local_u * (u1 - u0),
                                             v0 + local_v * (v1 - v0)),
                    )


# --- Tangents and normals -----------------------------------------------------------------


def test_surface_partials_match_finite_differences():
    """S_u and S_v via the forward-difference net (the surface hodograph) against a central
    difference -- the two-parameter version of the curve tangent from the last lerp."""
    net = sample_net(3, 4, seed=9)
    h = 1e-6
    for u, v in ((0.2, 0.8), (0.5, 0.5), (0.9, 0.1)):
        s_u, s_v = bezier_surface_partials(net, u, v)
        fd_u = (bezier_surface_point(net, u + h, v) - bezier_surface_point(net, u - h, v)) / (2 * h)
        fd_v = (bezier_surface_point(net, u, v + h) - bezier_surface_point(net, u, v - h)) / (2 * h)
        assert np.allclose(s_u, fd_u, rtol=1e-6)
        assert np.allclose(s_v, fd_v, rtol=1e-6)
        assert np.isclose(np.linalg.norm(bezier_surface_normal(net, u, v)), 1.0)


# --- Where the tensor-product structure shows its seams ------------------------------------


def test_bilinear_patch_is_doubly_ruled_but_not_planar():
    """The bidegree-(1,1) patch on four non-coplanar corners: every isocurve is a straight
    line, yet the surface is a curved hyperbolic paraboloid. A 'flat-looking' net does not
    mean a flat patch."""
    net = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                    [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]])
    t = np.linspace(0, 1, 9)
    for fixed in (0.0, 0.25, 0.5, 1.0):
        for isocurve in (bezier_isocurve(net, u=fixed), bezier_isocurve(net, v=fixed)):
            points = bezier_curve(isocurve, t)
            direction = points[-1] - points[0]
            residual = points - points[0] - np.outer(t, direction)
            assert np.abs(residual).max() < 1e-12, "isocurves of a bilinear patch are lines"
    # ...but the patch is genuinely curved: the corners are not coplanar, and the centre of
    # the patch sits well off the plane through three of them. Here z(u, v) = u + v - 2uv,
    # so the centre is at z = 0.5 while that plane (x + y - z = 0) is at z = 1.
    normal = np.cross(net[0, 1] - net[0, 0], net[1, 0] - net[0, 0])
    assert not np.isclose(np.dot(net[1, 1] - net[0, 0], normal), 0.0), "corners are coplanar"
    centre = bezier_surface_point(net, 0.5, 0.5)
    assert np.isclose(centre[2], 0.5)
    distance_from_plane = abs(np.dot(centre - net[0, 0], normal)) / np.linalg.norm(normal)
    assert distance_from_plane > 0.25


def test_diagonal_of_a_patch_has_degree_n_plus_m():
    """Isocurves have degree n or m, but the diagonal u = v = t has degree n + m. Degree
    inflates as soon as you cut across a patch in any direction that isn't isoparametric --
    which is exactly what a surface/surface intersection curve does."""
    t = np.linspace(0, 1, 200)

    def exact_degree(values, limit=14):
        for degree in range(limit + 1):
            basis = bernstein_basis(degree, t)
            coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
            if np.abs(basis @ coefficients - values).max() < 1e-9:
                return degree
        return None

    for n, m in ((1, 1), (2, 2), (3, 3), (3, 2), (4, 1)):
        net = sample_net(n, m, seed=10 + n * 5 + m)
        diagonal = np.array([bezier_surface_point(net, tt, tt) for tt in t])
        assert exact_degree(diagonal) == n + m
        assert exact_degree(bezier_curve(bezier_isocurve(net, v=0.5), t)) == n


# --- Triangular Bezier patches -- the other generalization to two parameters --------------


def sample_triangle_net(n=3, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    return {
        (n - j - k, j, k): rng.uniform(-1.0, 1.0, size=dim)
        for j in range(n + 1)
        for k in range(n + 1 - j)
    }


def sample_barycentric_points(count, seed=0):
    """Random points inside the unit simplex, l0+l1+l2=1, l_i >= 0."""
    rng = np.random.default_rng(seed)
    l0 = rng.uniform(0.0, 1.0, size=count)
    l1 = rng.uniform(0.0, 1.0 - l0)
    l2 = 1.0 - l0 - l1
    return l0, l1, l2


def test_triangle_de_casteljau_matches_closed_bernstein_form():
    """Barycentric De Casteljau and the closed multinomial-Bernstein sum must agree -- the
    trinomial analogue of test_surface_bernstein_matches_de_casteljau, and of
    test_bernstein_matches_de_casteljau (curves) before that."""
    net = sample_triangle_net(n=3, seed=1)
    l0s, l1s, l2s = sample_barycentric_points(25, seed=2)
    for l0, l1, l2 in zip(l0s, l1s, l2s):
        assert np.allclose(
            bezier_triangle_point(net, l0, l1, l2), bezier_triangle(net, l0, l1)[0]
        )


def test_triangle_corner_interpolation():
    """The three barycentric corners are interpolated -- the triangular analogue of endpoint
    interpolation -- and no other control point is."""
    net = sample_triangle_net(n=3, seed=3)
    assert np.allclose(bezier_triangle_point(net, 1.0, 0.0, 0.0), net[(3, 0, 0)])
    assert np.allclose(bezier_triangle_point(net, 0.0, 1.0, 0.0), net[(0, 3, 0)])
    assert np.allclose(bezier_triangle_point(net, 0.0, 0.0, 1.0), net[(0, 0, 3)])


def test_triangle_bernstein_basis_is_partition_of_unity_and_non_negative():
    """Non-negativity and partition of unity, now from the multinomial theorem: the sum of the
    (l0+l1+l2)^n expansion's terms is 1^n = 1 for every barycentric (l0, l1, l2)."""
    l0s, l1s, l2s = sample_barycentric_points(50, seed=4)
    _, values = bernstein_triangle_basis(4, l0s, l1s, l2s)
    assert np.all(values >= 0.0)
    assert np.allclose(values.sum(axis=1), 1.0)


def test_triangle_stays_in_the_bounding_box_of_its_control_net():
    """Convex hull containment, inherited exactly as it was for the tensor-product case: convex
    weights (Fact above) mean the surface cannot leave the hull of the net."""
    net = sample_triangle_net(n=3, seed=5)
    l0s, l1s, l2s = sample_barycentric_points(200, seed=6)
    points = np.array(
        [bezier_triangle_point(net, l0, l1, l2) for l0, l1, l2 in zip(l0s, l1s, l2s)]
    )
    corners = np.array(list(net.values()))
    assert np.all(points >= corners.min(axis=0) - 1e-12)
    assert np.all(points <= corners.max(axis=0) + 1e-12)


def test_triangle_affine_invariance():
    """Transform the net, get the transformed patch -- same proof as the curve and
    tensor-product cases, since the basis is still a partition of unity."""
    net = sample_triangle_net(n=3, seed=7)
    angle = 0.4
    rotate = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    offset = np.array([0.5, -1.0, 0.25])
    transformed_net = {index: p @ rotate.T + offset for index, p in net.items()}
    for l0, l1, l2 in zip(*sample_barycentric_points(15, seed=8)):
        transformed_then_evaluated = bezier_triangle_point(transformed_net, l0, l1, l2)
        evaluated_then_transformed = bezier_triangle_point(net, l0, l1, l2) @ rotate.T + offset
        assert np.allclose(transformed_then_evaluated, evaluated_then_transformed)


def test_de_casteljau_triangle_stages_shrink_to_the_surface_point():
    """The lattice loses one row per level, exactly as the curve triangle loses one point per
    level, and the last level is the single point bezier_triangle_point returns."""
    net = sample_triangle_net(n=3, seed=9)
    l0, l1, l2 = 0.5, 0.3, 0.2
    levels = de_casteljau_triangle_stages(net, l0, l1, l2)
    assert [len(level) for level in levels] == [10, 6, 3, 1]
    assert np.allclose(levels[-1][(0, 0, 0)], bezier_triangle_point(net, l0, l1, l2))
