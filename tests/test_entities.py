import numpy as np

from cadkernel.geometry.entities import (
    NurbsCurve,
    Plane,
    circle_arcs,
    extruded_surface,
    polygon_segments,
    revolved_surface,
    turning_arc,
    xz,
)

T = np.linspace(0.0, 1.0, 101)


def test_circle_arcs_lie_on_the_circle_and_join_up():
    arcs = circle_arcs([0.5, -0.2], 1.3)
    for a, b in zip(arcs, arcs[1:] + arcs[:1]):
        assert np.allclose(a.end, b.start)
    for arc in arcs:
        assert np.allclose(np.linalg.norm(arc(T)[:, :2] - [0.5, -0.2], axis=1), 1.3)
        assert np.allclose(arc(T)[:, 2], 0.0)


def test_reversed_traverses_the_same_points_backwards():
    curve = NurbsCurve.bezier([[0, 0], [1, 2], [3, -1], [4, 1]])
    arc = circle_arcs([0, 0], 1.0)[1]
    for c in (curve, arc):
        assert np.allclose(c.reversed()(T), c(1.0 - T))
        assert np.allclose(c.reversed().derivative(T), -c.derivative(1.0 - T))


def test_extruded_surface_is_the_curve_plus_v_d():
    """S(u, v) = C(u) + v d exactly, rational curve included: the net [P_i, P_i + d] with the
    curve's weights in both columns (Extrude.md)."""
    d = np.array([0.4, -0.3, 2.0])
    v = np.linspace(0, 1, 7)
    for curve in (circle_arcs([0, 0], 1.0)[0], NurbsCurve.bezier([[0, 0], [1, 2], [3, -1]])):
        surface = extruded_surface(curve, d)
        expected = curve(T)[:, None, :] + v[None, :, None] * d
        assert np.allclose(surface(T, v), expected, atol=1e-14)
        s_u, s_v = surface.partials(T, v)
        assert np.allclose(s_v, np.broadcast_to(d, s_v.shape))
        assert np.allclose(s_u, np.repeat(curve.derivative(T)[:, None], len(v), axis=1))


def test_distance_to_surface_finds_points_on_it_and_off_it():
    """Gauss-Newton closest point: zero for points on the surface, the true offset for a point
    moved along the normal."""
    arc = circle_arcs([0, 0], 2.0)[0]
    surface = extruded_surface(arc, [0, 0, 1])
    on = surface([0.13, 0.5, 0.91], [0.2, 0.7])[:, 0]
    assert surface.distance(on).max() < 1e-12
    angle = np.pi / 6  # a point 0.25 outside the cylinder of radius 2
    off = np.array([2.25 * np.cos(angle), 2.25 * np.sin(angle), 0.4])
    assert np.isclose(surface.distance(off)[0], 0.25)


def test_plane_distance():
    plane = Plane([0, 0, 1], [0, 0, 2])
    assert np.allclose(plane.distance([[3, 4, 1], [0, 0, -1]]), [0, 2])


def test_polygon_segments_close():
    segs = polygon_segments([[0, 0], [1, 0], [0, 1]])
    assert np.allclose(segs[-1].end, segs[0].start) and len(segs) == 3


def test_rotated_and_turning_arc():
    """Rotating the control points rotates the curve; turning_arc is the path of a point turned
    about the z-axis: constant height and distance from the axis, ending at the rotated point."""
    curve = NurbsCurve.bezier([[1, 0, 0], [2, 1, 0.5], [1, 2, 1]])
    turned = curve.rotated(0.7)
    c, s = np.cos(0.7), np.sin(0.7)
    assert np.allclose(turned(T), curve(T) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T)
    point = np.array([1.0, 2.0, 0.5])
    for angle in (1.0, 2 * np.pi):
        arc = turning_arc(point, angle)(T)
        assert np.allclose(np.hypot(arc[:, 0], arc[:, 1]), np.sqrt(5), atol=1e-14)
        assert np.allclose(arc[:, 2], 0.5) and np.allclose(arc[0], point)
        assert np.allclose(arc[-1], NurbsCurve.line(point, point).rotated(angle).start)
    assert turning_arc(point, 2 * np.pi).closed and not turning_arc(point, 1.0).closed


def test_revolved_surface_contains_the_curve_and_its_copies():
    """v = 0 is the curve itself; every v = const curve is the curve rotated about the z-axis."""
    curve = xz([NurbsCurve.bezier([[1, 0], [2, 1], [1, 2]])])[0]
    surface = revolved_surface(curve, 2.0)
    assert np.allclose(surface(T, 0.0)[:, 0], curve(T))
    assert np.allclose(surface(T, 1.0)[:, 0], curve.rotated(2.0)(T))
    with np.errstate(all="raise"):
        assert np.allclose(surface.distance(curve.rotated(1.2)(np.linspace(0, 1, 7))), 0,
                           atol=1e-12)
