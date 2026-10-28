import numpy as np

from cadkernel.geometry.entities import NurbsCurve, circle_arcs, polygon_segments
from cadkernel.ops.extrude import extrude, profile_vector_area
from cadkernel.ops.measure import loop_vector_area, surface_vector_area, volume
from cadkernel.topology.halfedge import block_with_hole, cube, octahedron, signed_volume


def test_volume_agrees_with_signed_volume_on_polyhedra():
    """With no curves bound, volume() is the polyhedral formula of module 03."""
    for solid in (cube(2.0), block_with_hole(), octahedron()):
        assert np.isclose(volume(solid), signed_volume(solid))


def test_vector_area_of_a_circle():
    """1/2 \\oint x \\times dx around a counter-clockwise circle of radius r is pi r^2 e_z."""
    assert np.allclose(profile_vector_area(circle_arcs([0.3, 0.1], 1.5)), [0, 0, np.pi * 2.25])


def test_quadrature_convergence_on_a_rational_integrand():
    """Extruded unit circle, height 1: error of the volume against pi for n Gauss points per knot
    span. The arcs are rational, so no finite rule is exact; the error falls geometrically.
    Measured: 0.17, 7.1e-3, 2.9e-4, 1.2e-5, 1.9e-8, 2.9e-11, ~1e-15 for n = 1, 2, 3, 4, 6, 8, 12."""
    solid = extrude(circle_arcs([0, 0], 1.0), [0, 0, 1])
    errors = {n: abs(volume(solid, n) - np.pi) for n in (1, 2, 3, 4, 6, 8, 12)}
    assert np.isclose(errors[1], 0.172, atol=1e-3)
    assert np.isclose(errors[2], 7.15e-3, rtol=1e-2)
    assert 1e-5 < errors[4] < 2e-5
    assert 1e-11 < errors[8] < 1e-10
    assert errors[12] < 1e-14
    assert all(a > b for a, b in zip(list(errors.values()), list(errors.values())[1:]))


def test_quadrature_is_exact_for_polynomial_faces():
    """A cubic Bezier profile, extruded at a tilt: the side integrand S . (S_u x S_v) has degree
    2p - 1 = 5 in u (S_v = d is constant) and the cap integrand C x C' too, so N = p = 3 Gauss
    points (exact to degree 2N - 1) suffice; N = 2 is off by 0.12. The reference is the enclosed
    area times the height, the area from a dense polygon of the profile."""
    profile = [NurbsCurve.bezier([[1, 0], [1, 1.2], [-1, 1.2], [-1, 0]]),
               NurbsCurve.bezier([[-1, 0], [-1, -0.6], [1, -0.6], [1, 0]])]
    solid = extrude(profile, [0.3, 0.2, 2.0])
    polygon = np.vstack([c(np.linspace(0, 1, 20001))[:-1, :2] for c in profile])
    x, y = polygon.T
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert np.isclose(volume(solid, 3), 2.0 * area, rtol=1e-8)
    assert np.isclose(volume(solid, 3), volume(solid, 40), atol=1e-13)
    assert np.isclose(volume(solid, 40) - volume(solid, 2), 0.12, atol=1e-12)


def test_loop_vector_area_follows_halfedge_direction():
    """The bottom face's loop runs clockwise seen from above (its outward normal is -z), the top
    face's counter-clockwise: equal and opposite vector areas."""
    solid = extrude(circle_arcs([0, 0], 1.0), [0, 0, 1])
    planar = [f for f in solid.faces if hasattr(f.surface, "normal")]
    areas = sorted((loop_vector_area(f.outer) for f in planar), key=lambda a: a[2])
    assert np.allclose(areas[0], [0, 0, -np.pi]) and np.allclose(areas[1], [0, 0, np.pi])


def test_square_volume_needs_no_quadrature():
    solid = extrude(polygon_segments([[0, 0], [3, 0], [3, 2], [0, 2]]), [0, 0, 1.5])
    assert volume(solid, 1) == 9.0


def test_extrude_loops_run_the_way_their_sense_says():
    """Stokes on every curved face of an extrusion with holes: 1/2 \\oint x \\times dx around the
    face's loop equals sense * \\iint S_u x S_v du dv -- outer walls (sense +1) and hole walls
    (sense -1) alike. Revolve.md derives the identity."""
    solid = extrude(circle_arcs([0, 0], 3.0), [0.4, 0.2, 1.5], [circle_arcs([0, 0], 1.0)])
    for face in solid.faces:
        if face.surface is not None and not hasattr(face.surface, "normal"):
            boundary = sum(loop_vector_area(loop) for loop in face.loops)
            assert np.allclose(boundary, face.sense * surface_vector_area(face.surface), atol=1e-13)
