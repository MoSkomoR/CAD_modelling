import numpy as np
import pytest

from cadkernel.geometry.entities import NurbsCurve, Plane, circle_arcs, polygon_segments, xz
from cadkernel.ops.measure import loop_vector_area, surface_vector_area, volume
from cadkernel.ops.revolve import profile_area_and_centroid, revolve
from cadkernel.topology.halfedge import check, euler_poincare, vertex_star

S = np.sqrt(0.5)
QUARTER = [0, 0, 0, 1, 1, 1]
DISC = xz(circle_arcs([2.0, 0.0], 0.7))  # off the axis: a torus
SQUARE = xz(polygon_segments([[1, 0], [2, 0], [2, 1], [1, 1]]))  # off the axis: a tube
HALF_DISC = xz([NurbsCurve([[0, -1], [1, -1], [1, 0]], [1, S, 1], QUARTER, 2),
                NurbsCurve([[1, 0], [1, 1], [0, 1]], [1, S, 1], QUARTER, 2),
                NurbsCurve.line([0, 1], [0, -1])])  # on the axis: a sphere
TRIANGLE = xz(polygon_segments([[0, 0], [1, 0], [0, 2]]))  # on the axis: a cone
PROFILES = [DISC, SQUARE, HALF_DISC, TRIANGLE]
ANGLES = [2 * np.pi, np.pi / 3, 1.5 * np.pi]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("angle", ANGLES)
def test_every_step_is_valid_and_the_result_is_consistent(profile, angle):
    """Topology valid after every operator, seams and closed edges included; at the end every
    edge curve lies on both of its faces' surfaces."""
    steps = []
    solid = revolve(profile, angle, steps)
    assert all(step["problems"] == [] for step in steps)
    assert check(solid) == []


@pytest.mark.parametrize("profile, full, partial", [
    (DISC, dict(V=4, E=8, F=4, G=1), dict(V=8, E=12, F=6, G=0)),
    (SQUARE, dict(V=4, E=8, F=4, G=1), dict(V=8, E=12, F=6, G=0)),
    (HALF_DISC, dict(V=3, E=3, F=2, G=0), dict(V=4, E=6, F=4, G=0)),
    (TRIANGLE, dict(V=3, E=3, F=2, G=0), dict(V=4, E=6, F=4, G=0)),
])
def test_counts(profile, full, partial):
    """n profile edges. Full turn off the axis: V = n, E = 2n, F = n, G = 1. Full turn with an
    axis edge: V = n, E = 2n - 3, F = n - 1. Partial turn: the extrusion counts, minus one rail
    per pole and one side face for the axis edge (Revolve.md)."""
    for angle, expected in ((2 * np.pi, full), (1.0, partial)):
        c = revolve(profile, angle).counts()
        assert {k: c[k] for k in expected} == expected
        assert c["R"] == 0 and euler_poincare(c)[0] == euler_poincare(c)[1]


def test_a_sphere_from_one_semicircle_is_one_face_with_one_seam():
    """The smallest case: one profile edge from pole to pole (plus the axis edge, dropped).
    V = 2, E = 1, F = 1: a single face whose loop is the seam, out and back."""
    semicircle = xz([NurbsCurve([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1]], [1, S, 1, S, 1],
                                [0, 0, 0, 0.5, 0.5, 1, 1, 1], 2), NurbsCurve.line([0, 1], [0, -1])])
    solid = revolve(semicircle)
    c = solid.counts()
    assert (c["V"], c["E"], c["F"]) == (2, 1, 1) and check(solid) == []
    (face,) = solid.faces
    seam = solid.edges[0]
    assert seam.he1.face is face and seam.he2.face is face
    assert np.isclose(volume(solid), 4 * np.pi / 3, atol=1e-14)


def test_seams_and_closed_edges_in_the_half_edge_structure():
    """Torus: every profile edge is a seam (both half-edges in one face), every circle is a
    closed edge (both half-edges leave the same vertex, in two different faces), and the star of
    each vertex walks all four of its outgoing half-edges."""
    solid = revolve(DISC)
    seams = [e for e in solid.edges if e.he1.face is e.he2.face]
    closed = [e for e in solid.edges if e.he1.origin is e.he2.origin]
    assert len(seams) == 4 and len(closed) == 4
    assert all(e.curve.closed for e in closed) and not any(e.curve.closed for e in seams)
    assert all(e.he1.face is not e.he2.face for e in closed)
    assert all(len(vertex_star(v)) == 4 for v in solid.vertices)


def test_poles_are_vertices_without_a_circle():
    """Sphere from two quarter arcs: the two poles each have one outgoing half-edge (the seam
    leaves them once), the equator vertex has four."""
    solid = revolve(HALF_DISC)
    stars = sorted(len(vertex_star(v)) for v in solid.vertices)
    assert stars == [1, 1, 4]
    poles = [v for v in solid.vertices if len(vertex_star(v)) == 1]
    assert sorted(v.point[2] for v in poles) == [-1, 1]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("angle", ANGLES)
def test_pappus(profile, angle):
    """V = angle * x_bar * A, with A and x_bar from Green's theorem on the profile alone and V
    from the divergence theorem on the solid's faces: two independent computations. Measured
    agreement better than 1e-12 (the rational arcs limit both quadratures)."""
    area, x_bar = profile_area_and_centroid(profile)
    assert abs(volume(revolve(profile, angle)) - angle * x_bar * area) < 1e-12


def test_known_volumes():
    """Sphere 4/3 pi, cone 1/3 pi r^2 h, torus 2 pi^2 R r^2, tube pi (b^2 - a^2) h."""
    assert np.isclose(volume(revolve(HALF_DISC)), 4 * np.pi / 3, atol=1e-14)
    assert np.isclose(volume(revolve(TRIANGLE)), 2 * np.pi / 3, atol=1e-14)
    assert np.isclose(volume(revolve(DISC)), 2 * np.pi ** 2 * 2.0 * 0.49, atol=1e-13)
    assert np.isclose(volume(revolve(SQUARE)), 3 * np.pi, atol=1e-13)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("angle", ANGLES)
def test_every_loop_runs_the_way_its_sense_says(profile, angle):
    """Stokes: 1/2 \\oint x \\times dx around a face's loops equals sense * \\iint S_u x S_v du dv
    when the loops run counter-clockwise around the outward normal. The left side comes from the
    topology (which half-edges, which direction), the right from the geometry."""
    for face in revolve(profile, angle).faces:
        if isinstance(face.surface, Plane):
            continue
        boundary = sum(loop_vector_area(loop) for loop in face.loops)
        assert np.allclose(boundary, face.sense * surface_vector_area(face.surface), atol=1e-13)


def test_a_reversed_circle_is_caught_by_the_orientation_check():
    """Reverse one closed edge's curve: check() still passes (the circle is still on both
    surfaces), but the loop identity above fails for both faces -- orientation is a separate
    property, and only the Stokes test sees it."""
    solid = revolve(DISC)
    circle = next(e for e in solid.edges if e.he1.origin is e.he2.origin)
    circle.curve = circle.curve.reversed()
    assert check(solid) == []
    wrong = [f for f in solid.faces
             if not np.allclose(sum(loop_vector_area(lp) for lp in f.loops),
                                f.sense * surface_vector_area(f.surface), atol=1e-9)]
    assert set(wrong) == {circle.he1.face, circle.he2.face}


def test_orientation_does_not_depend_on_the_input():
    clockwise = [c.reversed() for c in reversed(TRIANGLE)]
    assert np.isclose(volume(revolve(clockwise)), volume(revolve(TRIANGLE)))
    assert all(f.sense == 1 for f in revolve(clockwise).faces)


def test_bad_input_is_rejected():
    with pytest.raises(ValueError, match="xz-plane"):
        revolve(polygon_segments([[1, 0], [2, 0], [2, 1]]))
    with pytest.raises(ValueError, match="crosses"):
        revolve(xz(polygon_segments([[-1, 0], [2, 0], [2, 1]])))
    with pytest.raises(ValueError, match="single vertex"):
        revolve(xz(polygon_segments([[0, 0], [2, -1], [2, 1]])))
    with pytest.raises(ValueError, match="at most one"):
        revolve(xz(polygon_segments([[0, 0], [1, 0], [0, 1], [0, 0.5]])))
    with pytest.raises(ValueError, match="angle"):
        revolve(SQUARE, 7.0)
