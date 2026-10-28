import numpy as np
import pytest

from cadkernel.geometry.entities import NurbsCurve, Plane, circle_arcs, polygon_segments
from cadkernel.ops.extrude import extrude
from cadkernel.ops.measure import volume
from cadkernel.topology.halfedge import check, euler_poincare

SQUARE = polygon_segments([[-2, -2], [2, -2], [2, 2], [-2, 2]])
CIRCLE = circle_arcs([0, 0], 1.0)
SLOT = polygon_segments([[1.0, -0.5], [1.6, -0.5], [1.6, 0.5], [1.0, 0.5]])


@pytest.mark.parametrize("profile, holes", [
    (SQUARE, []), (CIRCLE, []), (SQUARE, [CIRCLE]), (circle_arcs([0, 0], 3.0), [CIRCLE, SLOT]),
])
@pytest.mark.parametrize("d", [(0, 0, 1.5), (0.7, 0.3, 1.5), (0, 0, -1.5)])
def test_every_step_is_valid_and_the_result_is_consistent(profile, holes, d):
    """Topology valid after every operator; at the end, every edge curve lies on both of its faces'
    surfaces and the Euler-Poincare formula balances."""
    steps = []
    solid = extrude(profile, d, holes, steps=steps)
    assert all(step["problems"] == [] for step in steps)
    assert check(solid) == []
    lhs, rhs = euler_poincare(solid.counts())
    assert lhs == rhs


def test_counts_of_an_extrusion():
    """A profile of n edges with holes of m_1, m_2, ... edges: V = 2N, E = 3N, F = n + 2 + sum m
    (N = n + sum m), R = number of holes, G = number of holes."""
    solid = extrude(circle_arcs([0, 0], 3.0), (0, 0, 1), [CIRCLE, SLOT])
    c = solid.counts()
    assert {k: c[k] for k in "VEFRSG"} == dict(V=24, E=36, F=14, R=2 * 2, S=1, G=2)


def test_counts_match_the_formula_for_several_sizes():
    """V = 2n, E = 3n, F = n + 2 for a profile of n edges, so V - E + F = 2 for every n."""
    lens = [NurbsCurve.bezier([[0, 0], [1, 1], [2, 0]]), NurbsCurve.bezier([[2, 0], [1, -1], [0, 0]])]
    for profile in (lens, circle_arcs([0, 0], 2.0, 3), CIRCLE, circle_arcs([0, 0], 2.0, 7)):
        n = len(profile)
        c = extrude(profile, (0, 0, 1)).counts()
        assert (c["V"], c["E"], c["F"]) == (2 * n, 3 * n, n + 2)


def test_volumes():
    """Circle: pi r^2 h. Square with a round hole: (16 - pi) h. Disc of radius 3 with a round hole
    of radius 1 and a 0.6 x 1 slot: (9 pi - pi - 0.6) h."""
    h = 1.5
    assert np.isclose(volume(extrude(CIRCLE, (0, 0, h))), np.pi * h, atol=1e-14)
    assert np.isclose(volume(extrude(SQUARE, (0, 0, h), [CIRCLE])), (16 - np.pi) * h, atol=1e-13)
    solid = extrude(circle_arcs([0, 0], 3.0), (0, 0, h), [CIRCLE, SLOT])
    assert np.isclose(volume(solid), (8 * np.pi - 0.6) * h, atol=1e-12)


def test_oblique_extrusion_keeps_the_volume():
    """Cavalieri: tilting d keeps its component along the profile normal, so V = A (d . n) does
    not change (Extrude.md proves it as a shear)."""
    upright = volume(extrude(SQUARE, (0, 0, 1.5), [CIRCLE]))
    for d in [(0.7, 0.3, 1.5), (-2.0, 1.0, 1.5), (3.0, 3.0, 1.5)]:
        assert np.isclose(volume(extrude(SQUARE, d, [CIRCLE])), upright, atol=1e-12)


def test_orientation_does_not_depend_on_the_input():
    """A clockwise profile, or d pointing down, gives the same positive volume: extrude orients
    the profile counter-clockwise seen from the tip of d before building."""
    clockwise = [c.reversed() for c in reversed(CIRCLE)]
    for profile in (CIRCLE, clockwise):
        for d in ((0, 0, 2.0), (0, 0, -2.0)):
            assert np.isclose(volume(extrude(profile, d)), 2 * np.pi)


def test_sense_records_outward_normals():
    """S_u x S_v points away from the material where sense is +1 (outer wall: away from the axis)
    and into the hole where it is -1 (hole wall: towards the axis)."""
    solid = extrude(circle_arcs([0, 0], 3.0), (0, 0, 1), [CIRCLE])
    curved = [f for f in solid.faces if not isinstance(f.surface, Plane)]
    assert sorted(f.sense for f in curved) == [-1] * 4 + [1] * 4
    for face in curved:
        point = face.surface(0.5, 0.5)[0, 0]
        s_u, s_v = (d[0, 0] for d in face.surface.partials(0.5, 0.5))
        outward = face.sense * np.cross(s_u, s_v)
        radial = point[:2] / np.linalg.norm(point[:2])
        on_outer_wall = np.linalg.norm(point[:2]) > 2
        assert np.sign(outward[:2] @ radial) == (1 if on_outer_wall else -1)


def test_flipping_one_sense_breaks_the_volume():
    """sense is not decoration: flip one quarter-cylinder side and its flux enters with the wrong
    sign. Measured: pi becomes 2 pi / 3 (that face contributes pi / 6, and is now subtracted)."""
    solid = extrude(CIRCLE, (0, 0, 1))
    next(f for f in solid.faces if not isinstance(f.surface, Plane)).sense = -1
    assert np.isclose(volume(solid), 2 * np.pi / 3)


def test_check_catches_a_curve_that_leaves_a_face():
    """Pull the middle control point of one bottom arc sideways, within the bottom plane: the arc
    is still on the bottom face's plane, but no longer on the side face's cylinder -- and only
    that face is reported."""
    solid = extrude(CIRCLE, (0, 0, 1))
    edge = next(e for e in solid.edges if e.curve.p == 2 and np.allclose(e.curve.points[:, 2], 0))
    c = edge.curve
    edge.curve = NurbsCurve(c.points + [[0, 0, 0], [0.01, 0, 0], [0, 0, 0]], c.weights, c.knots, c.p)
    problems = check(solid)
    assert len(problems) == 1 and "leaves the surface" in problems[0]
    side = next(f for f in (edge.he1.face, edge.he2.face) if not isinstance(f.surface, Plane))
    assert str(side) in problems[0]


def test_bad_input_is_rejected():
    with pytest.raises(ValueError, match="close"):
        extrude(SQUARE[:3], (0, 0, 1))
    with pytest.raises(ValueError, match="plane"):
        extrude(SQUARE, (1, 0, 0))
    bent = [c.translated((0, 0, 0)) for c in SQUARE]
    bent[0] = NurbsCurve.bezier([[-2, -2, 0], [0, -2, 0.5], [2, -2, 0]])
    with pytest.raises(ValueError, match="one plane"):
        extrude(bent, (0, 0, 1))
