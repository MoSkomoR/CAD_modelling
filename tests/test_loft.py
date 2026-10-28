import numpy as np
import pytest

from cadkernel.geometry.entities import NurbsCurve, Plane, circle_arcs, polygon_segments
from cadkernel.ops.extrude import extrude
from cadkernel.ops.loft import loft
from cadkernel.ops.measure import loop_vector_area, surface_vector_area, volume
from cadkernel.topology.halfedge import check

CIRCLE = circle_arcs([0, 0], 1.0)
SQUARE = polygon_segments([[1, -1], [1, 1], [-1, 1], [-1, -1]])


def lifted(curves, z, scale=1.0):
    return [NurbsCurve(c.points * scale + [0, 0, z], c.weights, c.knots, c.p) for c in curves]


SQUARE_TO_CIRCLE = [SQUARE, lifted(CIRCLE, 1.0), lifted(CIRCLE, 2.6, 0.6), lifted(SQUARE, 3.0, 0.4)]


@pytest.mark.parametrize("sections", [
    [CIRCLE, lifted(CIRCLE, 2.0)],
    [CIRCLE, lifted(CIRCLE, 1.0, 1.5), lifted(CIRCLE, 2.0, 0.5)],
    SQUARE_TO_CIRCLE,
])
def test_every_step_is_valid_and_the_result_is_consistent(sections):
    """Valid after every operator; the longitudinal edges lie on both neighbouring skins, which
    holds only because all faces share one set of section parameters."""
    steps = []
    solid = loft(sections, steps=steps)
    assert all(step["problems"] == [] for step in steps)
    assert check(solid) == []
    c = solid.counts()
    n = len(sections[0])
    assert (c["V"], c["E"], c["F"]) == (2 * n, 3 * n, n + 2)


def test_two_equal_circles_give_the_extruded_cylinder():
    """Two sections, so degree 1 across: the loft is the extrusion, face for face."""
    lofted, extruded = loft([CIRCLE, lifted(CIRCLE, 2.0)]), extrude(CIRCLE, (0, 0, 2.0))
    assert np.isclose(volume(lofted), 2 * np.pi, atol=1e-14)
    nets = lambda s: sorted(np.round(f.surface.net, 12).tobytes() for f in s.faces
                            if not isinstance(f.surface, Plane))
    assert nets(lofted) == nets(extruded)


def test_two_circles_of_different_radius_give_the_frustum():
    """Degree 1 across two circles is the exact cone frustum: V = pi h (a^2 + a b + b^2) / 3."""
    solid = loft([CIRCLE, lifted(CIRCLE, 2.0, 0.5)])
    assert np.isclose(volume(solid), np.pi * 2.0 * (1 + 0.5 + 0.25) / 3, atol=1e-14)


def test_every_loop_runs_the_way_its_sense_says():
    """The Stokes identity of test_revolve, on the square-to-circle loft."""
    for face in loft(SQUARE_TO_CIRCLE).faces:
        if not isinstance(face.surface, Plane):
            boundary = sum(loop_vector_area(loop) for loop in face.loops)
            assert np.allclose(boundary, face.sense * surface_vector_area(face.surface), atol=1e-13)


def test_sections_lie_on_the_solid():
    """Every section curve, the middle ones included, lies on the side faces' surfaces."""
    solid = loft(SQUARE_TO_CIRCLE)
    sides = [f.surface for f in solid.faces if not isinstance(f.surface, Plane)]
    for section in SQUARE_TO_CIRCLE:
        for curve in section:
            points = curve(np.linspace(0, 1, 9))
            assert min(s.distance(points).max() for s in sides) < 1e-10


def test_loft_parametrization_comparison():
    """Circles of radius r(z) = 1 + 0.4 sin(1.3 z) at given heights; the error is the largest
    |r - r(z)| along a longitudinal edge. Measured (uniform, chord, centripetal):
    uneven [0, .3, .6, 2.5, 3]:         0.221, 0.0279, 0.169
    even [0, 1, 2, 3, 4]:               0.033, 0.0543, 0.0435
    pairs [0, .2, 1.5, 1.7, 3.2, 3.4]:  0.150, 0.00344, 0.101
    Chord wins both uneven sets, uniform the even one; centripetal never wins, and is never
    the worst."""
    r = lambda z: 1 + 0.4 * np.sin(1.3 * z)
    expected = {(0, 0.3, 0.6, 2.5, 3.0): (0.221, 0.0279, 0.169),
                (0, 1, 2, 3, 4): (0.033, 0.0543, 0.0435),
                (0, 0.2, 1.5, 1.7, 3.2, 3.4): (0.150, 0.00344, 0.101)}
    for heights, errors in expected.items():
        sections = [lifted(CIRCLE, z, r(z)) for z in heights]
        for method, error in zip(("uniform", "chord", "centripetal"), errors):
            solid = loft(sections, parametrization=method)
            rail = next(e.curve for e in solid.edges if e.curve.p == 3)
            p = rail(np.linspace(0, 1, 2001))
            measured = np.abs(np.hypot(p[:, 0], p[:, 1]) - r(p[:, 2])).max()
            assert np.isclose(measured, error, rtol=0.02)


def test_loft_overshoot_comparison():
    """The stack of scripts/04_modeling_ops/03_loft.py with its two circles pulled close: square
    (distance of the vertices from the axis rho = 1.4, z = 0), circle (1.0, 1.0), circle (0.7,
    1.25), square (0.6, 3.0). rho decreases section by section, by 0.8 in all; the excess of the
    longitudinal edge's total variation of rho over 0.8 measures the wiggles between sections.
    Measured (uniform, chord, centripetal): excess 0.0038, 0.745, 0.186; smallest rho 0.598,
    0.234, 0.507. Chord pinches a waist far thinner than any section; uniform wins here."""
    def section(shape, rho, z):
        if shape == "square":
            a = np.radians(-45 + 90 * np.arange(4))
            return lifted(polygon_segments(np.c_[rho * np.cos(a), rho * np.sin(a)]), z)
        return lifted([c.rotated(-np.pi / 4) for c in circle_arcs([0, 0], rho)], z)

    sections = [section("square", 1.4, 0.0), section("circle", 1.0, 1.0),
                section("circle", 0.7, 1.25), section("square", 0.6, 3.0)]
    expected = {"uniform": (0.00375, 0.598), "chord": (0.745, 0.234),
                "centripetal": (0.186, 0.507)}
    for method, (excess, smallest) in expected.items():
        solid = loft(sections, parametrization=method)
        rail = next(e.curve for e in solid.edges
                    if e.curve.p == 3 and abs(e.curve.start[2] - e.curve.end[2]) > 1)
        p = rail(np.linspace(0, 1, 2001))
        rho = np.hypot(p[:, 0], p[:, 1])
        assert np.isclose(np.abs(np.diff(rho)).sum() - 0.8, excess, rtol=0.02)
        assert np.isclose(rho.min(), smallest, rtol=0.01)


def test_bad_input_is_rejected():
    triangle = polygon_segments([[1, 0], [-0.5, 1], [-0.5, -1]])
    with pytest.raises(ValueError, match="same number"):
        loft([CIRCLE, lifted(triangle, 1.0)])
    with pytest.raises(ValueError, match="same way round"):
        loft([CIRCLE, lifted([c.reversed() for c in reversed(CIRCLE)], 1.0)])
    bent = lifted(CIRCLE, 0.0)
    bent[0] = NurbsCurve(bent[0].points + [[0, 0, 0], [0, 0, 0.3], [0, 0, 0]], bent[0].weights,
                         bent[0].knots, 2)
    with pytest.raises(ValueError, match="planar"):
        loft([bent, lifted(CIRCLE, 1.0), lifted(CIRCLE, 2.0)])
