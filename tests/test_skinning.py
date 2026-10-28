import numpy as np
import pytest

from cadkernel.geometry.entities import NurbsCurve, circle_arcs, extruded_surface
from cadkernel.geometry.skinning import compatible, skin_parameters, skinned_surface

T = np.linspace(0.0, 1.0, 101)
ARC = circle_arcs([0, 0], 1.0)[0]  # degree 2, rational, no interior knots
CUBIC = NurbsCurve([[1, 0], [1.2, 0.3], [0.8, 0.6], [0.3, 0.9], [0, 1]], np.ones(5),
                   [0, 0, 0, 0, 0.3, 1, 1, 1, 1], 3)
LINE = NurbsCurve.line([1, 0], [0, 1])


def test_compatible_keeps_every_shape_and_lines_up_the_representation():
    """After compatible(): one degree, one knot vector, one number of control points -- and every
    curve traces the same points at the same parameters as before."""
    curves = [ARC, CUBIC, LINE]
    out = compatible(curves)
    assert {c.p for c in out} == {3}
    assert all(np.array_equal(c.knots, out[0].knots) for c in out)
    assert len({len(c.points) for c in out}) == 1
    for before, after in zip(curves, out):
        assert np.allclose(before(T), after(T), atol=1e-14)


def test_skin_passes_through_every_section():
    """S(u, v_s) = C^s(u): the isocurve at each section's parameter is that section, exactly."""
    sections = compatible([ARC, CUBIC.translated([0, 0, 1]), LINE.translated([0, 0, 1.5]),
                           ARC.translated([0, 0, 3.0])])
    params = skin_parameters([np.array([c.points[i] for c in sections])
                              for i in range(len(sections[0].points))])
    surface = skinned_surface(sections, 3, params)
    for curve, v in zip(sections, params):
        assert np.allclose(surface(T, v)[:, 0], curve(T), atol=1e-13)


def test_two_sections_give_the_ruled_surface():
    """Two sections a vector d apart, degree 1 across: the net is [P, P + d], the extruded
    surface of Extrude.md -- so two equal circles skin to the exact cylinder."""
    d = np.array([0.2, 0.1, 2.0])
    surface = skinned_surface([ARC, ARC.translated(d)], 1, np.array([0.0, 1.0]))
    reference = extruded_surface(ARC, d)
    assert np.allclose(surface.net, reference.net) and np.allclose(surface.weights, reference.weights)
    assert np.allclose(surface.knots_v, reference.knots_v)


def test_too_few_sections_for_the_degree():
    with pytest.raises(ValueError, match="at least 3"):
        skinned_surface([ARC, ARC.translated([0, 0, 1])], 2, np.array([0.0, 1.0]))
