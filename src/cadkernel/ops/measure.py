"""Volume of a solid with curved faces: the divergence theorem, by Gauss-Legendre quadrature.

The polyhedral version is signed_volume in topology/halfedge.py: V = 1/3 sum_f A_f . p_f, with the
vector area A_f from the shoelace formula. Bound geometry changes two things:

  * a planar face's boundary is now made of curves, so its vector area is the loop integral
    A = 1/2 \\oint x \\times dx, taken edge by edge along each edge curve;
  * a curved face has no single normal, so its share of the flux integral is taken over its
    parameter domain: 1/3 \\iint S . (S_u \\times S_v) du dv.

Both integrands are smooth inside each knot span (rational, not polynomial, when weights are not
all equal), so the integrals are split at the knots and each span gets an n-point Gauss-Legendre
rule. Derived and measured in notes/04_Modeling_Operations/Extrude.md.
"""
from __future__ import annotations

import numpy as np

from ..geometry.entities import Plane
from ..topology.halfedge import HalfEdge, Loop, Solid

GAUSS_POINTS = 12


def _gauss_on_spans(breaks: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of an n-point Gauss-Legendre rule on every interval between `breaks`."""
    x, w = np.polynomial.legendre.leggauss(n)
    nodes, weights = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def halfedge_points(he: HalfEdge, t) -> np.ndarray:
    """Points along the half-edge, in its own direction: the edge curve forwards for he1,
    backwards for he2 (attach_curve guarantees the curve runs he1.origin -> he2.origin).
    Without a curve, the straight segment."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    curve = he.edge.curve if he.edge is not None else None
    if curve is None:
        a, b = he.origin.point, he.target.point
        return a + t[:, None] * (b - a)
    return curve(t) if he is he.edge.he1 else curve(1.0 - t)


def loop_vector_area(loop: Loop, n: int = GAUSS_POINTS) -> np.ndarray:
    """A = 1/2 \\oint x \\times dx around the loop, summed half-edge by half-edge.

    Over a curved edge: 1/2 \\int_0^1 C(t) \\times C'(t) dt, by quadrature on the knot spans,
    negated when the half-edge runs against the curve. Over a straight edge it is exact in closed
    form, 1/2 a \\times b -- the shoelace term of the polyhedral case.
    """
    total = np.zeros(3)
    for he in loop.halfedges():
        curve = he.edge.curve if he.edge is not None else None
        if curve is None:
            total += 0.5 * np.cross(he.origin.point, he.target.point)
            continue
        t, w = _gauss_on_spans(curve.spans(), n)
        piece = 0.5 * (w[:, None] * np.cross(curve(t), curve.derivative(t))).sum(axis=0)
        total += piece if he is he.edge.he1 else -piece
    return total


def surface_flux(surface, n: int = GAUSS_POINTS) -> float:
    """\\iint S . (S_u \\times S_v) du dv over [0, 1]^2, by tensor Gauss-Legendre per knot span."""
    spans_u, spans_v = surface.spans()
    u, wu = _gauss_on_spans(spans_u, n)
    v, wv = _gauss_on_spans(spans_v, n)
    s = surface(u, v)
    s_u, s_v = surface.partials(u, v)
    integrand = np.einsum("abk,abk->ab", s, np.cross(s_u, s_v))
    return float(wu @ integrand @ wv)


def surface_vector_area(surface, n: int = GAUSS_POINTS) -> np.ndarray:
    """\\iint (S_u \\times S_v) du dv over [0, 1]^2: the integral of the unit normal times the area
    element. By Stokes' theorem it equals 1/2 \\oint x \\times dx around the patch's boundary,
    traversed counter-clockwise in (u, v) -- which is how the tests check that a face's loop runs
    the way its sense says (Revolve.md, "Checking the orientation")."""
    spans_u, spans_v = surface.spans()
    u, wu = _gauss_on_spans(spans_u, n)
    v, wv = _gauss_on_spans(spans_v, n)
    s_u, s_v = surface.partials(u, v)
    return np.einsum("a,abk,b->k", wu, np.cross(s_u, s_v), wv)


def volume(solid: Solid, n: int = GAUSS_POINTS) -> float:
    """V = 1/3 \\oint_{boundary} x . n dA, face by face.

    Planar faces (a Plane, or no surface at all): x . n is constant over the face, so the share
    is 1/3 A . p with A the vector area of all its loops (rings subtract themselves, running the
    other way) and p any point of the face. Curved faces: 1/3 sense * surface_flux, which assumes
    the face is the *whole* patch [0, 1]^2 (untrimmed) -- true of every face extrude makes.
    Positive iff every face is oriented outward.
    """
    total = 0.0
    for face in solid.faces:
        if face.surface is None or isinstance(face.surface, Plane):
            area = sum(loop_vector_area(loop, n) for loop in face.loops)
            total += float(area @ face.outer.he.origin.point) / 3.0
        else:
            total += face.sense * surface_flux(face.surface, n) / 3.0
    return total
