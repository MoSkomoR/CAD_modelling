"""Geometry objects that topology can carry: a NURBS curve on an edge, a surface on a face.

Modules 01 and 02 work with bare arrays -- (control points, weights, knots, degree) passed to
functions. A B-rep needs to *attach* geometry to its edges and faces and ask the same few
questions of all of it, whatever its kind: where is the point at parameter t, what are the
derivatives there, how far is this point from you. These classes bundle the arrays and answer
those questions by calling the existing functions (nurbs_curve, nurbs_surface, ...); no new
evaluation algorithm lives here.

Conventions: every curve and surface is clamped with knots normalized to [0, 1], so a parameter
always runs over [0, 1]. Points are 3D.

Notes: notes/04_Modeling_Operations/Modeling_Operations.md, notes/04_Modeling_Operations/Extrude.md,
       notes/04_Modeling_Operations/Revolve.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bspline import clamped_knots
from .nurbs import nurbs_arc, nurbs_curve, nurbs_derivative, rational_arc_quadratic
from .nurbs_surfaces import nurbs_surface, nurbs_surface_partials, surface_of_revolution


def _as_3d(points) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] == 2:
        points = np.c_[points, np.zeros(len(points))]
    return points


@dataclass(frozen=True)
class NurbsCurve:
    """C(t) = sum_i R_{i,p}(t) P_i on t in [0, 1]: the arrays of nurbs.py, kept together."""

    points: np.ndarray
    weights: np.ndarray
    knots: np.ndarray
    p: int

    def __post_init__(self):
        object.__setattr__(self, "points", _as_3d(self.points))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))
        object.__setattr__(self, "knots", np.asarray(self.knots, dtype=float))

    def __call__(self, t) -> np.ndarray:
        return nurbs_curve(self.points, self.weights, self.knots, self.p, np.atleast_1d(t))

    def derivative(self, t) -> np.ndarray:
        return nurbs_derivative(self.points, self.weights, self.knots, self.p, np.atleast_1d(t))

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    @property
    def closed(self) -> bool:
        return bool(np.allclose(self.start, self.end, atol=1e-9))

    def spans(self) -> np.ndarray:
        """The distinct knots: inside each span the curve is a single rational polynomial."""
        return np.unique(self.knots)

    def reversed(self) -> NurbsCurve:
        """The same point set traversed backwards: C(1 - t). Reverse the control data and
        reflect the knots, t -> 1 - t."""
        return NurbsCurve(self.points[::-1], self.weights[::-1], 1.0 - self.knots[::-1], self.p)

    def translated(self, d) -> NurbsCurve:
        """Translation is affine, so moving the control points moves the curve exactly."""
        return NurbsCurve(self.points + np.asarray(d, dtype=float), self.weights, self.knots,
                          self.p)

    def rotated(self, angle: float) -> NurbsCurve:
        """Rotation about the z-axis by `angle`: linear, so rotating the control points (weights
        unchanged) rotates the curve exactly."""
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return NurbsCurve(self.points @ rotation.T, self.weights, self.knots, self.p)

    @classmethod
    def line(cls, a, b) -> NurbsCurve:
        return cls(np.array([a, b], dtype=float), np.ones(2), [0, 0, 1, 1], 1)

    @classmethod
    def bezier(cls, points) -> NurbsCurve:
        """A polynomial Bezier curve of degree len(points) - 1, as a NURBS with no interior
        knots and unit weights."""
        points = _as_3d(points)
        return cls(points, np.ones(len(points)), clamped_knots(len(points), len(points) - 1),
                   len(points) - 1)


def circle_arcs(center, radius: float, pieces: int = 4) -> list[NurbsCurve]:
    """A full circle in the xy-plane as `pieces` exact rational quadratic arcs, counter-clockwise.

    One closed curve would need one edge that starts and ends at the same vertex, which the
    Euler operators of module 03 cannot make (mev always creates a second vertex). Splitting the
    circle into arcs gives each piece its own two vertices; the price is extra edges and faces.
    """
    center = np.asarray(center, dtype=float)[:2]
    angles = np.linspace(0.0, 2 * np.pi, pieces + 1)
    arcs = []
    for a0, a1 in zip(angles[:-1], angles[1:]):
        points, weights = rational_arc_quadratic(center, radius, a0, a1)
        arcs.append(NurbsCurve(points, weights, [0, 0, 0, 1, 1, 1], 2))
    return arcs


def xz(curves) -> list[NurbsCurve]:
    """Curves drawn in a 2D sketch with coordinates (r, z), placed in the xz-plane: (r, 0, z).
    Revolve takes its profile there, with r the distance from the z-axis."""
    return [NurbsCurve(np.c_[c.points[:, 0], np.zeros(len(c.points)), c.points[:, 1]],
                       c.weights, c.knots, c.p) for c in curves]


def turning_arc(point, angle: float) -> NurbsCurve:
    """The arc that `point` traces when turned about the z-axis by `angle`, starting at its own
    polar angle: an exact degree-2 NURBS (nurbs_arc), closed for angle = 2 pi."""
    point = np.asarray(point, dtype=float)
    radius = np.hypot(point[0], point[1])
    arc, weights, knots, p = nurbs_arc(np.zeros(2), radius, angle)
    curve = NurbsCurve(np.c_[arc, np.full(len(arc), point[2])], weights, knots, p)
    return curve.rotated(np.arctan2(point[1], point[0]))


def polygon_segments(corners) -> list[NurbsCurve]:
    """A closed polygon as straight segments, corner k to corner k + 1."""
    corners = _as_3d(corners)
    return [NurbsCurve.line(a, b) for a, b in zip(corners, np.roll(corners, -1, axis=0))]


@dataclass(frozen=True)
class Plane:
    """The plane through `origin` with unit normal `normal`. Unbounded: a planar face is the
    region of it that the face's loops enclose, so the face's extent lives in its edges."""

    origin: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=float)
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float))
        object.__setattr__(self, "normal", n / np.linalg.norm(n))

    def distance(self, points) -> np.ndarray:
        return np.abs((np.atleast_2d(points) - self.origin) @ self.normal)


@dataclass(frozen=True)
class NurbsSurface:
    """S(u, v) = sum_ij R_ij(u, v) P_ij on [0, 1]^2: the arrays of nurbs_surfaces.py, together."""

    net: np.ndarray
    weights: np.ndarray
    knots_u: np.ndarray
    knots_v: np.ndarray
    p: int
    q: int

    def __post_init__(self):
        for name in ("net", "weights", "knots_u", "knots_v"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))

    def _arrays(self):
        return self.net, self.weights, self.knots_u, self.knots_v, self.p, self.q

    def __call__(self, u, v) -> np.ndarray:
        """Points on the grid u x v, shape (len(u), len(v), 3)."""
        return nurbs_surface(*self._arrays(), np.atleast_1d(u), np.atleast_1d(v))

    def partials(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        """S_u and S_v on the grid u x v."""
        return nurbs_surface_partials(*self._arrays(), np.atleast_1d(u), np.atleast_1d(v))

    def spans(self) -> tuple[np.ndarray, np.ndarray]:
        return np.unique(self.knots_u), np.unique(self.knots_v)

    def closest_parameters(self, x, iterations: int = 30) -> tuple[float, float]:
        """(u, v) of the surface point nearest to x: a grid search, then Gauss-Newton.

        Gauss-Newton linearizes S(u + du, v + dv) ~ S + S_u du + S_v dv and solves the 3 x 2
        least-squares system [S_u S_v] (du, dv) = x - S. When x lies on the surface the residual
        goes to zero, and Gauss-Newton then converges like Newton's method (quadratically; a
        standard result, cited). That is the case the consistency check needs: is this edge
        point on this face?
        """
        x = np.asarray(x, dtype=float)
        grid = np.linspace(0.0, 1.0, 17)
        samples = self(grid, grid)
        i, j = np.unravel_index(np.argmin(np.linalg.norm(samples - x, axis=-1)), samples.shape[:2])
        u, v = grid[i], grid[j]
        for _ in range(iterations):
            s = self(u, v)[0, 0]
            s_u, s_v = (d[0, 0] for d in self.partials(u, v))
            step, *_ = np.linalg.lstsq(np.c_[s_u, s_v], x - s, rcond=None)
            u, v = np.clip(u + step[0], 0.0, 1.0), np.clip(v + step[1], 0.0, 1.0)
            if np.abs(step).max() < 1e-14:
                break
        return float(u), float(v)

    def distance(self, points) -> np.ndarray:
        points = np.atleast_2d(points)
        return np.array([np.linalg.norm(self(*self.closest_parameters(x))[0, 0] - x)
                         for x in points])


def extruded_surface(curve: NurbsCurve, d) -> NurbsSurface:
    """The surface swept by `curve` translated along the vector d: S(u, v) = C(u) + v d.

    Exactly a NURBS surface of degree (p, 1): the net has two columns, the curve's control points
    and the same points moved by d, with the curve's weights in both. Because the rational basis
    R_i(u) sums to 1, sum_i R_i(u) (P_i + v d) = C(u) + v d. Derived in Extrude.md.
    """
    d = np.asarray(d, dtype=float)
    net = np.stack([curve.points, curve.points + d], axis=1)
    weights = np.stack([curve.weights, curve.weights], axis=1)
    return NurbsSurface(net, weights, curve.knots, [0, 0, 1, 1], curve.p, 1)


def revolved_surface(curve: NurbsCurve, angle: float = 2 * np.pi) -> NurbsSurface:
    """The surface swept by a curve in the xz-plane (y = 0, x >= 0) turning about the z-axis by
    `angle`: surface_of_revolution of module 02, with u along the curve and v around the axis,
    S(u, v) = (x(u) cos phi(v), x(u) sin phi(v), z(u)), where (x(u), 0, z(u)) is the curve and
    phi(v) increases from 0 to `angle` (the NURBS arc's angle, not proportional to v)."""
    if np.abs(curve.points[:, 1]).max() > 1e-12:
        raise ValueError("a revolved curve must lie in the xz-plane")
    arrays = surface_of_revolution(curve.points[:, [0, 2]], curve.weights, curve.knots, curve.p,
                                   angle)
    return NurbsSurface(*arrays)
