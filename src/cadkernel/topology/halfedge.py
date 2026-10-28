"""Half-edge boundary representation (B-rep) of polyhedral solids, built with Euler operators.

Topology is *who touches whom*: which faces meet along which edges, which edges meet at which
vertices. Geometry is *where*: vertex positions, and optionally a curve on each edge and a surface
on each face (bound by the modeling operations of module 04; a solid built here in module 03 has
none and is polyhedral). The two are kept apart so the same connectivity can carry any geometry.

The structure is Mantyla's half-edge data structure (An Introduction to Solid Modeling, 1988):

    Solid  -> faces, edges, vertices
    Face   -> loops: one outer loop, plus inner loops ("rings") for holes in the face
    Loop   -> one of its half-edges; the rest follow by `next`
    HalfEdge -> origin vertex, its loop, next / prev in that loop, and its edge
    Edge   -> its two half-edges, one per side; each is the other's `twin`
    Vertex -> position, and one outgoing half-edge

Every edge is split into two directed half-edges, one in each of the two faces it bounds, running
in opposite directions. That one decision makes every adjacency query a walk of pointer steps:
the face across an edge is `he.twin.loop.face`, the edges around a vertex are `he.twin.next`
repeated. Faces are oriented counter-clockwise seen from outside (outward normal by the right-hand
rule); inner loops run the other way, so the face material is always on the left of a half-edge.

Solids are only ever built and changed through *Euler operators* (mvfs, mev, mef, kemr, kfmrh and
their inverses). Each one changes the counts (V, E, F, R, S, G) in a way that keeps the
Euler-Poincare formula

    V - E + F - R = 2 (S - G)        (R = rings = inner loops, S = shells, G = genus)

true, so every intermediate state is a valid topology by construction.

Notes: notes/03_Topology/B_Rep_Topology.md, notes/03_Topology/Half_Edge.md,
       notes/03_Topology/Euler_Operators.md
"""
from __future__ import annotations

from itertools import count

import numpy as np

_ids = count()


class Vertex:
    __slots__ = ("point", "he", "id")

    def __init__(self, point):
        self.point = np.asarray(point, dtype=float)
        self.he: HalfEdge | None = None  # one outgoing half-edge (None while isolated)
        self.id = next(_ids)

    def __repr__(self):
        return f"V{self.id}"


class Edge:
    __slots__ = ("he1", "he2", "id", "curve")

    def __init__(self):
        self.he1: HalfEdge | None = None
        self.he2: HalfEdge | None = None
        self.id = next(_ids)
        self.curve = None  # optional geometry; runs from he1.origin to he2.origin (attach_curve)


class HalfEdge:
    __slots__ = ("origin", "edge", "loop", "next", "prev")

    def __init__(self, origin: Vertex, loop: Loop):
        self.origin = origin
        self.edge: Edge | None = None  # None only for the lone half-edge of a bare-vertex loop
        self.loop = loop
        self.next: HalfEdge = self
        self.prev: HalfEdge = self

    @property
    def twin(self) -> HalfEdge:
        return self.edge.he2 if self is self.edge.he1 else self.edge.he1

    @property
    def target(self) -> Vertex:
        return self.next.origin

    @property
    def face(self) -> Face:
        return self.loop.face

    def __repr__(self):
        return f"{self.origin}->{self.target}"


class Loop:
    __slots__ = ("face", "he")

    def __init__(self, face: Face):
        self.face = face
        self.he: HalfEdge | None = None

    def halfedges(self) -> list[HalfEdge]:
        out, he = [], self.he
        while True:
            out.append(he)
            he = he.next
            if he is self.he or len(out) > 100_000:
                return out

    def vertices(self) -> list[Vertex]:
        return [he.origin for he in self.halfedges()]


class Face:
    __slots__ = ("solid", "outer", "loops", "id", "surface", "sense")

    def __init__(self, solid: Solid):
        self.solid = solid
        self.outer: Loop | None = None
        self.loops: list[Loop] = []
        self.id = next(_ids)
        self.surface = None  # optional geometry: a Plane or a NurbsSurface (geometry/entities.py)
        self.sense = 1  # +1 if S_u x S_v points out of the solid, -1 if into it
        solid.faces.append(self)

    @property
    def rings(self) -> list[Loop]:
        return [loop for loop in self.loops if loop is not self.outer]

    def new_loop(self) -> Loop:
        loop = Loop(self)
        self.loops.append(loop)
        if self.outer is None:
            self.outer = loop
        return loop

    def __repr__(self):
        return f"F{self.id}"


class Solid:
    def __init__(self):
        self.faces: list[Face] = []
        self.edges: list[Edge] = []
        self.vertices: list[Vertex] = []
        self.genus = 0  # tracked through kfmrh / mfkrh: topology alone does not store it

    def halfedges(self) -> list[HalfEdge]:
        return [he for face in self.faces for loop in face.loops for he in loop.halfedges()]

    def counts(self) -> dict[str, int]:
        """V, E, F, L (all loops), R (rings = L - F), S (shells, by connectivity), G (genus)."""
        loops = sum(len(face.loops) for face in self.faces)
        return {"V": len(self.vertices), "E": len(self.edges), "F": len(self.faces),
                "L": loops, "R": loops - len(self.faces), "S": count_shells(self),
                "G": self.genus}


def euler_poincare(c: dict[str, int]) -> tuple[int, int]:
    """The two sides of V - E + F - R = 2 (S - G)."""
    return c["V"] - c["E"] + c["F"] - c["R"], 2 * (c["S"] - c["G"])


def count_shells(solid: Solid) -> int:
    """Connected boundary surfaces: vertices joined by edges, *and* the loops of one face joined
    to each other. Edges alone are not enough: a ring (the boundary of a hole in a face) can be
    connected to the rest of the solid only through the face it lies in."""
    parent = {v: v for v in solid.vertices}

    def find(v):
        while parent[v] is not v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for e in solid.edges:
        a, b = find(e.he1.origin), find(e.he2.origin)
        parent[a] = b
    for face in solid.faces:
        for ring in face.rings:
            a, b = find(face.outer.he.origin), find(ring.he.origin)
            parent[a] = b
    return len({find(v) for v in solid.vertices})


# --- Low-level Euler operators, on half-edges (after Mantyla's lmev, lmef, ...) -------------------


def _add_halfedge(edge: Edge, origin: Vertex, where: HalfEdge, first: bool) -> HalfEdge:
    """Insert a half-edge of `edge`, starting at `origin`, just before `where` in its loop.

    A bare-vertex loop (one half-edge with no edge) is reused instead: that is how the first edge
    grows out of the seed vertex of mvfs.
    """
    if where.edge is None:
        he = where
    else:
        he = HalfEdge(origin, where.loop)
        he.next, he.prev = where, where.prev
        where.prev.next = he
        where.prev = he
    he.edge, he.origin = edge, origin
    if first:
        edge.he1 = he
    else:
        edge.he2 = he
    return he


def _remove_halfedge(he: HalfEdge) -> HalfEdge:
    """Unlink `he` from its loop; return a half-edge still in the loop (the loop's new anchor).

    If `he` was alone, it stays as the bare-vertex half-edge of its loop (edge = None).
    """
    if he.next is he:
        he.edge = None
        return he
    he.prev.next = he.next
    he.next.prev = he.prev
    return he.prev


def mvfs(point) -> tuple[Solid, Face, Vertex]:
    """Make vertex, face, solid: one vertex and one face whose only loop is that bare vertex.

    Counts: V+1, F+1, S+1. Not a solid anyone would draw -- the seed every construction grows from.
    """
    solid = Solid()
    face = Face(solid)
    loop = face.new_loop()
    v = Vertex(point)
    solid.vertices.append(v)
    loop.he = HalfEdge(v, loop)
    return solid, face, v


def kvfs(solid: Solid) -> None:
    """Inverse of mvfs: only allowed on the bare seed."""
    if len(solid.vertices) != 1 or solid.edges or len(solid.faces) != 1:
        raise ValueError("kvfs needs a solid that is a single bare vertex")
    solid.vertices.clear()
    solid.faces.clear()


def lmev(he1: HalfEdge, he2: HalfEdge, point) -> Vertex:
    """Make edge and vertex: split the vertex he1.origin, moving the half-edges from he1 round to
    he2 (exclusive) onto a new vertex at `point`, joined to the old one by a new edge.

    With he1 == he2 nothing moves, and the new edge is a "spike" from he1.origin into he1's loop:
    the loop gains old->new and new->old just before he1. Counts: V+1, E+1.
    """
    solid = he1.loop.face.solid
    new = Vertex(point)
    solid.vertices.append(new)
    edge = Edge()
    solid.edges.append(edge)
    he = he1
    while he is not he2:
        he.origin = new
        he = he.twin.next
    old = he2.origin
    a = _add_halfedge(edge, old, he1, first=False)
    b = _add_halfedge(edge, new, he2, first=True)
    new.he = b if b.edge is not None else a
    old.he = he2 if he2.edge is not None else a
    if old.he.origin is not old:
        old.he = a
    return new


def lkev(he1: HalfEdge, he2: HalfEdge) -> None:
    """Inverse of lmev: kill the edge of he1/he2 (twins) and the vertex he1.origin, whose other
    half-edges are handed to he2.origin. Counts: V-1, E-1."""
    solid = he1.loop.face.solid
    dead, keep = he1.origin, he2.origin
    he = he2.next
    while he is not he1:
        he.origin = keep
        he = he.twin.next
    edge = he1.edge
    he1.loop.he = _remove_halfedge(he1)
    he2.loop.he = _remove_halfedge(he2)
    solid.edges.remove(edge)
    solid.vertices.remove(dead)
    keep.he = _outgoing(solid, keep)


def lmef(he1: HalfEdge, he2: HalfEdge) -> Face:
    """Make edge and face: split the loop of he1 and he2 by a new edge from he2.origin to
    he1.origin. The half-edges from he1 up to he2 (exclusive), closed by the new edge, become the
    outer loop of a new face; the rest stay. Counts: E+1, F+1."""
    if he1.loop is not he2.loop:
        raise ValueError("lmef needs two half-edges of the same loop")
    old_loop = he1.loop
    solid = old_loop.face.solid
    face = Face(solid)
    loop = face.new_loop()
    edge = Edge()
    solid.edges.append(edge)
    he = he1
    while he is not he2:
        he.loop = loop
        he = he.next
    n1 = _add_halfedge(edge, he2.origin, he1, first=False)
    n2 = _add_halfedge(edge, he1.origin, he2, first=True)
    n1.prev.next = n2
    n2.prev.next = n1
    n1.prev, n2.prev = n2.prev, n1.prev
    loop.he = n1
    old_loop.he = n2
    he1.origin.he, he2.origin.he = n2, n1
    return face


def lkef(he1: HalfEdge, he2: HalfEdge) -> None:
    """Inverse of lmef: kill the edge of he1/he2 (twins, in different faces) and he2's face,
    merging its loop into he1's. Counts: E-1, F-1."""
    loop1, loop2 = he1.loop, he2.loop
    face1, face2 = loop1.face, loop2.face
    if face1 is face2:
        raise ValueError("lkef needs an edge between two different faces")
    solid = face1.solid
    for he in loop2.halfedges():
        he.loop = loop1
    he1.prev.next = he2
    he2.prev.next = he1
    he1.prev, he2.prev = he2.prev, he1.prev
    a, b = he1.origin, he2.origin
    _remove_halfedge(he2)
    loop1.he = _remove_halfedge(he1)
    solid.edges.remove(he1.edge)
    for ring in face2.rings:  # face2's holes move with it
        ring.face = face1
        face1.loops.append(ring)
    solid.faces.remove(face2)
    a.he, b.he = _outgoing(solid, a), _outgoing(solid, b)


def lkemr(he1: HalfEdge, he2: HalfEdge) -> Loop:
    """Kill edge, make ring: the edge of he1/he2 (twins, both in the same loop) is a bridge;
    removing it splits the loop in two, and the part starting after he2 becomes an inner loop of
    the same face. Counts: E-1, R+1."""
    old = he1.loop
    if he2.loop is not old:
        raise ValueError("lkemr needs an edge whose two sides are in the same loop")
    ring = old.face.new_loop()
    h3 = he1.next
    he1.next = he2.next
    he2.next.prev = he1
    he2.next = h3
    h3.prev = he2
    he = he2
    while True:
        he.loop = ring
        he = he.next
        if he is he2:
            break
    a, b = he1.origin, he2.origin
    old.he = _remove_halfedge(he1)
    ring.he = _remove_halfedge(he2)
    solid = old.face.solid
    solid.edges.remove(he1.edge)
    a.he, b.he = _outgoing(solid, a), _outgoing(solid, b)
    return ring


def lmekr(he1: HalfEdge, he2: HalfEdge) -> None:
    """Inverse of lkemr: a new edge from he1.origin to he2.origin joins he1's loop and the ring
    holding he2 into one loop. Counts: E+1, R-1."""
    loop1, ring = he1.loop, he2.loop
    if loop1.face is not ring.face or loop1 is ring:
        raise ValueError("lmekr needs two different loops of the same face")
    solid = loop1.face.solid
    edge = Edge()
    solid.edges.append(edge)
    for he in ring.halfedges():
        he.loop = loop1
    n1 = _add_halfedge(edge, he1.origin, he1, first=True)
    n2 = _add_halfedge(edge, he2.origin, he2, first=False)
    n1.next, n2.next = he2, he1
    he2.prev, he1.prev = n1, n2
    ring.face.loops.remove(ring)
    if ring.face.outer is ring:
        ring.face.outer = loop1


def lkfmrh(keep: Face, kill: Face) -> None:
    """Kill face, make ring and hole: `kill`'s outer loop becomes an inner loop of `keep`, and
    the solid gains a handle (genus + 1). Counts: F-1, R+1, G+1."""
    if kill.rings:
        raise ValueError("lkfmrh needs a face without rings")
    loop = kill.outer
    loop.face = keep
    keep.loops.append(loop)
    keep.solid.faces.remove(kill)
    keep.solid.genus += 1


def lmfkrh(ring: Loop) -> Face:
    """Inverse of lkfmrh: the ring becomes the outer loop of a new face. Counts: F+1, R-1, G-1."""
    old = ring.face
    if ring is old.outer:
        raise ValueError("lmfkrh needs an inner loop")
    old.loops.remove(ring)
    face = Face(old.solid)
    face.loops.append(ring)
    face.outer = ring
    ring.face = face
    old.solid.genus -= 1
    return face


def _outgoing(solid: Solid, v: Vertex) -> HalfEdge | None:
    for face in solid.faces:
        for loop in face.loops:
            for he in loop.halfedges():
                if he.origin is v:
                    return he
    return None


# --- Addressing by vertices: what builders and users actually say ----------------------------------


def find_halfedge(face: Face, origin: Vertex, target: Vertex | None = None) -> HalfEdge:
    """The half-edge of `face` leaving `origin` (towards `target`, if given). Must be unique."""
    found = [he for loop in face.loops for he in loop.halfedges()
             if he.origin is origin and (target is None or he.target is target)]
    if len(found) != 1:
        raise ValueError(f"{len(found)} half-edges of {face} leave {origin} towards {target}")
    return found[0]


def mev(face: Face, v: Vertex, point, towards: Vertex | None = None) -> Vertex:
    """A spike: new edge from v (inside `face`) to a new vertex at `point`."""
    he = find_halfedge(face, v, towards)
    return lmev(he, he, point)


def mef(face: Face, v1: Vertex, v2: Vertex, v1_to: Vertex | None = None,
        v2_to: Vertex | None = None) -> Face:
    """New edge v2 -> v1 across `face`; the new face takes the boundary path from v1 to v2."""
    return lmef(find_halfedge(face, v1, v1_to), find_halfedge(face, v2, v2_to))


def kemr(face: Face, v1: Vertex, v2: Vertex) -> Loop:
    """Remove the bridge edge v1 - v2 of `face`, leaving the far side as a ring."""
    return lkemr(find_halfedge(face, v1, v2), find_halfedge(face, v2, v1))


# --- Queries ---------------------------------------------------------------------------------------


def vertex_star(v: Vertex) -> list[HalfEdge]:
    """Outgoing half-edges around v, in order: he -> he.twin.next -> ... Constant work per edge."""
    out, he = [], v.he
    if he is None or he.edge is None:
        return []
    while True:
        out.append(he)
        he = he.twin.next
        if he is v.he or len(out) > 10_000:
            return out


def edge_faces(edge: Edge) -> tuple[Face, Face]:
    return edge.he1.face, edge.he2.face


def adjacent_faces(face: Face) -> list[Face]:
    """Faces across each edge of `face`: one twin step per half-edge, no search."""
    out = []
    for loop in face.loops:
        for he in loop.halfedges():
            if he.edge is not None and he.twin.face not in out and he.twin.face is not face:
                out.append(he.twin.face)
    return out


def loop_points(loop: Loop) -> np.ndarray:
    return np.array([v.point for v in loop.vertices()])


def face_normal(face: Face) -> np.ndarray:
    """Newell's vector area of the outer loop, normalized: robust for any planar polygon."""
    p = loop_points(face.outer)
    q = np.roll(p, -1, axis=0)
    n = np.cross(p, q).sum(axis=0)
    length = np.linalg.norm(n)
    return n / length if length > 0 else n


def signed_volume(solid: Solid) -> float:
    """Divergence theorem: V = 1/3 sum_f (vector area_f . point on f), rings included.

    Positive iff the faces are oriented outward (counter-clockwise seen from outside).
    """
    total = 0.0
    for face in solid.faces:
        area = sum(0.5 * np.cross(p := loop_points(loop), np.roll(p, -1, axis=0)).sum(axis=0)
                   for loop in face.loops)
        total += np.dot(area, face.outer.he.origin.point) / 3.0
    return float(total)


# --- Binding geometry ------------------------------------------------------------------------------


def attach_curve(edge: Edge, curve, tol: float = 1e-9, along: HalfEdge | None = None) -> None:
    """Give `edge` its curve, oriented to run from edge.he1.origin to edge.he2.origin.

    The direction is fixed by the topology, not by how the caller happened to build the curve:
    then a half-edge traverses the curve forwards exactly when it is he1, which is what loop
    integrals (vector area, volume) need. A closed curve (a full circle on a closed edge, whose
    two ends are the same vertex) cannot be oriented by its endpoints: the caller says which
    half-edge `along` runs with the curve, and it is stored so that he1 does.
    """
    a, b = edge.he1.origin.point, edge.he2.origin.point
    if np.allclose(curve.start, curve.end, atol=tol):
        if along is None or along.edge is not edge:
            raise ValueError("a closed curve needs `along`: the half-edge that runs with it")
        if not np.allclose(curve.start, a, atol=tol):
            raise ValueError("the closed curve does not pass through the edge's vertex")
        edge.curve = curve if along is edge.he1 else curve.reversed()
    elif np.allclose(curve.start, a, atol=tol) and np.allclose(curve.end, b, atol=tol):
        edge.curve = curve
    elif np.allclose(curve.start, b, atol=tol) and np.allclose(curve.end, a, atol=tol):
        edge.curve = curve.reversed()
    else:
        raise ValueError("the curve's endpoints are not the edge's vertices")


def _geometry_problems(solid: Solid, tol: float, samples: int = 5) -> list[str]:
    """Is the bound geometry consistent with the topology?

    A vertex of a face must lie on the face's surface; an edge curve must start and end at its
    vertices and lie on the surfaces of *both* faces it separates. That last condition is what
    makes a B-rep watertight: the two faces meet exactly along the edge, with no gap or overlap.
    Tested at a few interior parameters of each curve -- a check, not a proof.
    """
    problems = []
    t = np.linspace(0.0, 1.0, samples + 2)[1:-1]
    for face in solid.faces:
        if face.surface is None:
            continue
        p = np.vstack([loop_points(loop) for loop in face.loops])
        if face.surface.distance(p).max() > tol:
            problems.append(f"a vertex of {face} is off its surface")
    for edge in solid.edges:
        if edge.curve is None:
            continue
        a, b = edge.he1.origin.point, edge.he2.origin.point
        if not (np.allclose(edge.curve.start, a, atol=tol)
                and np.allclose(edge.curve.end, b, atol=tol)):
            problems.append(f"curve of edge {edge.id} does not run between its vertices")
        inside = edge.curve(t)
        for face in edge_faces(edge):
            if face.surface is not None and face.surface.distance(inside).max() > tol:
                problems.append(f"curve of edge {edge.id} leaves the surface of {face}")
    return problems


# --- Validity --------------------------------------------------------------------------------------


def check(solid: Solid, geometric: bool = True, planar_tol: float = 1e-9,
          geometry_tol: float = 1e-7) -> list[str]:
    """Every structural invariant, as a list of violations (empty = valid).

    `geometric=False` checks topology only. Intermediate states of a construction (a wire, a
    lamina, a face with spikes sticking out of its plane) are valid topology but not valid
    geometry, so the construction steps are checked that way. With geometry bound (module 04),
    faces carrying a surface are checked against it instead of for planarity.
    """
    problems = []
    halfedges = solid.halfedges()
    for he in halfedges:
        if he.next.prev is not he or he.prev.next is not he:
            problems.append(f"next/prev not inverse at {he}")
        if he.loop.face not in solid.faces:
            problems.append(f"{he} belongs to a face not in the solid")
        if he.edge is None:
            continue
        if he.twin.twin is not he:
            problems.append(f"twin not an involution at {he}")
        if he.twin.origin is not he.target:
            problems.append(f"twin of {he} does not run the other way")
    edge_halves = {id(he) for e in solid.edges for he in (e.he1, e.he2)}
    real = [he for he in halfedges if he.edge is not None]
    if {id(he) for he in real} != edge_halves:
        problems.append("half-edges in loops and half-edges of edges differ")
    for face in solid.faces:
        if face.outer not in face.loops:
            problems.append(f"{face} outer loop missing")
        for loop in face.loops:
            if loop.face is not face or any(he.loop is not loop for he in loop.halfedges()):
                problems.append(f"loop of {face} has foreign half-edges")
        p = np.vstack([loop_points(loop) for loop in face.loops])
        if geometric and face.surface is None and len(p) >= 3:
            n = face_normal(face)
            if np.linalg.norm(n) > 0 and np.abs((p - p[0]) @ n).max() > planar_tol:
                problems.append(f"{face} is not planar")
    for v in solid.vertices:
        outgoing = sum(he.origin is v for he in real)
        star = vertex_star(v)
        if any(he.origin is not v for he in star):
            problems.append(f"star of {v} leaves the vertex")
        elif len(star) != outgoing:
            problems.append(f"{v} is non-manifold: star of {len(star)}, {outgoing} outgoing")
    lhs, rhs = euler_poincare(solid.counts())
    if lhs != rhs:
        problems.append(f"Euler-Poincare fails: {lhs} != {rhs}")
    if geometric:
        problems += _geometry_problems(solid, geometry_tol)
    return problems


# --- Building from face lists ----------------------------------------------------------------------


def from_polygons(points, faces, genus: int = 0) -> Solid:
    """Half-edge solid from face-vertex lists (each face counter-clockwise from outside).

    Twins are paired by matching each directed edge (a, b) with (b, a). A directed edge used twice
    means inconsistent orientation or a non-manifold edge; one with no partner means a hole in the
    surface. Both are rejected. Genus cannot be read off the lists without the Euler-Poincare
    formula itself, so it is passed in and the formula is then a genuine check.
    """
    solid = Solid()
    solid.genus = genus
    verts = [Vertex(p) for p in points]
    solid.vertices.extend(verts)
    directed: dict[tuple[int, int], HalfEdge] = {}
    for indices in faces:
        face = Face(solid)
        loop = face.new_loop()
        hes = [HalfEdge(verts[i], loop) for i in indices]
        for k, he in enumerate(hes):
            he.next, he.prev = hes[(k + 1) % len(hes)], hes[k - 1]
            key = (indices[k], indices[(k + 1) % len(indices)])
            if key in directed:
                raise ValueError(f"directed edge {key} used twice")
            directed[key] = he
            verts[indices[k]].he = he
        loop.he = hes[0]
    for (a, b), he in directed.items():
        if he.edge is not None:
            continue
        if (b, a) not in directed:
            raise ValueError(f"edge {(a, b)} has no twin: the surface is not closed")
        edge = Edge()
        edge.he1, edge.he2 = he, directed[(b, a)]
        he.edge = directed[(b, a)].edge = edge
        solid.edges.append(edge)
    return solid


# --- Constructions by Euler operators --------------------------------------------------------------


def snapshot(solid: Solid, label: str) -> dict:
    """Drawable copy of the current state: loops as point arrays, edges as segments, counts."""
    return {
        "label": label,
        "faces": [[loop_points(face.outer)] + [loop_points(r) for r in face.rings]
                  for face in solid.faces],
        "edges": [np.array([e.he1.origin.point, e.he2.origin.point]) for e in solid.edges],
        "vertices": np.array([v.point for v in solid.vertices]),
        "counts": solid.counts(),
        "problems": check(solid, geometric=False),
    }


def _record(steps, solid, label):
    if steps is not None:
        steps.append(snapshot(solid, label))


def _box(lo, hi, steps=None):
    """A box as a sweep: square lamina (mvfs, 3 mev, mef), 4 vertical mevs, 4 side mefs.

    The lamina starts at the bottom. Its first face, `cap`, is the one the vertical edges grow
    into, so after the sweep `cap` is the top face; the other lamina face stays as the bottom.
    The square runs counter-clockwise seen from above, which makes every face outward-oriented.
    """
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    solid, cap, v0 = mvfs((x0, y0, z0))
    _record(steps, solid, "mvfs: one vertex, one face, one shell")
    v1 = mev(cap, v0, (x1, y0, z0))
    v2 = mev(cap, v1, (x1, y1, z0))
    v3 = mev(cap, v2, (x0, y1, z0))
    _record(steps, solid, "3 x mev: a wire of three edges")
    bottom = mef(cap, v3, v0)
    _record(steps, solid, "mef: close the square -- a lamina, two faces back to back")
    up = [mev(cap, v, (*v.point[:2], z1)) for v in (v0, v1, v2, v3)]
    _record(steps, solid, "4 x mev: four vertical spikes")
    for a, b in zip(up, up[1:] + up[:1]):
        mef(cap, a, b)
        _record(steps, solid, "mef: a side face")
    return solid, cap, bottom, up


def cube(size: float = 1.0, steps: list | None = None) -> Solid:
    solid, *_ = _box((0, 0, 0), (size, size, size), steps)
    return solid


def tetrahedron(steps: list | None = None) -> Solid:
    p = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, 0.8]])
    solid, f, v0 = mvfs(p[0])
    _record(steps, solid, "mvfs")
    v1 = mev(f, v0, p[1])
    v2 = mev(f, v1, p[2])
    _record(steps, solid, "2 x mev: a wire")
    mef(f, v2, v0)
    _record(steps, solid, "mef: a triangle lamina")
    v3 = mev(f, v0, p[3])
    _record(steps, solid, "mev: a spike to the apex")
    mef(f, v3, v1)
    _record(steps, solid, "mef: a side face")
    mef(f, v3, v2)
    _record(steps, solid, "mef: the last face")
    return solid


def block_with_hole(steps: list | None = None) -> Solid:
    """A 3 x 3 x 1 block with a 1 x 1 square through-hole: genus 1.

    Box; on the top face, a spike to the hole's corner and a square wire around it, closed by mef
    into a square face; kemr cuts the spike, leaving the square's boundary as a ring of the top;
    the square face is swept down through the block (4 mev, 4 mef) to the bottom; kfmrh merges it
    into the bottom face as a ring -- and that one step opens the hole.
    """
    solid, top, bottom, corners = _box((0, 0, 0), (3, 3, 1), steps)
    a = mev(top, corners[0], (1, 1, 1))
    b = mev(top, a, (2, 1, 1))
    c = mev(top, b, (2, 2, 1))
    d = mev(top, c, (1, 2, 1))
    _record(steps, solid, "5 x mev on the top: a spike and a square wire")
    plug = mef(top, a, d, v1_to=b)
    _record(steps, solid, "mef: close the square -- a new face inside the top")
    kemr(top, corners[0], a)
    _record(steps, solid, "kemr: cut the spike -- the square is now a ring of the top")
    low = [mev(plug, v, (*v.point[:2], 0.0)) for v in (a, b, c, d)]
    _record(steps, solid, "4 x mev: sweep the square face down")
    for p, q in zip(low, low[1:] + low[:1]):
        mef(plug, p, q)
        _record(steps, solid, "mef: a wall of the hole")
    lkfmrh(bottom, plug)
    _record(steps, solid, "kfmrh: the square becomes a ring of the bottom -- genus 1")
    return solid


def torus_grid(n: int = 8, m: int = 6, major: float = 2.0, minor: float = 0.7) -> Solid:
    """An n x m quad mesh on a torus via from_polygons: closed, genus 1, every face planar
    (each quad's two parallel-circle chords are parallel)."""
    points = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        for j in range(m):
            phi = 2 * np.pi * j / m
            r = major + minor * np.cos(phi)
            points.append((r * np.cos(theta), r * np.sin(theta), minor * np.sin(phi)))
    idx = lambda i, j: (i % n) * m + (j % m)
    faces = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
             for i in range(n) for j in range(m)]
    return from_polygons(points, faces, genus=1)


def octahedron() -> Solid:
    points = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
             [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    return from_polygons(points, faces)
