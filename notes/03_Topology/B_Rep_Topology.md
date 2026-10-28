Code: [[../../scripts/03_topology/01_halfedge_navigation.py]], [[../../scripts/03_topology/02_euler_operators.py]] — `cadkernel/topology/halfedge.py`; tests in `tests/test_halfedge.py`

# B-rep topology — the motivation hub

Modules 01 and 02 produce shapes: curves and surfaces. A solid model needs more than a pile of
surfaces. This note is about the "more", and the two notes below it do the work:
[[Half_Edge]] (the data structure) and [[Euler_Operators]] (how solids are built and changed).

## What geometry alone cannot answer

Take six square patches placed as the faces of a cube. As geometry they are six unrelated
surfaces. A kernel needs to ask questions about how they relate:

- **Is it closed?** Does every edge of every face meet exactly one other face? Only then is
  there an inside, a volume, a solid.
- **What is next to what?** A fillet on an edge needs the two faces meeting there; a boolean has
  to know how the pieces it cuts connect; a tessellator must give two neighbouring faces the same
  points along their shared edge, or the mesh has cracks.
- **Which way is out?** Every face needs a consistent outward side.

Answering these from coordinates means comparing points within a tolerance, and that breaks the
moment two points are close but meant to be different. So a boundary representation (B-rep)
stores the answers explicitly, as **topology**: who is connected to whom, kept separate from
**geometry**, which says where things are.

The split pays three ways:
- The same topology can carry different geometry: a cube's 6 faces, 12 edges and 8 vertices can
  be bound to planes, or to curved patches from [[../02_Surfaces/NURBS_Surfaces]]. Geometry can
  be edited (a vertex moved) without touching connectivity.
- Topological validity is a matter of integers and pointers, so it can be checked **exactly**
  (`check` in `halfedge.py`). Geometry can only be checked to a tolerance.
- Algorithms can navigate without geometry at all, as [[Half_Edge]] shows.

In module 03, vertices carry positions and nothing else. Edges and faces get curves and surfaces
in module 04 ([[../04_Modeling_Operations/Modeling_Operations]]).

## The hierarchy

A solid is bounded by **shells** (closed surfaces; one, unless the solid has internal voids). A
shell is made of **faces**. A face is bounded by **loops**: one outer loop, plus one inner loop
("ring") for each hole in the face. A loop is a cycle of **edges**, and an edge joins two
**vertices**.

This repo follows Mäntylä, *An Introduction to Solid Modeling* (1988), where each edge is used
twice — once by each face it bounds — as two directed **half-edges**. Production kernels use the
same layers under other names. OpenCASCADE's `TopoDS` has solids, shells, faces, wires (loops),
edges and vertices; ACIS calls the half-edge a "coedge". Both are cited, not tested here. The repo
doesn't store shells as objects: every solid it builds has one, and `count_shells` counts them by
connectivity.

## Why the half-edge, and not something simpler

**A triangle soup** (the STL format) stores each face as its own corner coordinates and nothing
else. There is no adjacency at all. Even "is it closed?" means matching coordinates within a
tolerance — exactly the fragility topology exists to avoid.

**Face-vertex lists** (the OBJ and PLY formats) share vertices by index, so they *do* know which
faces meet at an edge. But nothing points from a face to its neighbours, so finding them means
inspecting every other face. Measured on closed quad meshes of rising size, finding the faces next
to one face took $15, 63, 255, 1023$ face inspections, against $4$ half-edge steps every time
(`test_adjacency_cost_face_lists_versus_half_edges`).

To be fair to face-vertex lists, a hash map from edges to faces would also make that lookup
constant time on average. That is half of a half-edge structure, built on the side. What it still
lacks:
- the *cyclic order*: the edges around a vertex in order, the edges of a face in order;
- a way to make local edits — split a face, add an edge, open a hole — that update only the few
  pointers involved ([[Euler_Operators]]).

**Winged edge** (Baumgart, 1975, cited) is the classic answer that came first. Each edge stores
its two faces, its two vertices and the four neighbouring "wing" edges. It works, but an edge has
one direction and two faces, so every traversal step has to test which side it is on before it
knows which wing to follow. The half-edge splits the edge into two directed halves, one per face.
Each half has exactly one face and one direction, so traversal needs no case analysis: `next`,
`prev` and `twin` are always the right pointers ([[Half_Edge]]).

## The manifold assumption

Everything here assumes a **2-manifold** boundary:
- every edge has exactly two faces;
- the faces around every vertex form a single fan.

That is what makes "the face across this edge" and "the next edge around this vertex" well-defined.
`check` rejects violations. Two tetrahedra touching at one vertex have a twin for every edge, but
the shared vertex has two separate fans, and `check` reports it as non-manifold
(`test_check_catches_a_non_manifold_vertex`). Non-manifold modelling is listed in
[[../99_Not_Covered_In_Code]].

**What `check` does not check:** whether the geometry intersects itself. Two faces may pass through
each other while every pointer is consistent. That is a geometric test with tolerances; real
kernels run one, and this repo does not ([[../99_Not_Covered_In_Code]]).

**Status**: exact (polyhedral; vertices carry positions only). Implemented and tested in
`tests/test_halfedge.py`: the half-edge structure, Euler operators, queries, `check`,
`from_polygons`. Scripts 01 and 02 visualize navigation and construction. Measured: the adjacency
cost of face lists against half-edges, and non-manifold detection. Cited, not tested: Mäntylä's
structure, the OpenCASCADE and ACIS names, winged edge (Baumgart 1975).
