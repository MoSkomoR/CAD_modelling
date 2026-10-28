Code: [[../../src/cadkernel/geometry/entities.py]] (curves and surfaces as objects), `attach_curve` and `check` in [[../../src/cadkernel/topology/halfedge.py]], [[../../src/cadkernel/ops/measure.py]] (volume); tests in `tests/test_entities.py`, `tests/test_measure.py`

# Modeling operations — the motivation hub

Modules 01 and 02 built curves and surfaces. Module 03 built topology: who touches whom, with
vertices as the only geometry. A real solid needs both at once. This module joins them, through
the operations a CAD user actually performs: draw a closed profile, then **extrude**, **revolve**,
**loft** or **sweep** it into a solid.

The operations are written up one per note:
- [[Extrude]] — a profile moved along a vector;
- [[Revolve]] — a profile turned about an axis: seams, closed edges and poles;
- [[Loft]] — a solid through a stack of sections: compatible curves and skinning;
- sweep — planned.

Extrude, a partial revolve and loft share one topological construction (`sweep_lamina`); they
differ only in the curves and surfaces they attach.

## Why solids are made from profiles

A kernel could ask the user for faces directly: six patches for a box, glued along their edges.
Nobody models that way, for two reasons:
- **Validity is hard to get by hand.** Every edge must be shared by exactly two faces, and every
  edge curve must lie on both of its surfaces ([[../03_Topology/B_Rep_Topology]]). Placing
  patches so that this holds is the kernel's job, not the user's.
- **A profile is the design intent.** A shaft is "this cross-section, this long". The profile
  plus one parameter (a length, an angle, a path) says that directly, and editing the parameter
  rebuilds the solid.

So an operation takes a profile and produces a solid that is valid *by construction*. Its
topology comes from Euler operators, which keep it valid at every step
([[../03_Topology/Euler_Operators]]). Its geometry comes from the profile curves, which fix
every surface exactly.

## Binding geometry to topology

Each topological element now carries geometry:

| element | carries | in this repo |
|---|---|---|
| vertex | a point | `Vertex.point` |
| edge | a curve | `Edge.curve`, a `NurbsCurve` on $t \in [0, 1]$; closed for a full circle |
| face | a surface | `Face.surface`, a `Plane` or a `NurbsSurface` on $[0,1]^2$ |

**The rejected alternative: faceting.** An operation could stay polyhedral, for example revolving
a profile in $N$ discrete steps. That would reuse module 03 unchanged. But a revolved circle
would become a polygon, which throws away exactly what modules 01–02 built: a cylinder is a
NURBS surface *exactly* ([[../02_Surfaces/Surfaces_of_Revolution]]). Facets are what a kernel
produces at the very end, for display or export, from exact geometry. They are not what it
stores.

Binding geometry raises three questions that pure topology never had to answer.

**1. Do the pieces agree?** The topology says "edge $e$ separates faces $f$ and $g$". The geometry
must say the same thing:
- every vertex of a face lies on the face's surface;
- every edge curve starts and ends at its two vertices;
- every edge curve lies on the surfaces of **both** faces it separates.

The last condition is what makes the boundary watertight: two faces meet exactly along their
shared edge, with no gap and no overlap. `check` now tests all three, at sample points of each
curve (a check, not a proof). The distance from a point $x$ to a NURBS surface $S$ is found by
Gauss–Newton:

- linearize: $S(u + \Delta u, v + \Delta v) \approx S + S_u\,\Delta u + S_v\,\Delta v$, where
  $S_u$, $S_v$ are the partial derivatives;
- solve the $3 \times 2$ least-squares system $[S_u\ S_v]\,(\Delta u, \Delta v)^T = x - S$;
- repeat until the step is negligible.

For a point that does lie on the surface the residual tends to 0 and the iteration converges
fast, which is the case the check is about (`test_distance_to_surface_finds_points_on_it_and_off_it`).
The failure case is in [[Extrude#Checking the result]].

**2. Which way is out?** A parametric surface comes with one normal direction, $S_u \times S_v$.
A face of a solid needs its *outward* normal, and the two need not agree: the wall of a hole
bulges towards the hole's axis, while the outward normal of the solid points into the hole. Each
face therefore stores a **sense**, $+1$ if $S_u \times S_v$ points out of the solid and $-1$ if it
points in. The surface stays as it is; only the flag changes. OpenCASCADE calls this the
orientation of a face (`TopAbs_FORWARD` / `TopAbs_REVERSED`; cited, not checked here).
Whether a face's loop really runs counter-clockwise around $\sigma\,(S_u \times S_v)$, where
$\sigma$ is the sense, is not something `check` sees; the tests check it with a Stokes identity
([[Revolve#Checking the orientation: Stokes]]).

**3. Which way does an edge curve run?** The two half-edges of an edge run in opposite
directions, so the curve can't run "with" both of them. `attach_curve` fixes the convention: the
curve runs from `edge.he1.origin` to `edge.he2.origin`, reversing it if it was built the other
way. A half-edge then traverses its curve forwards exactly when it is `he1`. Loop integrals need
this (see volume, below). A closed curve (a full circle) has the same start and end, so its
direction can't be read off the vertices; the caller names the half-edge that runs with it.

**A constraint from module 03.** `mev` always creates a *new* vertex at the far end of the new
edge, so a profile built edge by edge with `mev` can't contain an edge that starts and ends at
the same vertex. A circle in a profile therefore becomes several arcs, each with its own two
vertices (`circle_arcs`, four quarter arcs by default). Real kernels store a full circle as one
closed edge with one vertex. That costs extra edges and faces here, and nothing else.

`mef` and `mekr` *can* make a closed edge: they join two half-edges' origins, and when both
half-edges leave the same vertex the new edge is closed. A full revolve uses exactly that, along
with *seams*, edges whose two half-edges lie in the same face ([[Revolve#A full turn: seams and closed edges]]).

## Measuring a solid with curved faces: volume

[[../03_Topology/Half_Edge]] derived the volume of a polyhedron from the divergence theorem:
$$\text{Vol} = \frac13 \oint_{\partial} x \cdot n \, dA ,$$
where $x$ is the position on the boundary $\partial$ and $n$ the outward unit normal. Bound
geometry changes how each face's share is computed, and [[Extrude#Volume]] derives both new
forms:
- **a curved face**, the whole patch $[0,1]^2$ with sense $\sigma$:
  $\tfrac13\,\sigma \iint S \cdot (S_u \times S_v)\, du\, dv$;
- **a planar face bounded by curves:** $\tfrac13\,\mathbf A \cdot p$, where $p$ is any point of
  the face and $\mathbf A = \tfrac12 \oint x \times dx$ is the vector area of its loops.

`volume` evaluates both by Gauss–Legendre quadrature on each knot span. With no curves bound it
reduces to the polyhedral formula, and agrees with `signed_volume` on the module 03 solids
(`test_volume_agrees_with_signed_volume_on_polyhedra`).

**Status**: exact geometry (planes and NURBS), untrimmed curved faces only. Implemented and tested:
the geometry objects, `attach_curve` (closed curves included), the consistency checks in `check`,
`volume`, `surface_vector_area`. Proved inline
here: nothing new (the derivations are in [[Extrude]]). Stated and measured: Gauss–Newton
convergence on and off a surface; agreement of `volume` with the polyhedral formula. Cited, not
checked: OpenCASCADE's face orientation flag.
