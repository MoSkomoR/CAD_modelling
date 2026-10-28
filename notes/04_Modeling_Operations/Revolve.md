Code: [[../../scripts/04_modeling_ops/02_revolve.py]] — `revolve` in [[../../src/cadkernel/ops/revolve.py]], `revolved_surface`, `turning_arc`, `xz` in [[../../src/cadkernel/geometry/entities.py]], `nurbs_arc` in [[../../src/cadkernel/geometry/nurbs.py]], `surface_vector_area` in [[../../src/cadkernel/ops/measure.py]]; tests in `tests/test_revolve.py`, `tests/test_measure.py`, `tests/test_nurbs.py`, `tests/test_entities.py`

# Revolve — a profile turned about an axis

Every part made on a lathe is a solid of revolution: shafts, flanges, pulleys, bottles, balls. A
CAD user sketches half the cross-section, on one side of an axis, and turns it. Every point of
the profile then moves on a circle about the axis, so every profile edge sweeps a surface of
revolution. That surface is an exact NURBS surface
([[../02_Surfaces/Surfaces_of_Revolution]]).

**The rejected alternative: a construction per primitive.** A kernel could build a cylinder, a
cone, a sphere and a torus each by its own code. Revolve makes all of them with one
construction: a rectangle gives a tube, a triangle on the axis a cone, a half-disc a ball, a disc
a torus. Real kernels also store these four as special analytic surface types (OpenCASCADE's
`Geom_CylindricalSurface`, `Geom_SphericalSurface`, ...; cited, not used here), but they still
build them by revolving. Here every face is a NURBS surface.

Revolve is also where the topology stops being an extrusion. A full turn brings the profile back
onto itself, and that needs three things the earlier constructions never did: **seams**,
**closed edges** and **poles**.

## The input

- **The profile:** a closed loop of NURBS curves in the $xz$-plane, with $x \ge 0$. The
  coordinate $x = r$ is the distance from the axis, the $z$-axis. A sketch drawn in coordinates
  $(r, z)$ is placed there by `xz`, which maps $(r, z) \mapsto (r, 0, z)$.
- **The angle:** $a \in (0, 2\pi]$. The value $a = 2\pi$ is a full turn.

The profile may touch the axis in only one way: **one whole edge lying on the axis**. Every other
point must have $r > 0$. `revolve` checks this through the control points:
- the profile lies in the $xz$-plane and has $x \ge 0$ if its control points do. A curve point is
  a combination of control points with coefficients $R_{i,p}(t) \ge 0$ summing to 1 (the argument
  of [[Extrude#The input]]);
- a curve whose *interior* control points have $x > 0$ has $x(t) > 0$ for $0 < t < 1$. For such
  $t$, at least one interior control point has a positive coefficient (a stated property of
  clamped B-spline bases, not proved here).

Why the restriction: a profile that touches the axis at a single vertex turns into two surfaces
meeting at one point, like two cones tip to tip. A neighbourhood of that point is not a disc, so
the boundary is not a 2-manifold, and the half-edge structure can't represent it
([[../03_Topology/B_Rep_Topology]]). Two edges on the axis would split the solid the same way.
Both are rejected (`test_bad_input_is_rejected`).

**Orientation.** At angle 0, a point $(r, 0, z)$ of the profile starts to move in the direction
$+e_y$. `revolve` orients the profile counter-clockwise seen from the tip of $e_y$. That is
extrude's rule, with $\mathbf d$ replaced by $e_y$. As there, input of either orientation gives
the same solid (`test_orientation_does_not_depend_on_the_input`).

## The revolved surface

Profile edge $k$, the curve $(x(u), 0, z(u))$, sweeps
$$S(u, v) = \big(x(u)\cos\varphi(v),\; x(u)\sin\varphi(v),\; z(u)\big), \qquad u, v \in [0,1],$$
where $\varphi(v)$ increases from 0 to $a$. Here $\varphi$ is the angle of the NURBS arc. It is
not proportional to $v$, since no rational parametrization of the circle has constant speed
([[../01_Curves/NURBS]]).

[[../02_Surfaces/Surfaces_of_Revolution]] proves this is exact for the full circle. The proof uses
only one fact about the circle: it is an exact NURBS curve whose weights multiply with the
profile's. So it holds unchanged for any exact arc. `nurbs_arc` builds the arc from 0 to $a$ out
of $\lceil a / (\pi/2) \rceil$ equal pieces. Each piece is a `rational_arc_quadratic`, exact by
[[../01_Curves/NURBS_Derivations]], and the pieces are joined with every interior knot doubled.

Why pieces of at most a quarter turn: a single quadratic piece of angle $\delta$ has middle weight
$\cos(\delta/2)$ and middle control point at distance $r / \cos(\delta/2)$. As $\delta \to \pi$
the weight goes to 0 and the point to infinity. Quarter turns keep the middle weight at least
$\cos 45°$. This is also the piece size Piegl & Tiller use (Algorithm A7.1; cited). For
$a = 2\pi$ the result is `nurbs_circle`'s four quarters again
(`test_arc_of_a_full_turn_is_the_circle`). Every sweep tested stays on its circle to $10^{-14}$
(`test_arcs_of_any_sweep_are_exact`), and so does the revolved surface
(`test_partial_revolution_turns_by_the_angle`).

## Orientation: every face has sense +1

**Claim.** With the profile oriented as above, $S_u \times S_v$ points out of the solid on every
face, wherever $x(u) > 0$.

*Proof.* Write $e_r = (\cos\varphi, \sin\varphi, 0)$ for the unit vector pointing away from the axis
and $e_\varphi = (-\sin\varphi, \cos\varphi, 0)$ for the direction of turning. Then
$$S_u = x'\,e_r + z'\,e_z, \qquad S_v = \varphi'(v)\,x\,e_\varphi ,$$
and with $e_r \times e_\varphi = e_z$ and $e_z \times e_\varphi = -e_r$,
$$S_u \times S_v = \varphi'\,x\,\big(x'\,e_z - z'\,e_r\big).$$
In the half-plane spanned by $e_r$ and $e_z$ (the plane of the profile turned by $\varphi$), the
profile's tangent is $(x', z')$ in the coordinates $(e_r, e_z)$. The vector $(-z', x')$ is that
tangent turned by $+90°$, which is its left side when $e_r$ points right and $e_z$ up. Since
$\varphi' > 0$ and $x > 0$, the claim is that the *outside* of the profile region lies to the left of
the tangent. In other words, the profile runs clockwise in the picture with $e_r$ right and $e_z$
up.

That picture is the profile's plane seen from $-e_y$, because $e_x \times e_z = -e_y$. Clockwise
seen from $-e_y$ is counter-clockwise seen from $+e_y$, which is the convention. $\blacksquare$

The caps of a partial turn are planes. The start cap has outward normal $-e_y$, since the
material lies on the side the profile moves into. The end cap has outward normal
$e_\varphi(a) = (-\sin a, \cos a, 0)$.

## A partial turn: the extrusion construction

For $a < 2\pi$ the far copy of the profile is a different set of points, and the topology is
extrude's: a lamina, a rail from every vertex, a side face for every edge. `sweep_lamina` in
[[../../src/cadkernel/ops/extrude.py]] is that construction, and extrude, revolve and loft all
call it, each with its own geometry:

| | extrude | partial revolve |
|---|---|---|
| rail of vertex $v$ | segment to $v + \mathbf d$ | `turning_arc(v, a)`, the arc $v$ traces |
| far copy of edge $k$ | $C_k + \mathbf d$ | $C_k$ turned by $a$ (rotation is linear, so turning the control points is exact) |
| side face $k$ | `extruded_surface` | `revolved_surface(C_k, a)` |
| caps | planes with normals $\mp\mathbf n$ | planes with normals $-e_y$ and $e_\varphi(a)$ |

**Poles in a partial turn.** A vertex on the axis doesn't move, so it gets no rail, and its far
copy is itself. The edge on the axis sweeps nothing, so it gets no side face. It stays an edge,
shared by the two caps. Let $n$ be the number of profile edges and $P$ the number of vertices on
the axis (0 or 2), and set $\varepsilon = 1$ if an edge lies on the axis, 0 if not. Then
$$V = 2n - P, \qquad E = 3n - P - \varepsilon, \qquad F = n + 2 - \varepsilon .$$
Without the axis, $V - E + F = 2$ as for an extrusion. With it ($P = 2$, $\varepsilon = 1$):
$(2n-2) - (3n-3) + (n+1) = 2$. Measured: a disc or a square gives $V = 8$, $E = 12$, $F = 6$, and
a half-disc of $n = 3$ edges gives $V = 4$, $E = 6$, $F = 4$ (`test_counts`).

## A full turn: seams and closed edges

At $a = 2\pi$ the far copy of every vertex *is* the vertex, and the far copy of every edge is the
edge. Two things change:
- **The rail of a vertex is a closed curve**, a full circle starting and ending at the vertex. Its
  edge is a **closed edge**: both of its half-edges leave the same vertex.
- **Side face $k$ meets itself** along profile edge $k$: $S(u, 0) = S(u, 1)$. That edge borders
  the face on both sides, so both of its half-edges lie in the face's loop. It is a **seam**.

**The rejected alternative: split the turn.** Two half-turns would avoid both, just as
`circle_arcs` avoids a closed edge by cutting the circle into arcs. Kernels don't do that. The
surface of revolution has no natural place to be cut. One face per profile edge is also what the
user means: "the cylinder", not "two half-cylinders". OpenCASCADE stores a cylindrical face as one
periodic face whose seam edge appears twice in its wire (cited).

**Seams are valid half-edge topology.** None of the invariants of [[../03_Topology/Half_Edge]]
requires an edge's two half-edges to lie in different faces. Each half-edge has its twin, the
twins run opposite ways, and `adjacent_faces` already skips the face itself.

**Closed edges come from mef and mekr.** [[Modeling_Operations]] noted that `mev` cannot close an
edge, because it always makes a new vertex. `lmef(h1, h2)` and `lmekr(h1, h2)`, however, join
`h2.origin` to `h1.origin`. If both half-edges leave the *same* vertex, the new edge is closed.
Revolve uses exactly that.

**The construction, for a profile off the axis.** This follows Mäntylä's rotational sweep
(*An Introduction to Solid Modeling*, 1988; cited). Write $e_k^+$ for the half-edge of profile
edge $k$ that runs with the profile, from vertex $k$ to vertex $k+1$, and $e_k^-$ for its twin.

| step | operator | what happens | $V, E, F, R, G$ | $V - E + F - R$ |
|---|---|---|---|---|
| 1 | lamina | front loop $e_0^+ \dots e_{n-1}^+$, back loop $e_{n-1}^- \dots e_0^-$ | $n, n, 2, 0, 0$ | 2 |
| 2 | `kfmrh` | the back face's loop becomes a ring of the front face | $n, n, 1, 1, 1$ | 0 |
| 3 | `mekr`($e_0^+$, $e_{n-1}^-$) | a closed edge $c_0$ at vertex 0 joins the two loops into one | $n, n+1, 1, 0, 1$ | 0 |
| 4 | `mef`($e_{k-1}^-$, $e_k^+$), for $k = 1, \dots, n-1$ | a closed edge $c_k$ at vertex $k$ splits off the face of edge $k-1$ | $n, 2n, n, 0, 1$ | 0 |

Throughout, the right side is $2(S - G)$: 2 before step 2 and 0 from then on. Step 2 is where the
genus becomes 1. The finished solid has a hole through it, around the axis.

*Why the loops come out right.* After step 3 the single loop reads
$$c_0,\; e_{n-1}^-, e_{n-2}^-, \dots, e_0^-,\; \bar c_0,\; e_0^+, e_1^+, \dots, e_{n-1}^+ ,$$
where $c_0$ and $\bar c_0$ are the two half-edges of the new closed edge. Step 4 at $k = 1$ cuts
this loop between $e_0^-$ (which leaves vertex 1) and $e_1^+$ (which also leaves vertex 1). The new
face takes $e_0^-, \bar c_0, e_0^+$ and the new closed edge $c_1$. That is the face of edge 0:
its seam $e_0$ twice, plus the circles of vertices 0 and 1. Each later step peels off the next
face the same way, and what remains at the end is the face of edge $n-1$. Every step passes the
topological check (`test_every_step_is_valid_and_the_result_is_consistent`).

**Counts.** $V = n$, $E = 2n$, $F = n$, $G = 1$: a torus-like solid for every $n$. A disc and a
square, $n = 4$ each, give $V = 4$, $E = 8$, $F = 4$ (`test_counts`). Each vertex has four outgoing
half-edges: the two seam halves $e_k^+$, $e_{k-1}^-$ and both halves of its circle. Its
neighbourhood is a disc cut into four quadrants, and `vertex_star` walks all four
(`test_seams_and_closed_edges_in_the_half_edge_structure`).

## A full turn with an edge on the axis: poles

The edge on the axis would sweep the axis segment itself, a "surface" of zero area. It is
dropped. Its two ends sweep nothing either, so they become **poles**: vertices with no circle,
where the surface of revolution collapses a whole row of its net to one point
([[../02_Surfaces/Surfaces_of_Revolution]]).

**The construction.** Number the profile so that the axis edge is the last, $n - 1$, joining
vertices $n-1$ and 0.
1. `mvfs` and $n-1$ × `mev` build the profile *without* its axis edge, as an open wire. Its one
   face has a single loop that runs out along the wire and back:
   $e_0^+, \dots, e_{n-2}^+, e_{n-2}^-, \dots, e_0^-$.
2. For $k = 1, \dots, n-2$: `mef`($e_k^+$, $e_{k-1}^-$), both leaving vertex $k$. This makes the
   closed edge $c_k$, and leaves behind the face of edge $k-1$: $e_{k-1}^+, c_k, e_{k-1}^-$
   (plus $c_{k-1}$ for $k \ge 2$).

Counts: $V = n$, $E = (n-1) + (n-2) = 2n - 3$, $F = n - 1$, so $V - E + F = 2$ and $G = 0$. A
half-disc of three edges (two quarter arcs and the axis edge) gives $V = 3$, $E = 3$, $F = 2$: two
hemispheres, two seams, one equator circle (`test_counts`). The smallest case is a single
semicircle plus the axis edge: $V = 2$, $E = 1$, $F = 1$, a sphere as one face whose loop is its
seam, out and back (`test_a_sphere_from_one_semicircle_is_one_face_with_one_seam`).

**What a pole looks like in the half-edge structure.** The face of edge 0 has the loop
$e_0^+, c_1, e_0^-$. It passes the pole once, between $e_0^-$ arriving and $e_0^+$ leaving. The
pole has one outgoing half-edge, and its star has length 1: the face wraps all the way around
it (`test_poles_are_vertices_without_a_circle`). `check` confirms the pole lies on the surface,
and the surface does pass through it.

OpenCASCADE puts a *degenerated edge* at a pole instead: an edge with no 3D curve, only a curve
in the face's parameter rectangle, along the collapsed row (cited). This repo stores no
parameter-space curves ([[../99_Not_Covered_In_Code]]), so it has nothing to put there, and the
pole is just a vertex.

## Orientation of the closed edges

A circle has two directions, and a closed edge can't be oriented by its endpoints. So
`attach_curve` takes the half-edge `along` that runs with the curve.

**Claim.** If the circle of vertex $k$ runs with *increasing* angle along its half-edge in the face
of edge $k-1$, every face's loop runs counter-clockwise around the outward normal.

*Proof.* Take the face of edge $k$, with the loop $\bar c_k, e_k^+, c_{k+1}, e_k^-$ in this cyclic
order. Here $\bar c_k$ is the half of vertex $k$'s circle that lies in this face, and $c_{k+1}$ the
half of vertex $(k+1)$'s circle that lies in it. Map the loop to the parameter rectangle, $u$
along edge $k$ and $v$ around the axis. The seam's two halves are the same points in 3D, so put
$e_k^+$ on the side $v = 0$, running from $(0,0)$ to $(1,0)$, and $e_k^-$ on $v = 1$, running from
$(1,1)$ to $(0,1)$.

Then $c_{k+1}$ must run from $(1,0)$ to $(1,1)$, with increasing angle. It lies in the face of
edge $k$, which is the face *before* vertex $k+1$, as the claim says. And $\bar c_k$ must run from
$(0,1)$ to $(0,0)$, with decreasing angle, which is what its twin's increasing direction implies.
The loop $(0,1) \to (0,0) \to (1,0) \to (1,1) \to (0,1)$ is counter-clockwise in $(u, v)$. So it
runs positively around $S_u \times S_v$, which is outward by the previous section. $\blacksquare$

## Checking the orientation: Stokes

`check` tests that the geometry *lies* where the topology says. It does not test that a face's
loop runs the way its sense says. A reversed circle still lies on both its faces and passes.
Volume doesn't catch it either: on a full turn every face is curved, and the flux integral uses
only the surfaces and their senses. A separate identity does catch it.

**Claim.** For an oriented surface patch $\Sigma$ with unit normal $n$, and boundary $\partial\Sigma$
traversed counter-clockwise around $n$,
$$\frac12 \oint_{\partial\Sigma} x \times dx = \iint_\Sigma n\, dA .$$

*Proof.* Fix a constant vector $c$. Then $c \cdot (x \times dx) = (c \times x)\cdot dx$, and Stokes'
theorem (cited) gives
$\oint (c \times x)\cdot dx = \iint \big(\nabla \times (c \times x)\big)\cdot n\, dA$. For constant
$c$ the curl is $\nabla \times (c \times x) = c\,(\nabla\cdot x) - (c\cdot\nabla)\,x = 3c - c = 2c$.
So $c \cdot \tfrac12\oint x \times dx = c \cdot \iint n\, dA$ for every $c$. $\blacksquare$

For a planar loop this is the vector-area formula of [[Extrude#Volume]]. For a face it compares two
computations from different sources:
- **left side, from the topology:** `loop_vector_area` over the face's loops. Which half-edges
  they contain, and which way each traverses its curve, decide the result. The two halves of a
  seam cancel; each circle counts once.
- **right side, from the geometry:** `surface_vector_area`, which is
  $\sigma \iint S_u \times S_v\, du\, dv$ by quadrature.

They agree to $10^{-13}$ on every curved face of every revolve tested, partial and full, with and
without poles (`test_every_loop_runs_the_way_its_sense_says`). The same holds for the extrusions
with holes (`test_extrude_loops_run_the_way_their_sense_says`) and for the lofts of [[Loft]].
Reversing one circle of a torus leaves `check` silent but breaks the identity for exactly the two
faces the circle borders (`test_a_reversed_circle_is_caught_by_the_orientation_check`).

## Volume: Pappus's theorem

**Claim (Pappus).** Turning a region of area $A$, lying in a half-plane bounded by the axis,
through an angle $a$ sweeps volume
$$V = a\,\bar x\,A ,$$
where $\bar x$ is the distance of the region's centroid from the axis.

*Proof.* Use cylindrical coordinates $(r, \varphi, z)$, with $x = r\cos\varphi$, $y = r\sin\varphi$.
The Jacobian of $(r, \varphi, z) \mapsto (x, y, z)$ is
$\det\begin{pmatrix}\cos\varphi & -r\sin\varphi \\ \sin\varphi & r\cos\varphi\end{pmatrix} = r$, so
$dV = r\,dr\,d\varphi\,dz$. The solid is the region times $[0, a]$ in $\varphi$, so
$$V = \int_0^a \iint_{\text{region}} r\, dr\, dz\, d\varphi = a \iint_{\text{region}} r\, dA = a\,\bar x\,A ,$$
by the definition $\bar x = \frac1A \iint r\, dA$. $\blacksquare$

`profile_area_and_centroid` computes $A$ and $\bar x\,A$ from the profile curves alone, by
Green's theorem (cited), for a boundary counter-clockwise in $(x, z)$:
$$A = \frac12\oint (x\,dz - z\,dx), \qquad \iint x\, dA = \frac12 \oint x^2\, dz .$$
The second follows from Green's $\oint M\,dz = \iint \partial_x M\, dA$ with $M = x^2/2$.

This gives two independent computations of one number. Pappus uses the 2D profile. `volume` uses
the divergence theorem over the faces of the 3D solid ([[Extrude#Volume]]). They agree to better
than $10^{-12}$ for four profiles at three angles each (`test_pappus`). The remaining difference
is quadrature error on the rational arcs, which enter both sides. The textbook volumes come out
to $10^{-13}$ or better (`test_known_volumes`):
- the unit ball, $\tfrac43\pi$;
- a cone of radius 1 and height 2, $\tfrac23\pi$;
- a torus with $R = 2$, $r = 0.7$, $2\pi^2 R r^2$;
- a tube with $1 \le r \le 2$ and height 1, $3\pi$.

Script 02 prints both computations under the solid, and they stay equal through every drag and
angle.

## What revolve doesn't do

- **Any axis but $z$, or a profile outside the $xz$-plane.** That is a rigid motion before and
  after, and isn't implemented.
- **Holes in the profile.** Each would add a through-hole on a full turn. Not implemented.
- **Self-intersecting profiles**, as for extrude.
- **Helical motion** (turning while moving along the axis): that is a sweep, not a revolve.

**Status**: exact (planar and revolved NURBS faces, untrimmed). Implemented and tested: `revolve`
for partial and full turns, with and without an edge on the axis, in `tests/test_revolve.py`;
`nurbs_arc` and the partial surface of revolution. Script 02 shows seams, closed edges and poles
appearing at the full turn, and compares Pappus with the measured volume.
- *Proved inline:* the outward orientation of $S_u \times S_v$; the counts and Euler–Poincaré for
  partial turns, full turns and poles; the loop structure the Euler operators produce; the
  direction rule for closed edges; the vector-area identity (given Stokes); Pappus's theorem.
- *Stated and measured:* exactness of arcs and partial surfaces; validity after every step; the
  stars at seams and poles; the Stokes identity on every face; Pappus against the divergence
  theorem; the textbook volumes.
- *Cited, not proved:* Stokes' and Green's theorems; the positivity of interior basis
  coefficients used in the input check; Piegl & Tiller's arc piece size; Mäntylä's rotational
  sweep; OpenCASCADE's analytic surface types, seams and degenerated edges.
