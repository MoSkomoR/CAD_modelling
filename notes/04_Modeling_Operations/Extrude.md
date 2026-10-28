Code: [[../../scripts/04_modeling_ops/01_extrude.py]] — `extrude` in [[../../src/cadkernel/ops/extrude.py]], `extruded_surface` in [[../../src/cadkernel/geometry/entities.py]], `volume` in [[../../src/cadkernel/ops/measure.py]]; tests in `tests/test_extrude.py`, `tests/test_measure.py`

# Extrude — a profile moved along a vector

Extrusion is the most-used operation in mechanical CAD: sketch a closed profile, pull it a
distance, and get a plate, a bar, a bracket. It is also the simplest operation that produces a
solid with exact curved faces, which is why it comes first. Why operations work from profiles at
all, and what binding geometry to topology involves, is in the hub [[Modeling_Operations]].

## The input

- **The profile:** a closed loop of curves $C_0, \dots, C_{n-1}$, each a NURBS curve on
  $t \in [0,1]$, with $C_k(1) = C_{k+1}(0)$ (indices mod $n$), all lying in one plane. The
  junction points are the profile's vertices; each curve becomes an edge.
- **The extrusion vector:** $\mathbf d$, not parallel to the profile's plane.
- **Holes (optional):** further closed loops in the same plane, inside the profile.

Planarity is checked through the control points. A NURBS curve point is a combination of the
control points whose coefficients $R_{i,p}(t)$ sum to 1, and such a combination of points in a
plane stays in that plane. So control points in one plane imply a curve in that plane.

Let $\mathbf n$ be the unit normal of the profile's plane. Throughout, the profile is oriented
**counter-clockwise seen from the tip of $\mathbf d$**: its vector area $\mathbf A$ (defined
under [[#Volume]]) satisfies $\mathbf A \cdot \mathbf d > 0$. `extrude` reverses a profile that
runs the other way, so the caller doesn't have to care. Clockwise input and a downward
$\mathbf d$ give the same solid (`test_orientation_does_not_depend_on_the_input`).

## The construction

The topology is the box sweep of [[../03_Topology/Euler_Operators#Two constructions, step by step]],
generalized from 4 edges to $n$, with geometry attached as each operator runs:

| step | operators | creates | geometry attached |
|---|---|---|---|
| 1 | `mvfs`, then $n-1$ × `mev` | the profile as an open wire | edge $k$ gets $C_k$ |
| 2 | `mef` | closes the wire into a **lamina**: two faces back to back | last edge gets $C_{n-1}$; one face becomes the bottom, plane through the profile with normal $-\mathbf n$ |
| 3 | $n$ × `mev` | a straight edge from every vertex $v$ to $v + \mathbf d$ | the segment |
| 4 | $n$ × `mef` | side face $k$, between the straight edges at vertices $k$ and $k+1$ | the top copy of edge $k$ gets $C_k + \mathbf d$; the face gets the extruded surface (next section) |

After step 4 the remaining lamina face is the top, the plane moved by $\mathbf d$ with normal
$+\mathbf n$. Steps 1–4 are `sweep_lamina`, which [[Revolve]] (a partial turn) and [[Loft]] call
with their own rails and surfaces.

**Counts.** A profile of $n$ edges gives $V = 2n$ (each profile vertex and its copy), $E = 3n$
(profile edges, top copies, straight edges) and $F = n + 2$ (the sides, top, bottom). Then
$$V - E + F = 2n - 3n + n + 2 = 2 ,$$
for every $n$ (`test_counts_match_the_formula_for_several_sizes`, $n = 2, 3, 4, 7$). With $n = 2$
the profile has two edges joining the same two vertices, a "lens", and the construction still
works.

**Holes.** Each hole of $m$ edges is drilled the way `block_with_hole` does it:
1. a spike from a top corner to the hole outline, then the outline itself, all by `mev`;
2. `mef` closes the outline into a face inside the top;
3. `kemr` cuts the spike, which leaves the outline as a ring of the top;
4. that face is swept down with $m$ `mev` and $m$ `mef`;
5. `kfmrh` merges it into the bottom as a second ring.

Net change per hole: $\Delta V = 2m$, $\Delta E = 3m$, $\Delta F = m$, $\Delta R = 2$,
$\Delta G = 1$. The left side of Euler–Poincaré changes by $2m - 3m + m - 2 = -2$, and the right
side by $2(0 - 1) = -2$, so the formula keeps holding. A disc with a round hole and a square slot
has $V = 24$, $E = 36$, $F = 14$, $R = 4$, $G = 2$ (`test_counts_of_an_extrusion`). Every
intermediate state passes the topological check
(`test_every_step_is_valid_and_the_result_is_consistent`).

## The extruded surface

Side face $k$ is the set of points $C_k(u) + v\,\mathbf d$, for $u, v \in [0,1]$: every point of
the profile curve, moved along $\mathbf d$.

**Claim.** If $C(u) = \sum_{i=0}^{n} R_{i,p}(u)\,P_i$, with control points $P_i$, weights $w_i$
and rational basis $R_{i,p}$, then the NURBS surface of degree $(p, 1)$ with
- control net $P_{i0} = P_i$, $P_{i1} = P_i + \mathbf d$ (two columns),
- weights $w_{i0} = w_{i1} = w_i$,
- knots: the curve's knots in $u$, and $0, 0, 1, 1$ in $v$,

is exactly $S(u, v) = C(u) + v\,\mathbf d$.

*Proof.* The degree-1 basis on knots $0,0,1,1$ is $N_{0,1}(v) = 1 - v$, $N_{1,1}(v) = v$. Since the
weights don't depend on $j$, the rational surface basis factors:
$$R_{ij}(u, v) = \frac{w_i N_{i,p}(u) N_{j,1}(v)}{\sum_k \sum_l w_k N_{k,p}(u) N_{l,1}(v)}
= \frac{w_i N_{i,p}(u)}{\sum_k w_k N_{k,p}(u)} \cdot \frac{N_{j,1}(v)}{(1-v) + v}
= R_{i,p}(u)\, N_{j,1}(v).$$
So
$$S(u,v) = \sum_i R_{i,p}(u)\big[(1-v)\,P_i + v\,(P_i + \mathbf d)\big]
= \sum_i R_{i,p}(u)\,P_i + v\,\mathbf d \sum_i R_{i,p}(u) = C(u) + v\,\mathbf d,$$
using $\sum_i R_{i,p}(u) = 1$. $\blacksquare$

Measured on a rational arc and a polynomial curve, to $10^{-14}$, together with
$S_u = C'(u)$ and $S_v = \mathbf d$ (`test_extruded_surface_is_the_curve_plus_v_d`). The surface is
**ruled**: for fixed $u$, the curve $v \mapsto S(u,v)$ is a straight segment. Script 01 draws
these segments ("rulings") and the two-column net ("side-face nets").

## Orientation: the sense of each side face

The surface's own normal direction is $S_u \times S_v = C'(u) \times \mathbf d$. Is it outward?

**Claim.** On the outer boundary, $C'(u) \times \mathbf d$ points out of the solid (sense $+1$).
On a hole's wall it points into the material (sense $-1$).

*Proof.* At a point of the profile, the outward direction *within the profile's plane* is
$C' \times \mathbf n$. That is what counter-clockwise seen from $\mathbf n$ means: the enclosed
region lies to the left of the tangent, in direction $\mathbf n \times C'$, so the outside lies in
the opposite direction $C' \times \mathbf n$. Compare the surface normal with that direction using
the identity $(\mathbf a \times \mathbf b)\cdot(\mathbf c \times \mathbf e) = (\mathbf a\cdot\mathbf c)(\mathbf b\cdot\mathbf e) - (\mathbf a\cdot\mathbf e)(\mathbf b\cdot\mathbf c)$:
$$(C' \times \mathbf d)\cdot(C' \times \mathbf n) = |C'|^2\,(\mathbf d \cdot \mathbf n) - (C' \cdot \mathbf n)(\mathbf d \cdot C') = |C'|^2\,(\mathbf d \cdot \mathbf n) > 0 .$$
Here $C' \cdot \mathbf n = 0$ because the profile lies in the plane, and $\mathbf d \cdot \mathbf n > 0$
by the orientation convention. So the surface normal has a positive component along the outward
direction. That holds even when $\mathbf d$ is tilted, and then $C' \times \mathbf d$ is not
horizontal.

A hole is also oriented counter-clockwise, but the material lies *outside* it. The direction
$C' \times \mathbf n$ then points into the material, so the side surface's normal does too, and
the hole's walls get sense $-1$. $\blacksquare$

Measured: on a disc with a hole, the four outer walls have $\sigma\,(S_u \times S_v)$ pointing away
from the axis and the four hole walls pointing towards it
(`test_sense_records_outward_normals`). Each wall's loop also runs the way its sense says, by the
Stokes identity of [[Revolve#Checking the orientation: Stokes]]
(`test_extrude_loops_run_the_way_their_sense_says`). The sense matters: flipping it on one quarter-cylinder of
an extruded unit circle (height 1) turns the volume $\pi$ into $2\pi/3$. That face's share,
$\pi/6$, is subtracted instead of added (`test_flipping_one_sense_breaks_the_volume`).

## Volume

The divergence theorem gives $\text{Vol} = \frac13 \oint_\partial x \cdot n\, dA$
([[../03_Topology/Half_Edge#The invariants]] derives it). Each face contributes one integral.

**A curved face.** For a surface patch $S(u,v)$ the vector $S_u \times S_v$ is normal to it, and
its length is the area scale factor of the map $(u,v) \mapsto S(u,v)$. So $n\, dA = (S_u \times S_v)\,du\,dv$,
the standard change of variables for surface integrals (cited from calculus), and
$$\iint_{\text{face}} x\cdot n\, dA = \sigma \int_0^1\!\!\int_0^1 S(u,v)\cdot\big(S_u \times S_v\big)\,du\,dv ,$$
where the sense $\sigma = \pm 1$ makes the normal outward. This assumes the face is the whole
patch $[0,1]^2$ (it is **untrimmed**), which is true of every face `extrude` makes.

**A planar face.** On a plane $x \cdot n$ is constant, equal to $p \cdot n$ for any point $p$ of the
face, so the integral is $(p\cdot n)\cdot \text{Area} = p \cdot \mathbf A$ with $\mathbf A = n \cdot \text{Area}$,
the face's **vector area**.

**Claim.** For a closed curve in a plane with unit normal $n$, enclosing area $\text{Area}$
counter-clockwise seen from $n$,
$$\mathbf A = \frac12 \oint x \times dx = n \cdot \text{Area}.$$

*Proof.* Write points of the plane as $x = p_0 + a\,e_1 + b\,e_2$, where $p_0$ is a fixed point of
the plane, $(e_1, e_2, n)$ is a right-handed orthonormal frame, and $(a, b)$ are coordinates in
the plane. Then $dx = da\,e_1 + db\,e_2$ and
$$x \times dx = p_0 \times dx + (a\,db - b\,da)\, n ,$$
using $e_1 \times e_2 = n$ and $e_1 \times e_1 = e_2 \times e_2 = 0$. Around a closed curve
$\oint dx = 0$, so the first term integrates to $p_0 \times \oint dx = 0$. Green's theorem (cited)
gives $\frac12\oint (a\,db - b\,da) = \text{Area}$. $\blacksquare$

For a face with rings, the rings run clockwise, so their vector areas subtract the holes. Along a
straight edge from $a$ to $b$ the integral is exactly $\tfrac12\,a \times b$: the shoelace term of
the polyhedral formula, which is why `volume` agrees with `signed_volume` when no curves are bound.
Along a curved edge it is $\tfrac12\int_0^1 C(t)\times C'(t)\,dt$, negated when the half-edge runs
against the curve.

**Evaluating the integrals.** Both integrands are evaluated by Gauss–Legendre quadrature on each
knot span (on each rectangle of knot spans, for surfaces). An $N$-point rule is exact for
polynomials up to degree $2N - 1$ (cited, a standard result of numerical analysis). Inside a knot
span:
- **Polynomial (B-spline) faces.** $S \cdot (S_u \times S_v)$ is a polynomial. On an extruded
  face $S$ has degree $p$ in $u$, $S_u = C'$ has degree $p - 1$, and $S_v = \mathbf d$ is constant,
  so the integrand has degree $2p - 1$ in $u$ (and 2 in $v$). The cap integrand $C \times C'$ has
  the same degree $2p - 1$. So $N = p$ points are exact. A cubic Bezier profile, tilted, comes out
  exact from $N = 3$, while $N = 2$ is off by 0.12 (`test_quadrature_is_exact_for_polynomial_faces`).
- **Rational faces.** The integrand is a quotient of polynomials, and no finite rule is exact.
  The error falls geometrically instead. On an extruded unit circle of height 1
  (`test_quadrature_convergence_on_a_rational_integrand`):

| $N$ | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| $\lvert V - \pi\rvert$ | 0.17 | $7.1\cdot10^{-3}$ | $2.9\cdot10^{-4}$ | $1.2\cdot10^{-5}$ | $1.9\cdot10^{-8}$ | $2.9\cdot10^{-11}$ | $\sim 10^{-15}$ |

So `volume` uses $N = 12$ by default.

Measured volumes (`test_volumes`), all with height $h = 1.5$:
- the unit circle gives $\pi h$;
- a $4 \times 4$ square with a round hole of radius 1 gives $(16 - \pi)\,h$;
- a disc of radius 3 with a round hole of radius 1 and a $0.6 \times 1$ slot gives
  $(9\pi - \pi - 0.6)\,h$.

## Tilting does not change the volume

**Claim.** The extrusion of a profile with area $\text{Area}$ along $\mathbf d$ has volume
$\text{Area} \cdot (\mathbf d \cdot \mathbf n)$, whatever the tilt of $\mathbf d$.

*Proof.* Choose coordinates with $\mathbf n = e_z$ and the profile in $z = 0$, and write
$\mathbf d = (a, b, h)$ with $h = \mathbf d \cdot \mathbf n > 0$. The map
$$\phi(x, y, z) = \Big(x + \frac{a}{h} z,\; y + \frac{b}{h} z,\; z\Big)$$
sends the upright prism, profile $\times\, [0, h]$, onto the tilted one: the point at height $z$
above profile point $q$ goes to $q + (z/h)\,\mathbf d$. Its Jacobian matrix is triangular with
ones on the diagonal, so it has determinant 1 and preserves volume. The upright prism has volume
$\text{Area}\cdot h$. $\blacksquare$

This is Cavalieri's principle: equal cross-sections at every height give equal volumes. Measured
for three tilts, to $10^{-12}$ (`test_oblique_extrusion_keeps_the_volume`). In script 01, drag the
"tilt" slider and watch the volume in the title stay fixed.

## Checking the result

After building, `check` verifies the topology and that the geometry agrees with it: vertices on
their faces' surfaces, and each edge curve running between its vertices and lying on both of its
faces ([[Modeling_Operations#Binding geometry to topology]]). Every construction in the tests
passes, including tilted ones and ones with holes.

The check is local, and its report says *where* a problem is. Pull the middle control point of one
bottom arc of an extruded circle 0.01 sideways, within the bottom plane. The arc still lies on the
bottom face's plane, so the bottom face has no complaint. It no longer lies on the neighbouring
quarter-cylinder. `check` reports exactly one problem, naming that side face
(`test_check_catches_a_curve_that_leaves_a_face`).

## What extrude doesn't check

- **A self-intersecting profile.** A figure-eight profile produces faces passing through each
  other. Every pointer is consistent and every edge lies on its faces, so `check` passes it. This
  is the same gap as in [[../03_Topology/B_Rep_Topology]].
- **Holes that leave the profile or touch each other.** Same reason.
- **Extruding to a face** ("up to next", "through all"). That needs intersections, which this repo
  doesn't do ([[../99_Not_Covered_In_Code]]).

**Status**: exact (planar and extruded NURBS faces, untrimmed). Implemented and tested:
`extrude`, with holes, upright and tilted, in `tests/test_extrude.py`; quadrature in
`tests/test_measure.py`. Script 01 lets you drag the profile and watch the solid follow.
- *Proved inline:* the extruded surface is exactly a degree-$(p,1)$ NURBS; the counts and
  Euler–Poincaré for any $n$ and any number of holes; the sense of outer and hole walls; the
  vector-area formula (given Green's theorem); the polynomial degree of the flux integrand; the
  volume under tilting.
- *Stated and measured:* the quadrature error on rational faces; the volumes; validity after every
  step; that the consistency check reports the faulty face.
- *Cited, not proved:* Green's theorem, the change of variables for surface integrals, and the
  exactness degree of Gauss–Legendre quadrature.
