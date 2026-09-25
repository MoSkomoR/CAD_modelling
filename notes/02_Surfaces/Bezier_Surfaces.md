Code: [[../../scripts/02_surfaces/01_bezier_surface_de_casteljau.py]] — `cadkernel/geometry/surfaces.py`

# Bezier surfaces — De Casteljau, run twice

Module 01 ended with a curve scheme that works: control points that are geometry, a convex hull
to cull with, subdivision to recurse on, and an algorithm made of nothing but lerp. The question
this note answers is how to get *surfaces* out of that, and the answer is startlingly cheap —
which is exactly why it is the answer kernels use.

## The design question first

A surface needs two parameters, $S(u,v)$. There is no shortage of ways to build one, and the
three obvious ones are all worse:

**Implicit surfaces**, $f(x,y,z)=0$. Compact, trivially answers "is this point inside?", and
closed under booleans (min/max of the $f$'s). But they have no natural parametrization, so you
cannot walk the surface, texture it, or tessellate it without root-finding at every step, and
you cannot *bound* one cheaply. A kernel that must trim, sew and mesh faces needs the
parametrization.

**A mesh of triangles.** This is the output format, not the representation. It is an
approximation fixed at the moment you build it — you cannot refine it later, offset it exactly,
or ask what surface it was.

**Invent a genuinely two-dimensional scheme.** Possible, and one such scheme is worth knowing
about (see "the other generalization" below) — but it means re-deriving every property and
rewriting every algorithm.

**The tensor-product construction** takes the fourth option: don't invent anything. Start from
$C(t) = \sum_i B_i^n(t)P_i$ and let the *coefficients themselves be curves*:
$$S(u,v) = \sum_{i=0}^{n} B_i^n(u)\,\underbrace{\left[\sum_{j=0}^{m} B_j^m(v)\,P_{ij}\right]}_{\text{a Bezier curve in } v}
= \sum_{i=0}^{n}\sum_{j=0}^{m} B_i^n(u)\,B_j^m(v)\,P_{ij}.$$
A curve is a row of control points; a surface is a **rectangular net** $P_{ij}$, degree $n$ in
$u$ and $m$ in $v$ (its *bidegree*). Nothing else changed. The whole argument for this
construction is that the double sum factors — and everything below is a consequence of that one
fact.

## Yes, De Casteljau constructs it — twice

The double sum has a factored inner bracket, and De Casteljau's intermediate results are
**points**. So they can be fed straight back into De Casteljau. Evaluating $S(u,v)$ is:

> **Pass 1 (collapse $u$).** Each *column* $P_{0j},\dots,P_{nj}$ of the net is an ordinary
> degree-$n$ Bezier control polygon. Run De Casteljau at $u$ down each of the $m+1$ columns.
> What survives is $m+1$ points $Q_0,\dots,Q_m$.
>
> **Pass 2 (collapse $v$).** Those $m+1$ points are themselves a control polygon — of a
> degree-$m$ Bezier curve. Run De Casteljau on them at $v$. The single point left is $S(u,v)$.

That is the entire algorithm (`bezier_surface_point`). No binomials, no new recursion, no second
derivation — just $\tfrac{n(n+1)}{2}\,(m+1) + \tfrac{m(m+1)}{2}$ lerps. Run
[[../../scripts/02_surfaces/01_bezier_surface_de_casteljau.py]] and the intermediate polygon
$Q_j$ is drawn on the patch as you move the sliders.

**The order does not matter.** Collapse $v$ first and $u$ second — De Casteljau along each *row*,
then on the results — and you land on the same point (measured at $1.1\times10^{-16}$, i.e.
rounding only; `test_surface_evaluation_order_does_not_matter`, and the "other order" checkbox
draws both constructions at once). This is immediate from the double sum: the two summations
are independent, so they commute. It is worth stating explicitly because it is what makes
"the surface" a single well-defined object rather than an artefact of an evaluation order.

**What pass 1 leaves behind is an isocurve.** The points $Q_j$ are not scratch work — they are
the exact control points of the curve $v \mapsto S(u_0, v)$ (`bezier_isocurve`,
`test_isocurve_is_a_bezier_curve_of_the_other_direction_degree`). So a surface, held at fixed
$u$, *is* a Bezier curve of degree $m$, with control points De Casteljau already computed. Every
curve routine from module 01 — subdivision, hull bounds, the flatness test, ray/curve
intersection — applies to it unchanged. That is the payoff: the second dimension came free.

## What carries over, and why

All of it descends from the same two facts as the curve case
([[../01_Curves/Bernstein_Basis_Properties]]), because a **product of Bernstein bases is still a
partition of unity of non-negative functions**:
$$B_{ij}^{nm}(u,v) = B_i^n(u)B_j^m(v) \ge 0, \qquad
\sum_{i}\sum_{j} B_i^n(u)B_j^m(v) = \Big(\sum_i B_i^n(u)\Big)\Big(\sum_j B_j^m(v)\Big) = 1\cdot1 = 1.$$
(`test_tensor_basis_is_a_partition_of_unity_and_non_negative`.) So $S(u,v)$ is a convex
combination of the $(n+1)(m+1)$ net points, and every corollary follows verbatim:

| property | surface form | why it matters here |
|---|---|---|
| endpoint → **corner** interpolation | the patch passes through the four *corner* net points only | the other net points are pure handles |
| **boundary curves** | the four edges of the patch are the Bezier curves of the four edges of the net | this is what lets patches be sewn edge-to-edge into a B-rep face |
| **convex hull** | $S(u,v)$ lies in the hull of the whole net | the net's bounding box culls surface/surface pairs before any evaluation |
| **affine invariance** | transform the net, get the transformed patch | one transform of $(n+1)(m+1)$ points, parametrization untouched |
| **subdivision** | split every column at $u$, then every row at $v$ → four sub-patches | subdivide-and-cull, now in 2D — the backbone of real surface intersection |
| **global support** | every net point moves every interior point of the patch | the same ceiling as curves, and the same fix (B-splines) |

Subdivision is worth dwelling on, since it was the decisive argument for De Casteljau on curves
([[../01_Curves/Bezier_Curves]]). It survives *because* the structure is a tensor product: apply
the curve algorithm one direction at a time and you get four patches of the same bidegree,
covering the four quarters of the parameter square, reproducing the original exactly
(`test_surface_subdivide_reproduces_original`; the "subdivide" checkbox colours them). Surface/
surface intersection by recursive box-culling is now available, and it is the same code as the
curve case.

## What it costs — the honest section

The tensor product is not free of consequences, and three of them shape everything downstream.

**1. The patch is a rectangle, and only a rectangle.** A tensor-product patch is topologically a
square: four corners, four edges, a parameter domain of $[0,1]^2$. You cannot build a
three-sided or five-sided patch, a patch with a hole, or a closed sphere out of one. Real models
are full of exactly those. The kernel's answer is **trimming** — keep the rectangular patch as
the underlying geometry, and carry a separate set of boundary curves in parameter space saying
which part of it is really there. That split, geometry underneath and topology on top, is not an
implementation detail: it is *why* B-rep exists, and it is what module 03 is about.

**2. Degree inflates the moment you leave the isoparametric directions.** The isocurves have
degree $n$ and $m$ — but the diagonal $u=v=t$ has degree $n+m$, measured exactly for bidegrees
$(1,1),(2,2),(3,3),(3,2),(4,1)$ in `test_diagonal_of_a_patch_has_degree_n_plus_m`. Worse, the
normal field $S_u\times S_v$ has bidegree $(2n-1,\,2m-1)$: for an ordinary bicubic patch that is
a bidegree-$(5,5)$ vector field, 36 coefficients per component, against the patch's own 16
points. Anything that is not a query along a grid line — an intersection curve, a silhouette, an
offset, a fillet — lands in a much higher-degree world than the surface you started from. This
is the concrete reason exact surface/surface intersection is hard, and it is why module 05
settles for the simplified mesh-CSG tier.

**3. "Flat-looking net" is not "flat patch".** The simplest possible patch, bidegree $(1,1)$ on
four corner points, is a *bilinear* patch. If the four corners are not coplanar it is a
hyperbolic paraboloid: every isocurve is a perfectly straight line, and yet the surface is
curved and misses the plane through three of its corners by a wide margin
(`test_bilinear_patch_is_doubly_ruled_but_not_planar`). Straight isocurves buy you nothing about
the surface between them.

And one property genuinely **does not** generalize: **variation diminishing**. The curve version
(no line crosses the curve more often than the control polygon) has no true analogue for
tensor-product surfaces — the natural statement about planes cutting the patch versus the net is
false in general. This is stated here from the literature, not derived or measured in this repo;
see [[../99_Not_Covered_In_Code]].

## The other generalization: triangular patches

There is a second, genuinely different way to push De Casteljau into two dimensions, and it is
worth knowing that the tensor product is a *choice*.

Instead of lerping along one direction, lerp over a triangle in **barycentric coordinates**
$(\lambda_0,\lambda_1,\lambda_2)$ with $\lambda_0+\lambda_1+\lambda_2=1$. The control points sit
on a triangular lattice, and each step of the recursion blends *three* neighbours at a time:
$$P^k_{\mathbf{i}} = \lambda_0 P^{k-1}_{\mathbf{i}+e_0} + \lambda_1 P^{k-1}_{\mathbf{i}+e_1}
+ \lambda_2 P^{k-1}_{\mathbf{i}+e_2}.$$
The triangle shrinks by one row per level and collapses to a point, exactly as before. This is
the **Bezier triangle**, its basis is the *bivariate Bernstein* (multinomial rather than binomial)
basis, and it keeps hull containment, affine invariance and subdivision. It has a single total
degree instead of a bidegree, and — the point — it is naturally three-sided, which is precisely
what tensor-product patches cannot be.

Kernels standardize on the tensor product anyway: it composes with the B-spline knot machinery
that fixes global support, it is what STEP and IGES interchange, and its grid structure makes
evaluation, rendering and trimming straightforward. Bezier triangles remain the tool of choice in
finite elements and in subdivision-surface work. Not implemented here —
[[../99_Not_Covered_In_Code]].

## Where this goes

The two limitations that motivated NURBS for curves are both still here, unchanged, because the
tensor product inherited them along with everything else:

- **Global support** → the same fix, applied per direction: B-spline bases in $u$ and $v$, giving
  local control and letting one patch cover what would otherwise need many.
- **No exact conics** → and now with a corollary worth naming. A sphere, a cylinder and a torus
  all have circular isocurves; a polynomial curve is never a circle
  ([[../01_Curves/Lines_and_Arcs]], with proof); therefore **no polynomial tensor-product patch is
  exactly a sphere, cylinder, cone or torus.** The most common surfaces in mechanical CAD are all
  outside reach until weights arrive.

Add both fixes to the tensor product and you have the NURBS surface — one representation holding
planes, cylinders, spheres, tori and free-form patches alike, with the curve algorithms of module
01 still doing the work underneath, one direction at a time.

**Status**: implemented exactly in code (`bezier_surface_point`, `bezier_surface`,
`de_casteljau_surface_stages`, `bezier_isocurve`, `bezier_surface_partials`,
`bezier_surface_normal`, `bezier_surface_subdivide`); every claim above with a number attached is
covered in `tests/test_surfaces.py`, except the variation-diminishing remark, which is flagged as
stated-not-verified. Triangular Bezier patches and rational/B-spline surfaces are not implemented.
