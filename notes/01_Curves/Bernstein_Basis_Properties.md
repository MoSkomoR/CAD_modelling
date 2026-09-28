Code: [[../../scripts/01_curves/03_bezier_properties.py]] — `bernstein_basis` in `cadkernel/geometry/curves.py`; tests in `tests/test_curves.py`

# The Bernstein basis, and the proofs of the Bezier properties

Every useful property of a Bezier curve comes from two facts about its basis functions. Get
those two facts and the rest is bookkeeping. Run
[[../../scripts/01_curves/03_bezier_properties.py]] alongside this note — each section below is
one panel of that figure.

## The basis

$$B_i^n(t) = \binom{n}{i}(1-t)^{n-i}t^i, \qquad i = 0,\dots,n,$$
with the convention $B_j^k \equiv 0$ for $j<0$ or $j>k$, and the recursion
$B_j^k = (1-t)B_j^{k-1} + t\,B_{j-1}^{k-1}$ (Pascal's rule — derived in
[[De_Casteljau_Derivation]]).

**Fact 1 — non-negativity.** For $t \in [0,1]$: $\binom{n}{i} > 0$, $(1-t)^{n-i} \ge 0$ and
$t^i \ge 0$, so $B_i^n(t) \ge 0$. ∎

**Fact 2 — partition of unity.** By the binomial theorem,
$$\sum_{i=0}^n B_i^n(t) = \sum_{i=0}^n \binom{n}{i}(1-t)^{n-i}t^i = \big((1-t) + t\big)^n = 1. \qquad\blacksquare$$

Together: for each $t$, the numbers $B_0^n(t),\dots,B_n^n(t)$ are **non-negative and sum to 1** —
they are convex weights. So $C(t) = \sum_i B_i^n(t)P_i$ is, at every parameter value, a convex
combination of the control points. Almost everything below is a corollary.

(Tested: `test_bernstein_non_negative`, `test_bernstein_partition_of_unity`.)

## Endpoint interpolation

At $t=0$, $B_0^n(0)=1$ and $B_i^n(0)=0$ for $i>0$ (each carries a factor $t^i$); so $C(0)=P_0$.
Symmetrically $B_n^n(1)=1$ and the rest vanish, so $C(1)=P_n$. ∎

The interior control points are *not* interpolated — they only pull. That asymmetry is why
designers think of them as handles rather than as points on the shape.

## Convex hull property

**Claim.** For every $t\in[0,1]$, $C(t)$ lies in the convex hull of $\{P_0,\dots,P_n\}$.

**Proof.** The convex hull of a finite point set is *by definition* the set of all convex
combinations $\sum_i \lambda_i P_i$ with $\lambda_i \ge 0$ and $\sum_i \lambda_i = 1$. By Facts
1 and 2 the coefficients $\lambda_i = B_i^n(t)$ satisfy exactly those two conditions for each
fixed $t$. Hence $C(t)$ is a convex combination of the control points, so it lies in their
convex hull. ∎

That is the entire proof — which is the point: the property is not a geometric coincidence, it
is what "non-negative weights summing to one" *means*.

### What this buys a kernel

This is the property that does the most work in real code, for three reasons:

1. **A free conservative bounding volume.** The control points bound the curve, so the axis-aligned
   box of the control points bounds the curve too — no curve evaluation required. If two curves'
   boxes are disjoint, the curves provably do not intersect, and intersection code can discard
   the pair before any root-finding happens. Most candidate pairs die here.
2. **Guaranteed tessellation error bounds.** Subdivide ([[De_Casteljau_Derivation]]) and the
   control polygons converge to the curve. Because the curve is trapped in the hull, "this
   control polygon is flat to within ε" *implies* "the curve is within ε of that polygon" —
   a guarantee, not an estimate. This flatness test is how a kernel turns a spline into a
   polyline within a stated tolerance for display, export or meshing.
3. **Containment answers without evaluation.** Clearance and interference questions ("can this
   curve possibly leave this region?") get a sound conservative answer from the hull alone.

Combined with subdivision, this is the subdivide-and-cull pattern that makes B-rep intersection
algorithms tractable and robust. The top-left panel of script 03 shades the hull live: drag the
control points as hard as you like, the curve never escapes.

## Affine invariance

**Claim.** For any affine map $T(x) = Ax + b$: transforming the control points and then
evaluating gives the same curve as evaluating and then transforming.

**Proof.**
$$T\big(C(t)\big) = A\left(\sum_i B_i^n(t) P_i\right) + b = \sum_i B_i^n(t)\,A P_i + b.$$
Now use partition of unity to write $b = \left(\sum_i B_i^n(t)\right) b$:
$$= \sum_i B_i^n(t)\,A P_i + \sum_i B_i^n(t)\, b = \sum_i B_i^n(t)\big(A P_i + b\big) = \sum_i B_i^n(t)\, T(P_i). \qquad\blacksquare$$

The translation is exactly where $\sum_i B_i^n = 1$ is needed; without it the $b$ would not
distribute back into the sum. This is why partition of unity is not a technicality — it is the
property that makes control points a *geometric* handle rather than an arbitrary parametrization
of coefficients.

Practically: a kernel can transform a curve by transforming $n+1$ points, never touching the
parametrization, and every downstream algorithm still applies. (Tested to machine precision in
`test_bezier_affine_invariance`; the bottom-left panel of script 03 prints the residual live as
you rotate and shear.)

**Where it stops.** The proof uses only affine maps. Under a *projective* map (perspective) the
argument breaks — the denominators don't distribute. Polynomial Bezier curves are affinely but
not projectively invariant. Rational curves are both, because a projective map is just a linear
map upstairs in homogeneous coordinates, which is precisely where `rational_bezier` does its
work. One more reason NURBS is the representation kernels standardize on ([[Bezier_Curves]]).

## Global support (and why B-splines exist)

**Claim.** Moving any single control point changes the curve at *every* interior parameter.

**Proof.** $C(t)$ depends on $P_i$ only through the term $B_i^n(t)P_i$, so the sensitivity is
$$\frac{\partial C(t)}{\partial P_i} = B_i^n(t)\, I.$$
For $t \in (0,1)$ both $t^i > 0$ and $(1-t)^{n-i} > 0$, so $B_i^n(t) > 0$ strictly. Hence the
sensitivity is non-zero for every interior $t$ and every $i$: no control point has a region of
the curve it leaves alone. ∎

The support of $B_i^n$ is the whole interval $[0,1]$ — its only zeros there are the endpoints.
(Tested: `test_bernstein_global_support`; the bottom-right panel of script 03 plots
$\|C_{\text{nudged}}(t) - C(t)\|$ and reports the *minimum* interior displacement, which stays
strictly positive.)

This is the practical ceiling on plain Bezier curves. To model anything intricate you need many
control points, hence high degree, and then every one of them is a global edit: nudge a point to
fix a detail at one end and the far end moves too. A designer cannot work that way, and neither
can a solver.

The fix is not a better basis of the same kind but a **piecewise** one: replace the single
polynomial by segments glued with continuity conditions, and choose basis functions with
**compact support** — each nonzero on only a few spans. That is the B-spline basis
$N_{i,p}$, nonzero only on $[u_i, u_{i+p+1})$, so moving a control point perturbs at most $p+1$
spans and nothing else. Same convex-hull and affine-invariance guarantees, now local — built
and measured in [[B_Splines]].

## Variation diminishing

Stated without proof (it follows from the corner-cutting view of subdivision): **no straight
line crosses the curve more often than it crosses the control polygon.** So the curve cannot
oscillate more than its control polygon suggests — no Runge-style wiggles appearing between the
points you placed. This is the formal version of "Bezier curves do what they look like they will
do", and it is a large part of why the control-point paradigm is usable for design at all.

**Status**: proofs complete for the four properties above; variation diminishing is stated and
referenced, not derived. All four are exercised interactively by
[[../../scripts/01_curves/03_bezier_properties.py]] and covered by tests.
