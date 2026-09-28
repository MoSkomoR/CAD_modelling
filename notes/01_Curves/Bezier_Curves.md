Code: [[../../scripts/01_curves/02_bezier_de_casteljau.py]], [[../../scripts/01_curves/03_bezier_properties.py]]

# Bezier curves — why this representation

The motivation hub for module 01. The algorithm is in [[De_Casteljau_Derivation]], the property
proofs in [[Bernstein_Basis_Properties]], and the case against analytic-only geometry in
[[Lines_and_Arcs]]. This note answers the three "why" questions those don't.

A Bezier curve of degree $n$ is defined by $n+1$ control points and evaluated either as
$C(t) = \sum_{i=0}^n B_i^n(t)P_i$ or by De Casteljau's algorithm (proved equivalent).
Degree = number of control points − 1.

---

## Why a control-point representation at all?

A free-form curve has to be *some* finite set of numbers. The obvious candidates are worse than
they look.

**The power basis.** Write $C(t) = a_0 + a_1 t + a_2 t^2 + \dots$ and store the $a_i$. This is
the representation a numerical analyst reaches for first, and it fails as a *design* format:
- The coefficients have no geometric meaning. $a_3$ is not anywhere; you cannot draw it, drag
  it, or explain it to a user. There is no handle.
- It is numerically poor. The monomials $1, t, t^2, \dots$ become nearly parallel as functions on
  $[0,1]$ (the Hilbert-matrix problem), so the coefficients are badly conditioned: small changes
  in the curve mean wild changes in the $a_i$, and vice versa.
- None of the useful guarantees hold. No hull containment, no obvious bound on where the curve  goes, nothing a kernel can cull with.

**Interpolation through points.** Store $k$ points and pass a single polynomial through all of
them (Lagrange). Now the numbers *are* geometric — but:
- **Runge oscillation**: high-degree interpolants wiggle violently between the points, worst near
  the ends. The curve does things you did not ask for.
- Still no local control, and no containment guarantee.
- Interpolation is the wrong interface anyway: a designer wants to *pull* a shape into place,
  not to guess a sequence of points the curve must hit exactly.

**The Bernstein basis fixes exactly these.** Same polynomial curves — it is a change of basis,
nothing more — but:
- The coefficients **are points in space**. That is the whole trick: $C(t)$ is a convex
  combination of the $P_i$ at every $t$ ([[Bernstein_Basis_Properties]]), so the coefficients
  live in the same space as the curve and can be drawn, dragged, transformed and bounded.
- The curve is **predictable**: it stays in the convex hull, it is variation-diminishing (no
  oscillation the control polygon doesn't suggest), and it transforms with its control points.
- The basis is **well conditioned** — in fact Farouki and Rajan (1987) showed the Bernstein
  basis is *optimally* conditioned among all non-negative polynomial bases on $[0,1]$. The
  design-friendly basis and the numerically stable basis turn out to be the same one.

So the control-point paradigm is not a UI convenience layered over the maths; it *is* the
maths, in the basis where the coefficients happen to be geometry.

---

## What the basis actually looks like

The closed form $B_i^n(t) = \binom{n}{i}(1-t)^{n-i}t^i$ is compact enough to hide how simple the
individual functions are. Written out — factored on the left, expanded into the power basis on
the right:

**$n=1$ (a line — this *is* the lerp):**
$$B_0^1 = 1-t, \qquad B_1^1 = t.$$

**$n=2$ (quadratic — 3 control points):**
$$\begin{aligned}
B_0^2 &= (1-t)^2 &&= 1 - 2t + t^2\\
B_1^2 &= 2t(1-t) &&= 2t - 2t^2\\
B_2^2 &= t^2 &&= t^2
\end{aligned}$$

**$n=3$ (cubic — the workhorse of CAD and of every font and drawing program):**
$$\begin{aligned}
B_0^3 &= (1-t)^3 &&= 1 - 3t + 3t^2 - t^3\\
B_1^3 &= 3t(1-t)^2 &&= 3t - 6t^2 + 3t^3\\
B_2^3 &= 3t^2(1-t) &&= 3t^2 - 3t^3\\
B_3^3 &= t^3 &&= t^3
\end{aligned}$$

**$n=4$:**
$$\begin{aligned}
B_0^4 &= (1-t)^4 &&= 1 - 4t + 6t^2 - 4t^3 + t^4\\
B_1^4 &= 4t(1-t)^3 &&= 4t - 12t^2 + 12t^3 - 4t^4\\
B_2^4 &= 6t^2(1-t)^2 &&= 6t^2 - 12t^3 + 6t^4\\
B_3^4 &= 4t^3(1-t) &&= 4t^3 - 4t^4\\
B_4^4 &= t^4 &&= t^4
\end{aligned}$$

Things worth noticing, with the functions in front of you:

- **The leading coefficients are a row of Pascal's triangle** — $1$; $1,2,1$; $1,3,3,1$;
  $1,4,6,4,1$. That is not decoration: Pascal's rule is precisely what makes De Casteljau's
  recursion equal the closed form ([[De_Casteljau_Derivation]]).
- **Only the first and last survive at the endpoints.** Every $B_i^n$ with $0<i<n$ carries both a
  $t^i$ and a $(1-t)^{n-i}$ factor, so it vanishes at both ends. That single observation *is* the
  endpoint-interpolation proof.
- **Each one is a single bump.** $B_i^n$ rises to its unique maximum at $t = i/n$ and falls away —
  e.g. $B_1^3$ peaks at $t=1/3$ with value $4/9 \approx 0.444$. So $P_i$ has most influence
  roughly $i/n$ of the way along the curve, which is why dragging a control point feels local
  even though it strictly is not ([[Bernstein_Basis_Properties]]).
- **They are mirror images**: $B_i^n(t) = B_{n-i}^n(1-t)$, so reversing the control points
  reverses the curve and nothing else.
- **The expanded column is the argument against the power basis, made concrete.** Those
  right-hand coefficients — $3, -6, 3$ — are what you would store if you kept the curve in the
  power basis. They are signed, they cancel against each other, and not one of them is anywhere
  you could point at on the screen. The left-hand column is non-negative and sums to 1; the
  right-hand one is neither. Same curves, and all the structure lives in the basis you choose.

The top-right panel of [[../../scripts/01_curves/03_bezier_properties.py]] plots the whole family
for the curve on screen, with the control-point slider highlighting one $B_i^n$ at a time — worth
watching next to the bottom-right panel, which shows how much of the curve that same point moves.

---

## Why De Casteljau rather than the closed Bernstein form?

Both compute the same curve — that equivalence is proved in [[De_Casteljau_Derivation]]. The
closed form is shorter and asymptotically cheaper ($O(n)$ with a Horner-style scheme against
$O(n^2)$). Kernels use De Casteljau anyway. Before the reasons, it is worth doing both by hand
once, on the same curve, to see exactly what each one spends its work on.

### The same evaluation, both ways

Take the cubic with control points
$$P_0=(0,0),\quad P_1=(0,3),\quad P_2=(3,3),\quad P_3=(3,0)$$
and evaluate at $t=\tfrac13$ (so $1-t=\tfrac23$). Every number below is exact.

**Route 1 — the closed Bernstein form.** Evaluate the four basis functions, then take one
weighted sum of the control points:
$$B_0^3=\left(\tfrac23\right)^3=\tfrac{8}{27},\quad
B_1^3=3\left(\tfrac13\right)\left(\tfrac23\right)^2=\tfrac{12}{27}=\tfrac49,\quad
B_2^3=3\left(\tfrac13\right)^2\left(\tfrac23\right)=\tfrac{6}{27}=\tfrac29,\quad
B_3^3=\left(\tfrac13\right)^3=\tfrac{1}{27}.$$
(Check: $\tfrac{8+12+6+1}{27}=1$ — partition of unity, and note $B_1^3=\tfrac49$ is its peak
value, since $t=\tfrac13=i/n$.) Then
$$C\!\left(\tfrac13\right)=\tfrac{8}{27}(0,0)+\tfrac{12}{27}(0,3)+\tfrac{6}{27}(3,3)+\tfrac{1}{27}(3,0)
=\tfrac{1}{27}\big(0+0+18+3,\ 0+36+18+0\big)=\left(\tfrac79,\ 2\right).$$

**Route 2 — De Casteljau.** No binomials, no powers: six lerps at the same $t=\tfrac13$, each one
$\tfrac23 A+\tfrac13 B$.

| level | points |
|---|---|
| $k=0$ | $(0,0)\quad (0,3)\quad (3,3)\quad (3,0)$ |
| $k=1$ | $(0,1)\quad (1,3)\quad (3,2)$ |
| $k=2$ | $\left(\tfrac13,\tfrac53\right)\quad \left(\tfrac53,\tfrac83\right)$ |
| $k=3$ | $\left(\tfrac79,\,2\right)$ |

e.g. $P_0^1=\tfrac23(0,0)+\tfrac13(0,3)=(0,1)$, and at the top
$P_0^3=\tfrac23\left(\tfrac13,\tfrac53\right)+\tfrac13\left(\tfrac53,\tfrac83\right)=\left(\tfrac{2+5}{9},\tfrac{10+8}{9}\right)=\left(\tfrac79,2\right).$

Same point, $\left(\tfrac79, 2\right)$, as it must be. (Pinned in
`test_worked_cubic_example_from_the_notes`, exact fractions and all.)

**What the two computations actually differ in:**

|                                 | closed Bernstein form                      | De Casteljau                                                                                                          |
| ------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| what is computed                | $n+1$ basis values, then one weighted sum  | $n(n+1)/2$ lerps — 6 for a cubic                                                                                      |
| cost per parameter value        | $\Theta(n)$ with a Horner-style scheme     | $\Theta(n^2)$                                                                                                         |
| needs binomials & powers        | yes                                        | no — only $t$ and $1-t$                                                                                               |
| intermediate quantities         | scalars: numbers with no geometric meaning | **points**: every one lies in the hull, and the whole triangle is on screen in script 02                              |
| by-products                     | none                                       | both half-curves' control polygons, and $C'(t)=n\,(P_1^{n-1}-P_0^{n-1})$ — the last lerp's direction *is* the tangent |
| extends to surfaces / rationals | needs new algebra                          | unchanged (affine combinations only)                                                                                  |

The bottom two rows are the whole argument. Route 1 turns the control points into numbers and
gives you back a point. Route 2 never leaves the space the geometry lives in, so its scratch work
is itself geometry you can use.

### The three reasons

**1. Subdivision comes for free — this is the big one.** Evaluating at $t=s$ *already computes*
the control polygons of both halves of the curve, split at $s$: they are the two outer edges of
the triangle. Nothing equivalent falls out of the closed form.

That single fact, combined with the convex hull property, is the foundation of most robust
curve algorithms in a kernel: intersection by recursive subdivide-and-cull (compare boxes,
discard disjoint pairs, subdivide the rest), adaptive tessellation with *guaranteed* error
bounds via the flatness test, ray casting, trimming. An evaluator gives you points on a curve;
subdivision gives you a divide-and-conquer strategy for reasoning about the whole curve.

**2. Numerical stability — but measure it before believing the usual story.** The textbook
argument is that De Casteljau is stable because every step is a convex combination with weights
in $[0,1]$, so intermediates stay inside the hull of their inputs and nothing can cancel
catastrophically, whereas the closed form multiplies enormous binomials by tiny powers
($\binom{50}{25}\approx1.26\times10^{14}$ against $t^{25}$) and relies on the magnitudes
cancelling back down.

The first half is true. The implied conclusion is not. Compared against exact rational arithmetic
on the same control points, the naive closed form is **just as accurate as De Casteljau** — both
sit at $\sim10^{-16}$ relative error at degree 3, 20, 50, and still at degree 800, and both hold
up when extrapolating outside $[0,1]$ where signs finally do alternate. The closed form's only
hard failure is *overflow*: `math.comb(n, n/2)` exceeds the float64 range at $n = 1030$, and
`bernstein_basis` raises. Every degree a CAD kernel will ever see is four hundred times below
that.

The reason is the one Farouki and Rajan gave: on $[0,1]$ the Bernstein basis functions are all
non-negative, so the weighted sum has **no subtractive cancellation between terms** — the large
binomials are multiplied by correspondingly tiny powers and the products come out ordinary-sized.
The stability belongs to the *basis*, which both routes share, not to the algorithm. So this is a
real property of De Casteljau and a bad reason to prefer it. (Measured in
`test_closed_form_and_de_casteljau_agree_at_high_degree`.)

**3. Generality.** It is nothing but repeated lerp, so it extends without re-derivation: to any
dimension, to tensor-product surfaces (run it in $u$, then in $v$), and to rational curves (run
it in homogeneous coordinates and project). The closed form needs new algebra each time.

At the degrees CAD actually uses — 2 to 5 — the asymptotic cost difference is noise (6 lerps
against 4 basis values), and subdivision is decisive on its own. The honest ranking is that
reason 1 carries this argument, reason 3 supports it, and reason 2 is a property De Casteljau has
rather than an advantage it wins.

---

## Properties, and the one that limits us

All proved in [[Bernstein_Basis_Properties]], all visible live in
[[../../scripts/01_curves/03_bezier_properties.py]]:

- **Endpoint interpolation** — $C(0)=P_0$, $C(1)=P_n$; interior points only pull.
- **Convex hull containment** — the curve never leaves the hull of its control points. This is
  what lets a kernel bound and cull before evaluating anything.
- **Affine invariance** — transform the control points, get the transformed curve. Turns on
  partition of unity; *fails* for projective maps, which rational curves fix.
- **Variation diminishing** — no line crosses the curve more often than the control polygon.
- **Global support** — and this one is a problem.

Every basis function $B_i^n$ is strictly positive on all of $(0,1)$, so **every control point
affects every interior point of the curve**. Model something intricate and you need many control
points, hence high degree, and then every edit is global: nudge one point to fix a detail at one
end and the far end moves. Worse, high-degree single polynomials get numerically delicate and
expensive to intersect.

---

## Where this goes: NURBS as *Bezier + two fixes*

Each letter of NURBS is a specific repair to a specific limitation established above.

**Piecewise, with local control (the B and the NU).** The fix for global support is not a better
basis of the same kind but a *piecewise* one: chain low-degree segments with continuity
conditions, using basis functions with **compact support**. The B-spline basis $N_{i,p}$ is
non-zero only on $[u_i, u_{i+p+1})$, so a control point perturbs at most $p+1$ spans and nothing
else — local editing, bounded degree, and the same hull and affine guarantees. The knot vector
that positions those spans is the "non-uniform" part, and it also lets the curve be shaped or
made to interpolate its ends by repeating knots.

**Rational (the R).** The fix for the fact that *no polynomial of any degree is exactly a
circle* ([[Lines_and_Arcs]], with proof). One weight per control point, evaluated in homogeneous
coordinates:
$$C(t) = \frac{\sum_i B_i^n(t)\,w_i P_i}{\sum_i B_i^n(t)\,w_i}.$$
This buys the exact conics — circles, ellipses, parabolas, hyperbolas — at three control points
instead of fourteen, and it upgrades affine invariance to **projective** invariance, since a
projective map is just a linear map one dimension up. The weights are also an extra shape handle
in their own right.

The two fixes compose, and the result is the single representation kernels standardize on and
STEP/IGES interchange: NURBS can express every line, arc and conic *exactly*, every free-form
shape to tolerance, with local control, one code path for intersection, offsetting and
tessellation — which is exactly the closure [[Lines_and_Arcs]] argued a kernel cannot do without.

**Status**: implemented exactly in code (`bezier_curve`, `bezier_de_casteljau`,
`de_casteljau_triangle`, `bezier_subdivide`, `bernstein_basis`, plus `rational_bezier` as an
early taste of the R); tested in `tests/test_curves.py` — the explicit expansions, the peak and
symmetry facts, the worked $t=\tfrac13$ evaluation (both routes, exact fractions), the tangent
by-product and the high-degree accuracy comparison each have a test. B-splines are implemented in
[[B_Splines]], [[De_Boor_Algorithm]] and [[B_Spline_Interpolation]]; full NURBS not yet — see
[[../00_Map_of_Content|Map of Content]].
