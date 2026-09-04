Code: [[../../scripts/01_curves/01_lines_and_arcs.py]], [[../../scripts/01_curves/04_circle_needs_rational.py]]

# Lines and arcs — and why analytic geometry isn't enough on its own

The two curve types every CAD kernel treats as **exact analytic geometry**: evaluated from a
closed-form equation, not approximated by control points.

## The two forms

**Line.** $C(t) = (1-t)\,P_0 + t\,P_1,\ t\in[0,1]$ — interpolation between two endpoints. The
same lerp is the single operation De Casteljau's algorithm repeats to evaluate a Bezier curve of
any degree ([[De_Casteljau_Derivation]]).

**Arc.** $C(t) = c + r\big(\cos\theta(t), \sin\theta(t)\big)$ with
$\theta(t) = (1-t)\theta_0 + t\theta_1$. Exact and cheap — but note it is a **trigonometric**,
not polynomial, parametrization. That distinction turns out to matter enormously.

Both are exact, compact, and completely rigid: two endpoints, or a centre and two angles. Drag
the script's handles and you can feel how little they let you say. Anything that isn't a
straight line or a piece of a circle has no home here at all.

## Why not just add more analytic types?

The obvious response is to keep extending the catalogue — lines, arcs, ellipses, parabolas,
cylinders, cones, tori — and give each its own exact formula. Kernels really do carry such a
catalogue. But it cannot be the *only* representation, for two independent reasons.

**1. Analytic types are not closed under the operations a kernel must perform.** The catalogue
has to be closed under everything the modelling operations produce, and it isn't:
- **Intersection.** Two cylinders meeting at an oblique angle intersect in a quartic space curve
  that is not a line, arc, or conic. Intersections of analytic surfaces routinely leave the
  catalogue, and B-rep booleans are *built* on such intersections.
- **Offsetting.** The offset of a circle is a circle — fine. The offset of almost anything else
  is not of the same type as the original, and generally has no closed form at all.
- **Sweeps and lofts.** Sweep an arbitrary profile along an arbitrary path, or loft through a
  stack of cross-sections, and the result is whatever it is. There is no analytic name for it.

Since the results of these operations must be storable and then fed back in as inputs to further
operations, a kernel needs a representation **closed** under them — one general enough to hold
whatever comes out, at least to within tolerance. NURBS is that closure.

**2. The combinatorial explosion.** With $N$ curve types and $M$ surface types, a
special-cased intersector needs on the order of $N\times M$ routines — every pair written,
tested and maintained separately. With one universal representation you write one, and every
new capability applies to all geometry at once.

## The honest version: kernels keep both layers

The conclusion is *not* "analytic geometry is bad, NURBS is good". Real kernels keep both, and
for good reasons. OpenCASCADE has exact `Geom_Line`, `Geom_Circle`, `Geom_CylindricalSurface`
and so on, with special-cased routines that are faster and more accurate than the general path,
and it falls back to the general NURBS machinery when no analytic answer exists.

Keeping the analytic layer buys:
- **Exactness.** A circle that *is* a circle has exactly the right radius everywhere, exact
  tangency with its neighbours, and exact symmetry — none of it dependent on a tolerance.
- **Semantics.** A drilled hole should still be recognisably a cylinder: feature recognition,
  CAM toolpath selection, GD&T and manufacturing all ask "what kind of surface is this?" and
  need a truthful answer. A hole approximated by a spline patch has silently lost that.
- **Compactness and speed.** Three numbers instead of a control net, and closed-form
  intersections against other analytic types.

So: **analytic where you can, NURBS as the universal fallback.** The rest of this module is
about building that fallback.

## A circle is not a polynomial — proof

The gap between the two layers is not a matter of degree or effort. It is a theorem.

**Claim.** There is no non-constant pair of real polynomials $x(t), y(t)$ with
$x(t)^2 + y(t)^2 = r^2$ identically.

**Proof.** Suppose such polynomials exist and let $d = \max(\deg x, \deg y) \ge 1$. Write the
degree-$d$ coefficients of $x$ and $y$ as $\alpha$ and $\beta$ (one of them may be $0$ if that
polynomial has lower degree, but not both, by the choice of $d$). The coefficient of $t^{2d}$ in
$x(t)^2 + y(t)^2$ is then $\alpha^2 + \beta^2$. Since the sum is the constant $r^2$ and
$2d \ge 2 > 0$, that coefficient must vanish:
$$\alpha^2 + \beta^2 = 0.$$
Over the reals a sum of squares vanishes only if each term does, so $\alpha = \beta = 0$ —
contradicting that at least one of them is the leading coefficient of a degree-$d$ polynomial.
Hence $d = 0$: both are constants. $\blacksquare$

So **no polynomial Bezier or B-spline curve, of any degree, is exactly a circle.** Not "hard to
find" — nonexistent.

## What the demo actually shows

[[../../scripts/01_curves/04_circle_needs_rational.py]] fits polynomial Beziers of rising degree
to a quarter circle. The measured result is worth stating carefully, because the naive version
of this argument is wrong:

| degree | control points | max radial error |
|---|---|---|
| 2 | 3 | 2.0e-02 |
| 9 | 10 | 8.8e-11 |
| 13 | 14 | 2.8e-15 |
| 20 | 21 | 2.1e-15 (plateau) |
| **rational quadratic** | **3 (+3 weights)** | **3.3e-16** |

Convergence is *geometric* — roughly an order of magnitude per degree — and by degree 13 the
error has reached ~1e-15 and stops improving, because it has hit the floor of double-precision
arithmetic rather than because it reached the circle. **Polynomials approximate circles
extremely well.** Accuracy is not the argument.

The argument is:
- **Never exact, in principle** (the proof above) — you are permanently carrying an
  approximation whose error has to live in the kernel's tolerance bookkeeping.
- **Cost.** Machine-precision agreement costs 14 control points against 3. On tensor-product
  surfaces the gap squares: a sphere patch at ~14×14 = 196 control points versus 3×3 = 9.
- **Semantics and closure**, as above — the fitted curve is not a circle, cannot say it is one,
  and degrades under offsetting, transformation and re-intersection.

The fix is one extra number per control point. A weight $w_i$ per point, evaluated as a
polynomial curve in homogeneous coordinates and projected back down, gives
$$C(t) = \frac{\sum_i B_i^n(t)\,w_i P_i}{\sum_i B_i^n(t)\,w_i},$$
and a circular arc becomes an *exact* rational quadratic: three control points (the two arc
endpoints and the intersection of their tangents) with weights $(1, \cos(\delta/2), 1)$ for a
sweep of $\delta$. Verified to 3.3e-16 in `test_rational_quadratic_is_an_exact_circular_arc`.

That is the **R** in NURBS, and it is why the letter is there: rational is not a refinement of
the polynomial case, it is the thing that makes conics representable at all. The **NU** —
non-uniform knots, giving local control — is the other half of the story, motivated at the end
of [[Bernstein_Basis_Properties]] and in [[Bezier_Curves]].

**Status**: lines, arcs, and the exact rational quadratic arc implemented in code
(`cadkernel.geometry.curves.line`, `.arc`, `.rational_arc_quadratic`, `.rational_bezier`);
impossibility proof complete above; approximation behaviour measured, not assumed.
