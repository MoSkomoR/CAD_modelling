Code: [[../../scripts/01_curves/08_nurbs_weights.py]] — `nurbs_curve`, `nurbs_basis`, `nurbs_derivative`, `nurbs_insert_knot`, `nurbs_circle`, `nurbs_projective_map`, `rational_bezier`, `rational_arc_quadratic` in `cadkernel/geometry/nurbs.py`; tests in `tests/test_nurbs.py`

# NURBS — B-splines with weights

The motivation hub for the last piece of module 01. The proofs are in [[NURBS_Derivations]].

## Where we are

[[Bezier_Curves]] ended with two limitations and promised a fix for each ("NURBS as *Bezier + two
fixes*"):

1. **Global support** — every control point moves the whole curve. Fixed by [[B_Splines]]: a
   piecewise basis on a knot vector, local control, bounded degree. That is the **NU** and the **B**.
2. **No polynomial is a circle** — proved in [[Lines_and_Arcs]], for any degree, and B-splines are
   still polynomial on every span. That is what is left, and it is the **R**.

## Why rational, and not something else

The problem is narrow: circles, and conics in general, are the most common curves in mechanical
CAD, and no polynomial can be one. The candidate fixes:

- **Fit them with polynomials to tolerance.** [[Lines_and_Arcs]] measured this honestly: it works
  numerically (machine precision by degree ~13), and that is *not* the argument against it. The
  argument is that it is never exact, costs 14 control points against 3, and cannot answer "is this
  a circle?" — the semantics a kernel needs for offsets, tangency and feature recognition.
- **Keep analytic circles beside the polynomials.** Real kernels do this — analytic where you
  can ([[Lines_and_Arcs]]). But then every algorithm has a separate case for every pair of types,
  and a construction that leaves the analytic world (a perspective view of a circle, a blend between
  an arc and a free-form curve) has nowhere to go. A kernel still needs one general type.
- **Trigonometric or other non-polynomial bases.** Exact for circles, but they give up everything
  module 01 built: no De Casteljau, no de Boor, no knot insertion, no Bezier decomposition.

**Rational** is the smallest extension that keeps all of it. Give each control point a weight, and
evaluate an ordinary B-spline one dimension up, on the lifted points $(w_iP_i, w_i)$; then divide by
the last coordinate:
$$C(t) = \frac{\sum_i N_{i,p}(t)\,w_iP_i}{\sum_j N_{j,p}(t)\,w_j} = \sum_i R_{i,p}(t)\,P_i,
\qquad R_{i,p} = \frac{w_iN_{i,p}}{\sum_j w_jN_{j,p}} .$$
Every algorithm from the polynomial case runs unchanged on the lifted points, and one division
comes at the end ([[NURBS_Derivations]] §1). `nurbs.py` contains no new evaluation algorithm at
all — `nurbs_curve` is lift, `bspline_curve`, project.

## What carries over, what is new, what is lost

Section numbers (§) refer to [[NURBS_Derivations]].

| | property | status with weights |
|---|---|---|
| carries over | partition of unity, local support of $R_{i,p}$ | for **any** weights (§2) |
| carries over | convex hull, affine invariance, the hull-based culling | for **positive** weights only (§2) |
| carries over | knot insertion, Bezier decomposition, de Boor | applied to the lifted points (§9) |
| carries over | the B-spline and the rational Bezier | the special cases $w_i = 1$, and Bezier knots |
| **new** | exact conics: arcs, full circle, ellipse, parabola, hyperbola | §3, §4, §7 |
| **new** | projective invariance | §5 |
| lost | derivative is a quotient, degree $2n-2$ over $2n$ | §6 |
| lost | unique control data: weights carry pure parametrization | §8 |

**The two new properties.**

- **Exact conics.** A quarter arc is three control points with weights $(1, \cos 45°, 1)$, exact to
  rounding (`test_rational_quadratic_is_an_exact_circular_arc`). The whole circle is one degree-2
  NURBS on nine points (`test_full_circle_is_exact`). And the type of conic is set by one number:
  middle weight $<1$ ellipse, $=1$ parabola, $>1$ hyperbola
  (`test_middle_weight_selects_the_conic_type`) — the middle panel of script 08.
- **Projective invariance.** Affine invariance was already true of Bezier curves and B-splines
  ([[Bernstein_Basis_Properties]] said where it stops: perspective). A projective map is linear on
  the lifted points, so a NURBS is mapped exactly by mapping its control data — but the *weights
  change* with it, which is why a polynomial curve, having no weights, cannot follow. Exact to
  $10^{-14}$, against an error above $0.4$ for moving the points alone
  (`test_nurbs_is_projectively_invariant`). Tick "perspective" in script 08. The perspective image
  of a circle is an exact ellipse (`test_perspective_image_of_a_circle_is_an_exact_ellipse`). For a
  kernel this is less about rendering than about closure: a class of curves that is mapped into
  itself by every transformation the kernel uses.

## What it costs — the honest part

- **Exact shape, inexact speed.** The NURBS circle is the circle, but it is not traversed at
  constant speed: $|C'(t)|$ varies by a factor $4 - 2\sqrt2 \approx 1.17$, and the angle drifts from
  $2\pi t$ by $0.92°$ (`test_circle_is_not_arc_length_parametrized`; the ticks in script 08's right
  panel). Anything that needs arc length — constant-speed toolpaths, equally spaced points — still
  has to reparametrize numerically. (No rational parametrization of the circle has constant speed;
  that is cited, not proved here, see [[../99_Not_Covered_In_Code]].)
- **Derivatives get expensive.** $C' = (A'w - Aw')/w^2$: a quotient of degree $2n-2$ over $2n$, not a
  curve of degree $n-1$ (`test_derivative_degree_inflates`). The curve analogue of the normal-field
  blow-up in [[../02_Surfaces/Bezier_Surfaces]]; everything built from derivatives (curvature,
  offsets, intersections) inherits it.
- **Weights are a second non-geometric input, and partly redundant.** Like knots, they are not
  visible on screen. Worse, they are not even unique: scaling them geometrically, $w_i \to \lambda^iw_i$,
  leaves the shape alone and only reparametrizes (`test_scaling_weights_geometrically_only_reparametrizes`).
  Two different weight vectors can describe the same curve.
- **Operations must happen upstairs** — in homogeneous space, on the lifted points
  $(w_iP_i,\ w_i)$, not on $P_i$ and $w_i$ apart. Knot insertion is the example: it replaces points
  by lerps $(1-\alpha)\,\cdot_{i-1} + \alpha\,\cdot_i$ of their neighbours, with $\alpha \in [0,1]$ set by
  the knots, and that keeps the curve unchanged *only for the B-spline whose control points were
  lerped*. With $w_i' = (1-\alpha)w_{i-1} + \alpha w_i$ the new weight in both cases, the new point is
  $$Q_i^{\text{upstairs}} = \frac{(1-\alpha)\,w_{i-1}P_{i-1} + \alpha\,w_iP_i}{w_i'}
  \qquad\text{vs.}\qquad Q_i^{\text{separately}} = (1-\alpha)\,P_{i-1} + \alpha\,P_i .$$
  The upstairs version preserves both $A = \sum_i N_{i,p}w_iP_i$ and $w = \sum_i N_{i,p}w_i$, hence
  $C = A/w$. The separate version preserves $w$ and $\sum_i N_{i,p}P_i$ — the unweighted B-spline,
  which appears nowhere in $C$. The correct point is a *weight-biased* lerp, pulled toward the heavier neighbour; the two agree
  only when $w_{i-1} = w_i$, which is why a test with equal weights would never catch the mistake.
  The wrong one moves the curve by $0.22$ (`test_knot_insertion_must_happen_in_homogeneous_space`).
  The bug is silent: the result is still a valid NURBS with positive weights, smooth and close to
  the original — just the wrong curve. Subdivision, Bezier decomposition and degree elevation obey
  the same rule ([[NURBS_Derivations]] §9).
- **Non-positive weights forfeit the guarantees.** Zero switches a control point off; negative ones
  leave the convex hull (to $y = -2$ under a triangle in $y \ge 0$,
  `test_negative_weight_leaves_the_convex_hull`) or put a pole on the curve (weights $(1,-1,1)$,
  `test_weight_minus_one_puts_a_pole_on_the_curve`). Negating the middle weight of a circular arc
  draws the *complementary* arc, exactly on the circle and entirely outside its control triangle
  (`test_negative_middle_weight_gives_the_complementary_arc`). Kernels therefore require positive
  weights.

## Are weights a design handle?

[[Bezier_Curves]] called weights "an extra shape handle in their own right". Measured, they are a
real one: raising $w_i$ pulls the curve toward $P_i$ — for the quadratic, to distance
$|P_1 - M|/(1+w)$ from it, $M$ the chord midpoint (`test_large_weight_pulls_the_curve_to_its_point`) —
and only over the support of $N_{i,p}$, the same locality as moving the point
(`test_changing_one_weight_is_local`). Script 08's left panel does exactly this.

But the honest verdict is weaker than the promise. The same pull is available by moving the control
point, which is visible and intuitive; the weight is neither, and part of what it changes is pure
parametrization (§8). What weights do that points cannot is the *exact* conic, and the usual account
in the literature is that this is where weights actually come from: produced by constructions —
arcs, circles, conics, surfaces of revolution — rather than set by hand (Piegl & Tiller build conics
and circles this way, ch. 7). This repo cannot measure how designers use software, so that last
claim is cited, not tested.

## Where this goes

Module 01 is now complete: Bezier → B-splines → NURBS, each fix proved or measured. The same lift
works for tensor-product surfaces, one direction at a time, and closes the gap
[[../02_Surfaces/Bezier_Surfaces]] named: sphere, cylinder, cone and torus all have circular
isocurves, so none of them is a polynomial patch — and all of them are NURBS surfaces. That is the
next step, in module 02.

**Status**: exact. `nurbs_curve`, `nurbs_basis`, `nurbs_derivative`, `nurbs_insert_knot`,
`nurbs_circle`, `nurbs_projective_map`, `rational_bezier`, `rational_arc_quadratic` implemented and
tested in `tests/test_nurbs.py`; script 08 visualizes weight pull, conic type, projective invariance
and the full circle. Proved inline in [[NURBS_Derivations]]: the homogeneous form, the rational basis
properties, exact arcs, projective invariance, the derivative and its degree bound, the
reparametrization freedom. Stated and measured: the speed variation of the circle, that the degree
bound is attained. Cited, not proved: the classification of conics by their points at infinity, that
no rational circle has constant speed, and how weights are used in practice.
