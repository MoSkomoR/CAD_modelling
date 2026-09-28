Code: [[../../scripts/01_curves/05_bspline_basis.py]], [[../../scripts/01_curves/06_bspline_local_control.py]] — `cadkernel/geometry/bspline.py`; tests in `tests/test_bspline.py`

# B-splines — Bezier, allowed to break into pieces

The motivation hub for B-splines. The evaluation algorithm, knot insertion and degree elevation
are in [[De_Boor_Algorithm]]; fitting a curve *through* points is in [[B_Spline_Interpolation]].
The rational extension (the R in NURBS) comes after.

## The problem being fixed

[[Bernstein_Basis_Properties]] ended on the one Bezier property that is a liability: **global
support**. Every $B_i^n$ is strictly positive on all of $(0,1)$, so every control point moves
every interior point of the curve. Measured on a 10-point zig-zag polygon
(`test_moving_a_control_point_changes_only_p_plus_one_spans`): raising $P_5$ by one unit moves
**100%** of the degree-9 Bezier curve — and moves it by at most **0.26**, because a degree-9
Bernstein bump is low and wide. The handle is global *and* weak.

The goal is a representation where a control point is a local handle, the degree stays low no
matter how many points there are, and every guarantee from module 01 (hull, affine invariance,
an algorithm made of lerps) survives.

## Why not the two obvious fixes

**1. Just use more control points on one Bezier curve.** That *is* the problem: more points means
higher degree, and degree-$n$ support is always global. No choice of points fixes it.

**2. Chain several low-degree Bezier curves end to end.** Now each piece is local — but the joins
are your problem. For a $C^1$ join between cubics, the last edge of one polygon and the first edge
of the next must be collinear *with a fixed length ratio*; for $C^2$, a further condition ties
three points on each side. Move one point near a join and the constraints break; restoring them
means moving neighbours, which can break the next join. Continuity is a **constraint you
maintain**, and every editing operation has to know about it.

Count the freedom that is left. $S$ cubic pieces have $4S$ control points; $C^2$ at the $S-1$
joins imposes three conditions each (position, first and second derivative), leaving
$4S - 3(S-1) = S + 3$ free points. **B-splines are a basis for exactly that constrained space**: a
cubic B-spline with $S$ spans has $S+3$ control points, and *every* placement of them gives a
$C^2$ curve. Nothing to maintain — the continuity is in the basis functions, not in the data. That
is the whole idea, and everything below is making it precise.

## The knot vector

A B-spline of degree $p$ with control points $P_0,\dots,P_n$ needs a non-decreasing **knot
vector** $u_0 \le u_1 \le \dots \le u_m$, $m = n+p+1$. The knots are the parameter values where
the polynomial pieces meet, and they carry every design decision the Bezier curve lacked:

- **spacing** — where along the parameter the pieces change (the "non-uniform" in NURBS);
- **multiplicity** — how smooth each join is (below);
- **the ends** — repeating the first and last knot $p+1$ times (**clamped** knots,
  `clamped_knots`) makes the curve start at $P_0$ and end at $P_n$
  (`test_clamped_bspline_interpolates_its_end_points`). Unclamped knots are legal too; the curve
  then starts somewhere inside the polygon, which is useful for closed curves and useless for
  most CAD edges.

The curve is defined on $[u_p, u_{n+1}]$.

## The basis: Cox–de Boor

$$N_{i,0}(t) = \begin{cases}1 & u_i \le t < u_{i+1}\\ 0 & \text{otherwise}\end{cases}$$
$$N_{i,d}(t) = \frac{t-u_i}{u_{i+d}-u_i}\,N_{i,d-1}(t) + \frac{u_{i+d+1}-t}{u_{i+d+1}-u_{i+1}}\,N_{i+1,d-1}(t),$$
with $0/0 := 0$ (a repeated knot makes a span empty, and a function built on an empty span
contributes nothing). The curve is
$$C(t) = \sum_{i=0}^n N_{i,p}(t)\,P_i.$$

Read the recursion as a construction. Degree 0 is a row of **box functions**, one per knot span.
Each step blends two neighbouring functions with linear ramps, producing one that is a span wider
and one degree smoother. Script 05 draws it level by level — the "level d" slider turns boxes into
tents into bumps.

### The three facts, proved

**Local support.** $N_{i,p}(t) = 0$ for $t \notin [u_i, u_{i+p+1})$.

*Proof by induction on $d$.* $N_{i,0}$ is zero outside $[u_i,u_{i+1})$. If $N_{i,d-1}$ vanishes
outside $[u_i,u_{i+d})$ and $N_{i+1,d-1}$ outside $[u_{i+1},u_{i+d+1})$, their weighted sum
vanishes outside the union $[u_i,u_{i+d+1})$. $\blacksquare$

$N_{i,p}$ was built from $p+1$ boxes and cannot reach further. So **moving $P_i$ changes the curve
only on $[u_i, u_{i+p+1})$ — at most $p+1$ spans**, and by *exactly* zero elsewhere, not merely a
small amount: measured, on the same 10-point polygon as above but as a cubic B-spline, raising
$P_5$ changes 4 of the 7 spans (57% of the parameter range), the change outside is `0.0` to the
last bit, and the peak response is **0.67** instead of 0.26 — local *and* strong. Script 06 shows
it live. (`test_moving_a_control_point_changes_only_p_plus_one_spans`,
`test_bspline_basis_has_local_support`.)

**Non-negativity.** $N_{i,p}(t) \ge 0$.

*Proof by induction.* Boxes are $\ge 0$. In the recursion, the first ramp $\frac{t-u_i}{u_{i+d}-u_i}$
multiplies $N_{i,d-1}$, which is non-zero only where $t \ge u_i$ — so the ramp is $\ge 0$ wherever
it matters. Likewise the second ramp is $\ge 0$ wherever $N_{i+1,d-1} \ne 0$, since there
$t < u_{i+d+1}$. A sum of non-negative terms is non-negative. $\blacksquare$

**Partition of unity.** $\sum_i N_{i,p}(t) = 1$ for $t$ in the domain $[u_p, u_{n+1}]$.

*Proof.* Sum the recursion over $i$ and shift the index of the second term ($i+1 \to i$):
$$\sum_i N_{i,d}(t) = \sum_i \left[\frac{t-u_i}{u_{i+d}-u_i} + \frac{u_{i+d}-t}{u_{i+d}-u_i}\right] N_{i,d-1}(t) = \sum_i N_{i,d-1}(t).$$
The bracket is exactly 1: the two ramps meeting on the same function are complementary. (On the
domain, the terms that fall off either end of the index range are zero by local support, so the
shift loses nothing.) Repeating down to degree 0 leaves $\sum_i N_{i,0}(t) = 1$, since $t$ lies in
exactly one span. $\blacksquare$

(`test_bspline_basis_is_non_negative_and_a_partition_of_unity`, including uneven and repeated
knots; the dashed "sum" line in script 05 is flat at 1 however you drag the knots.)

### What follows, for free

These are the same two facts Bezier curves rested on, so the same corollaries hold — and one gets
sharper:

- **Affine invariance** — identical proof to [[Bernstein_Basis_Properties]]
  (`test_bspline_affine_invariance`).
- **Strong convex hull.** At $t$ in span $[u_k,u_{k+1})$ only $N_{k-p,p},\dots,N_{k,p}$ are
  non-zero, so $C(t)$ is a convex combination of just the $p+1$ points $P_{k-p},\dots,P_k$. Each
  piece lies in the hull of its *own* $p+1$ points — a far tighter bound than the hull of the whole
  polygon, and a better one to cull with (`test_strong_convex_hull`).
- **Bezier is the special case.** With no interior knots, $0^{p+1}1^{p+1}$, every ramp is $t$ or
  $1-t$, Cox–de Boor becomes Pascal's rule from [[De_Casteljau_Derivation]], and $N_{i,p} = B_i^p$
  (`test_bezier_is_a_special_case_of_bspline`, to $10^{-14}$ for degrees 1, 2, 3, 5). A B-spline
  is not a different kind of curve — it is a Bezier curve that has been allowed to break into
  pieces.

## Continuity is read off the knot vector

**Claim.** Across a knot of multiplicity $k$, a degree-$p$ B-spline is $C^{p-k}$ — and no better,
for generic control points.

The intuition: across a simple knot, the two neighbouring pieces share $p$ of their $p+1$ active
control points, and that shared data pins down the value and the first $p-1$ derivatives. Every
repeat of the knot removes one shared point, and with it one order of continuity. At multiplicity
$p$ the pieces share a single point: $C^0$, a corner is allowed, and the curve passes *through*
that control point (this is also why the clamped ends interpolate). This is stated rather than
proved here — the proof goes through the derivative formula in [[De_Boor_Algorithm]] applied $p-k+1$
times — and **measured** instead: for a cubic with a knot of multiplicity 1, 2, 3 at $t=0.5$,
the derivative jumps are below $10^{-6}$ (finite-difference noise at $\varepsilon=10^{-9}$) up to
order $3-k$, and $\ge 1$ — in practice tens to hundreds — at order $3-k+1$
(`test_continuity_is_p_minus_multiplicity`).

This is the design lever the chained-Bezier approach made painful. Want a crease? Raise one knot's
multiplicity. Want the curve to hit a control point? Multiplicity $p$. Want it smooth again? One
knot. In script 05, raise "middle mult." and watch the functions around it narrow until one of
them reaches 1.

## What it costs — the honest part

- **The knot vector is a second, less intuitive input.** Control points are geometry; knots are
  not anywhere on screen. Most users never touch them, which is why clamped uniform knots are the
  default in almost every tool — and why fitting ([[B_Spline_Interpolation]]) has to *choose*
  them.
- **Still polynomial.** Every piece is a polynomial, so the circle proof in [[Lines_and_Arcs]]
  applies unchanged: no B-spline of any degree or knot vector is exactly a circle. That is the
  remaining limitation, and the one weights fix.
- **Evaluation needs a span search.** Bezier evaluation starts immediately; a B-spline must first
  find which span $t$ is in (`find_span`, a binary search). Cheap, but a source of real bugs at the
  right end of the domain, where the half-open spans would otherwise leave $t = u_{n+1}$ uncovered
  (`test_find_span_covers_the_closed_domain`).

**Status**: exact. `clamped_knots`, `find_span`, `bspline_basis_levels`, `bspline_basis`,
`bspline_curve` implemented; every claim above with a number is covered in
`tests/test_bspline.py`. Local support, non-negativity and partition of unity are proved inline;
the $C^{p-k}$ continuity theorem is stated and measured, not proved. Script 05 visualizes the
basis, script 06 local control.
