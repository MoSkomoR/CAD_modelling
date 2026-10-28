Code: [[../../scripts/04_modeling_ops/03_loft.py]] — `loft` in [[../../src/cadkernel/ops/loft.py]], `compatible`, `skin_parameters`, `skinned_surface` in [[../../src/cadkernel/geometry/skinning.py]]; tests in `tests/test_loft.py`, `tests/test_skinning.py`

# Loft — a solid through a stack of sections

Some shapes are easiest to describe by their cross-sections: a duct that is square at one end and
round at the other, a bottle's neck, an aircraft fuselage, a hull drawn station by station. The
designer fixes a few sections, and the kernel must fill in a smooth surface between them. That is
a **loft**, and its surface is a **skinned** surface.

**Two rejected alternatives.**
- **Straight lines between consecutive sections** (a ruled surface per gap). It is simple, but the
  surface has a crease, a jump in its tangent plane, at every inner section. `loft` does produce
  this with `degree=1`, for comparison.
- **A separate patch per gap**, each fitted on its own: the same creases, plus a seam between
  patches.

The standard construction, used here, is one surface through *all* the sections at once
(Piegl & Tiller, *The NURBS Book*, §10.3; cited). Its curves across the sections are $C^2$ cubic
interpolants, so there are no creases. That interpolation is the curve interpolation of
[[../01_Curves/B_Spline_Interpolation]], and it brings the parametrization question back in a new
direction.

## The input

- **Sections:** $K + 1 \ge 2$ closed profiles $\mathcal C^0, \dots, \mathcal C^K$. Each is a loop of
  the same number $n$ of curves, and $C_k^s$ is curve $k$ of section $s$.
- **Correspondence by index:** curve $k$ of every section belongs to side face $k$, and vertex
  $k$ (the start of curve $k$) of every section lies on one longitudinal edge. All sections must
  run the same way round. `loft` reverses all of them together if needed, which keeps the
  correspondence.
- **Caps:** the first and last sections must be planar. The ones in between need not be.
- **End weights 1:** every section curve starts and ends with weight 1. That is true of lines,
  Beziers and `circle_arcs`; the reason is under [[#The longitudinal edges]].

## Step 1: compatible sections

A tensor-product surface
$$S(u, v) = \frac{\sum_i \sum_j N_{i,p}(u)\,N_{j,q}(v)\,w_{ij}\,P_{ij}}{\sum_i \sum_j N_{i,p}(u)\,N_{j,q}(v)\,w_{ij}}$$
has *one* basis $N_{i,p}(u)$ in $u$, shared by every $v$. Here $P_{ij}$ is the control net, $w_{ij}$
its weights, $p, q$ the degrees and $N$ the B-spline basis functions. So to put the sections into
one surface, the curves $C_k^0, \dots, C_k^K$ must first be written on one degree and one knot
vector.

In the square-to-circle loft of script 03, edge $k$ is a straight line in the squares (degree 1)
and a rational quadratic arc in the circles (degree 2). `compatible` rewrites them without
changing their shapes:
1. **Degree:** each curve is raised to the highest degree present, by `elevate_degree`
   ([[../01_Curves/De_Boor_Algorithm]]).
2. **Knots:** every knot value is inserted into every curve until it has, in every curve, the
   highest multiplicity it has in any curve. This uses Boehm insertion, `insert_knot`.

Both steps act on the lifted points $(w_i P_i, w_i)$, as every NURBS operation must
([[../01_Curves/NURBS]]). Afterwards curve $C_k^s$ has control points
$P_{0}^s, \dots, P_{m}^s$, the same $m$ for every $s$. Measured on a rational arc, a cubic with an
interior knot and a line: one degree, one knot vector, and the same points at the same parameters
to $10^{-14}$ (`test_compatible_keeps_every_shape_and_lines_up_the_representation`).

There is a cost. With no knot removal ([[../99_Not_Covered_In_Code]]), elevated curves keep knots
they don't need, and the net grows.

## Step 2: interpolation across the sections

Choose parameters $0 = v_0 < v_1 < \dots < v_K = 1$, one per section. For each $i$, the points
$P_i^0, \dots, P_i^K$ are data, and a B-spline of degree $q$ in $v$ through them gives column $i$ of
the net. Written out, with knots $\tau$ averaged from the $v_s$ as in curve interpolation, the
unknowns $Q_{ij}$ solve
$$\sum_{j=0}^{K} N_{j,q}(v_s)\,Q_{ij}^w = (P_i^s)^w, \qquad s = 0, \dots, K,$$
where $^w$ marks lifted points. That is one $(K+1) \times (K+1)$ matrix $M_{sj} = N_{j,q}(v_s)$ for
every column. It is invertible by Schoenberg–Whitney (cited, as in
[[../01_Curves/B_Spline_Interpolation]]). `skinned_surface` solves for all columns at once.

**Claim.** The surface passes through every section: $S(u, v_s) = C^s(u)$.

*Proof.* Upstairs,
$S^w(u, v_s) = \sum_i N_{i,p}(u) \sum_j N_{j,q}(v_s)\,Q_{ij}^w = \sum_i N_{i,p}(u)\,(P_i^s)^w$,
which is the lifted section curve. Projecting gives $C^s(u)$. $\blacksquare$

Measured to $10^{-13}$ on four mixed sections (`test_skin_passes_through_every_section`). On the
solid, every section, the middle ones included, lies on the side faces to $10^{-10}$
(`test_sections_lie_on_the_solid`). The middle sections are not edges, only curves inside the
faces; script 03 draws them in black.

**The degree across** is $q = \min(\text{degree}, K)$, since $q + 1$ data points are needed.
- With two sections, $q = 1$ and $Q_{i0} = P_i^0$, $Q_{i1} = P_i^1$.
- If the second section is the first moved by $\mathbf d$, this net is $[P_i, P_i + \mathbf d]$:
  exactly the extruded surface of [[Extrude]]. Two equal circles therefore loft to the
  extrusion, face for face (`test_two_sections_give_the_ruled_surface`,
  `test_two_equal_circles_give_the_extruded_cylinder`).
- Two circles of radii $a$ and $b$ loft to the exact cone frustum. When the weights of the two
  sections agree, the degree-1 interpolant upstairs projects to $(1-v)\,C^0 + v\,C^1$. Its volume
  $\pi h (a^2 + ab + b^2)/3$ comes out to $10^{-14}$
  (`test_two_circles_of_different_radius_give_the_frustum`).

## The longitudinal edges

Side faces $k-1$ and $k$ meet along the longitudinal edge through vertex $k$ of every section. For
the solid to be watertight, that edge must lie on both skins. Two things make it so.

**One set of parameters for the whole loft.** Face $k$'s boundary at $u = 0$ interpolates the
vertices; so does face $(k-1)$'s boundary at $u = 1$. They are the same curve only if they
interpolate at the same $v_s$ on the same knots. So `skin_parameters` computes the $v_s$ once, from
every column of every face. Each column's parameters come from `interpolation_parameters`, and
then they are averaged (Piegl & Tiller's averaging; cited).

**End weights 1.** At $u = 0$ only $N_{0,p}$ is non-zero (clamped knots), so
$S^w(0, v) = \sum_j N_{j,q}(v)\,Q_{0j}^w$. That is the interpolant of the lifted vertices
$(w_0^s V^s, w_0^s)$, where $V^s$ is the vertex and $w_0^s$ the end weight of section $s$.

**Claim.** If all end weights are 1, the interpolated weights are all 1. The longitudinal edge is
then a polynomial B-spline through the vertices, the same for both faces.

*Proof.* The weight column solves $M w = (1, \dots, 1)^T$. Since $\sum_j N_{j,q}(v_s) = 1$ (partition
of unity), $w = (1, \dots, 1)$ is a solution, and $M$ is invertible, so it is the only one.
$\blacksquare$

With other end weights, the two faces would interpolate data scaled differently. A rational curve
is unchanged when all its weights are multiplied by one constant, but each section could be scaled
by a different constant, and the interpolant across would change. So `loft` requires end weights
of 1.

`check` confirms every longitudinal edge lies on both of its skins
(`test_every_step_is_valid_and_the_result_is_consistent`, three lofts).

## The topology: an extrusion

Topologically a loft is an extrusion. `loft` calls `sweep_lamina` ([[Revolve#A partial turn: the extrusion construction]])
with:
- the first section as the profile;
- the last section as the far copy;
- the longitudinal edges as rails, each one taken as the $u = 0$ column of a skin;
- the skins as side faces.

So $V = 2n$, $E = 3n$ and $F = n + 2$ (`test_every_step_is_valid_and_the_result_is_consistent`).

Every side face gets sense $+1$. The argument of [[Extrude#Orientation: the sense of each side face]]
applies at the first section, where $S_v$ points from the first section towards the next. A
regular skin ($S_u \times S_v \ne 0$ everywhere) cannot change the side its normal points to, so
the sense holds over the whole face. Twisted or folded skins can break regularity, and that isn't
checked. On the square-to-circle loft, the Stokes identity of [[Revolve#Checking the orientation: Stokes]]
holds on every side face to $10^{-13}$ (`test_every_loop_runs_the_way_its_sense_says`).

## Spacing the sections: two measurements, two different winners

The $v_s$ come from one of the three rules of [[../01_Curves/B_Spline_Interpolation]]: uniform,
chord length or centripetal. The usual textbook choice for skinning is chord length (Piegl &
Tiller; cited). Two experiments, with four circular or square sections, compare them. In both,
$\rho$ is the distance of the vertices from the $z$-axis.

**1. Error against a known profile.** Circles of radius $r(z) = 1 + 0.4\sin(1.3\,z)$ are placed at
given heights $z_s$. The error is the largest $\lvert\rho - r(z)\rvert$ along a longitudinal edge
(`test_loft_parametrization_comparison`):

| heights $z_s$ | uniform | chord | centripetal |
|---|---|---|---|
| 0, 0.3, 0.6, 2.5, 3 | 0.221 | 0.0279 | 0.169 |
| 0, 1, 2, 3, 4 | 0.033 | 0.0543 | 0.0435 |
| 0, 0.2, 1.5, 1.7, 3.2, 3.4 | 0.150 | 0.00344 | 0.101 |

**2. Wiggles on the script's stack.** The stack is: a square ($\rho = 1.4$, $z = 0$), two circles
pulled close ($\rho = 1.0$ at $z = 1.0$, $\rho = 0.7$ at $z = 1.25$), and a square ($0.6$, $3.0$).
From section to section $\rho$ only decreases, by 0.8 in total. Any extra total variation of $\rho$
along a longitudinal edge is a wiggle the data doesn't have (`test_loft_overshoot_comparison`):

| | uniform | chord | centripetal |
|---|---|---|---|
| total variation of $\rho$ beyond 0.8 | 0.0038 | 0.745 | 0.186 |
| smallest $\rho$ along the edge | 0.598 | 0.234 | 0.507 |

With chord spacing, the solid pinches into a waist of radius 0.23, far thinner than its thinnest
section (0.6). Script 03 shows it when the two circles are pulled together.

**What the tables say.**
- Chord length wins both unevenly spaced sets of the first experiment, beating the next best by
  factors of 6 and 29, and is by far the worst in the second.
- Uniform wins the second experiment and the evenly spaced set, and is the worst on the uneven
  sets of the first.
- Centripetal is never the best and never the worst.

A plausible reading of the first experiment, not proved here: there the data are samples of a
graph $\rho = r(z)$ whose slope stays small. Chord length is then close to $z$ itself, the
parameter in which the data are a smooth function. In the second experiment the data turn sharply,
and the long gap above the close pair gets a long stretch of $v$ to swing in.

**The default.** `loft` and `skin_parameters` default to **centripetal**: the one rule that was
never the worst. This departs from the textbook's chord length, which lost badly on the second
experiment. It is also the default of curve interpolation, for the related reason measured there.
Two experiments are evidence, not a theorem.

## What loft doesn't do

- **Sections with different numbers of edges.** Kernels split edges to make the counts match.
- **Choosing the vertex correspondence.** Matching is by index, and the user has to avoid twist.
  The script's "twist top" box turns the top square by $30°$ to show what a mismatched
  correspondence does.
- **Guide curves** that steer the surface between sections.
- **Closed lofts**, where the last section joins back to the first to make a ring. That needs
  seams, as in [[Revolve]].
- **Detecting folds or self-intersections.** Interpolated weights that come out non-positive are
  rejected, but nothing else is checked.

**Status**: exact (planar caps and skinned NURBS faces, untrimmed; the skin is exact through the
sections, and in between it is whatever the interpolation makes it). Implemented and tested:
`compatible`, `skinned_surface`, `loft`, in `tests/test_skinning.py` and `tests/test_loft.py`.
Script 03 shows the sections, the spacing rules and the degree across.
- *Proved inline:* the skin passes through every section; the longitudinal edges are polynomial
  and shared by both neighbouring faces; two sections give the ruled surface (the extrusion, or
  the frustum); the counts.
- *Stated and measured:* compatibility keeps the shapes; validity and consistency after every
  step; the Stokes orientation identity; the two parametrization experiments.
- *Cited, not proved:* Schoenberg–Whitney; Piegl & Tiller's skinning and parameter averaging; that
  chord length is the textbook choice for skinning.
