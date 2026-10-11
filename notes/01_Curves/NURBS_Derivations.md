Code: [[../../scripts/01_curves/08_nurbs_weights.py]] — `cadkernel/geometry/nurbs.py`; tests in `tests/test_nurbs.py`

# NURBS — the derivations

The proofs behind [[NURBS]], kept separate the way [[De_Casteljau_Derivation]] and
[[Bernstein_Basis_Properties]] are separate from [[Bezier_Curves]]. *Why* weights exist, and what
they cost, is argued in the hub; here is why each claim it makes is true.

**Notation** (Piegl & Tiller, as in [[B_Splines]]): control points $P_0,\dots,P_n$, weights
$w_0,\dots,w_n$, degree $p$, knots $u_0,\dots,u_m$ with $m = n+p+1$. Points are in $\mathbb{R}^d$
(the examples are 2D). Two abbreviations used throughout:
$$w(t) = \sum_{i=0}^{n} N_{i,p}(t)\,w_i \quad\text{(the weight function)}, \qquad
A(t) = \sum_{i=0}^{n} N_{i,p}(t)\,w_iP_i .$$
$w(t)$ is a scalar B-spline, $A(t)$ a B-spline in $\mathbb{R}^d$; both are polynomial on each span.

## 1. The definition, and why it is "a B-spline one dimension up"

**Definition.**
$$C(t) = \frac{A(t)}{w(t)} = \frac{\sum_i N_{i,p}(t)\,w_iP_i}{\sum_j N_{j,p}(t)\,w_j}
= \sum_{i=0}^{n} R_{i,p}(t)\,P_i, \qquad R_{i,p}(t) = \frac{w_i\,N_{i,p}(t)}{w(t)}.$$

**Homogeneous form.** Lift each control point to $\mathbb{R}^{d+1}$ as $P_i^w = (w_iP_i,\ w_i)$.
The ordinary B-spline on the lifted points is
$$C^w(t) = \sum_i N_{i,p}(t)\,P_i^w = \big(A(t),\ w(t)\big),$$
because the sum acts on each coordinate separately. Dividing the first $d$ coordinates by the last
gives $A(t)/w(t) = C(t)$. $\blacksquare$

The notes call $\mathbb{R}^{d+1}$, where the lifted points and $C^w$ live, **upstairs**, and the
division back to $\mathbb{R}^d$ the projection.

That one line is the whole implementation strategy of `nurbs.py`: lift, run the existing
polynomial machinery ([[De_Boor_Algorithm]]), divide. Every result below that says "carries over"
is this line applied once more. With all $w_i = 1$, $w(t) = \sum_i N_{i,p} = 1$ by partition of
unity and $C$ is the B-spline (`test_nurbs_with_unit_weights_is_the_bspline`); on Bezier knots it is
the rational Bezier curve (`test_rational_bezier_is_nurbs_on_bezier_knots`).

**Only ratios of weights matter.** Multiplying every $w_i$ by the same $c \neq 0$ multiplies $A$ and
$w$ by $c$, which cancels (`test_rational_bezier_reduces_to_polynomial_when_weights_equal`).

## 2. The rational basis

**Partition of unity.** $\sum_i R_{i,p}(t) = \frac{\sum_i w_iN_{i,p}(t)}{w(t)} = 1$, for *any*
weights — the denominator is by definition the sum of the numerators. $\blacksquare$

**Local support.** $R_{i,p}$ has the factor $N_{i,p}$, so it is zero outside $[u_i, u_{i+p+1})$,
exactly like the B-spline basis. Hence changing $w_i$ (like moving $P_i$) only changes the curve
on that interval (`test_rational_basis_is_a_partition_of_unity_with_local_support`,
`test_changing_one_weight_is_local`).

**Non-negativity — only for positive weights.** If every $w_i > 0$, each $R_{i,p}$ is a ratio of
non-negative terms with a positive denominator, so $R_{i,p} \ge 0$. With partition of unity, every
corollary in [[Bernstein_Basis_Properties]] that used only "non-negative weights summing to one" —
convex hull containment, affine invariance — survives verbatim. A negative $w_i$ makes $R_{i,p}$
negative somewhere and the hull argument is gone; section 7 measures what that does.

## 3. Exact circular arcs

**Claim.** For a sweep $\delta = 2\varphi$ with $0 < \varphi < \pi/2$, the quadratic with control points
on the endpoint tangents and weights $(1, \cos\varphi, 1)$ is exactly the circular arc.

*Proof.* By affine invariance (weights untouched, section 5) it is enough to take the unit circle
and the arc symmetric about the $x$-axis:
$$P_0 = (\cos\varphi, -\sin\varphi), \quad P_1 = \Big(\frac{1}{\cos\varphi},\, 0\Big), \quad
P_2 = (\cos\varphi, \sin\varphi), \qquad w = (1, \cos\varphi, 1).$$
($P_1$ is where the two endpoint tangents meet.) Write $a = (1-t)^2 + t^2$ and $b = 2t(1-t)$, so
$a + b = 1$ and $a - b = (1-2t)^2$. The homogeneous coordinates $(X, Y, W) = (A_x, A_y, w)$ are
$$X = a\cos\varphi + b, \qquad Y = (2t-1)\sin\varphi, \qquad W = a + b\cos\varphi .$$
(The middle term of $X$ is $b \cdot \cos\varphi \cdot \frac{1}{\cos\varphi} = b$.) Then, using
$(2t-1)^2 = a - b = (a-b)(a+b) = a^2 - b^2$,
$$X^2 + Y^2 = a^2\cos^2\varphi + 2ab\cos\varphi + b^2 + (a^2-b^2)\sin^2\varphi
= a^2 + 2ab\cos\varphi + b^2\cos^2\varphi = W^2 .$$
So $(X/W)^2 + (Y/W)^2 = 1$ for every $t$: the curve lies on the circle. $\blacksquare$

Measured: radial error below $10^{-13}$ on a quarter arc
(`test_rational_quadratic_is_an_exact_circular_arc`).

**The negative middle weight traces the rest of the circle.** Replace $\cos\varphi$ by $-\cos\varphi$
in the weight only: $X = a\cos\varphi - b$, $W = a - b\cos\varphi$, and the identical computation gives
$X^2 + Y^2 = (a - b\cos\varphi)^2 = W^2$. Still on the circle — but at $t = \tfrac12$ ($a = b = \tfrac12$)
the point is $X/W = -1$, $Y = 0$: the antipode of the arc's midpoint. Weights $(1, -\cos\varphi, 1)$
trace the *complementary* arc, entirely outside the control triangle
(`test_negative_middle_weight_gives_the_complementary_arc`).

## 4. The full circle

Four quarter arcs ($\varphi = \pi/4$, middle weight $\cos\tfrac\pi4 = \tfrac{\sqrt2}{2}$) joined into one
degree-2 NURBS (`nurbs_circle`): nine control points around the circumscribed square — edge
midpoints on the circle with weight 1, corners with weight $\tfrac{\sqrt2}{2}$ — and knots
$$0,0,0,\ \tfrac14,\tfrac14,\ \tfrac12,\tfrac12,\ \tfrac34,\tfrac34,\ 1,1,1 .$$
Each interior knot has multiplicity $2 = p$, so by [[De_Boor_Algorithm]] ("Decomposition into
Bezier pieces") the curve on each quarter of $[0,1]$ *is* the rational Bezier on three consecutive
control points — one of the arcs of section 3. Radial error below $10^{-15}$
(`test_full_circle_is_exact`).

**Smoother than the knots promise.** Multiplicity 2 on degree 2 only guarantees $C^0$
([[B_Splines]]). But the end derivative of a rational Bezier is, from the quotient rule of section 6
at $t=0$ (where $A = w_0P_0$, $A' = n(w_1P_1 - w_0P_0)$, $w' = n(w_1 - w_0)$):
$$C'(0) = \frac{A'(0) - w'(0)\,C(0)}{w(0)} = n\,\frac{w_1}{w_0}\,(P_1 - P_0),$$
and symmetrically $C'(1) = n\frac{w_{n-1}}{w_n}(P_n - P_{n-1})$. At a joint of the circle both sides
see the same square edge vector and the same weight ratio $\tfrac{\sqrt2}{2}$, and both quarters are
reparametrized from $[0,1]$ to a quarter of the parameter range (the same factor 4). So the
derivative matches: the circle is $C^1$ (`test_circle_is_c1_at_its_double_knots`). $\blacksquare$

## 5. Projective invariance

A **projective map** of the plane is a $3\times3$ invertible matrix $H$ acting on homogeneous
coordinates: $x \mapsto \operatorname{proj}\big(H\,(x, 1)\big)$, where $\operatorname{proj}(X,Y,W) = (X/W, Y/W)$.
Affine maps are the special case whose last row is $(0,0,1)$; a perspective view is the typical
non-affine one.

**Claim.** Mapping the curve point by point equals mapping the homogeneous control points:
$$\operatorname{proj}\big(H\,(C(t),1)\big) \;=\; \text{the NURBS with control data } HP_i^w .$$

*Proof.* $(C(t), 1)$ and $C^w(t) = w(t)\,(C(t), 1)$ represent the same point, and $\operatorname{proj}$
ignores scalar factors. So
$$\operatorname{proj}\big(H\,(C(t),1)\big) = \operatorname{proj}\big(H\,C^w(t)\big)
= \operatorname{proj}\Big(\sum_i N_{i,p}(t)\,HP_i^w\Big),$$
since $H$ is linear and passes through the sum. The right side is the NURBS whose lifted control
points are $HP_i^w$. $\blacksquare$

Reading off the new control data: $HP_i^w = w_i\,H(P_i,1) = w_ih_i\,(Q_i, 1)$, where $Q_i$ is the
mapped point and $h_i$ the last coordinate of $H(P_i,1)$. So **the control points map as points, and
the weights get multiplied by $h_i$** (`nurbs_projective_map`). For an affine map every $h_i = 1$
and the weights are untouched (`test_affine_maps_leave_the_weights_alone`). For a perspective they
change, which is exactly why a polynomial B-spline cannot be projectively invariant: it has no
weights to absorb the $h_i$. Measured: the exact construction agrees with the mapped curve to below
$10^{-14}$; moving only the control points is off by more than $0.4$, for a NURBS and for a plain
B-spline alike (`test_nurbs_is_projectively_invariant`). A perspective image of the circle is an
exact ellipse (`test_perspective_image_of_a_circle_is_an_exact_ellipse`).

*Fine print.* The map must not send a point of the curve to infinity ($h \neq 0$ along the curve);
and if some $h_i < 0$ the new weights are not all positive, so the hull guarantee is lost even
though the curve is exact.

## 6. The derivative, and why its degree grows

Differentiate $A(t) = w(t)\,C(t)$ with the product rule: $A' = w'C + wC'$, so
$$C'(t) = \frac{A'(t) - w'(t)\,C(t)}{w(t)} = \frac{A'(t)\,w(t) - A(t)\,w'(t)}{w(t)^2}.$$
$A'$ and $w'$ are the hodographs of polynomial B-splines, which [[De_Boor_Algorithm]] already
provides; `nurbs_derivative` computes them on the lifted control points and applies this formula
(matches central differences to $10^{-8}$, `test_nurbs_derivative_matches_finite_differences`).

**Degree.** On one Bezier span of degree $n$, $A$ and $w$ are polynomials of degree $n$, so the
numerator $A'w - Aw'$ has degree at most $2n-1$. Its $t^{2n-1}$ coefficient vanishes: if $A$ has
leading coefficient $\alpha$ and $w$ has $\gamma$, it is $n\alpha\gamma - \alpha n\gamma = 0$. So
$$C'(t) = \frac{\text{degree } \le 2n-2}{\text{degree } 2n} .$$
A polynomial curve's derivative has degree $n-1$; the rational one's is a quotient of roughly twice
the degree. That the bound is attained for generic data is measured, not proved: fitting the
numerator needs exactly degree $2n-2$ for $n = 2,\dots,5$ (`test_derivative_degree_inflates`).

## 7. What the weights do to the shape

**A large weight pulls the curve to its point.** For $t$ inside the support of $N_{i,p}$,
$$R_{i,p}(t) = \frac{w_iN_{i,p}(t)}{w_iN_{i,p}(t) + \sum_{j\neq i} w_jN_{j,p}(t)} \;\longrightarrow\; 1
\qquad (w_i \to \infty),$$
so $C(t) \to P_i$. For the quadratic with weights $(1, w, 1)$ at $t = \tfrac12$:
$A = \tfrac14P_0 + \tfrac{w}{2}P_1 + \tfrac14P_2$ and $w(\tfrac12) = \tfrac{1+w}{2}$, so
$$C(\tfrac12) = \frac{M + wP_1}{1+w}, \qquad M = \tfrac12(P_0 + P_2),$$
a point on the segment from the chord midpoint $M$ to $P_1$, at distance $|P_1 - M|/(1+w)$ from $P_1$.
For the test triangle that is $2/(1+w)$, and measured it is also the curve's closest approach
(`test_large_weight_pulls_the_curve_to_its_point`).

**One middle weight picks the conic type.** The rational quadratic with weights $(1, w, 1)$ and
non-collinear $P_0, P_1, P_2$ is a conic, and which one is read off $w$:
$$|w| < 1:\ \text{ellipse}, \qquad |w| = 1:\ \text{parabola}, \qquad |w| > 1:\ \text{hyperbola}.$$
*Proof.* Use the lifted points $b_0 = (P_0,1)$, $b_1 = w\,(P_1,1)$, $b_2 = (P_2,1)$ as a basis of
$\mathbb{R}^3$ (independent, since the $P_i$ are not collinear and $w \neq 0$). In that basis
$C^w(t)$ has coordinates $(x_0, x_1, x_2) = \big((1-t)^2,\ 2t(1-t),\ t^2\big)$, which satisfy
$x_1^2 = 4x_0x_2$ — a conic in homogeneous coordinates; changing basis is a linear map, and a
linear map takes the conic $x^\top M x = 0$ to the conic $y^\top (H^{-\top}MH^{-1})\,y = 0$. As $t$ runs
over all of $\mathbb{R}\cup\{\infty\}$ it covers the whole conic; $[0,1]$ is the arc. The conic's points
at infinity are where the weight function vanishes. Writing $t$ homogeneously as $(1-t, t) = (a, b)$,
$$w(a,b) = a^2 + 2w\,ab + b^2, \qquad \text{discriminant } 4(w^2 - 1),$$
which has no real root, one double root, or two real roots as $|w| <, =, > 1$. The last step is
**cited, not proved**: a real non-degenerate conic is an ellipse, parabola or hyperbola according as
it meets the line at infinity in 0, 1 (tangentially) or 2 points (standard projective geometry).
$\blacksquare$

Measured by fitting an implicit conic through sampled points: $B^2 - 4AC$ is negative for
$w = 0.3, 0.7$, zero to $10^{-12}$ at $w = 1$, positive for $w = 1.5, 4$
(`test_middle_weight_selects_the_conic_type`). At $w = -1$ the double root is $(a,b) = (1,1)$, i.e.
$t = \tfrac12$ — *inside* the parameter range: the denominator is $(1-2t)^2$ and the curve runs off
to infinity at the middle (`test_weight_minus_one_puts_a_pole_on_the_curve`).

**Zero and negative weights.** $w_1 = 0$ switches $P_1$ off: $C = \frac{(1-t)^2P_0 + t^2P_2}{(1-t)^2+t^2}$
lies on the chord. $w_1 = -\tfrac12$ gives, at $t = \tfrac12$, $C = P_0 - P_1 + P_2$ — for the test
triangle $(1, -2)$, below a triangle that lies in $y \ge 0$
(`test_negative_weight_leaves_the_convex_hull`).

## 8. The weights are not unique

**Claim.** For a rational Bezier of degree $n$ and any $\lambda > 0$, the weights $\lambda^i w_i$ give
the same curve, reparametrized by $t = \dfrac{\lambda s}{(1-s) + \lambda s}$.

*Proof.* With $D = (1-s) + \lambda s$: $1 - t = \frac{1-s}{D}$ and $t = \frac{\lambda s}{D}$, so
$$B_i^n(t) = \binom{n}{i}\frac{(1-s)^{n-i}\lambda^i s^i}{D^n} = \frac{\lambda^i\,B_i^n(s)}{D^n}.$$
Hence $\sum_i B_i^n(t)\,w_i\,(P_i, 1) = D^{-n}\sum_i B_i^n(s)\,\lambda^iw_i\,(P_i,1)$, and the common factor
$D^{-n}$ disappears in the division. $\blacksquare$

So one degree of freedom in the weights (per Bezier segment) changes only the *speed*, never the
shape. Measured: pointwise the two curves differ by more than $0.1$; after the substitution they
agree to $10^{-14}$ (`test_scaling_weights_geometrically_only_reparametrizes`).

## 9. Knot insertion, and why it must be done upstairs

The lifted curve $C^w$ is a polynomial B-spline, so Boehm insertion on the lifted control points
leaves $C^w$ unchanged ([[De_Boor_Algorithm]], proved there given blossoming), and so its projection
$C$ is unchanged too. Unlifting the new points gives the new weights (lerps of the old weights) and
new control points (lerps of $w_iP_i$, divided by the new weight) — `nurbs_insert_knot`, unchanged to
below $10^{-14}$. Inserting into the points and the weights *separately* is wrong, because
$(1-\alpha)w_{i-1}P_{i-1} + \alpha w_iP_i$ is not $\big((1-\alpha)w_{i-1} + \alpha w_i\big)\big((1-\alpha)P_{i-1} + \alpha P_i\big)$;
measured, the curve moves by more than $0.1$
(`test_knot_insertion_must_happen_in_homogeneous_space`). The same rule holds for every operation
in [[De_Boor_Algorithm]]: do it to $P_i^w$, never to $P_i$ and $w_i$ apart.

**Status**: exact. Proved inline: the homogeneous form, partition of unity and local support of
$R_{i,p}$, the exact arc (and its complementary arc), $C^1$ of the full circle, projective invariance,
the derivative formula and its degree bound, the weight-pull limit, the reparametrization freedom,
and knot insertion via the lift (given the blossoming theorem cited in [[De_Boor_Algorithm]]).
Cited, not proved: the classification of conics by their points at infinity. Stated and measured:
that the derivative degree bound is attained, and that $t = \tfrac12$ is the closest approach in the
pull example. Every number is from `tests/test_nurbs.py`.
