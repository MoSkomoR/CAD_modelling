Code: [[../../scripts/01_curves/06_bspline_local_control.py]] — `de_boor`, `de_boor_stages`, `bspline_derivative`, `insert_knot`, `bspline_to_bezier`, `elevate_degree` in `cadkernel/geometry/bspline.py`; `bezier_elevate` in `cadkernel/geometry/curves.py`

# De Boor's algorithm — and the tools built on it

The algorithm note for B-splines, the counterpart of [[De_Casteljau_Derivation]]. *Why*
B-splines exist is in [[B_Splines]]. Here: how a point is evaluated, why that equals the basis
sum, and the three curve operations a kernel builds on the same recursion — knot insertion,
decomposition into Bezier pieces, and degree elevation.

## The recursion

To evaluate at $t$, first find the span $k$ with $u_k \le t < u_{k+1}$ (`find_span`). Only
$P_{k-p},\dots,P_k$ matter there (local support). Then lerp, $p$ times:
$$P_i^0 = P_i, \qquad
P_i^r = (1-\alpha_{i,r})\,P_{i-1}^{r-1} + \alpha_{i,r}\,P_i^{r-1}, \qquad
\alpha_{i,r} = \frac{t-u_i}{u_{i+p+1-r}-u_i},$$
for $r = 1,\dots,p$ and $i = k-p+r,\dots,k$. One point is left: $C(t) = P_k^p$.

It is De Casteljau with one change: the lerp ratio is not $t$ itself but **where $t$ sits inside
a window of knots**, and the window shrinks by one knot per level. Every step is still an affine
combination with weights in $[0,1]$, so every intermediate point is geometry — script 06 draws the
$p+1 \to p \to \dots \to 1$ collapse. On Bezier knots $0^{p+1}1^{p+1}$ every window is $[0,1]$,
every $\alpha$ is $t$, and this *is* De Casteljau (`test_bezier_is_a_special_case_of_bspline`).
Cost: $p(p+1)/2$ lerps, independent of how many control points the curve has
(`test_de_boor_touches_only_p_plus_one_points`).

## Equivalence to the basis sum

**Claim.** For $1 \le r \le p$,
$$\sum_i N_{i,p}(t)\,P_i = \sum_i N_{i,p-r}(t)\,P_i^{r}.$$
At $r = p$ the right side is $\sum_i N_{i,0}(t)P_i^p = P_k^p$, since exactly one box is on.

*Proof (one level; repeat).* Substitute Cox–de Boor ([[B_Splines]]) for $N_{i,d}$, $d = p-r+1$:
$$\sum_i N_{i,d}P_i^{r-1} = \sum_i \frac{t-u_i}{u_{i+d}-u_i}N_{i,d-1}P_i^{r-1}
+ \sum_i \frac{u_{i+d+1}-t}{u_{i+d+1}-u_{i+1}}N_{i+1,d-1}P_i^{r-1}.$$
Shift the second sum's index ($i+1 \to i$) so both multiply $N_{i,d-1}$:
$$= \sum_i N_{i,d-1}\left[\frac{t-u_i}{u_{i+d}-u_i}P_i^{r-1} + \frac{u_{i+d}-t}{u_{i+d}-u_i}P_{i-1}^{r-1}\right]
= \sum_i N_{i,d-1}\Big[\alpha_{i,r}P_i^{r-1} + (1-\alpha_{i,r})P_{i-1}^{r-1}\Big],$$
since $u_{i+d} = u_{i+p+1-r}$. The bracket is $P_i^r$. $\blacksquare$

It is the same move as the Bernstein proof in [[De_Casteljau_Derivation]] — re-index, and the two
complementary ramps become one lerp — which is why the two algorithms look alike. Measured:
de Boor against the basis sum agree to $10^{-14}$ on uneven knots
(`test_de_boor_matches_the_basis_sum`).

## The derivative

Differentiating Cox–de Boor gives the B-spline analogue of the Bezier hodograph: $C'(t)$ is a
degree-$(p-1)$ B-spline on the knot vector with its first and last knot removed, with control points
$$Q_i = \frac{p}{u_{i+p+1}-u_{i+1}}\,(P_{i+1}-P_i), \qquad i = 0,\dots,n-1.$$
On Bezier knots every denominator is 1 and this is $n(P_{i+1}-P_i)$. It is stated here, not
derived; `bspline_derivative` implements it and matches central differences to $10^{-7}$
(`test_bspline_derivative_matches_finite_differences`). Applying it repeatedly is how the continuity
claim in [[B_Splines]] was measured.

## Knot insertion — the B-spline form of subdivision

For Bezier curves, the decisive by-product of De Casteljau was subdivision
([[De_Casteljau_Derivation]]). The B-spline equivalent is **knot insertion** (Boehm, 1980): add a
knot $\hat t$, gain a control point, and do not change the curve. Refining the knot vector only
*enlarges* the space of piecewise polynomials (one more place where the pieces may change), so the
old curve is certainly in the new space; the question is only what its new control points are.

$$Q_i = \begin{cases} P_i & i \le k-p\\[2pt] (1-\alpha_i)P_{i-1} + \alpha_i P_i,\quad \alpha_i = \dfrac{\hat t-u_i}{u_{i+p}-u_i} & k-p+1 \le i \le k\\[2pt] P_{i-1} & i \ge k+1\end{cases}$$

Look at the middle row: those $\alpha_i$ are exactly the **first level of de Boor at $\hat t$**.
Inserting a knot splices the first level of the evaluation triangle into the polygon. Insert the
same knot again and you splice in the second level; insert it $p$ times and the whole triangle's
outer edges are in the polygon — the curve now passes through a control point at $\hat t$ and falls
apart into two independent B-splines. That is subdivision, reached one level at a time.

*Why the formula is right.* The cleanest proof uses **blossoming** (Ramshaw, 1987), and this is
the one fact imported rather than proved here: a degree-$p$ polynomial piece has a unique
symmetric, multi-affine "blossom" $f(x_1,\dots,x_p)$ with $f(t,\dots,t) = C(t)$, and B-spline
control points are its values on consecutive knot windows, $P_i = f(u_{i+1},\dots,u_{i+p})$.
Granting that: after inserting $\hat t$, the new windows are the old ones with $\hat t$ slotted in,
and multi-affinity in that one slot gives
$$f(u_{i+1},\dots,\hat t,\dots,u_{i+p-1}) = (1-\alpha_i)\,f(u_i,\dots,u_{i+p-1}) + \alpha_i\, f(u_{i+1},\dots,u_{i+p}),\quad \alpha_i = \frac{\hat t-u_i}{u_{i+p}-u_i},$$
because $\hat t = (1-\alpha_i)u_i + \alpha_i u_{i+p}$. That is the middle row. $\blacksquare$ (Given
the imported fact.) Measured directly: inserting a new knot and then raising an existing one twice
leaves the curve unchanged to below $10^{-14}$ (`test_knot_insertion_leaves_the_curve_unchanged`).

## Decomposition into Bezier pieces

Insert every interior knot until it has multiplicity $p$ (`bspline_to_bezier`). Now each span
$[a,b]$ sits in the knot window $a^p\, b^p$, so in de Boor on that span every
$\alpha = (t-a)/(b-a)$: de Boor *is* De Casteljau at the local parameter $s = (t-a)/(b-a)$, and the
span's $p+1$ control points are its **Bezier control polygon**. Neighbouring pieces share their
end point. Reproduced span by span to machine precision ($3\times10^{-16}$ in a run)
(`test_bspline_to_bezier_segments_reproduce_each_span`; the "Bezier segments" toggle in script 06).

This is how a kernel gets the whole of module 01 for B-splines without rewriting it: decompose,
then use the Bezier hull, De Casteljau subdivision and the flatness test on each piece.

## Degree elevation

**Why a kernel needs it.** Lofting a surface through two curves, or joining them into one, needs
them *compatible*: same degree, same knot vector. A cubic and a quadratic cannot be blended control
point by control point. Knot insertion equalizes the knots; degree elevation equalizes the degree.

**Bezier first.** Multiply $C(t)$ by $1 = (1-t) + t$. Two identities do the work:
$$(1-t)\,B_i^n = \frac{n+1-i}{n+1}\,B_i^{n+1}, \qquad t\,B_i^n = \frac{i+1}{n+1}\,B_{i+1}^{n+1}$$
(expand the binomials: $\binom{n}{i}\frac{n+1}{n+1-i} = \binom{n+1}{i}$ and
$\binom{n}{i}\frac{n+1}{i+1} = \binom{n+1}{i+1}$). Collecting the coefficient of $B_i^{n+1}$:
$$Q_i = \frac{i}{n+1}\,P_{i-1} + \Big(1-\frac{i}{n+1}\Big)P_i, \qquad Q_0 = P_0,\ Q_{n+1} = P_n. \qquad\blacksquare$$
Corner cutting again — the new polygon lies inside the old one's hull
(`bezier_elevate`, `test_bezier_degree_elevation_keeps_the_curve`).

**B-splines.** `elevate_degree` decomposes into Bezier pieces, elevates each, and stitches them
back (`test_bspline_degree_elevation_keeps_the_curve`). **The honest caveat:** the stitched knot
vector has every breakpoint at multiplicity $p+1$, which by [[B_Splines]] only *promises* $C^0$.
The curve itself is as smooth as before — elevation never changes a curve — but the representation
has forgotten it and carries redundant knots and control points. Getting them back out needs **knot
removal**, which is not implemented ([[../99_Not_Covered_In_Code]]). Production implementations
(e.g. algorithm A5.9 in Piegl & Tiller's *The NURBS Book*) do the same decompose-and-elevate, then
remove the redundant knots in the same pass.

**Status**: exact. `de_boor`, `de_boor_stages`, `bspline_derivative`, `insert_knot`,
`bspline_to_bezier`, `bezier_elevate`, `elevate_degree` implemented and tested in
`tests/test_bspline.py`. Proved inline: de Boor ≡ basis sum, Bezier degree elevation. Knot insertion
is proved *given* the blossoming theorem, which is cited, not proved; the derivative formula is
stated and verified numerically. Knot removal is notes-only.
