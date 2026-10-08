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

**Claim.** For each fixed level $r \in \{0, 1, \dots, p\}$ — a separate equation per $r$, with
**no summation over $r$** —
$$\sum_{i=k-p}^{k} N_{i,p}(t)\,P_i \;=\; \sum_{i=k-p+r}^{k} N_{i,p-r}(t)\,P_i^{r}.$$
The left side is $C(t)$ and does not involve $r$; the claim is that the right side has the *same
value at every level*. Each level lowers the basis degree by one and replaces the points by lerped
ones, and the two changes cancel — a conserved quantity, like $\sum_i N_{i,d}$ in the
partition-of-unity proof. At $r = 0$ it is trivial ($P_i^0 = P_i$); at $r = p$ the right side is
$N_{k,0}(t)\,P_k^p = P_k^p$, since exactly one box is on.

The sum ranges matter: $P_i^r$ only exists for $i = k-p+r,\dots,k$, and that is exactly the range
on which $N_{i,p-r}(t) \neq 0$ by local support. So nothing outside the range is being dropped —
those terms are zero. (Below, $\sum_i$ is shorthand for these ranges.)

*Proof.* It suffices to show that level $r-1$ and level $r$ give the same value, for each
$r = 1,\dots,p$; chaining these equalities from the trivial $r=0$ case up to $r=p$ finishes it.
Fix $r$, write $d = p-r+1$ (the basis degree at level $r-1$), and substitute Cox–de Boor
([[B_Splines]]) for $N_{i,d}$:
$$\sum_i N_{i,d}P_i^{r-1} = \sum_i \frac{t-u_i}{u_{i+d}-u_i}N_{i,d-1}P_i^{r-1}
+ \sum_i \frac{u_{i+d+1}-t}{u_{i+d+1}-u_{i+1}}N_{i+1,d-1}P_i^{r-1}.$$
Both sums run over $i = k-p+r-1,\dots,k$, where $P_i^{r-1}$ exists. To combine them they must
multiply the same function, so substitute $j = i+1$ in the second sum — replacing *every* $i$,
including the knot indices and the point's index — and rename $j$ back to $i$. That only
relabels the terms; the sum's value is unchanged. The second sum becomes
$\sum_{i=k-p+r}^{k+1}\frac{u_{i+d}-t}{u_{i+d}-u_i}N_{i,d-1}P_{i-1}^{r-1}$: same function as the first
sum now, but its range has slid up by one. The two ranges overlap on $k-p+r,\dots,k$, and the
one leftover term at each end is zero by local support:
- second sum, $i = k+1$: $N_{k+1,d-1}$ is supported on $[u_{k+1},\dots)$, but $t < u_{k+1}$;
- first sum, $i = k-p+r-1$: $N_{i,d-1}$ is supported on $[u_i, u_{i+d}) = [u_i, u_k)$, but $t \ge u_k$.

So both sums run over $k-p+r,\dots,k$ and can be added term by term:
$$= \sum_i N_{i,d-1}\left[\frac{t-u_i}{u_{i+d}-u_i}P_i^{r-1} + \frac{u_{i+d}-t}{u_{i+d}-u_i}P_{i-1}^{r-1}\right]
= \sum_i N_{i,d-1}\Big[\alpha_{i,r}P_i^{r-1} + (1-\alpha_{i,r})P_{i-1}^{r-1}\Big],$$
since $u_{i+d} = u_{i+p+1-r}$. The bracket is $P_i^r$. $\blacksquare$

It is the same move as the Bernstein proof in [[De_Casteljau_Derivation]] — re-index, and the two
complementary ramps become one lerp — which is why the two algorithms look alike. Measured:
de Boor against the basis sum agree to $10^{-14}$ on uneven knots
(`test_de_boor_matches_the_basis_sum`).

## The derivative

For Bezier curves, [[De_Casteljau_Derivation]] ("The derivative, for free") derives the
derivative in two equivalent forms: **constant points times a basis**,
$C'(t) = \sum_j B_j^{n-1}(t)\,n(P_{j+1}-P_j)$ (the *hodograph*), and **the last level of the
algorithm**, $C'(t) = n(P_1^{n-1} - P_0^{n-1})$. B-splines have both forms too, and they are derived
here in the same order: first the derivative of one basis function, then the curve, then the
algorithm.

Throughout, $0/0 := 0$ as in Cox–de Boor, and derivatives are taken inside a span
$u_k < t < u_{k+1}$. At a knot the curve may only be $C^{p-\text{mult}}$ ([[B_Splines]]), so there
the formulas give the one-sided derivative.

### Step 1 — the derivative of one basis function

**Claim.** For $d \ge 1$,
$$N_{i,d}'(t) = d\left[\frac{N_{i,d-1}(t)}{u_{i+d}-u_i} - \frac{N_{i+1,d-1}(t)}{u_{i+d+1}-u_{i+1}}\right].$$
This is the B-spline version of $\frac{d}{dt}B_i^n = n\big[B_{i-1}^{n-1} - B_i^{n-1}\big]$: a degree-$d$
function differentiates to $d$ times a difference of its two degree-$(d-1)$ parents, except that each
parent is now divided by the width of its own support.

*Proof by induction on $d$.* Write the two Cox–de Boor ramps as
$$\omega_{i,d}(t) = \frac{t-u_i}{u_{i+d}-u_i}, \qquad\text{so}\qquad
N_{i,d} = \omega_{i,d}\,N_{i,d-1} + (1-\omega_{i+1,d})\,N_{i+1,d-1},$$
since $1-\omega_{i+1,d} = \frac{u_{i+d+1}-t}{u_{i+d+1}-u_{i+1}}$. The ramps are linear, with
$\omega_{i,d}' = \frac{1}{u_{i+d}-u_i}$.

*Base case* ($d=1$): boxes are constant inside a span, so only the ramps get differentiated:
$N_{i,1}' = \frac{N_{i,0}}{u_{i+1}-u_i} - \frac{N_{i+1,0}}{u_{i+2}-u_{i+1}}$ — the hat function rises
with slope $1/(\text{width})$ on its first span and falls on its second. ✓

*Inductive step* ($d \ge 2$, claim true for $d-1$). Product rule on Cox–de Boor:
$$N_{i,d}' = \underbrace{\frac{N_{i,d-1}}{u_{i+d}-u_i} - \frac{N_{i+1,d-1}}{u_{i+d+1}-u_{i+1}}}_{\text{ramps differentiated}}
\;+\; \underbrace{\omega_{i,d}\,N_{i,d-1}' + (1-\omega_{i+1,d})\,N_{i+1,d-1}'}_{\text{parents differentiated}}.$$
The first group is one copy of the bracket in the claim. So it suffices to show that the second
group is $d-1$ more copies:
$$\omega_{i,d}\,N_{i,d-1}' + (1-\omega_{i+1,d})\,N_{i+1,d-1}'
\;\overset{?}{=}\; (d-1)\left[\frac{N_{i,d-1}}{u_{i+d}-u_i} - \frac{N_{i+1,d-1}}{u_{i+d+1}-u_{i+1}}\right].$$
(This is the same split as the Bezier "factor $n$" argument: the last lerp gives one copy, the
moving parents give the rest.) Both sides are combinations of the three degree-$(d-2)$ functions
$N_{i,d-2}$, $N_{i+1,d-2}$, $N_{i+2,d-2}$: expand the left side with the induction hypothesis, the
right side with Cox–de Boor, and compare coefficients. Abbreviate the four knots involved,
$$a = u_i,\quad b = u_{i+1},\quad c = u_{i+d},\quad e = u_{i+d+1},$$
and write $u_{i+d-1}$ and $u_{i+2}$ out where they appear. Dropping the common factor $d-1$:

- $N_{i,d-2}$: left $\dfrac{\omega_{i,d}}{u_{i+d-1}-a} = \dfrac{t-a}{(c-a)(u_{i+d-1}-a)}$; right
  $\dfrac{\omega_{i,d-1}}{c-a} = \dfrac{t-a}{(u_{i+d-1}-a)(c-a)}$. ✓
- $N_{i+2,d-2}$: left $-\dfrac{1-\omega_{i+1,d}}{e-u_{i+2}} = -\dfrac{e-t}{(e-b)(e-u_{i+2})}$; right
  $-\dfrac{1-\omega_{i+2,d-1}}{e-b} = -\dfrac{e-t}{(e-u_{i+2})(e-b)}$. ✓
- $N_{i+1,d-2}$ (the middle one, which both parents share): left
  $\dfrac{1-\omega_{i,d}-\omega_{i+1,d}}{c-b} = \dfrac{1}{c-b}\left[\dfrac{c-t}{c-a} - \dfrac{t-b}{e-b}\right]$;
  right $\dfrac{1-\omega_{i+1,d-1}}{c-a} - \dfrac{\omega_{i+1,d-1}}{e-b}
  = \dfrac{c-t}{(c-b)(c-a)} - \dfrac{t-b}{(c-b)(e-b)}$. ✓

(The middle line uses $1-\omega_{i,d} = \frac{c-t}{c-a}$ and $\omega_{i+1,d-1} = \frac{t-b}{c-b}$.)
All three coefficients match, so the second group is $d-1$ copies of the bracket, and
$N_{i,d}' = d\,[\dots]$. $\blacksquare$

### Step 2 — the curve: constant points times a basis

Differentiate $C(t) = \sum_{i=0}^{n} N_{i,p}(t)\,P_i$ term by term with Step 1:
$$C'(t) = \sum_{i=0}^{n} \frac{p\,N_{i,p-1}}{u_{i+p}-u_i}\,P_i \;-\; \sum_{i=0}^{n} \frac{p\,N_{i+1,p-1}}{u_{i+p+1}-u_{i+1}}\,P_i.$$
To combine them, make both multiply $N_{i+1,p-1}$: substitute $i \to i+1$ in the first sum, so it
runs over $i = -1,\dots,n-1$ and its term becomes $\frac{p\,N_{i+1,p-1}}{u_{i+p+1}-u_{i+1}}P_{i+1}$ —
the *same* coefficient as the second sum's. Two leftover terms remain, and on a clamped knot vector
both vanish:
- first sum, $i = -1$: $N_{0,p-1}$, supported on $[u_0, u_p)$, which is empty because $u_0 = \dots = u_p$;
- second sum, $i = n$: $N_{n+1,p-1}$, supported on $[u_{n+1}, u_{n+p+1}) = [u_{n+1}, u_m)$, empty because
  $u_{n+1} = \dots = u_m$.

What is left runs over $i = 0,\dots,n-1$:
$$C'(t) = \sum_{i=0}^{n-1} N_{i+1,p-1}(t)\,Q_i, \qquad
Q_i = \frac{p}{u_{i+p+1}-u_{i+1}}\,(P_{i+1}-P_i). \qquad\blacksquare$$
**Why the end knots disappear.** $N_{i+1,p-1}$ is built from the knots $u_{i+1},\dots,u_{i+p+1}$; as $i$
runs over $0,\dots,n-1$, these use $u_1,\dots,u_{m-1}$ and never $u_0$ or $u_m$. Renaming $u'_j = u_{j+1}$
turns $N_{i+1,p-1}$ on the old knots into $N_{i,p-1}$ on the trimmed vector $u'_0,\dots,u'_{m-2}$, so
$C'$ is a degree-$(p-1)$ B-spline with control points $Q_0,\dots,Q_{n-1}$ on the knot vector with
its first and last knot removed. The count checks: one point and one degree fewer is two knots
fewer, $m' = (n-1) + (p-1) + 1 = m - 2$.

As for Bezier, the $Q_i$ are constant — scaled edge vectors of the control polygon — and all of the
$t$-dependence is in the basis. The scale is $p$ divided by $u_{i+p+1}-u_{i+1}$, the width of the
knot window $P_i$ and $P_{i+1}$ share, where Bezier had just $n$. On Bezier knots $0^{p+1}1^{p+1}$
every such window is $[0,1]$ and $n = p$, so the formula is exactly the Bezier one.
`bspline_derivative` implements it and matches central differences to $10^{-7}$
(`test_bspline_derivative_matches_finite_differences`). Applying it repeatedly is how the continuity
claim in [[B_Splines]] was measured.

### Step 3 — the last level of de Boor

**Claim.** In the span $u_k \le t < u_{k+1}$,
$$C'(t) = \frac{p}{u_{k+1}-u_k}\,\big(P_k^{p-1}(t) - P_{k-1}^{p-1}(t)\big).$$

The Bezier proof read the two points off a split sum. Here the cleaner route is to show something
stronger: **scaled differences of de Boor points are the de Boor points of the derivative.** Define,
at every level $r = 0,\dots,p-1$,
$$D_i^r = \frac{p}{u_{i+p-r}-u_i}\,\big(P_i^r - P_{i-1}^r\big), \qquad i = k-p+r+1,\dots,k.$$
(The range is where both $P_i^r$ and $P_{i-1}^r$ exist. The denominators are never zero: for these
$i$, $u_i \le u_k$ and $u_{i+p-r} \ge u_{k+1}$, so the window contains the non-empty span.)

- *Level 0.* $D_i^0 = \frac{p}{u_{i+p}-u_i}(P_i - P_{i-1}) = Q_{i-1}$: the control points of the
  derivative from Step 2, for $i = k-p+1,\dots,k$ — that is $Q_{k-p},\dots,Q_{k-1}$, exactly the $p$
  points a degree-$(p-1)$ curve uses in this span.
- *Each level is one de Boor step of the derivative.* Substitute the recursion
  $P_i^r = (1-\alpha_{i,r})P_{i-1}^{r-1} + \alpha_{i,r}P_i^{r-1}$ for both points and collect, checking
  the coefficient of $P_{i-1}^{r-1}$ (which appears in both):
  $$P_i^r - P_{i-1}^r = \alpha_{i,r}\big(P_i^{r-1} - P_{i-1}^{r-1}\big) + (1-\alpha_{i-1,r})\big(P_{i-1}^{r-1} - P_{i-2}^{r-1}\big).$$
  Replace each difference by its $D$ (the window at level $r-1$ is one knot wider), and use
  $\alpha_{i,r}\,(u_{i+p+1-r}-u_i) = t-u_i$ and $(1-\alpha_{i-1,r})\,(u_{i+p-r}-u_{i-1}) = u_{i+p-r}-t$:
  $$D_i^r = \frac{(t-u_i)\,D_i^{r-1} + (u_{i+p-r}-t)\,D_{i-1}^{r-1}}{u_{i+p-r}-u_i}
  = (1-\beta_{i,r})\,D_{i-1}^{r-1} + \beta_{i,r}\,D_i^{r-1}, \qquad \beta_{i,r} = \frac{t-u_i}{u_{i+p-r}-u_i}.$$
  And $\beta_{i,r}$ is exactly de Boor's ratio for a degree-$(p-1)$ curve on the trimmed knots
  ($\alpha'_{j,r} = \frac{t-u'_j}{u'_{j+p-r}-u'_j}$ with $j = i-1$, $u'_j = u_{j+1}$).

So $D^0, D^1, \dots, D^{p-1}$ is de Boor's algorithm run on the derivative curve of Step 2, and
by the equivalence proved above, its last point is that curve's value: $C'(t) = D_k^{p-1}$, which
is the claim. $\blacksquare$

The one difference from Bezier is the scale: De Casteljau's last lerp runs over $[0,1]$, de Boor's
over the span $[u_k, u_{k+1}]$ (its $\alpha_{k,p} = (t-u_k)/(u_{k+1}-u_k)$), and dividing by the span
width converts "per unit of $\alpha$" into "per unit of $t$". On Bezier knots every window is
$[0,1]$, $D$ is just $n$ times the difference, and this proof *is* a second proof of the Bezier
result. The two forms agree to $1.1\times10^{-14}$ for degrees 1–4 on uneven knots
(`test_derivative_from_the_last_de_boor_level`). So de Boor, like De Casteljau, hands out the
tangent at no extra cost.

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
`tests/test_bspline.py`. Proved inline: de Boor ≡ basis sum, the basis derivative, both derivative forms (hodograph and
last de Boor level), Bezier degree elevation; the derivative forms are also measured. Knot insertion
is proved *given* the blossoming theorem, which is cited, not proved. Knot removal is notes-only.
