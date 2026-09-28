Code: [[../../scripts/01_curves/07_bspline_interpolation.py]] — `interpolate_points`, `interpolation_parameters`, `averaged_knots` in `cadkernel/geometry/bspline.py`; tests in `tests/test_bspline.py`

# B-spline interpolation — when the points are data, not handles

## The tension with the Bezier note

[[Bezier_Curves]] argued that interpolation is the **wrong interface** for design: a designer
wants to *pull* a shape into place, not to guess points the curve must hit, and one polynomial
through many points oscillates (Runge). Both statements stand. But a kernel also has to accept
points that are **data**: a measured profile from a scanner, sections a loft must pass through,
a sketch snapped to fixed points, a path that must hit given waypoints. For those, "pass through
these points" is the specification, and the control points are an *output*.

So the question is not whether to interpolate but **with what**. The answer is a B-spline, and the
reason is exactly the one that made B-splines worth building ([[B_Splines]]): low degree, piecewise,
local.

## Why not one polynomial

Runge's function $1/(1+25x^2)$, sampled at equally spaced $x$ on $[-1,1]$, interpolated by one
polynomial through all the points versus a cubic B-spline through the same points and parameters
(`test_piecewise_interpolation_escapes_runge`):

| points | one polynomial: max error | cubic B-spline: max error |
|---|---|---|
| 7 | 0.617 | 0.132 |
| 13 | 3.66 | 0.0069 |
| 21 | 59.8 | 0.0032 |

Adding data makes the polynomial **worse** and the spline better. The polynomial's degree grows
with the data and its global support lets every point pull on the whole curve; the spline's degree
stays 3 and each point only answers to its neighbours. On the less tidy data in script 07, the same
contrast is 4.1 units of stray for the polynomial against 0.72 for the spline
(`test_single_polynomial_strays_further_than_the_spline`).

## The linear system

Given data $Q_0,\dots,Q_n$, choose a parameter $t_k$ for each point and a knot vector, then demand
$C(t_k) = Q_k$:
$$\sum_{i=0}^{n} N_{i,p}(t_k)\,P_i = Q_k, \qquad k = 0,\dots,n.$$
That is $n+1$ linear equations in the $n+1$ unknown control points — one solve per coordinate,
sharing one matrix $A_{ki} = N_{i,p}(t_k)$. By local support each row has at most $p+1$ non-zeros,
so $A$ is **banded**; a production kernel uses a banded solver in $O(np^2)$, while
`interpolate_points` uses a dense `np.linalg.solve`, which gives the same answer at this size.
Every curve hits every point to $\sim10^{-16}$ (`test_interpolation_passes_through_every_point`).

Two choices are hidden in that setup, and the matrix is only invertible if they are made
sensibly:

**The knots.** They are placed by **averaging** (de Boor): each interior knot is the mean of $p$
consecutive parameters, $u_{j+p} = \frac1p\sum_{i=j}^{j+p-1} t_i$. This puts every parameter $t_i$
inside the support of "its" basis function $N_{i,p}$, which is the Schoenberg–Whitney condition
for $A$ to be non-singular. That condition is cited, not proved here.

**The parameters.** This is the choice that actually decides what the curve looks like.

## Choosing the parameters — measured, not assumed

Three standard rules for $t_k$, all normalized to run from 0 to 1:

- **uniform** — equal steps, ignoring geometry;
- **chord length** — steps proportional to $|Q_k - Q_{k-1}|$;
- **centripetal** — steps proportional to $\sqrt{|Q_k - Q_{k-1}|}$ (Lee, 1989).

Chord length is the default in most textbooks. Measured on three deliberately awkward data sets
(cubic; "stray" is the largest distance from the curve to the data polyline)
(`test_parametrization_comparison`):

| data set | uniform | chord length | centripetal |
|---|---|---|---|
| sharp turn | stray 0.10 | stray 0.15 | **stray 0.02** |
| uneven spacing | stray 0.25, **self-intersects** | stray **5.09** | stray 0.72 |
| tight cluster | stray 0.14 | stray **5.61**, **self-intersects** | **stray 0.13** |

Centripetal is the only rule that is clean on all three; chord length, the usual default, is the
*worst* on two of them. So `interpolate_points` defaults to centripetal.

Why, qualitatively: the parameters decide how much "time" the curve spends between consecutive
points. **Uniform** gives a tiny gap and a huge gap the same time, so the curve crawls through
clustered points and races across the gaps — and a curve that must hit several close points while
moving fast overshoots and loops. **Chord length** fixes that by making speed roughly constant, but
constant speed through a sharp change of direction at close points means a wide swing to get
round the corner. **Centripetal** sits between the two (a square root compresses the ratio of long
to short steps), which is why it tends to behave. The honest limit of this section: three data
sets, one degree. It is evidence, not a theorem, and there are data sets where chord length does
better. The point that does generalize is that **the parametrization is a first-class design
choice**, not a detail, and a kernel that interpolates must make it deliberately.

Script 07 lets you drag the data and watch all three disagree; the "control polygon" toggle shows
the solved control points, which sit nowhere near the data — they are the output of the solve,
not something anyone would draw.

## What is not here

Exact interpolation passes through every point, noise included. Real measured data wants
**least-squares approximation** instead — fewer control points than data points, a curve that
passes *near* the points within a tolerance — which is a different solve and is not implemented
([[../99_Not_Covered_In_Code]]). End-derivative constraints (clamped tangents) are also left out.

**Status**: exact. `interpolation_parameters`, `averaged_knots`, `interpolate_points`
implemented; residuals, the Runge comparison, the parametrization table and the polynomial
contrast are all produced by `tests/test_bspline.py`. The Schoenberg–Whitney condition is cited,
not proved.
