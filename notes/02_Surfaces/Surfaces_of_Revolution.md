Code: [[../../scripts/02_surfaces/04_surface_of_revolution.py]] — `surface_of_revolution`, `nurbs_cylinder`, `nurbs_cone`, `nurbs_sphere`, `nurbs_torus` (and their `*_profile` functions) in `cadkernel/geometry/nurbs_surfaces.py`; tests in `tests/test_nurbs_surfaces.py`

# Surfaces of revolution — the analytic surfaces, as NURBS

[[NURBS_Surfaces]] is the general machinery. This note uses it for the surfaces that dominate
mechanical CAD — cylinder, cone, sphere, torus — and proves they come out exact.

## Why one construction, and not four formulas

Each of the four could be given its own rational parametrization. But all four are **surfaces of
revolution**: a curve in a half-plane through an axis, turned about that axis.

| surface | profile in the $(r, z)$ half-plane |
|---|---|
| cylinder | a segment parallel to the axis |
| cone | a segment ending on the axis |
| sphere | a semicircle with both ends on the axis |
| torus | a full circle off the axis |

So one construction — "revolve a NURBS profile" — plus four profiles covers them. It also covers
everything a lathe makes: any NURBS profile revolves exactly the same way. Script 04 lets you drag
the profile and watch this.

## 1. The construction

Take a profile, a NURBS curve in the $(r, z)$ half-plane ($r$ is the distance from the $z$-axis):
control points $(r_i, z_i)$, weights $a_i$, $i = 0..n$, degree $p$, knots in $u$. Take the exact
NURBS circle of [[../01_Curves/NURBS_Derivations]] §4: nine control points $Q_j$ on the unit
square's corners and edge midpoints, weights $c_j$ alternating $1$ and $\tfrac{\sqrt2}{2}$, degree 2,
knots in $v$.

Each profile control point is carried round on its own circle — its own square, scaled by $r_i$
and lifted to height $z_i$:
$$P_{ij} = \big(r_i\,Q_j,\ z_i\big), \qquad w_{ij} = a_i\,c_j .$$
The weights **multiply**. On the sphere this makes the corner weights
$\tfrac{\sqrt2}{2}\cdot\tfrac{\sqrt2}{2} = \tfrac12$, and the whole $5 \times 9$ weight table is
exactly the outer product of the two weight vectors
(`test_sphere_net_is_five_by_nine_with_product_weights`). Script 04 prints this product for any
net point you pick.

## 2. Why it is exact

**Claim.** With $(r(u), z(u))$ the profile curve and $(\cos\theta(v), \sin\theta(v))$ the NURBS circle,
$$S(u,v) = \big(r(u)\cos\theta(v),\ r(u)\sin\theta(v),\ z(u)\big).$$

*Proof.* Write $N_i = N_{i,p}(u)$ and $M_j = N_{j,2}(v)$. The homogeneous numerator of $S$ is
$\sum_{i,j} N_iM_j\,w_{ij}P_{ij}$. Substitute $w_{ij} = a_ic_j$ and $P_{ij} = (r_iQ_j, z_i)$; every
term splits into an $i$-part times a $j$-part, so the double sums factor:
$$\text{horizontal part: } \sum_{i,j} N_iM_j\,a_ic_j\,r_iQ_j = \Big(\sum_i N_ia_ir_i\Big)\Big(\sum_j M_jc_jQ_j\Big),$$
$$\text{height: } \sum_{i,j} N_iM_j\,a_ic_j\,z_i = \Big(\sum_i N_ia_iz_i\Big)\Big(\sum_j M_jc_j\Big),$$
$$\text{denominator: } \sum_{i,j} N_iM_j\,a_ic_j = \Big(\sum_i N_ia_i\Big)\Big(\sum_j M_jc_j\Big).$$
Divide. The horizontal part becomes
$\dfrac{\sum_i N_ia_ir_i}{\sum_i N_ia_i}\cdot\dfrac{\sum_j M_jc_jQ_j}{\sum_j M_jc_j} = r(u)\,(\cos\theta(v), \sin\theta(v))$,
because the first factor is the profile's $r$-coordinate and the second is the NURBS circle. The
height becomes $\dfrac{\sum_i N_ia_iz_i}{\sum_i N_ia_i} = z(u)$, the $\sum_j M_jc_j$ cancelling. $\blacksquare$

This is the factoring case of [[NURBS_Surfaces]] ("the basis no longer factors"): because the
weights are a product, the surface is a product, and the proof needs nothing beyond the curve
results.

**Consequences, each one line from the claim.**
- A parallel ($u$ fixed) is a circle of radius $r(u)$ at height $z(u)$.
- A meridian ($v$ fixed) is the profile, turned to angle $\theta(v)$; it stays in one plane through
  the axis.
- Any equation the profile satisfies in $(r, z)$ becomes an equation of the surface with
  $r = \sqrt{x^2 + y^2}$.

Measured: a parallel of the sphere has constant height and constant distance from the axis, and a
meridian constant angle, all below $10^{-14}$ (`test_revolution_factors_into_profile_times_circle`).

## 3. The four primitives

Each implicit equation follows from the profile's, with $r = \sqrt{x^2+y^2}$:

| surface | profile | degree $p \times q$, net | implicit equation | measured |
|---|---|---|---|---|
| cylinder, radius $R$, height $h$ | segment $(R, 0)$ to $(R, h)$ | $1 \times 2$, $2 \times 9$ | $x^2 + y^2 = R^2$ | below $10^{-14}$ |
| cone, base radius $R$, height $h$ | segment $(R, 0)$ to $(0, h)$ | $1 \times 2$, $2 \times 9$ | $\sqrt{x^2+y^2} = R\,(1 - z/h)$ | below $10^{-14}$ |
| sphere, radius $R$ | two quarter arcs, $(0,-R)$ via $(R,0)$ to $(0,R)$ | $2 \times 2$, $5 \times 9$ | $x^2 + y^2 + z^2 = R^2$ | below $10^{-14}$ |
| torus, radii $R > \rho$ | the NURBS circle, centre $(R, 0)$, radius $\rho$ | $2 \times 2$, $9 \times 9$ | $\big(\sqrt{x^2+y^2} - R\big)^2 + z^2 = \rho^2$ | below $10^{-13}$ |

All four rows: `test_the_four_primitives_satisfy_their_implicit_equations`.

The sphere's profile is the semicircle built like the full circle of
[[../01_Curves/NURBS_Derivations]] §4, but with two quarter arcs instead of four: control points
$(0,-R), (R,-R), (R,0), (R,R), (0,R)$, weights $1, \tfrac{\sqrt2}{2}, 1, \tfrac{\sqrt2}{2}, 1$, knots
$0,0,0,\tfrac12,\tfrac12,1,1,1$.

## 4. Poles: a smooth surface with a singular parametrization

Two of the profiles touch the axis: the sphere at both ends, the cone at its apex. A profile control
point with $r_i = 0$ has a square of size zero, so **all nine control points of row $i$ coincide**.

**Claim.** If row $0$ of the net collapses to one point $O$ (and the profile is clamped, so it starts
at that control point), then $S(0, v) = O$ for every $v$, $S_v(0, v) = 0$, and $S_u \times S_v = 0$ there.

*Proof.* A clamped curve interpolates its first control point, so every meridian starts at $O$:
$S(0, v) = O$ for all $v$. A function constant in $v$ has $v$-derivative $0$, and the cross product
of anything with the zero vector is zero. $\blacksquare$

So at the pole the formula for the normal gives the zero vector, even though the sphere is
perfectly smooth there. The singularity belongs to the parametrization, not to the shape. Measured
(`test_sphere_pole_is_a_singular_point`):
- every point of the first and last rows is the same point, exactly;
- $S_v = 0$ exactly at $u = 0$ and $u = 1$, while $|S_u|$ stays above $1$ — the meridians still move,
  they just all start from the same point;
- the normal at $u = 0$ is the zero vector, and at $u = 10^{-2}, 10^{-4}, 10^{-6}$ it approaches the
  limit $(0, 0, 1)$, the error shrinking with $u$ (below $10^{-5}$ at $u = 10^{-6}$).

A kernel therefore has to treat poles specially: take the normal as a limit, avoid degenerate
triangles when tessellating, and expect a whole edge of the parameter rectangle to map to one point.
Tick "poles" in script 04 to see the collapsed rows and the parallels shrinking into them.

**Could a cleverer construction avoid poles?** Not with a single rectangular patch covering the
whole sphere. Sketch: if it had no singular points, $S_v$ would be a tangent vector field on the sphere that
vanishes nowhere, and the **hairy ball theorem** says no such field exists. That theorem is cited,
not proved here. Kernels avoid the poles only by using more than one patch, or by keeping the sphere
analytic ([[NURBS_Surfaces]], last section).

**Orientation.** With $u$ running south to north and $v$ counter-clockwise, $S_u \times S_v$ points
inward: away from the poles the unit normal is exactly $-S/R$ (`test_sphere_normal_is_radial_inward`).
Swapping the roles of $u$ and $v$ flips it, because the cross product is antisymmetric. Which side
is "outside" is a convention a kernel has to store; the parametrization alone does not decide it.

## 5. Exact shape, inexact speed — in both directions

The meridians are made of NURBS arcs and the parallels are NURBS circles, so both inherit the
circle's speed variation: $|S_u|$ along a meridian and $|S_v|$ along the equator both vary by the
factor $4 - 2\sqrt2 \approx 1.17$ (`test_sphere_speed_is_not_uniform_in_either_direction`, matching
`test_circle_is_not_arc_length_parametrized` for the curve). That is why script 04's slider says
"how far round" and not "angle".

**Status**: exact. `surface_of_revolution` and the four primitives are implemented and tested in
`tests/test_nurbs_surfaces.py`; script 04 visualizes the construction, the product weights and the
poles. Proved inline: the factorization $S = (r\cos\theta, r\sin\theta, z)$ and the implicit
equations that follow from it, that a collapsed row makes $S_v$ and the normal vanish, and that
swapping $u$ and $v$ flips the orientation. Stated and measured: the implicit-equation errors, the
shape of parallels and meridians, the limit of the normal at the pole, the inward orientation, and
the speed factor. Cited, not proved: the hairy ball theorem.
