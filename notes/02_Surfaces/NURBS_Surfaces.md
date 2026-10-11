Code: [[../../scripts/02_surfaces/03_bspline_surface_tensor.py]], [[../../scripts/02_surfaces/04_surface_of_revolution.py]] — `bspline_surface`, `bspline_isocurve`, `nurbs_surface`, `nurbs_surface_basis`, `nurbs_surface_partials`, `nurbs_surface_normal`, `nurbs_surface_insert_knot` in `cadkernel/geometry/nurbs_surfaces.py`; tests in `tests/test_nurbs_surfaces.py`

# NURBS surfaces — the curve fixes, applied per direction

The motivation hub for B-spline and NURBS surfaces. The analytic surfaces of mechanical CAD,
built from this, are in [[Surfaces_of_Revolution]].

## Where we are

[[Bezier_Surfaces]] ended where [[../01_Curves/Bezier_Curves]] did, with the same two limitations
inherited through the tensor product:

1. **Global support.** Every control point moves the whole patch.
2. **No exact quadrics.** A sphere, cylinder, cone or torus has circular isocurves, no polynomial
   is a circle, so no polynomial patch is any of them.

Module 01 fixed both for curves: knots ([[../01_Curves/B_Splines]]) and weights
([[../01_Curves/NURBS]]). The question for surfaces is only how to carry the two fixes over, and
the answer is the one [[Bezier_Surfaces]] already gave for De Casteljau: run the curve algorithm
once per direction.

## Why the tensor product again, and not something else

- **Analytic surface types only** (a sphere is a centre and a radius). Exact and compact, and real
  kernels do keep them — see the last section. But on their own they cannot represent a free-form
  patch, a blend between a cylinder and a free-form face, or the perspective image of a sphere,
  and every algorithm needs a case for every pair of types. A kernel still needs one general type,
  exactly the argument of [[../01_Curves/NURBS]].
- **Polynomial patches fitted to tolerance.** Never exact, and the semantics are lost; the same
  argument [[../01_Curves/Lines_and_Arcs]] measured for curves, unchanged.
- **Triangular patches** ([[Bezier_Triangles]]). A real alternative, but without the row/column
  structure there is no "knot vector per direction", so the B-spline and NURBS curve algorithms do
  not transfer row by row. Kernels standardize on the tensor product.

The tensor product is the only option on the list that reuses all of module 01 unchanged.

## A surface is a curve of curves

A B-spline surface has a rectangular net $P_{ij}$, $i = 0..n$ along $u$ and $j = 0..m$ along $v$,
a degree and knot vector per direction ($p$ on $u_0..u_{n+p+1}$, $q$ on $v_0..v_{m+q+1}$), and
$$S(u,v) = \sum_{i=0}^{n}\sum_{j=0}^{m} N_{i,p}(u)\,N_{j,q}(v)\,P_{ij} .$$
Group the sum by $j$:
$$S(u,v) = \sum_{j=0}^{m} N_{j,q}(v)\,\underbrace{\Big[\sum_{i=0}^{n} N_{i,p}(u)\,P_{ij}\Big]}_{Q_j(u)} .$$
$Q_j(u)$ is the point at $u$ on the B-spline curve whose control polygon is column $j$ of the net.
So for fixed $u$, the $m+1$ points $Q_j(u)$ are the control polygon of the curve $v \mapsto S(u,v)$:
a B-spline curve of degree $q$ on the $v$-knots. Grouping by $i$ instead gives the same with the
roles swapped; both are the same finite double sum, so both orders give the same point. $\blacksquare$

Measured: the two-pass evaluation in either order and the closed tensor form agree below
$10^{-14}$ (`test_two_passes_match_the_grid_in_either_order`). This is the picture script 03 draws:
the columns as thin curves in $u$, their points at the current $u$ as squares, and the isocurve
through them. Move the $u$ slider and that isocurve sweeps out the surface.

**Weights change nothing in this picture.** As for curves, lift each control point to
$(w_{ij}P_{ij},\ w_{ij})$, evaluate the polynomial surface one dimension up, and divide by the last
coordinate ([[../01_Curves/NURBS_Derivations]] §1, word for word with a double sum):
$$S(u,v) = \frac{\sum_{i,j} N_{i,p}(u)N_{j,q}(v)\,w_{ij}P_{ij}}{\sum_{i,j} N_{i,p}(u)N_{j,q}(v)\,w_{ij}}
= \sum_{i,j} R_{ij}(u,v)\,P_{ij} .$$

## What carries over

Each of these is the curve result applied per direction, or the lift applied once more.

| property | why | measured |
|---|---|---|
| Bezier surface is the special case | no interior knots in either direction | `test_bspline_surface_on_bezier_knots_is_the_bezier_surface`, below $10^{-14}$ |
| B-spline surface is the special case | all $w_{ij} = 1$, partition of unity | `test_unit_weights_give_the_bspline_surface`, below $10^{-14}$ |
| partition of unity of $R_{ij}$ | the denominator is the sum of the numerators | `test_rational_tensor_basis_is_a_partition_of_unity`, below $10^{-14}$ |
| local support | see below | `test_moving_one_control_point_is_local_in_both_directions` |
| knot insertion | Boehm on every column (or row) of the lifted net | `test_knot_insertion_in_either_direction_keeps_the_surface`, below $10^{-14}$ |
| partial derivatives | the curve quotient rule, once per direction | `test_partials_match_finite_differences`, below $10^{-8}$ |

**Local support is a rectangle.** $N_{i,p}(u)N_{j,q}(v)$ is zero unless both factors are, so
$P_{ij}$ acts only on
$$[u_i, u_{i+p+1}) \times [v_j, v_{j+q+1}),$$
the product of the two curve supports ([[../01_Curves/B_Splines]]). With weights the same holds,
since $w_{ij}$ only scales that one term. Moving one control point leaves the surface outside the
rectangle unchanged to the last bit — the change there is exactly $0$ — while inside it moves
(`test_moving_one_control_point_is_local_in_both_directions`). Script 03 shows the rectangle twice:
on the surface, and in the $(u,v)$ parameter square with the knot lines drawn.

**One thing does not carry over cleanly: the basis no longer factors.** For polynomial surfaces the
basis is a product of a $u$-function and a $v$-function. The rational basis
$$R_{ij}(u,v) = \frac{w_{ij}N_{i,p}(u)N_{j,q}(v)}{\sum_{k,l} w_{kl}N_{k,p}(u)N_{l,q}(v)}$$
is a product only when the weights are. If $w_{ij} = a_ib_j$, the denominator factors as
$\big(\sum_k a_kN_{k,p}\big)\big(\sum_l b_lN_{l,q}\big)$ and
$$R_{ij} = \frac{a_iN_{i,p}(u)}{\sum_k a_kN_{k,p}(u)} \cdot \frac{b_jN_{j,q}(v)}{\sum_l b_lN_{l,q}(v)},$$
two rational *curve* bases. $\blacksquare$ For general weights there is no such split. The surfaces
of revolution are exactly the factoring case, and that is why their proof is short
([[Surfaces_of_Revolution]] §2).

## What it costs — the honest part

All measured on the NURBS sphere ([[Surfaces_of_Revolution]]):

- **Size.** A sphere is a centre and a radius, four numbers. As a NURBS surface it is a $5 \times 9$
  net with 45 weights and two knot vectors (`test_sphere_net_is_five_by_nine_with_product_weights`).
- **Poles.** The construction collapses a whole row of control points onto each pole, and there
  the surface has no tangent plane from $S_u \times S_v$: it is $0$ at the pole, and the normal
  exists only as a limit (`test_sphere_pole_is_a_singular_point`). The shape is smooth; the
  parametrization is not. Every algorithm that uses normals — shading, offsets, intersections —
  needs a special case there.
- **Seams.** The NURBS circle starts and ends at the same point, so the first and last columns of
  every revolved net coincide (both are the profile, at angle $0$). A closed surface is stored as an
  open patch with a seam.
- **Speed.** Neither direction is traversed at constant speed: the parallels and the meridians
  both inherit the circle's factor $4 - 2\sqrt2 \approx 1.17$
  (`test_sphere_speed_is_not_uniform_in_either_direction`). Equal steps in $(u, v)$ are not equal
  steps on the surface.
- **Derivatives** are quotients, as for curves ([[../01_Curves/NURBS]]), now in two directions.

**So real kernels keep the analytic types as well.** A cylinder stored as a cylinder answers "what
is your axis and radius?" directly, has no poles or seams to special-case in its own formulas, and
offsets to another cylinder. Two concrete examples, cited rather than tested here: the STEP
geometry schema (ISO 10303-42) has `cylindrical_surface`, `conical_surface`, `spherical_surface`
and `toroidal_surface` alongside `b_spline_surface`; OpenCASCADE has `Geom_CylindricalSurface`,
`Geom_SphericalSurface` and friends alongside `Geom_BSplineSurface`. NURBS is the common
denominator every analytic surface *can* be converted to — which is what the next note does — not
the only representation in use.

**Status**: exact. Implemented and tested in `tests/test_nurbs_surfaces.py`: `bspline_surface`,
`bspline_isocurve`, `nurbs_surface`, `nurbs_surface_basis`, `nurbs_surface_partials`,
`nurbs_surface_normal`, `nurbs_surface_insert_knot`. Script 03 visualizes the curve of curves and
local support. Proved inline: the curve-of-curves form (order independence), partition of unity, the rectangle
of local support, and that the rational basis factors when the weights do. Stated and measured:
the special cases, knot insertion (proved for curves, applied per direction) and the partials. Cited, not proved: the STEP and
OpenCASCADE surface types.
