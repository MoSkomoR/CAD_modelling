**Code**: [[../../scripts/02_surfaces/02_bezier_triangle_de_casteljau.py]] — `bezier_triangle_point`, `de_casteljau_triangle_stages`, `bernstein_triangle_basis`, `bezier_triangle` in `cadkernel/geometry/surfaces.py`; tests in `tests/test_surfaces.py`

# Bezier triangles — De Casteljau over three neighbours

This note is about the *algorithm*, the same split [[De_Casteljau_Derivation]] makes for curves.
*Why* a three-sided patch is wanted at all, and why kernels use it anyway only in finite elements
rather than in mainstream B-rep, is argued in [[Bezier_Surfaces]] ("The other generalization").
Here: what the recursion is, and why it still computes a Bernstein sum.

## The recursion

A tensor-product patch lerps along two independent directions, $u$ and $v$. A Bezier triangle
instead lerps over a single triangular lattice, in **barycentric coordinates**
$(\lambda_0,\lambda_1,\lambda_2)$ with $\lambda_0+\lambda_1+\lambda_2=1$. Control points are
indexed by a multi-index $\mathbf{i}=(i,j,k)$ with $i+j+k=n$ — there is no "rows and columns"
here, only the triangular lattice itself (`bezier_triangle_point` therefore keys its control net
by a dict `{(i,j,k): point}`, not an array).

Let $e_0=(1,0,0)$, $e_1=(0,1,0)$, $e_2=(0,0,1)$. The recursion blends **three** neighbours per
step, at the same fixed weights, instead of two:
$$P^0_{\mathbf{i}} = P_{\mathbf{i}}, \qquad
P^{s}_{\mathbf{i}} = \lambda_0 P^{s-1}_{\mathbf{i}+e_0} + \lambda_1 P^{s-1}_{\mathbf{i}+e_1}
+ \lambda_2 P^{s-1}_{\mathbf{i}+e_2}, \qquad s=1,\dots,n.$$
Level $s$ is defined over every $\mathbf{i}$ with $|\mathbf{i}| = n-s$: the lattice loses one row
per level exactly as the curve triangle loses one point per level ([[De_Casteljau_Derivation]]),
and after $n$ levels a single point remains, indexed $(0,0,0)$:
$$C(\lambda_0,\lambda_1,\lambda_2) = P^n_{(0,0,0)}.$$
That is the whole algorithm — the curve recursion with one extra term per step.

## Equivalence to the trivariate Bernstein form

The closed form replaces the binomial with a **multinomial**:
$$C(\boldsymbol\lambda) = \sum_{i+j+k=n} B^n_{ijk}(\boldsymbol\lambda)\,P_{ijk}, \qquad
B^n_{ijk}(\boldsymbol\lambda) = \binom{n}{i,j,k}\lambda_0^i\lambda_1^j\lambda_2^k, \qquad
\binom{n}{i,j,k} = \frac{n!}{i!\,j!\,k!}.$$

**Claim.** $\displaystyle P^s_{\mathbf{i}} = \sum_{a+b+c=s} B^s_{abc}(\boldsymbol\lambda)\,P_{\mathbf{i}+(a,b,c)}$ for $s=0,\dots,n$. Setting $\mathbf i=(0,0,0)$, $s=n$ gives the claim above.

*Base case* ($s=0$): $P^0_{\mathbf i}=P_{\mathbf i}$ and $B^0_{000}=1$. ✓

*Inductive step*: assume the claim for $s-1$. Substituting into the recursion and re-indexing
each of the three sums so all three land on the same point $P_{\mathbf i + (a,b,c)}$ (exactly the
$j\to j-1$ shift in the curve proof, done three ways) leaves
$$P^s_{\mathbf i} = \sum_{a+b+c=s}\Big[\lambda_0 B^{s-1}_{a-1,b,c} + \lambda_1 B^{s-1}_{a,b-1,c}
+ \lambda_2 B^{s-1}_{a,b,c-1}\Big] P_{\mathbf i+(a,b,c)},$$
using the same $B\equiv 0$-outside-range convention as the curve case. So it suffices to prove
the **trinomial Pascal rule**
$$\lambda_0 B^{s-1}_{a-1,b,c} + \lambda_1 B^{s-1}_{a,b-1,c} + \lambda_2 B^{s-1}_{a,b,c-1}
= B^s_{abc}.$$
Expanding each term with $\binom{s-1}{a-1,b,c}=\frac{(s-1)!}{(a-1)!\,b!\,c!}$ etc., every term
shares the factor $\lambda_0^a\lambda_1^b\lambda_2^c$, and what's left is
$$\frac{(s-1)!}{(a-1)!\,b!\,c!} + \frac{(s-1)!}{a!\,(b-1)!\,c!} + \frac{(s-1)!}{a!\,b!\,(c-1)!}
= \frac{(s-1)!\,(a+b+c)}{a!\,b!\,c!} = \frac{s!}{a!\,b!\,c!} = \binom{s}{a,b,c},$$
using $a+b+c=s$ in the last step — the three-term generalization of Pascal's rule
$\binom{n-1}{i}+\binom{n-1}{i-1}=\binom{n}{i}$, proved the same way. $\blacksquare$

So barycentric De Casteljau computes the same surface as the multinomial Bernstein sum, for the
same reason the binomial case did: the recursion *is* Pascal's rule, run as repeated
interpolation. Checked directly against `bernstein_triangle_basis`
(`test_triangle_de_casteljau_matches_closed_bernstein_form`).

## What carries over unchanged

$B^n_{ijk}\ge 0$ on the simplex (a product/quotient of non-negative terms) and
$\sum_{i+j+k=n} B^n_{ijk}(\boldsymbol\lambda) = (\lambda_0+\lambda_1+\lambda_2)^n = 1$ by the
**multinomial theorem** — the trivariate analogue of the binomial theorem that closed the curve
case (`test_triangle_bernstein_basis_is_partition_of_unity_and_non_negative`). Every corollary
that only used "non-negative weights summing to one" therefore survives verbatim:

- **Corner interpolation**: at $(\lambda_0,\lambda_1,\lambda_2)=(1,0,0)$ (and the other two unit
  vectors), every term with $i<n$ vanishes ($\lambda_0^i=0$), leaving $C=P_{n,0,0}$
  (`test_triangle_corner_interpolation`) — the *edges* of the lattice are not interpolated, only
  the three corners, same asymmetry as a curve's interior control points.
- **Convex hull containment**: $C(\boldsymbol\lambda)$ is a convex combination of the control net,
  so it never leaves the net's bounding box (`test_triangle_stays_in_the_bounding_box_of_its_
  control_net`).
- **Affine invariance**: the same partition-of-unity argument from
  [[Bernstein_Basis_Properties]] applies term for term (`test_triangle_affine_invariance`).

`de_casteljau_triangle_stages` returns every level of the shrinking lattice rather than only the
last, exactly as `de_casteljau_triangle` (curves) and `de_casteljau_surface_stages` (tensor
patches) do — it is what the "De Casteljau stages" toggle in the script draws collapsing.

## What is different from the tensor product, and what is not built here

A triangular patch has one **total** degree $n$, not a bidegree $(n,m)$ — the three barycentric
directions are interchangeable, unlike $u$ and $v$. That symmetry is also *why* the patch is
naturally three-sided rather than four (the geometric argument lives in [[Bezier_Surfaces]]).

Not implemented: **subdivision**. Splitting a Bezier triangle into sub-triangles is a real
algorithm (it also falls out of De Casteljau, run at more than one interior point), but it is a
different derivation from the tensor-product case — column-wise curve subdivision has nothing to
lerp along here — and is left for [[../99_Not_Covered_In_Code]] alongside degree elevation.
Evaluation, corner interpolation, hull containment and affine invariance are exercised above;
subdivision and degree elevation are not.

**Status**: exact (evaluation via both `bezier_triangle_point` and the closed
`bernstein_triangle_basis` / `bezier_triangle`, agreeing to machine precision). Corner
interpolation, hull containment, affine invariance and the Bernstein equivalence are all tested
in `tests/test_surfaces.py`; the interactive script draws the collapsing lattice and lets a
control point be dragged to show global support. Subdivision and degree elevation are notes-only.
