Code: [[../../scripts/01_curves/02_bezier_de_casteljau.py]] — `bezier_de_casteljau`, `de_casteljau_triangle`, `bezier_subdivide` in `cadkernel/geometry/curves.py`

# De Casteljau's algorithm — the algorithm, and why kernels use it

This note is about the **algorithm**: what it computes, why it equals the Bernstein form, and
the two things it gives us that the closed form doesn't. The *properties* of the resulting curve
(convex hull, affine invariance, global support) are proved separately in
[[Bernstein_Basis_Properties]]; the *motivation* for Bezier curves at all is in [[Bezier_Curves]].

## The recursion

Given control points $P_0, \dots, P_n$ and a parameter $t$:

$$P_i^0 = P_i \qquad \text{(level 0 = the control points themselves)}$$
$$P_i^k = (1-t)\,P_i^{k-1} + t\,P_{i+1}^{k-1}, \qquad k = 1, \dots, n,\ \ i = 0, \dots, n-k$$

Each level lerps every *adjacent pair* of the level below, at the same fixed $t$. The triangle
loses one point per level; after $n$ levels a single point remains, and

$$C(t) = P_0^n.$$

That is the whole algorithm: repeated linear interpolation, nothing else. Move the `t` slider in
the script to watch the triangle collapse.

## Equivalence to the Bernstein form

The textbook definition of a Bezier curve is
$$C(t) = \sum_{i=0}^n B_i^n(t)\, P_i, \qquad B_i^n(t) = \binom{n}{i} (1-t)^{n-i} t^i.$$
De Casteljau's recursion never mentions binomial coefficients, so the equivalence is worth
actually proving.

Throughout, adopt the standard convention
$$B_j^k \equiv 0 \quad\text{for } j < 0 \text{ or } j > k,$$
which lets the boundary cases below be handled by the same identity as the interior ones.

**Claim.** $\displaystyle P_i^k(t) = \sum_{j=0}^{k} B_j^k(t)\, P_{i+j}$ for $k = 0, \dots, n$.

Setting $i=0,\ k=n$ gives $P_0^n = \sum_j B_j^n(t) P_j$, which is what we want.

**Proof by induction on $k$.**

*Base case* ($k=0$): $P_i^0 = P_i$ and $B_0^0(t) = 1$. ✓

*Inductive step*: assume the claim for $k-1$. Then
$$P_i^k = (1-t) P_i^{k-1} + t\, P_{i+1}^{k-1} = (1-t)\sum_{j=0}^{k-1} B_j^{k-1}(t) P_{i+j} \;+\; t\sum_{j=0}^{k-1} B_j^{k-1}(t) P_{i+1+j}.$$
Re-index the second sum ($j \to j-1$) so both are indexed by the same point $P_{i+j}$:
$$P_i^k = \sum_{j=0}^{k} \Big[(1-t)\, B_j^{k-1}(t) + t\, B_{j-1}^{k-1}(t)\Big] P_{i+j},$$
where the convention above makes the $j=0$ and $j=k$ terms come out right automatically. So it
suffices to prove the **Bernstein recursion**
$$(1-t)\, B_j^{k-1}(t) + t\, B_{j-1}^{k-1}(t) = B_j^k(t).$$
Expanding both terms,
$$(1-t)\binom{k-1}{j}(1-t)^{k-1-j}t^j + t\binom{k-1}{j-1}(1-t)^{k-j}t^{j-1} = (1-t)^{k-j}t^j\left[\binom{k-1}{j} + \binom{k-1}{j-1}\right],$$
and **Pascal's rule** $\binom{k-1}{j} + \binom{k-1}{j-1} = \binom{k}{j}$ gives
$\binom{k}{j}(1-t)^{k-j}t^j = B_j^k(t)$. $\blacksquare$

**The boundary cases, explicitly.** They are worth checking by hand rather than waving at, since
this is exactly where the convention is doing work:
- $j = 0$: $B_{-1}^{k-1} \equiv 0$, so the identity reduces to
  $(1-t)B_0^{k-1} = (1-t)\,(1-t)^{k-1} = (1-t)^k = B_0^k$. ✓
- $j = k$: $B_k^{k-1} \equiv 0$, so it reduces to
  $t\,B_{k-1}^{k-1} = t\cdot t^{k-1} = t^k = B_k^k$. ✓

So De Casteljau's algorithm and the Bernstein sum compute the same curve — the algorithm is
Pascal's rule, run as repeated interpolation instead of as binomial bookkeeping. Verified
numerically in `tests/test_curves.py::test_bernstein_matches_de_casteljau`.

## Subdivision, for free

This is the result that makes the algorithm worth more than the formula. Running the recursion
at $t = s$ doesn't just produce the point $C(s)$ — the two outer edges of the triangle are
**exactly the control polygons of the two halves of the curve**, split at $s$:

$$\text{left} = [P_0^0,\, P_0^1,\, \dots,\, P_0^n], \qquad \text{right} = [P_0^n,\, P_1^{n-1},\, \dots,\, P_n^0].$$

Both halves are Bezier curves of the same degree $n$, and reparametrized they reproduce the
original exactly (`test_bezier_subdivide_reproduces_original`). Tick "subdivision" in the script
to see the two polygons appear along the edges of the triangle already on screen.

Why a kernel cares: combined with the convex hull property, subdivision turns hard global
questions into cheap recursive local ones. Want to know whether two curves intersect? Compare
their control-point boxes; if the boxes are disjoint, they provably cannot intersect — discard.
If not, subdivide both and recurse. The same subdivide-and-cull pattern underlies adaptive
tessellation (subdivide until each control polygon is flat to within tolerance — and the hull
property is what makes "flat polygon ⇒ flat curve" a *guarantee* rather than a hope), ray
casting, and trimming. Nothing comparable falls out of evaluating $\sum_i B_i^n(t) P_i$.

## Numerical stability

Every step of the recursion is a convex combination with weights $(1-t)$ and $t$, both in
$[0,1]$. Intermediate points therefore stay inside the convex hull of the points they came from,
and no step can produce catastrophic cancellation.

Direct evaluation of the Bernstein sum, by contrast, multiplies large binomial coefficients by
tiny powers — at degree 50, $\binom{50}{25} \approx 1.26\times10^{14}$ against $t^{25}$ — and
relies on those magnitudes coming back down. De Casteljau never forms them at all.

**But this argument is weaker than it sounds, and the measurement says so.** Against exact
rational arithmetic, the naive closed form matches De Casteljau at $\sim10^{-16}$ relative error
at degree 3, 20, 50 and 800, and outside $[0,1]$ as well. The reason is that on $[0,1]$ the
Bernstein basis functions are all non-negative, so there is no subtractive cancellation *between*
terms: the huge binomials meet correspondingly tiny powers and the products are ordinary-sized.
The stability is a property of the **basis**, which both routes share (Farouki–Rajan again), not
of the algorithm. The closed form's only real failure mode is overflow — `math.comb(n, n/2)`
leaves the float64 range at $n = 1030$ — which no CAD kernel will ever reach.
(`test_closed_form_and_de_casteljau_agree_at_high_degree`.)

So the trade is not stability against cost. It is: De Casteljau is $O(n^2)$ per evaluated point
against $O(n)$ for a Horner-style Bernstein scheme, and it buys **subdivision** with that extra
work. At the degrees CAD actually uses (2–5) the cost difference is noise and subdivision is
decisive; the stability is real but is not what settles the choice.

## Generality

The recursion only ever takes affine combinations of points, so it carries over unchanged to
places the closed form would need re-deriving:
- **any dimension or affine space** — nothing in it is 2D-specific;
- **tensor-product surfaces** — run it in $u$ across rows of the control net, then in $v$ on the
  results;
- **rational curves** — run it in homogeneous coordinates and project at the end, which is
  exactly how `rational_bezier` handles the weights that make exact circles possible
  ([[Lines_and_Arcs]]).

**Status**: implemented exactly in code; the equivalence proof above is complete, and the
subdivision and stability claims are covered by tests.
