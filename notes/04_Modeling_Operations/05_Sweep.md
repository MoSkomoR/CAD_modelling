Code: [[../../scripts/04_modeling_ops/04_sweep.py]] — `sweep`, `path_frames`, `place`, `arc_length` in [[../../src/cadkernel/ops/sweep.py]], `frenet_frames`, `rotation_minimizing_frames`, `curvature_torsion` in [[../../src/cadkernel/geometry/frames.py]], `nurbs_derivatives` in [[../../src/cadkernel/geometry/nurbs.py]]; tests in `tests/test_sweep.py`, `tests/test_frames.py`, `tests/test_nurbs.py`

# Sweep — a profile carried along a path

A pipe, a handrail, a cable, the rim of a wheel, an O-ring groove: each is one cross-section
carried along a curve. The designer draws the profile and the path, and the kernel must place a
copy of the profile at every point of the path. That is a **sweep**.

Placing the profile at a point $C(t)$ of the path needs a plane to put it in, and a direction in
that plane for its $x$-axis. The plane is the natural one, normal to the path's tangent. The
direction is the real question of this note: a **moving frame** along the path.

**Two rejected alternatives.**
- **The Frenet frame**, the frame every differential-geometry course builds from the path's
  derivatives. It is undefined on straight stretches, flips by half a turn at an inflection, and
  turns the profile about the path wherever the path has torsion. Each of these is measured
  below.
- **An exact swept surface.** The surface a sweep traces is in general not a NURBS surface, so it
  can't be stored exactly. A sweep here is therefore **simplified**: the profile is placed exactly
  at a number of sections, and a skin is interpolated through them, as in [[04_Loft]].

The frame used instead is the **rotation-minimizing frame** (RMF), computed by double reflection
(Wang, Jüttler, Zheng & Liu, *Computation of rotation minimizing frames*, ACM TOG 27(1), 2008;
cited).

## The swept surface

- **Path:** a regular curve $C(t)$, $t \in [0, 1]$: $C'(t) \ne 0$, so the unit tangent
  $T = C'/\lvert C'\rvert$ exists everywhere. Write $v(t) = \lvert C'(t)\rvert$ for its speed and
  $s$ for arc length, $ds = v\,dt$.
- **Profile:** a closed loop of curves in the $xy$-plane, with points $(x(u), y(u))$. The origin is
  the point that runs along the path.
- **Frame:** unit vectors $r(t)$ and $b(t) = T(t) \times r(t)$ with $r \perp T$, so that
  $(T, r, b)$ is orthonormal and right-handed at every $t$.

The **exact sweep** puts the profile's $x$-axis along $r$, its $y$-axis along $b$, and its normal
$z$ along $T$:
$$X(u, t) = C(t) + x(u)\,r(t) + y(u)\,b(t).$$
$T$ fixes the plane of each copy; $r$ fixes its rotation inside that plane, the one degree of
freedom the path does not decide.

## The Frenet frame

The obvious choice of $r$ takes it from the path's own derivatives:
$$T = \frac{C'}{\lvert C'\rvert}, \qquad B = \frac{C' \times C''}{\lvert C' \times C''\rvert}, \qquad N = B \times T,$$
the unit tangent, binormal and principal normal; $r = N$ and $b = B$. $N$ points to the side the
path turns towards. With the curvature $\kappa$ and torsion $\tau$
$$\kappa = \frac{\lvert C' \times C''\rvert}{\lvert C'\rvert^3}, \qquad \tau = \frac{(C' \times C'') \cdot C'''}{\lvert C' \times C''\rvert^2},$$
the frame moves by the **Frenet–Serret formulas** (do Carmo, *Differential Geometry of Curves and
Surfaces*, §1-5; cited):
$$T' = v\kappa N, \qquad N' = -v\kappa T + v\tau B, \qquad B' = -v\tau N,$$
where $'$ is $d/dt$. $\tau$ measures how fast the path leaves the plane of $T$ and $N$ (the
osculating plane). `curvature_torsion` computes both from $C', C'', C'''$, and `nurbs_derivatives`
supplies those for a NURBS path (Piegl & Tiller's eq. 4.8, the quotient rule repeated; checked
against finite differences and the curvature $1/r$ of a rational arc,
`test_higher_derivatives_match_finite_differences_and_the_circle`).

Three things go wrong.

**1. Undefined where $\kappa = 0$.** $B$ divides by $\lvert C' \times C''\rvert$. On a straight
line $C'' \parallel C'$ and there is no $N$ at all
(`test_frenet_is_undefined_on_a_line_and_flips_at_an_inflection`). A sweep along a straight
segment, the commonest path there is, has no Frenet frame
(`test_a_straight_path_gives_the_extrusion`).

**2. A flip at an inflection.** *Claim.* On a plane curve, $B = \pm e$, where $e$ is the plane's
unit normal, and the sign changes wherever the curve's turning direction does.

*Proof.* $C'$ and $C''$ lie in the plane, so $C' \times C'' = k(t)\,e$ with the scalar
$k = (C' \times C'') \cdot e$. Then $B = \operatorname{sign}(k)\,e$, and $N = B \times T$ changes
sign with it. At a simple inflection $k$ changes sign. $\blacksquare$

Measured on a planar S-shaped cubic: $10^{-6}$ before and after the inflection in $t$, the two
normals have dot product below $-0.999999$, and $B$ goes from $-e$ to $+e$
(`test_frenet_is_undefined_on_a_line_and_flips_at_an_inflection`). Along a planar S-path with 8
sections, consecutive Frenet normals have dot product above 0.5 everywhere except across the
inflection, where it is below $-0.99$ (`test_frenet_flips_the_profile_at_an_inflection`). The
profile turns half a turn between two sections and the skin twists through the path. Script 04
shows it at lift 0.

**3. Twist from torsion.** Even where it is defined, $N$ turns about $T$ at the rate $v\tau$:
that is the $v\tau B$ term in $N'$. The profile is turned about the path because the path's
osculating plane turns, not because anything asked for it. How much is made exact in the next
section.

## The rotation-minimizing frame

**Definition.** $r(t)$ is **rotation-minimizing** if $r \perp T$, $\lvert r\rvert = 1$ and
$r' \parallel T$: $r$ changes only as much as it must to stay normal to $T$, and never turns about
$T$.

Differentiating $r \cdot T = 0$ gives $r' \cdot T = -r \cdot T'$. So $r' \parallel T$ means
$$r' = -(r \cdot T')\,T ,$$
a linear differential equation for $r$. Given any $r(0) \perp T(0)$ it has one solution, wherever
$T$ is differentiable: a regular $C^2$ path suffices, with no condition on $\kappa$.

**Claim.** The solution stays a unit vector normal to $T$.

*Proof.* $\frac{d}{dt}(r \cdot T) = r' \cdot T + r \cdot T' = -(r \cdot T') + r \cdot T' = 0$, and then
$\frac{d}{dt}\lvert r\rvert^2 = 2\,r \cdot r' = -2\,(r \cdot T')(r \cdot T) = 0$. $\blacksquare$

**Claim (RMF against Frenet).** Where $\kappa > 0$, write $r = \cos\theta\,N + \sin\theta\,B$. Then
$r$ is rotation-minimizing if and only if
$$\theta' = -v\tau, \qquad\text{that is}\qquad \theta(t) = \theta(0) - \int_0^{s(t)} \tau\,ds .$$

*Proof.* By the Frenet–Serret formulas,
$$r' = -\theta'\sin\theta\,N + \cos\theta\,(-v\kappa T + v\tau B) + \theta'\cos\theta\,B - \sin\theta\,v\tau N
= -v\kappa\cos\theta\,T + (\theta' + v\tau)(-\sin\theta\,N + \cos\theta\,B).$$
The last vector is a unit vector normal to $T$, so $r' \parallel T$ exactly when
$\theta' + v\tau = 0$. $\blacksquare$

So the Frenet frame is turned against the RMF by exactly the integrated torsion. On a cubic
Bézier path with $\tau$ between 0.095 and 0.36, the computed RMF and the integrated torsion agree
to $10^{-8}$, and by $t = 1$ the two frames are 1.004 rad ($57.5°$) apart
(`test_frenet_turns_against_rmf_by_the_integrated_torsion`). Script 04 plots both against $t$.

**On a plane curve** $\tau = 0$, so $\theta$ is constant: an RMF started in the plane stays in it,
and $b = T \times r$ is the plane's normal $e$ throughout. Directly: $e$ is normal to $T$ and
$e' = 0 \parallel T$, so $e$ itself is rotation-minimizing. Measured, $b = \pm e$ to $10^{-15}$
along the planar S-path, where Frenet flips (`test_frenet_flips_the_profile_at_an_inflection`).

## Computing the RMF: double reflection

The ODE has no closed-form solution for a general path, so $r$ is computed at sample points
$C_i = C(t_i)$ with unit tangents $T_i$. Double reflection (Wang et al. 2008; cited) takes one step
from $r_i$ to $r_{i+1}$ by two reflections, each in a plane through the origin. Reflection in the
plane normal to $w$ is
$$\mathcal R_w(x) = x - \frac{2\,(w \cdot x)}{w \cdot w}\,w .$$
1. $w_1 = C_{i+1} - C_i$. Reflect: $r^L = \mathcal R_{w_1}(r_i)$, $T^L = \mathcal R_{w_1}(T_i)$.
2. $w_2 = T_{i+1} - T^L$. Reflect: $r_{i+1} = \mathcal R_{w_2}(r^L)$.

**Claim.** $r_{i+1}$ is a unit vector normal to $T_{i+1}$, exactly (up to rounding), with no
normalization.

*Proof.* A reflection is orthogonal, so $\mathcal R_{w_2}\mathcal R_{w_1}$ preserves lengths and dot
products, and $r_{i+1} \cdot \mathcal R_{w_2}(T^L) = r_i \cdot T_i = 0$. It remains to show
$\mathcal R_{w_2}(T^L) = T_{i+1}$. For unit vectors $a, c$ and $w = c - a$:
$w \cdot a = a \cdot c - 1$ and $w \cdot w = 2 - 2\,a \cdot c$, so
$\mathcal R_w(a) = a + (c - a) = c$. $T^L$ is a unit vector, since $\mathcal R_{w_1}$ is
orthogonal, so $\mathcal R_{w_2}(T^L) = T_{i+1}$. $\blacksquare$

Over 400 sample points along a helix, $\lvert r\rvert = 1$ and $r \cdot T = 0$ hold to $10^{-14}$
(`test_double_reflection_keeps_an_orthonormal_frame`).

The first reflection carries the geometry: $\mathcal R_{w_1}$ is the linear part of the mirror
in the plane that bisects $C_i$ and $C_{i+1}$, the mirror that swaps the two points, and it
carries the frame along with them. The second only corrects the tangent. Wang et al. show that
the result approximates the RMF to fourth order in the step (cited).

**The rejected alternative: projection.** The obvious scheme removes the $T_{i+1}$ component of
$r_i$ and normalizes. It stays orthonormal too, but it is less accurate. On a helix with
$a = 1$, $b = 0.5$ (curvature and torsion constant, so $\theta = -\tau s$ is known exactly), the
largest angle error over $t \in [0, 4\pi]$ with $m$ steps was (`test_rmf_convergence_on_a_helix`):

| $m$ | double reflection | projection |
| --- | ----------------- | ---------- |
| 8   | 7.86e-3           | 1.52       |
| 16  | 4.79e-4           | 0.345      |
| 32  | 2.98e-5           | 0.171      |
| 64  | 1.86e-6           | 0.0863     |
| 128 | 1.16e-7           | 0.0435     |

Doubling $m$ divides the double-reflection error by 16, consistent with fourth order, and the
projection error by 2: first order. At 128 steps that is a factor of $4 \times 10^5$.

## Closed paths: the RMF does not close

Carried once round a closed path, the RMF need not come back to where it started. *Claim.* Where
$\kappa > 0$ everywhere, it comes back turned about $T$ by $-\oint \tau\,ds$ (mod $2\pi$).

*Proof.* The Frenet frame is built from the derivatives at each point, so on a smooth closed path
it returns to itself. By the previous claim, the RMF's angle from $N$ changes by
$-\oint \tau\,ds$ on the way. $\blacksquare$

On the trefoil knot, $C(t) = ((2 + \cos 3t)\cos 2t, (2 + \cos 3t)\sin 2t, \sin 3t)$, the
mismatch is 3.0270 rad ($173.4°$). Double reflection finds it to $3 \times 10^{-4}$ with 64 steps
and to $8 \times 10^{-8}$ with 512; projection to 0.077 and $9.5 \times 10^{-4}$
(`test_rmf_does_not_close_on_a_trefoil`).

A sweep along a closed path must therefore add a twist to close up, spread along the path (Wang et
al. propose adding an angle that grows linearly in arc length; cited). Closed sweeps aren't built
here: see [[#What sweep doesn't do]].

## The surface: skinning exact sections

In general $X(u, t)$ is not a NURBS surface. $T = C'/\lvert C'\rvert$ involves
$\lvert C'\rvert$, the square root of a polynomial (of a rational function, for a NURBS path), and
the RMF solves a differential equation. Paths for which these are rational do exist: the
Pythagorean-hodograph curves (Farouki, *Pythagorean-Hodograph Curves*, 2008; cited). A generic
path isn't one.

What *is* exact is a single section. Placing the profile at $C(t_s)$ maps each profile point by
$$p \mapsto C(t_s) + x\,r(t_s) + y\,b(t_s) + z\,T(t_s),$$
a rotation followed by a translation. NURBS curves are invariant under affine maps
([[../01_Curves/08_NURBS]]), so mapping the control points, with the weights unchanged, maps the
curve exactly.

So `sweep` places $K$ sections at $t_s = s/(K-1)$ and skins them
([[04_Loft#Step 2: interpolation across the sections]]).
The result is exact at the sections and approximate in between:
- **Every section lies on the solid**, to $10^{-10}$ (`test_every_section_lies_on_the_solid`).
- **The rails are shared.** In a loft, end weights had to be 1 for neighbouring skins to share the
  longitudinal edge ([[04_Loft#The longitudinal edges]]). Here the sections are rigid copies of
  one profile, so every column of weights is constant across the sections. Interpolating a
  constant reproduces it (partition of unity, the same proof). Both skins next to a rail
  therefore interpolate the same points with the same constant weight, which makes them the same
  curve, whatever the profile's end weights.
- **A straight path gives the extrusion.** Along a line every frame is constant, the sections are
  translates, and cubic interpolation reproduces the linear motion exactly. The volume matches
  `extrude`'s to $10^{-15}$ (`test_a_straight_path_gives_the_extrusion`).

**Topology and orientation.** Topologically a sweep is an extrusion
([[02_Extrude#The construction]]): the first section as the profile, the last as the far copy,
the skins' boundary curves as rails. So $V = 2n$, $E = 3n$, $F = n + 2$. The profile is made
counter-clockwise seen from $+z$, which the placement carries to counter-clockwise seen from the
tip of $T$. The map has determinant $(r \times b) \cdot T = 1$, so it preserves orientation, and
every side face has sense $+1$ by the argument of [[04_Loft#The topology: an extrusion]]. The
Stokes identity of [[03_Revolve#Checking the orientation: Stokes]] holds on every side face, for
a rectangle and a circle, with both frames
(`test_every_step_is_valid_and_the_result_is_consistent`).

## Volume: Pappus along any path

**Claim.** Let the profile have area $A$ and its centroid at the origin, and let the path have
length $L$. If the exact sweep does not fold or cut through itself, its volume is
$$V = A\,L,$$
for *any* frame, Frenet or rotation-minimizing.

*Proof.* The sweep is the image of (profile region) $\times [0, 1]$ under
$\Phi(x, y, t) = C(t) + x\,r(t) + y\,b(t)$. Its Jacobian is
$$\det\left[\Phi_x, \Phi_y, \Phi_t\right] = (r \times b) \cdot \Phi_t = T \cdot (C' + x\,r' + y\,b') = v + x\,(r' \cdot T) + y\,(b' \cdot T).$$
From $r \cdot T = 0$, $r' \cdot T = -r \cdot T' = -v\kappa\,(r \cdot N)$, and the same for $b$. So
$$\det = v\,\bigl(1 - \kappa\,(x\,r \cdot N + y\,b \cdot N)\bigr).$$
Only $r \cdot N$ and $b \cdot N$ depend on the frame, and they multiply $x$ and $y$. Integrating
over the profile, $\iint x\,dA = \iint y\,dA = 0$ because the centroid is at the origin, so
$$V = \int_0^1 v\,A\,dt = A\,L .$$
The change of variables needs $\Phi$ to be one-to-one with $\det > 0$. Locally that holds when
$\kappa\,\rho < 1$, where $\rho$ is the profile's largest distance from the origin. That the sweep
doesn't cut through itself globally is assumed. $\blacksquare$

Revolve's Pappus theorem ([[03_Revolve#Volume: Pappus's theorem]]) is the special case of a
circular path: the centroid travels the arc $L = a\,\bar x$.

`arc_length` computes $L$ by Gauss–Legendre quadrature on each knot span; `volume` integrates over
the faces of the solid, independently. Their difference measures how far the skin is from the
exact sweep.

## How close the skin is

The rectangle $0.4 \times 0.2$ was swept along the cubic Bézier space path of the torsion
measurement above ($L = 4.4967$) with $K$ sections. Two measurements
(`test_sweep_converges_to_the_exact_sweep`):
- err: the largest distance from points of the exact sweep (profile edges placed by the same
  frames at 21 values of $t$) to the skin;
- $\Delta V = V - AL$.

| $K$ | err (RMF) | $\Delta V$ (RMF) | err (Frenet) | $\Delta V$ (Frenet) |
| --- | --------- | ---------------- | ------------ | ------------------- |
| 5   | 9.6e-3    | 3.1e-3           | 9.5e-3       | 4.3e-3              |
| 9   | 1.7e-4    | -6.6e-6          | 1.6e-4       | -1.2e-5             |
| 17  | 4.7e-6    | -1.5e-6          | 4.5e-6       | -1.9e-6             |
| 33  | 2.7e-7    | -8.0e-8          | 2.5e-7       | -1.0e-7             |

The error falls by about 17 at the last doubling, consistent with the fourth-order accuracy of
cubic interpolation; three doublings are evidence, not a proof. On this path, where the torsion
is moderate, the two frames do about equally well.

**Where Frenet turns fast, it needs far more sections.** The path of script 04 dips to a small
curvature near $t = 1/2$: $\kappa_{\min} = 0.078$ at lift 0.5, and 0.025 at lift 0.15. There Frenet's
frame turns by nearly half a turn against the RMF ($\int\tau\,ds = 3.005$ and 3.094 rad). Relative
volume errors $V/(AL) - 1$ (`test_frenet_needs_far_more_sections_where_it_turns_fast`):

| lift | $K$ | RMF     | Frenet   |
| ---- | --- | ------- | -------- |
| 0.5  | 12  | 4.8e-4  | -7.8e-3  |
| 0.5  | 24  | -1.3e-5 | -1.5e-3  |
| 0.5  | 48  | 7.0e-7  | -1.05e-4 |
| 0.15 | 12  | 6.8e-4  | -1.2e-2  |
| 0.15 | 24  | -1.4e-5 | -5.2e-3  |
| 0.15 | 48  | 1.2e-6  | -1.5e-3  |

At 48 sections Frenet is 150 and 1200 times worse. The skin has to follow the profile's fast turn
with sections spaced evenly in $t$, and the turn gets faster as the curvature dips towards zero.
This is the practical form of failure 2: well before the curvature reaches zero, the flip is
already a fast twist.

## Spacing the sections

The sections are equally spaced in $t$, so the parameters across can be the path's own,
$v_s = t_s$ ("path"), or come from the rules of
[[04_Loft#Spacing the sections: two measurements, two different winners]].
Here "path" is also "uniform". Relative volume errors on two paths, the second a cubic with three
control points crowded near its start, so that $\lvert C'\rvert$ varies strongly
(`test_sweep_spacing_comparison`):

| path       | spacing     | $K = 5$ | 9       | 17      | 33      |
| ---------- | ----------- | ------- | ------- | ------- | ------- |
| space path | path        | 8.6e-3  | -1.8e-5 | -4.1e-6 | -2.2e-7 |
| space path | chord       | 2.9e-2  | -6.4e-4 | -2.1e-5 | -6.0e-7 |
| space path | centripetal | 1.8e-2  | -2.0e-4 | -1.0e-5 | -4.0e-7 |
| uneven     | path        | 7.9e-6  | 2.5e-5  | 1.9e-6  | 1.3e-8  |
| uneven     | chord       | 5.0e-2  | -9.3e-5 | 3.6e-6  | 1.1e-7  |
| uneven     | centripetal | 4.3e-3  | 9.1e-6  | 3.2e-6  | 5.1e-8  |

"path" is the most accurate in 7 of the 8 columns; centripetal wins once. A plausible reading, not
proved: the exact sweep is a smooth function of $t$, and the sections are samples of it at
$t_s$, so interpolating at the $t_s$ themselves interpolates a smooth function in its own
variable. The other rules measure distances between control points, which a rigid motion along
a curve doesn't reflect well. Two paths are evidence, not a theorem.

## What sweep doesn't do

- **Closed paths.** They need a seam, as in a full revolve, and a correction twist for the RMF's
  mismatch.
- **Other frame laws** that CAD systems offer: a fixed binormal direction ("keep normal
  constant"), a twist law along the path, or a second guide curve that steers the profile.
- **Profiles placed anywhere else.** The profile is drawn in its own $xy$-plane and always placed
  normal to the path. A profile drawn somewhere in space, or kept at a constant angle to the
  path, is not supported.
- **Profiles with holes** (a tube), and **detecting folds or self-intersections**. Nothing checks
  $\kappa\,\rho < 1$.
- **Exact special cases.** A circle swept along a line or a circle is a cylinder or a torus,
  exactly NURBS. Kernels recognise such cases; here every sweep is skinned.

## In this implementation

**Input conventions** (`sweep`).
- The profile must be closed (extrude's `_check_profile`) and lie in $z = 0$, with control points
  within $10^{-9}$ of it. It is reoriented counter-clockwise seen from $+z$ by extrude's
  `_oriented`.
- The path is a `NurbsCurve` on $[0, 1]$. Regularity is checked only at the sample points:
  $\lvert C'\rvert = 0$ there is rejected, and nothing between them is checked.
- At least two sections. The degree across is $\min(3, K - 1)$.

**Frames** (`path_frames`).
- `"frenet"` evaluates $N$ at the section parameters directly. It is rejected where
  $\lvert C' \times C''\rvert \le 10^{-9}\,\lvert C'\rvert^3$ ($\kappa \le 10^{-9}$, a tolerance in
  units of inverse length). Script 04 hits this at lift 0 with 9 sections, which puts a section on
  the inflection.
- `"rmf"` starts at $r(0) = N(0)$, so both frames start alike and the script's angle plot starts
  at 0. If $N(0)$ doesn't exist (a straight start), it falls back to `default_normal`: the
  coordinate axis least aligned with $T(0)$, with its $T$ component removed.
- The RMF is carried by double reflection on a finer grid, `SUBSTEPS` $= 32$ steps per gap between
  sections, from $t = 0$. That accuracy was not measured on its own; the helix table above gives
  an idea ($1.2 \times 10^{-7}$ after 128 steps over two turns).
- `rotation_minimizing_frames` also offers `"projection"`, only so that the tests can measure it.

**The skin.** Sections at $t_s = s/(K-1)$, equally spaced in $t$, not in arc length. With
`parametrization="path"` (the default) the skin's $v_s$ are the $t_s$; `"chord"` and `"centripetal"`
call `skin_parameters` as loft does. The skin and rails come from `skin_faces`, shared with
`loft` (`ops/loft.py`). `compatible` isn't called, because rigid copies of one profile are already
compatible. The caps are planes through $C(0)$ and $C(1)$ with normals $-T(0)$ and $T(1)$.

**Derivatives.** `nurbs_derivatives` differentiates the lifted B-spline repeatedly
(`bspline_derivative`) and applies the quotient rule once per order. Past the degree $p$, the
lifted derivatives are set to zero.

**Status**: simplified. The faces are skins through exact sections, exact there and accurate to
the measured tables in between; the true swept surface is not NURBS in general. Implemented and
tested: `frenet_frames`, `rotation_minimizing_frames`, `curvature_torsion`, `nurbs_derivatives`,
`sweep`, in `tests/test_frames.py`, `tests/test_sweep.py`, `tests/test_nurbs.py`. Script 04 shows
the two frames, the flip, and the angle between them against the integrated torsion.
- *Proved inline:* Frenet's flip at a planar inflection; the RMF ODE keeps a unit normal; the RMF
  turns against Frenet by $-\int\tau\,ds$; a plane curve's RMF keeps the plane's normal; double
  reflection keeps the frame orthonormal exactly; the closed-path mismatch is $-\oint\tau\,ds$; the
  rails are shared without the end-weight condition; Pappus $V = AL$ for any frame; the counts.
- *Stated and measured:* the angle between Frenet and RMF against the integrated torsion; the
  convergence orders of double reflection (4) and projection (1); the trefoil's mismatch; the
  skin's convergence to the exact sweep; Frenet's loss of accuracy where it turns fast; the
  spacing comparison; validity and the Stokes identity after every step.
- *Cited, not proved:* the Frenet–Serret formulas and the curvature and torsion formulas (do
  Carmo); double reflection and its fourth-order accuracy, and the linear correction twist for
  closed paths (Wang et al. 2008); Pythagorean-hodograph curves (Farouki 2008); Piegl & Tiller's
  derivative formula.
