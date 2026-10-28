# Not covered in code

Concepts that are part of "how CAD kernels really work" but are deliberately kept notes-only in
this repo, either because they're intractable to hand-build in a learning timeframe or because
they're implementation/format details rather than modeling math. Add to this list whenever a
topic comes up during a script/note session that we decide not to implement.

- **Exact B-rep boolean operations** — real NURBS surface-surface intersection + trimming +
  robust tolerancing, as opposed to this repo's simplified BSP-tree mesh CSG
  ([[05_Booleans_and_Fillets/Real_Kernel_Comparison]], once written).
- **Variable-radius / general blend surfaces** — this repo only implements a constant-radius
  blend on simple polyhedral edges, not the general rolling-ball/spring-surface blending real
  kernels use for arbitrary edge networks.
- **Triangular Bezier patch subdivision and degree elevation** — evaluation, corner
  interpolation, hull containment and affine invariance for triangular Bezier patches are
  implemented in [[02_Surfaces/Bezier_Triangles]]; subdivision (splitting one Bezier triangle
  into sub-triangles) and degree elevation are a genuinely different derivation from the
  tensor-product case and are not built here. Kernels standardize on the tensor-product patch
  anyway, which is what this repo follows through to NURBS.
- **Variation diminishing for surfaces** — the curve property has no true analogue for
  tensor-product patches. Stated in [[02_Surfaces/Bezier_Surfaces]] from the literature; unlike
  every other property claim in that note, it is neither derived nor measured here.
- **Knot removal** — the inverse of knot insertion, within a tolerance. Without it,
  `elevate_degree` leaves every breakpoint at multiplicity p+1: the curve is unchanged, but the
  representation carries redundant knots ([[01_Curves/De_Boor_Algorithm]]).
- **Arc-length parametrization** — the NURBS circle is exact in shape but not in speed
  ($|C'|$ varies by a factor $\approx 1.17$, [[01_Curves/NURBS]]), and no rational parametrization
  of the circle has constant speed (cited, not proved). Kernels reparametrize numerically when
  arc length matters; not implemented here.
- **NURBS degree elevation, Bezier decomposition, knot removal** — all follow by applying the
  B-spline versions to the lifted points $(w_iP_i, w_i)$ ([[01_Curves/NURBS_Derivations]] §9); only
  knot insertion is wired up for NURBS (`nurbs_insert_knot`).
- **Trimmed surfaces** — a NURBS patch restricted to a region of its $(u, v)$ domain bounded by
  curves (a plate with a hole, a face cut by a boolean). Every B-rep face in a real kernel is one;
  this repo's curved faces are always the full parameter rectangle ([[02_Surfaces/NURBS_Surfaces]]).
  The one exception is the plane: a planar face is the region its edge curves enclose, which
  needs no trimming in parameter space ([[04_Modeling_Operations/Extrude]]).
- **NURBS surface degree elevation, Bezier decomposition, knot removal** — the curve versions
  applied per direction to the lifted net; only knot insertion is wired up for surfaces
  (`nurbs_surface_insert_knot`).
- **Pole-free sphere parametrizations** — multi-patch spheres (e.g. six patches on a cube) that
  avoid the collapsed rows of the single revolved patch ([[02_Surfaces/Surfaces_of_Revolution]]).
- **Least-squares curve approximation** — fitting fewer control points than data points, to a
  tolerance, for noisy measured data. Only exact interpolation is implemented
  ([[01_Curves/B_Spline_Interpolation]]).
- **Exchange formats** — STEP, IGES: how B-rep data is serialized/interchanged between kernels.
- **Robust numerical tolerancing** — how kernels handle floating-point tolerance in
  intersection/boolean algorithms so results stay topologically consistent.
- **Non-manifold topology, sheet bodies, lattice/cellular topology** — this repo's half-edge
  structure targets manifold solids only.
- **Multi-shell solids (internal voids)** — the operators allow more than one shell, but no
  construction here builds one, and shells are counted, not stored ([[03_Topology/B_Rep_Topology]]).
- **Self-intersection checks** — `check` validates topology, planarity and that edge curves lie on
  their faces, not whether faces pass through each other; that is a geometric test with
  tolerances ([[03_Topology/B_Rep_Topology]]). `extrude` likewise accepts a self-intersecting
  profile, or holes that leave the profile ([[04_Modeling_Operations/Extrude]]).
- **Extrude to a face** ("up to next", "through all") — needs the extrusion to be cut by other
  geometry, i.e. surface intersection ([[04_Modeling_Operations/Extrude]]).
- **Parameter-space curves (pcurves) and degenerated edges** — real kernels store, for every edge
  of a face, its curve in the face's $(u, v)$ rectangle as well as in 3D: two different pcurves for
  the two sides of a seam, and a pcurve with no 3D curve at a pole. This repo stores 3D curves
  only, so a pole is just a vertex ([[04_Modeling_Operations/Revolve]]).
- **Revolve about any axis, with holes in the profile** — `revolve` turns about the $z$-axis only,
  and its profile has no holes ([[04_Modeling_Operations/Revolve]]).
- **Loft refinements** — sections with different edge counts, automatic vertex correspondence
  (minimizing twist), guide curves, and closed lofts that join the last section back to the first
  ([[04_Modeling_Operations/Loft]]).
- **Adaptive tessellation** — real kernels tessellate curved surfaces adaptively (denser mesh
  where curvature is higher) for display/export; this repo uses uniform parameter-space
  sampling throughout.
- **Kernel-specific architecture** (ACIS, Parasolid, OpenCASCADE internals) — proprietary/
  large-codebase specifics beyond the general algorithms covered here.

This list is expected to grow as the project progresses — it's the honest record of the gap
between this repo and a production kernel, not a to-do list.