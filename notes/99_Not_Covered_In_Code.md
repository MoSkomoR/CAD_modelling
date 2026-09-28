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
- **Least-squares curve approximation** — fitting fewer control points than data points, to a
  tolerance, for noisy measured data. Only exact interpolation is implemented
  ([[01_Curves/B_Spline_Interpolation]]).
- **Exchange formats** — STEP, IGES: how B-rep data is serialized/interchanged between kernels.
- **Robust numerical tolerancing** — how kernels handle floating-point tolerance in
  intersection/boolean algorithms so results stay topologically consistent.
- **Non-manifold topology, sheet bodies, lattice/cellular topology** — this repo's half-edge
  structure targets manifold solids only.
- **Adaptive tessellation** — real kernels tessellate curved surfaces adaptively (denser mesh
  where curvature is higher) for display/export; this repo uses uniform parameter-space
  sampling throughout.
- **Kernel-specific architecture** (ACIS, Parasolid, OpenCASCADE internals) — proprietary/
  large-codebase specifics beyond the general algorithms covered here.

This list is expected to grow as the project progresses — it's the honest record of the gap
between this repo and a production kernel, not a to-do list.