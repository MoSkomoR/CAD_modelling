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