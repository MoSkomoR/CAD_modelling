# CAD Modelling — a hand-built CAD kernel, for learning

Parametric curves, surfaces, B-rep topology, and modeling operations (extrude/revolve/sweep/
loft/boolean/fillet), implemented from scratch in numpy and visualized with matplotlib — not a
wrapper around an existing CAD kernel. The goal is hands-on intuition for the math and data
structures real kernels (OpenCASCADE, Parasolid, ACIS) are built on.

## Layout
- `src/cadkernel/` — the reusable library (geometry, topology, ops, viz), built up module by module.
- `scripts/` — numbered, **interactive** scripts: drag the control points, move the sliders, watch
  the geometry respond (`python scripts/01_curves/02_bezier_de_casteljau.py`). They open a window
  and write no files. To check one runs without a display:
  `MPLBACKEND=Agg CADK_SMOKE=1 python scripts/01_curves/02_bezier_de_casteljau.py`.
- `tests/` — pytest sanity checks for the library (`pytest`).
- `notes/` — Obsidian vault content. Open the **repo root** as the Obsidian vault (not just
  `notes/`) so notes can link directly to the scripts/code they describe. Start at
  `notes/00_Map_of_Content.md` — it's the curriculum index and tracks, per topic, whether it's
  implemented exactly, simplified, or notes-only, and whether the note itself exists yet.
  `notes/99_Not_Covered_In_Code.md` lists concepts intentionally left out of the code.

## Setup
```
source .venv/bin/activate
pip install -e ".[dev]"
```

## Scope note
Booleans and fillets are implemented as **simplified but real** techniques (BSP-tree mesh CSG;
constant-radius blends on simple polyhedral edges) rather than the exact NURBS
intersection/trimming real kernels use — that's a multi-month undertaking on its own. The notes
for that section explicitly compare the simplified approach to what real kernels do.