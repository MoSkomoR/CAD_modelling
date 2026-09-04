# CLAUDE.md

Guidance for Claude Code sessions working in this repo.

## What this project is

A hand-built CAD kernel, for learning. Parametric curves/surfaces, B-rep topology, and modeling
operations (extrude/revolve/sweep/loft/boolean/fillet), implemented from scratch in numpy and
explored through interactive matplotlib scripts.

**Not** a wrapper around pythonocc/build123d/CadQuery. The goal is to see the explicit maths and
data structures a real kernel (OpenCASCADE/Parasolid/ACIS) is built on. Dependencies stay numpy
+ matplotlib + pytest — do not add a CAD library to the core curriculum. (Using one later purely
to cross-check a result against a real kernel is fine, if kept clearly separate.)

## Layout

```
src/cadkernel/{geometry,topology,ops,viz}/   reusable library, built up module by module
scripts/NN_topic/NN_name.py                  interactive, runnable demo scripts
tests/                                       pytest sanity checks
notes/                                       Obsidian vault content (vault root = repo root)
```

The Obsidian vault is the **repo root**, not `notes/`, so notes can link straight to the `.py`
files that implement them. `.obsidian/` is gitignored.

## Module sequencing

Curves → surfaces → topology (half-edge + Euler operators) → modeling ops → booleans/fillets.
Each stage depends on the ones before it.

Booleans and fillets are deliberately the **simplified tier**: BSP-tree mesh CSG, and
constant-radius blends on simple polyhedral edges — not the exact NURBS surface-surface
intersection and trimming real kernels use, which is out of scope. Notes in that section must
say plainly what is being approximated. Anything deliberately left unimplemented goes in
`notes/99_Not_Covered_In_Code.md` rather than being silently skipped.

## The workflow: one vertical slice at a time

Add a topic end to end, in this order — not many topics' code first and notes later:

1. **Library code** in `src/cadkernel/...`, docstring pointing at the note(s) that cover it.
2. **Tests** in `tests/` for every nonobvious claim (endpoint interpolation, partition of unity,
   subdivision correctness, Euler characteristic, watertightness). Run them.
3. **Interactive script** in `scripts/NN_topic/` (see below). Smoke-run it.
4. **Note(s)** in `notes/NN_Topic/`, describing *verified* behaviour — never assumed behaviour.
   If a note states a number, a test or a run should have produced it.
5. **Update `notes/00_Map_of_Content.md`** so the coverage table never drifts from what is on
   disk.

## Scripts are interactive, and produce no files

Scripts exist to be played with — drag control points, move sliders, watch the geometry respond.
They do **not** write PNGs or feed anything into Obsidian.

- Build on `cadkernel.viz.interactive`: `DraggableControlPoints`, `add_slider`,
  `add_checkbuttons`, and `run(fig, smoke_hook)`.
- End every script with `run(fig, smoke)` instead of `plt.show()`.
- **Every script needs a smoke hook.** Under `CADK_SMOKE=1` the script builds its figure, runs
  the hook (which must exercise the drag/slider callbacks and assert the properties being
  demonstrated), and exits without a window. This is the only way these scripts can be verified
  from a session with no display:
  ```
  MPLBACKEND=Agg CADK_SMOKE=1 python scripts/01_curves/03_bezier_properties.py
  ```
  The real visual check is the user running them normally, from an editor.
- Keep shared logic in `cadkernel`; scripts should demonstrate, not reimplement.

## Notes are motivation-first

The formulas are the easy part — the repo exists for the reasoning. Every note must answer
**"why this, and not the obvious alternative?"**, not just state what is true. Concretely:

- Lead with the motivation, and name the alternative being rejected and why (analytic-only
  geometry, the power basis, the closed Bernstein form, polynomial-only curves…).
- **Full derivations inline**, not just references — e.g. `De_Casteljau_Derivation.md` proves
  the Bernstein equivalence from Pascal's rule, `Lines_and_Arcs.md` proves no polynomial
  parametrizes a circle.
- **Separate the note types**: algorithm notes stay about the algorithm; property proofs live in
  their own properties note (`Bernstein_Basis_Properties.md`); one note per module acts as the
  motivation hub (`Bezier_Curves.md`).
- Open with a `Code:` link to the implementing script/module; close with a `**Status**:` line
  giving fidelity (exact / simplified / notes-only) and what is tested or visualized.
- Cross-link liberally with `[[wikilinks]]`.
- **Be honest when a demo undercuts the tidy story.** The polynomial-vs-circle case is the
  worked example: polynomial fits reach ~1e-15 by degree 13, so "not accurate enough" is the
  *wrong* argument for rational curves; the real arguments are exactness in principle, cost, and
  semantics. Measure first, then write the note around what actually happened.

## Style

Match the surrounding code: numpy-flavoured, type hints on public functions, docstrings that
explain *why* the algorithm is shaped the way it is (these docstrings are teaching material, and
carry more prose than production code would).
