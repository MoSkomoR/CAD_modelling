# Map of Content — Hand-built CAD kernel

Vault root = repo root, so links below resolve to both notes and the `.py` files that
implement them. See [[../README|README]] for how to run scripts. See
[[99_Not_Covered_In_Code]] for the running list of concepts that are notes-only by design.

Fidelity legend: **exact** = real math, no shortcuts · **simplified** = a real but simplified
stand-in technique (documented gap to the real kernel algorithm) · **notes-only** = not
implemented, conceptual notes only.

Scripts are **interactive** — run them and drag things; they produce no files. See
[[../CLAUDE|CLAUDE.md]] for the conventions.

## 01 — Curves
| Topic | Script | Fidelity | Note |
|---|---|---|---|
| Lines & arcs; why analytic-only isn't enough | [[../scripts/01_curves/01_lines_and_arcs.py]] | exact | [[01_Curves/Lines_and_Arcs]] |
| Bezier curves — the motivation hub | [[../scripts/01_curves/02_bezier_de_casteljau.py]] | exact | [[01_Curves/Bezier_Curves]] |
| De Casteljau: ⇔ Bernstein, subdivision, stability | [[../scripts/01_curves/02_bezier_de_casteljau.py]] | exact (proof) | [[01_Curves/De_Casteljau_Derivation]] |
| Bernstein basis & the property proofs | [[../scripts/01_curves/03_bezier_properties.py]] | exact (proofs) | [[01_Curves/Bernstein_Basis_Properties]] |
| Rational Bezier — why a circle needs it | [[../scripts/01_curves/04_circle_needs_rational.py]] | exact | [[01_Curves/Lines_and_Arcs]] |
| B-splines (Cox-de Boor) | not started | — | not started |
| NURBS (full: knots + weights) | not started | — | not started |

## 02 — Surfaces
| Topic | Script | Fidelity | Note |
|---|---|---|---|
| Bezier surfaces — tensor product, De Casteljau twice | [[../scripts/02_surfaces/01_bezier_surface_de_casteljau.py]] | exact | [[02_Surfaces/Bezier_Surfaces]] |
| Triangular Bezier patches (barycentric De Casteljau) | not started | notes-only | [[02_Surfaces/Bezier_Surfaces]] (section) |
| Analytic surfaces (plane/cylinder/sphere/cone/torus) | not started | — | not started |
| B-spline & NURBS surfaces | not started | — | not started |
| Normals & curvature | partial (`bezier_surface_normal`) | exact | [[02_Surfaces/Bezier_Surfaces]] |

## 03 — Topology
Not started. Planned: half-edge mesh structure, Euler operators, Euler's formula derivation,
shared-edge queries, solid/manifold validity checks.

## 04 — Modeling operations
Not started. Planned: extrude, revolve, sweep (Frenet frames & twist), loft.

## 05 — Booleans & fillets (simplified tier)
Not started. Planned: BSP-tree mesh CSG, boolean union/cut/intersect, constant-radius fillet
blend on simple polyhedral edges, and a dedicated note contrasting this simplified approach
with real exact-BRep boolean/blend algorithms.

## Legend for future entries
Every new topic added to this repo gets a row here the same day its script/note lands, so this
table always reflects what's actually on disk — not an aspirational curriculum.