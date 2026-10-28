# Open questions

Questions raised while working through the repo, parked here **unanswered on purpose**. We come
back to them once most of the theory and code is done, when they can be answered from the
implementations and from the lessons that carry over to modelling in any CAD program.

New questions are appended at the bottom, numbered, with the date and the note being worked on
when they came up. Answers, once written, go under each question.

## Q1 — Does the tensor product dictate how surfaces must be modelled?

*2026-10-25, while on [[02_Surfaces/NURBS_Surfaces]].*

Does the tensor product expression for surface affect how should I model the parametric
surfaces — do they need to be constructed in a way to suit the tensor product? Or do most modern
kernels have also other methods of constructing surfaces — which tools are usually used for those
purposes?

Just from the top of my head I can think of at least three scenarios:
1. One that works with the regular tensor product, that is a grid of $m \times n$ curves.
2. A triangular grid, which isn't covered by the regular tensor product.
3. A closed loop curve with additional curves that are bound to the loop and define the surface.
