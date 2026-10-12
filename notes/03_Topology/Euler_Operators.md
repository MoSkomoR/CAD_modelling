Code: [[../../scripts/03_topology/02_euler_operators.py]] — `mvfs`, `lmev`/`lkev`, `lmef`/`lkef`, `lkemr`/`lmekr`, `lkfmrh`/`lmfkrh`, and the builders `cube`, `tetrahedron`, `block_with_hole` in `cadkernel/topology/halfedge.py`

# Euler operators — building solids that are valid at every step

[[Half_Edge]] says what a valid structure is. This note is about how to *change* one without ever
leaving validity: how solids are built and edited.

## Why operators, and not pointer surgery

`from_polygons` builds a half-edge solid in one go from a finished list of faces. That works for
importing a complete mesh, but a modeller does something else: it edits. It splits a face, adds an
edge, opens a hole. Doing that by rewiring pointers by hand is where invalid structures come from.
A missed `prev` is invisible until a traversal loops forever.

**Euler operators** are a small vocabulary of local edits. Each one:
- changes a handful of pointers;
- changes the element counts by a fixed amount, chosen so the Euler–Poincaré formula below keeps
  holding;
- has an exact inverse, which gives undo for free.

A construction made only of operators is valid after every single step. Mäntylä built his
modeller on this idea; the operators here follow his (Mäntylä 1988, cited).

## Euler's formula

**Claim.** If a polyhedron's surface is a topological sphere (no handles, no holes in faces),
$$V - E + F = 2 .$$

*Proof.* Remove one face and flatten the rest of the surface onto the plane, with stereographic
projection from a point inside the removed face. The result is a connected graph drawn in the plane
with $V$ vertices and $E$ edges. It divides the plane into $F$ regions: the $F - 1$ remaining faces,
plus the unbounded outside, which stands for the removed face. Now induct on the number of edges.
- **The graph is a tree** (no cycles). Then $E = V - 1$ and there is one region, so
  $V - E + F = V - (V-1) + 1 = 2$.
- **Otherwise** pick an edge on a cycle. A cycle is a closed curve, so it separates the plane
  (Jordan curve theorem, cited), and the regions on its two sides are different. Removing the edge
  merges those two regions: $E$ drops by 1, $F$ drops by 1, and $V - E + F$ is unchanged. The graph
  stays connected, because the rest of the cycle still joins the edge's ends.

Each step removes an edge, so this ends at a tree, where the value is 2. $\blacksquare$

The tetrahedron $(4, 6, 4)$, the cube $(8, 12, 6)$ and the octahedron $(6, 12, 8)$ all give 2
(`test_euler_poincare_on_known_solids`).

## The general form: Euler–Poincaré

Solids with holes in their faces, or holes through them, need two more counts:
- $R$, the **rings** (inner loops of faces);
- $G$, the **genus** (number of handles: holes through the solid).

$S$ counts the **shells**. Then
$$V - E + F - R = 2\,(S - G).$$
This is stated here, not proved. It is Poincaré's generalization, in the form Mäntylä (1988) uses,
and is cited.

Measured on five solids (`test_euler_poincare_on_known_solids`):

| solid | $V$ | $E$ | $F$ | $R$ | $S$ | $G$ | both sides |
|---|---|---|---|---|---|---|---|
| tetrahedron | 4 | 6 | 4 | 0 | 1 | 0 | 2 |
| cube | 8 | 12 | 6 | 0 | 1 | 0 | 2 |
| octahedron | 6 | 12 | 8 | 0 | 1 | 0 | 2 |
| block with a square through-hole | 16 | 24 | 10 | 2 | 1 | 1 | 0 |
| $8 \times 6$ quad torus | 48 | 96 | 48 | 0 | 1 | 1 | 0 |

**The genus is not stored by the topology.** Nothing in the pointers says "this solid has a
handle". Mäntylä *computes* $G$ from the formula. Here the solid tracks $G$ through the operators
that change it, so the formula is a genuine check rather than a definition. For `from_polygons`
the genus is passed in.

**Shells must be counted through faces, not just edges.** On the block with a hole, right after
`kemr`, no edge connects the hole's square to the rest of the solid. The two are joined only
because both are loops of the top face. Counting connected components of the vertex–edge graph
alone gives $S = 2$ and breaks the formula; joining each face's loops gives $S = 1$
(`test_a_ring_is_connected_to_its_shell_only_through_its_face`). That bug happened during
development, and the formula caught it.

## The operators

Each operator changes $(V, E, F, R, S, G)$ by a fixed amount. Write $\Delta L$ for the change in
the left side of the formula, $V - E + F - R$, and $\Delta R_{\text{hs}}$ for the change in the
right side, $2(S - G)$:

| operator | what it does | $\Delta V$ | $\Delta E$ | $\Delta F$ | $\Delta R$ | $\Delta S$ | $\Delta G$ | $\Delta L$ | $\Delta R_{\text{hs}}$ |
|---|---|---|---|---|---|---|---|---|---|
| `mvfs` | make vertex, face, shell: the seed | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 2 |
| `mev` | make edge and vertex | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `mef` | make edge and face: split a loop in two | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `kemr` | kill an edge, make a ring | 0 | −1 | 0 | 1 | 0 | 0 | 0 | 0 |
| `kfmrh` | kill a face, make a ring and a hole | 0 | 0 | −1 | 1 | 0 | 1 | −2 | −2 |

**Claim.** Every operator, and every inverse, keeps $V - E + F - R = 2(S - G)$.

*Proof.* The last two columns are computed from the others ($\Delta L = \Delta V - \Delta E +
\Delta F - \Delta R$, $\Delta R_{\text{hs}} = 2(\Delta S - \Delta G)$), and they agree in every row.
An inverse negates its row, so its two columns agree as well. Since the empty solid satisfies
$0 = 0$, every sequence of operators starting from it satisfies the formula. $\blacksquare$

Measured: each operator changes the counts by exactly its row; its inverse restores the previous
counts **and** the exact connectivity, compared loop by loop (`test_mev_and_kev`,
`test_mef_and_kef`, `test_kemr_and_mekr`, `test_kfmrh_and_mfkrh`). The last test plugs the
through-hole with `mfkrh`, which leaves a block with a blind pocket. That is still valid, with
volume $9 - 1$.

## Two constructions, step by step

Script 02 replays these with a step slider, the counts and both sides of the formula alongside.

**A box** is a *sweep*:
1. `mvfs` plants one vertex.
2. Three `mev`s grow a wire.
3. `mef` closes it into a square: a **lamina**, two faces back to back, $(4, 4, 2)$, already
   satisfying $V - E + F = 2$.
4. Four `mev`s grow vertical edges out of one of the two faces.
5. Four `mef`s close the sides. The face the edges grew from ends up as the top.

**The block with a hole** continues from a $3 \times 3 \times 1$ box:
1. Five `mev`s on the top draw a spike to $(1, 1)$ and a square wire.
2. `mef` closes the square into a new face inside the top.
3. `kemr` removes the spike. The square's boundary is now a ring of the top face.
4. The square face is swept down through the block (four `mev`s, four `mef`s).
5. `kfmrh` makes it a ring of the bottom face, and the hole goes through.

The left side of the formula is 2 at every step except the last, where `kfmrh` alone drops it to
0 (`test_the_genus_one_step_is_kfmrh`).

**Valid topology is not valid geometry.** Every intermediate state passes the topological check
(`test_every_step_of_every_construction_is_valid`), including:
- a lone vertex;
- a wire;
- a lamina;
- a top face with four edges sticking out of its plane.

None of those is a solid anyone would draw, and the face with spikes is not even planar. The
operators guarantee topology only; putting the vertices in sensible places is the caller's job. Only
the finished solids are checked with geometry (planar faces, positive volume).

## Completeness

Can every valid solid be built this way? Mäntylä (1988) states that these operators are
**complete**: every orientable 2-manifold solid can be reached from `mvfs` by a sequence of them.
That is cited, not proved here. The constructions above are evidence only for the solids they
build.

**Status**: exact (topology). Implemented and tested in `tests/test_halfedge.py`: all ten operators,
the three constructions, `from_polygons`. Script 02 replays the constructions. Proved inline:
Euler's formula $V - E + F = 2$ for sphere-like polyhedra (given the Jordan curve theorem and the
flattening), and that each operator preserves Euler–Poincaré. Stated and measured: the counts of
five solids, each operator's count row and its exact inverse, validity after every step, the
genus-1 step, the shell count through rings. Cited, not proved: the Euler–Poincaré formula in
general, the Jordan curve theorem, and completeness of the operators (Mäntylä 1988).
