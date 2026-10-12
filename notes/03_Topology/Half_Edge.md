Code: [[../../scripts/03_topology/01_halfedge_navigation.py]] — `HalfEdge`, `Loop`, `Face`, `vertex_star`, `adjacent_faces`, `check`, `signed_volume`, `from_polygons` in `cadkernel/topology/halfedge.py`

# The half-edge data structure

Why a B-rep needs topology at all, and why the half-edge beats the alternatives, is argued in
[[B_Rep_Topology]]. This note is the structure itself: its pointers, the invariants that make it
valid, and the walks those pointers allow.

## The pointers

Every edge is split into two **half-edges**, one in each of the two faces it bounds, running in
opposite directions.

| object | stores |
|---|---|
| half-edge $h$ | `origin` (start vertex), `loop`, `next` and `prev` in that loop, `edge` |
| edge | its two half-edges; each is the other's `twin` |
| loop | one of its half-edges (the rest follow by `next`), and its face |
| face | one outer loop, and any inner loops (rings) |
| vertex | its position, and one outgoing half-edge |

Everything else is derived. A half-edge's end vertex is `h.next.origin`, and its face is
`h.loop.face`.

## The invariants

A structure is valid when all of these hold. `check` tests each one and reports violations.

1. **`twin` is an involution** and twins run in opposite directions: $h.\text{twin}.\text{twin} = h$,
   and $h.\text{twin}$ starts where $h$ ends.
2. **`next` and `prev` are inverse**, and following `next` from any half-edge comes back to it:
   every loop is a cycle.
3. **Every vertex is manifold:** the half-edges leaving it form a single cycle under the star walk
   below.
4. **Euler–Poincaré** holds for the counts ([[Euler_Operators]]).
5. **Faces are planar** — geometry, not topology, and checkable only to a tolerance. It is
   switched off for the intermediate states of a construction.

Breaking a single twin pointer, or a single `next`, is caught
(`test_check_catches_a_broken_twin`, `test_check_catches_a_broken_next`). The broken `next`
changes no count at all, so Euler–Poincaré still holds: **the formula is necessary, not
sufficient.** That is why the pointer invariants are checked separately.

**Orientation.** Each outer loop runs counter-clockwise seen from outside (the right-hand rule gives
the outward normal), so the face material lies to the **left** of every half-edge. Inner loops run
the other way, which keeps the material on the left of them too. Consistent orientation is the
same statement as "each directed edge $(a, b)$ occurs exactly once": two neighbouring faces that
both used $a \to b$ would be facing opposite ways. `from_polygons` pairs twins on exactly this
rule, and rejects a surface with one face flipped (a directed edge used twice) or one face missing
(an edge with no twin) (`test_from_polygons_rejects_open_and_inconsistent_surfaces`).

**Measuring orientation: signed volume.** The divergence theorem with $\nabla\cdot x = 3$ gives
$$\text{Vol} = \frac13 \oint_{\partial} x\cdot n\, dA = \frac13\sum_{f} \mathbf A_f\cdot p_f ,$$
because on a planar face $x\cdot n$ is constant: it equals $n\cdot p_f$ for any point $p_f$ of
the face. Here $\mathbf A_f$ is the face's vector area (area times unit normal),
$\tfrac12\sum_k p_k \times p_{k+1}$ summed over its loops (the 3D shoelace formula, cited). A ring
runs the other way, so it subtracts its hole automatically. The result is positive exactly when
the normals point outward. Measured:
- a cube of side 2 gives $8$;
- the octahedron gives $4/3$;
- the $3\times3\times1$ block with a $1\times1$ through-hole gives $9 - 1 = 8$

(`test_finished_solids_are_outward_oriented`).

## Walks

All of these take constant work per step, with no searching:

- **Round a face:** `next`, until back at the start. A cube face is a 4-cycle.
- **Across an edge:** `twin`, which lands in the neighbouring face. So `adjacent_faces(f)` is one
  `twin` per half-edge of $f$.
- **Round a vertex:** `twin.next`.

*Why `twin.next` turns round the vertex.* Let $h$ leave vertex $v$. Then $h.\text{twin}$ runs the
other way, so it *ends* at $v$, and it lies in the face on the other side of $h$'s edge. The `next`
of a half-edge starts where that half-edge ends, so $h.\text{twin}.\text{next}$ *leaves* $v$, in
that neighbouring face. Each step crosses exactly one face. At a manifold vertex the faces around
$v$ form one fan, so the walk visits every outgoing half-edge once and returns to $h$. $\blacksquare$

Measured: on a closed quad mesh every vertex star has 4 half-edges, and consecutive ones are
separated by exactly one face (`test_vertex_star_walks_round_the_vertex_face_by_face`). On the cube,
every vertex has 3 and every face has 4 neighbours (`test_cube_queries`). At a non-manifold vertex
the walk returns early, after one fan, and `check` compares its length with the number of outgoing
half-edges to catch exactly that (`test_check_catches_a_non_manifold_vertex`).

Script 01 has these as buttons: **next**, **prev**, **twin**, **twin.next**. On the cube,
next ×4, twin ×2 and twin.next ×3 each return to the starting half-edge.

**Status**: exact (topology), polyhedral geometry. Proved inline: why `twin.next` walks round a
vertex; the signed-volume formula from the divergence theorem; that consistent orientation means
each directed edge occurs once. Stated and measured: the invariants and their violations, the
volumes, the star structure. Cited, not proved: the 3D shoelace formula for vector area.
