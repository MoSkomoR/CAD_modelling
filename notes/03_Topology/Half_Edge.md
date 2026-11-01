Code: [[../../scripts/03_topology/01_halfedge_navigation.py]] — `HalfEdge`, `Loop`, `Face`, `vertex_star`, `adjacent_faces`, `check`, `signed_volume`, `from_polygons` in `cadkernel/topology/halfedge.py`

# The half-edge data structure

Why a B-rep needs topology at all, and why the half-edge beats the alternatives, is argued in
[[B_Rep_Topology]]. This note is the structure itself: its pointers, the invariants that make it
valid, and the walks those pointers allow.

## The pointers

Every edge is split into two **half-edges**, one in each of the two faces it bounds, running in
opposite directions. A "pointer" is a reference from one element to another; the structure is
nothing but these references.

The minimum a half-edge $h$ has to record is:
- its **origin**, the vertex it starts at;
- its **twin**, the other half of its edge, which runs the opposite way in the neighbouring face;
- its **next**, the half-edge that follows it round its loop;
- its **loop**, the boundary cycle it lies in. A face is bounded by one outer loop and possibly
  inner loops (rings), so a face is reached through its loops.

Each loop and each vertex also needs one entry point: one half-edge of the loop, one half-edge
leaving the vertex. Everything else can be derived from these:
- the end vertex of $h$ is $h.\text{next}.\text{origin}$, or equally $h.\text{twin}.\text{origin}$;
- the face of $h$ is the face of its loop;
- the half-edge *before* $h$ in its loop is found by following `next` round the loop until the
  step that lands on $h$. That costs one step per half-edge of the loop.

Implementations often store more than this minimum to make common steps cheaper. Every stored
extra is one more thing the edits must keep consistent. What this repo stores is under
[[#In this implementation]].

## The invariants

A structure is valid when all of these hold. `check` tests each one and reports violations.

1. **`twin` is an involution** and twins run in opposite directions: $h.\text{twin}.\text{twin} = h$,
   and $h.\text{twin}$ starts where $h$ ends.
2. **Every loop is a cycle:** following `next` from any half-edge comes back to it. If the
   predecessor is stored as well (`prev`), it must be the inverse of `next`.
3. **Every vertex is manifold:** the half-edges leaving it form a single cycle under the star walk
   below.
4. **Euler–Poincaré** holds for the counts ([[Euler_Operators]]).
5. **Faces are planar** — geometry, not topology, and checkable only to a tolerance. A
   polyhedral face is a piece of one plane; a construction in progress may violate this
   ([[Euler_Operators#Two constructions, step by step]]). Once faces carry surfaces
   (module 04), this becomes "the geometry agrees with the topology"
   ([[../04_Modeling_Operations/Modeling_Operations#Binding geometry to topology]]).

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

## In this implementation

`halfedge.py` follows the layout of Mäntylä's modeller (1988, cited). These are the stored
fields (`__slots__`):

| class | stored fields |
|---|---|
| `HalfEdge` | `origin`, `edge`, `loop`, `next`, `prev` |
| `Edge` | `he1`, `he2`, and `curve` (module 04) |
| `Loop` | `face`, and `he`, one of its half-edges |
| `Face` | `solid`, `loops`, `outer`, and `surface`, `sense` (module 04) |
| `Vertex` | `point`, and `he`, one outgoing half-edge |

**Derived, not stored.**
- `twin` is not a field. The half-edge stores its `edge`, and `he.twin` returns the edge's other
  half. One pointer per half-edge serves both "which edge" and "which twin".
- `target` is `next.origin`, and `face` is `loop.face`.

**Stored although derivable, and what each costs.**
- **`prev`.** It makes the backward step constant time, and lets an operator insert a half-edge
  just before a given one (`_add_halfedge`) without walking the loop. The price is invariant 2's
  second half: every operator must update `prev` alongside `next`. The "missed `prev`" danger in
  [[Euler_Operators#Why operators, and not pointer surgery]] exists only because this field is
  stored.
- **`Loop` as its own object**, between half-edges and faces. A half-edge points to its loop, not
  its face. Moving a whole loop to another face, which `kfmrh` does, then changes one field (the
  loop's `face`) instead of one field per half-edge.

**The seed vertex.** After `mvfs` there is a vertex and a face, but no edge. Its loop holds one
half-edge whose `edge` is `None`: a loop that is a bare vertex. `check` skips the twin tests for
it, and the first `mev` reuses it.

**Guards against corrupted pointers.** A broken `next` can make a cycle that never returns to its
start. `Loop.halfedges` stops after 100 000 steps and `vertex_star` after 10 000, so a corrupted
structure makes `check` report a problem instead of hanging.

**Not stored at all:** shells and the genus. See [[B_Rep_Topology#In this implementation]].

**Tolerances in `check`.** Invariants 1–4 are exact comparisons of references and integers.
Invariant 5 compares distances: a face is planar if its vertices lie within $10^{-9}$ of the plane
through them (`planar_tol`). `check(geometric=False)` skips it, which is how the intermediate
states of a construction are checked.

**Status**: exact (topology), polyhedral geometry. Proved inline: why `twin.next` walks round a
vertex; the signed-volume formula from the divergence theorem; that consistent orientation means
each directed edge occurs once. Stated and measured: the invariants and their violations, the
volumes, the star structure. Cited, not proved: the 3D shoelace formula for vector area.
