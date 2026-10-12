import numpy as np
import pytest

from cadkernel.topology.halfedge import (
    adjacent_faces,
    block_with_hole,
    check,
    cube,
    edge_faces,
    euler_poincare,
    find_halfedge,
    from_polygons,
    kemr,
    lkef,
    lkev,
    lkfmrh,
    lmekr,
    lmfkrh,
    mef,
    mev,
    octahedron,
    signed_volume,
    tetrahedron,
    torus_grid,
    vertex_star,
)


def signature(solid):
    """Connectivity as plain data: every face as its set of loops, every loop as its vertex
    cycle rotated to start at the smallest vertex id. Object identities are ignored."""
    def loop_key(loop):
        ids = [v.id for v in loop.vertices()]
        k = ids.index(min(ids))
        return tuple(ids[k:] + ids[:k])
    return sorted(tuple(sorted(loop_key(loop) for loop in face.loops)) for face in solid.faces)


def delta(before, after):
    return {k: after[k] - before[k] for k in ("V", "E", "F", "R", "S", "G")}


# --- Validity of every construction, at every step -------------------------------------------------


@pytest.mark.parametrize("builder", [cube, tetrahedron, block_with_hole])
def test_every_step_of_every_construction_is_valid(builder):
    """Euler operators keep the topology valid after *each* step -- wires, laminae and faces
    with spikes included -- and the finished solid is valid geometry too."""
    steps = []
    solid = builder(steps=steps)
    assert all(step["problems"] == [] for step in steps)
    assert all(euler_poincare(step["counts"])[0] == euler_poincare(step["counts"])[1]
               for step in steps)
    assert check(solid) == []


def test_finished_solids_are_outward_oriented():
    """Signed volume by the divergence theorem is positive iff every face is counter-clockwise
    seen from outside. The block with a hole is 3 x 3 x 1 minus 1 x 1 x 1."""
    assert np.isclose(signed_volume(cube(2.0)), 8.0)
    assert np.isclose(signed_volume(tetrahedron()), np.sqrt(3) / 4 * 0.8 / 3)
    assert np.isclose(signed_volume(block_with_hole()), 8.0)
    assert np.isclose(signed_volume(octahedron()), 4 / 3)


@pytest.mark.parametrize("solid, expected", [
    (tetrahedron(), dict(V=4, E=6, F=4, R=0, S=1, G=0)),
    (cube(), dict(V=8, E=12, F=6, R=0, S=1, G=0)),
    (octahedron(), dict(V=6, E=12, F=8, R=0, S=1, G=0)),
    (block_with_hole(), dict(V=16, E=24, F=10, R=2, S=1, G=1)),
    (torus_grid(8, 6), dict(V=48, E=96, F=48, R=0, S=1, G=1)),
])
def test_euler_poincare_on_known_solids(solid, expected):
    counts = solid.counts()
    assert {k: counts[k] for k in expected} == expected
    lhs, rhs = euler_poincare(counts)
    assert lhs == rhs
    assert check(solid) == []


def test_the_genus_one_step_is_kfmrh():
    """In the block-with-hole construction the left side of Euler-Poincare is 2 until the very
    last operator, kfmrh, which drops it to 0 -- the moment the hole goes through."""
    steps = []
    block_with_hole(steps=steps)
    sides = [euler_poincare(step["counts"])[0] for step in steps]
    assert sides[:-1] == [2] * (len(steps) - 1) and sides[-1] == 0
    assert steps[-1]["label"].startswith("kfmrh")


def test_a_ring_is_connected_to_its_shell_only_through_its_face():
    """After kemr cuts the spike, no edge joins the hole's boundary to the rest of the solid; it
    is still one shell, because the two loops belong to the same face."""
    steps = []
    block_with_hole(steps=steps)
    after_kemr = next(step for step in steps if step["label"].startswith("kemr"))
    assert after_kemr["counts"]["R"] == 1 and after_kemr["counts"]["S"] == 1


# --- Each operator: its count row, and its inverse --------------------------------------------------


def test_mev_and_kev():
    solid = cube()
    top = solid.faces[0]
    v = top.outer.he.origin
    before, sig = solid.counts(), signature(solid)
    new = mev(top, v, v.point + [0.2, 0.2, 0.0])
    assert delta(before, solid.counts()) == dict(V=1, E=1, F=0, R=0, S=0, G=0)
    he = find_halfedge(top, new)
    lkev(he, he.twin)
    assert solid.counts() == before and signature(solid) == sig and check(solid) == []


def test_mef_and_kef():
    solid = cube()
    top = solid.faces[0]
    a, _, c, _ = top.outer.vertices()
    before, sig = solid.counts(), signature(solid)
    new_face = mef(top, a, c)  # a diagonal: the top splits into two triangles
    assert delta(before, solid.counts()) == dict(V=0, E=1, F=1, R=0, S=0, G=0)
    assert check(solid) == []
    he = next(h for h in new_face.outer.halfedges() if h.twin.face is top)
    lkef(he.twin, he)
    assert solid.counts() == before and signature(solid) == sig and check(solid) == []


def test_kemr_and_mekr():
    steps = []
    solid = cube(steps=steps)
    top = solid.faces[0]
    corner = top.outer.he.origin
    a = mev(top, corner, (0.3, 0.3, 1.0))
    b = mev(top, a, (0.6, 0.3, 1.0))
    c = mev(top, b, (0.6, 0.6, 1.0))
    mef(top, a, c, v1_to=b)
    before, sig = solid.counts(), signature(solid)
    ring = kemr(top, corner, a)
    assert delta(before, solid.counts()) == dict(V=0, E=-1, F=0, R=1, S=0, G=0)
    assert check(solid) == []
    lmekr(find_halfedge(top, corner), next(h for h in ring.halfedges() if h.origin is a))
    assert solid.counts() == before and signature(solid) == sig and check(solid) == []


def test_kfmrh_and_mfkrh():
    solid = block_with_hole()
    bottom = next(f for f in solid.faces if f.rings and f.outer.he.origin.point[2] == 0.0)
    before, sig = solid.counts(), signature(solid)
    plug = lmfkrh(bottom.rings[0])
    assert delta(before, solid.counts()) == dict(V=0, E=0, F=1, R=-1, S=0, G=-1)
    assert check(solid) == []  # the hole is plugged: a genus-0 solid with a blind pocket
    assert np.isclose(signed_volume(solid), 9.0 - 1.0)
    lkfmrh(bottom, plug)
    assert solid.counts() == before and signature(solid) == sig and check(solid) == []


# --- Queries --------------------------------------------------------------------------------------


def test_cube_queries():
    solid = cube()
    assert all(len(vertex_star(v)) == 3 for v in solid.vertices)
    assert all(len(adjacent_faces(f)) == 4 for f in solid.faces)
    assert all(a is not b for a, b in map(edge_faces, solid.edges))


def test_vertex_star_walks_round_the_vertex_face_by_face():
    """Consecutive half-edges of a star are separated by exactly one face: he.twin and the next
    star half-edge (he.twin.next) lie in the same loop."""
    solid = torus_grid(8, 6)
    for v in solid.vertices:
        star = vertex_star(v)
        assert len(star) == 4
        for he, nxt in zip(star, star[1:] + star[:1]):
            assert nxt is he.twin.next and nxt.loop is he.twin.loop


# --- The check catches corruption ---------------------------------------------------------------


def test_check_catches_a_broken_twin():
    solid = cube()
    edge = solid.edges[3]
    edge.he2 = edge.he1
    assert any("twin" in p for p in check(solid))


def test_check_catches_a_broken_next():
    solid = cube()
    he = solid.faces[0].outer.he
    he.next = he.next.next
    assert any("next/prev" in p for p in check(solid))
    lhs, rhs = euler_poincare(solid.counts())
    assert lhs == rhs  # no count changed: the formula is necessary, not sufficient


def test_check_catches_a_non_manifold_vertex():
    """Two tetrahedra touching at one vertex: every edge has its twin, but the faces around the
    shared vertex form two separate fans, not one disc."""
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    tet = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    other = [[0, 5, 4], [0, 4, 6], [0, 6, 5], [4, 5, 6]]
    solid = from_polygons(points, tet + other)
    assert any("non-manifold" in p for p in check(solid))


def test_from_polygons_rejects_open_and_inconsistent_surfaces():
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    with pytest.raises(ValueError, match="no twin"):
        from_polygons(points, [[0, 2, 1], [0, 1, 3], [0, 3, 2]])  # one face missing
    with pytest.raises(ValueError, match="used twice"):
        from_polygons(points, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 3, 2]])  # one face flipped


# --- Why half-edges: adjacency without search --------------------------------------------------


def test_adjacency_cost_face_lists_versus_half_edges():
    """Faces adjacent to one face. With bare face-vertex lists every other face must be inspected
    for a shared edge, so the work grows with F; with half-edges it is one twin step per edge of
    the face, independent of the size of the model."""
    list_cost, halfedge_cost = [], []
    for n in (4, 8, 16, 32):
        solid = torus_grid(n, n)
        faces = [[v.id for v in f.outer.vertices()] for f in solid.faces]
        target = faces[0]
        target_edges = {frozenset(e) for e in zip(target, target[1:] + target[:1])}
        touched, found = 0, []
        for k, other in enumerate(faces[1:], start=1):
            touched += 1
            if target_edges & {frozenset(e) for e in zip(other, other[1:] + other[:1])}:
                found.append(k)
        list_cost.append(touched)
        steps = len(solid.faces[0].outer.halfedges())
        assert len(adjacent_faces(solid.faces[0])) == len(found) == 4
        halfedge_cost.append(steps)
    assert list_cost == [15, 63, 255, 1023]
    assert halfedge_cost == [4, 4, 4, 4]
