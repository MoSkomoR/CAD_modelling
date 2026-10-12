"""Walk a half-edge solid one pointer at a time.

Every edge of the solid is two half-edges, one in each face it bounds, running in opposite
directions; each is drawn as a small arrow pushed into its own face, so the two halves of an edge
sit on opposite sides of it. The red arrow is "where you are". Move with the buttons:

  * **next / prev** -- the following / preceding half-edge of the same loop: you walk round a face.
  * **twin** -- the other half of the same edge: you cross into the neighbouring face.
  * **twin.next** -- cross, then step on: you turn round the start vertex to its next edge.

Those four pointers are the whole navigation vocabulary. Tick "face loop" to see the loop the red
half-edge belongs to (on the block, the top and bottom faces also have an inner loop -- the rim
of the hole -- which runs the other way round), and "vertex star" for every half-edge leaving its
start vertex: the edges you reach by pressing twin.next repeatedly.

Notes: notes/03_Topology/Half_Edge.md
Run:   python scripts/03_topology/01_halfedge_navigation.py
"""

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.topology.halfedge import block_with_hole, cube, loop_points, vertex_star
from cadkernel.viz.interactive import (
    add_button,
    add_checkbuttons,
    add_radiobuttons,
    press,
    run,
)
from cadkernel.viz.topology import face_collection, halfedge_arrow

SOLIDS = {"cube": lambda: cube(2.0), "block with hole": block_with_hole}


def main():
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.0, 0.12, 0.75, 0.85], projection="3d")
    ax.view_init(elev=32, azim=-62)
    state = {}
    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def load(name):
        solid = SOLIDS[name]()
        state["solid"] = solid
        state["he"] = solid.faces[0].outer.he
        state["path"] = []
        lo = np.min([v.point for v in solid.vertices], axis=0)
        hi = np.max([v.point for v in solid.vertices], axis=0)
        ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
        ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
        ax.set_zlim(lo[2] - 0.3, hi[2] + 0.3)
        ax.set_box_aspect(hi - lo + 0.6)
        redraw()

    def arrows(halfedges, **style):
        if not halfedges:
            return
        starts, vectors = zip(*(halfedge_arrow(he) for he in halfedges))
        starts, vectors = np.array(starts), np.array(vectors)
        artists.append(
            ax.quiver(*starts.T, *vectors.T, arrow_length_ratio=0.25, **style)
        )

    def redraw(_=None):
        clear()
        solid, he = state["solid"], state["he"]
        show_all, show_loop, show_star = check.get_status()
        faces = [
            [loop_points(face.outer)] + [loop_points(r) for r in face.rings]
            for face in solid.faces
        ]
        colors = ["C1" if face is he.face else "0.75" for face in solid.faces]
        artists.append(ax.add_collection3d(face_collection(faces, colors, alpha=0.22)))
        for e in solid.edges:
            seg = np.array([e.he1.origin.point, e.he2.origin.point])
            artists.extend(ax.plot(*seg.T, color="0.3", lw=1))
        if show_all:
            arrows(solid.halfedges(), color="0.55", lw=0.8)
        if show_loop:
            arrows(he.loop.halfedges(), color="C0", lw=1.8)
        if show_star:
            arrows(vertex_star(he.origin), color="C2", lw=1.8)
        arrows([he], color="C3", lw=3.5)
        artists.extend(ax.plot(*he.origin.point[:, None], "o", color="C3", ms=9))

        kind = "outer loop" if he.loop is he.face.outer else "inner loop (ring)"
        v = lambda vertex: f"v{solid.vertices.index(vertex)}"
        f = lambda face: f"f{solid.faces.index(face)}"
        fig.suptitle(
            f"half-edge {v(he.origin)} -> {v(he.target)}, in face {f(he.face)} ({kind})\n"
            f"its twin is in face {f(he.twin.face)};  {v(he.origin)} has "
            f"{len(vertex_star(he.origin))} edges;  steps taken: {' '.join(state['path'][-12:])}",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def step(name, move):
        def on_click(_=None):
            state["he"] = move(state["he"])
            state["path"].append(name)
            redraw()

        return on_click

    buttons = {
        "next": add_button(
            fig, [0.06, 0.03, 0.1, 0.05], "next", step("next", lambda h: h.next)
        ),
        "prev": add_button(
            fig, [0.18, 0.03, 0.1, 0.05], "prev", step("prev", lambda h: h.prev)
        ),
        "twin": add_button(
            fig, [0.30, 0.03, 0.1, 0.05], "twin", step("twin", lambda h: h.twin)
        ),
        "twin.next": add_button(
            fig,
            [0.42, 0.03, 0.12, 0.05],
            "twin.next",
            step("twin.next", lambda h: h.twin.next),
        ),
    }
    check = add_checkbuttons(
        fig,
        [0.78, 0.55, 0.2, 0.15],
        ["all half-edges", "face loop", "vertex star"],
        [True, True, False],
        redraw,
    )
    radio = add_radiobuttons(fig, [0.78, 0.35, 0.2, 0.12], list(SOLIDS), 0, load)
    load("cube")

    def smoke():
        start = state["he"]
        for _ in range(4):  # a cube face is a 4-cycle
            press(buttons["next"])
        assert state["he"] is start
        press(buttons["twin"])
        press(buttons["twin"])
        assert state["he"] is start
        for _ in range(3):  # three edges meet at a cube vertex
            press(buttons["twin.next"])
        assert state["he"] is start
        press(buttons["prev"])
        assert state["he"] is start.prev
        for index in range(3):
            check.set_active(index)
        radio.set_active(1)
        ring_he = next(f for f in state["solid"].faces if f.rings).rings[0].he
        state["he"] = ring_he
        press(buttons["next"])  # walking a ring works the same way
        assert state["he"].loop is ring_he.loop
        print("  cube: next x4, twin x2, twin.next x3 all return to the start")

    run(fig, smoke)


if __name__ == "__main__":
    main()
