"""Build solids one Euler operator at a time, and watch V - E + F - R = 2(S - G) hold throughout.

Pick a construction and drag "step". Each step applies one Euler operator (or a few of the same
kind) and shows the result; red marks what that step created. The table on the right keeps the
counts:

    V vertices, E edges, F faces, R rings (inner loops of faces), S shells, G genus (holes through)

and the two sides of the Euler-Poincare formula. They stay equal at every step, because each
operator changes the counts in a way that keeps them equal -- including the strange intermediate
states that are not solids at all: a single vertex, a wire, a flat lamina with two faces back to
back, a face with edges sticking out of it.

The block with a hole is the interesting one. Watch for:
  * kemr -- the square cut into the top loses its only edge to the rest of the solid; its
    boundary survives as a *ring*, an inner loop of the top face (R goes up).
  * kfmrh -- the swept square merges into the bottom face as a second ring, and the hole goes
    through (G goes to 1). The left side of the formula drops from 2 to 0 in this single step.

Notes: notes/03_Topology/Euler_Operators.md
Run:   python scripts/03_topology/02_euler_operators.py
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from cadkernel.topology.halfedge import (
    block_with_hole,
    cube,
    euler_poincare,
    tetrahedron,
)
from cadkernel.viz.interactive import add_radiobuttons, add_slider, run
from cadkernel.viz.topology import face_collection

BUILDERS = {
    "tetrahedron": tetrahedron,
    "cube": cube,
    "block with hole": block_with_hole,
}


def build(name):
    steps = []
    BUILDERS[name](steps=steps)
    return steps


def main():
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.0, 0.12, 0.66, 0.85], projection="3d")
    ax_table = fig.add_axes([0.68, 0.35, 0.3, 0.5])
    ax_table.axis("off")
    ax.view_init(elev=28, azim=-60)
    state = {}
    artists = []

    def clear():
        while artists:
            artists.pop().remove()

    def load(name):
        state["steps"] = build(name)
        points = np.vstack([s["vertices"] for s in state["steps"]])
        lo, hi = points.min(axis=0) - 0.2, points.max(axis=0) + 0.2
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        s_step.valmax = len(state["steps"]) - 1
        s_step.ax.set_xlim(0, s_step.valmax)
        s_step.set_val(s_step.valmax)

    def redraw(_=None):
        clear()
        steps = state["steps"]
        k = min(int(round(s_step.val)), len(steps) - 1)
        now = steps[k]
        before = (
            steps[k - 1]
            if k > 0
            else {"faces": [], "edges": [], "vertices": np.zeros((0, 3))}
        )

        n_old = len(before["faces"])
        colors = ["0.7" if i < n_old else "C3" for i in range(len(now["faces"]))]
        if now["faces"]:
            artists.append(
                ax.add_collection3d(face_collection(now["faces"], colors, alpha=0.25))
            )
        old_edges = {tuple(np.round(e, 9).ravel()) for e in before["edges"]}
        old_edges |= {tuple(np.round(e[::-1], 9).ravel()) for e in before["edges"]}
        for e in now["edges"]:
            new = tuple(np.round(e, 9).ravel()) not in old_edges
            artists.extend(
                ax.plot(*e.T, color="C3" if new else "k", lw=3 if new else 1.4)
            )
        old_points = {tuple(np.round(p, 9)) for p in before["vertices"]}
        for p in now["vertices"]:
            new = tuple(np.round(p, 9)) not in old_points
            artists.extend(
                ax.plot(
                    *p[:, None], "o", color="C3" if new else "k", ms=8 if new else 4
                )
            )

        c = now["counts"]
        lhs, rhs = euler_poincare(c)
        rows = "\n".join(
            f"  {key} = {c[key]:3d}" for key in ("V", "E", "F", "R", "S", "G")
        )
        artists.append(
            ax_table.text(
                0.0,
                1.0,
                f"step {k} of {len(steps) - 1}\n{textwrap.fill(now['label'], 42)}\n\n{rows}\n\n"
                f"V - E + F - R = {c['V']} - {c['E']} + {c['F']} - {c['R']} = {lhs}\n"
                f"2 (S - G)     = 2 ({c['S']} - {c['G']})         = {rhs}",
                va="top",
                family="monospace",
                fontsize=10,
            )
        )
        state["sides"] = (lhs, rhs)
        fig.canvas.draw_idle()

    s_step = add_slider(
        fig, [0.1, 0.05, 0.5, 0.03], "step", 0, 1, 0, redraw, valfmt="%d"
    )
    radio = add_radiobuttons(
        fig, [0.70, 0.08, 0.18, 0.16], list(BUILDERS), 2, lambda name: load(name)
    )
    load("block with hole")

    def smoke():
        for index, name in enumerate(BUILDERS):
            radio.set_active(index)
            for k in range(len(state["steps"])):
                s_step.set_val(k)
                lhs, rhs = state["sides"]
                assert lhs == rhs, (name, k)
        assert state["sides"] == (0, 0)  # the finished block with a hole: genus 1
        s_step.set_val(len(state["steps"]) - 2)
        assert state["sides"] == (2, 2)  # one step earlier: still genus 0
        print(
            f"  {sum(len(build(n)) for n in BUILDERS)} steps, Euler-Poincare balanced at each"
        )

    run(fig, smoke)


if __name__ == "__main__":
    main()
