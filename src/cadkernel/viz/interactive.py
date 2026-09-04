"""Reusable interactive-plot machinery: draggable control points, sliders, and a run helper.

Scripts in this repo are meant to be *played with* -- drag a control point and watch the curve
follow -- rather than to produce images. Every later module (B-splines, NURBS weights, surfaces,
sweeps) needs the same widget plumbing, so it lives here once.

Headless verification: set CADK_SMOKE=1 and a script builds its figure, runs its smoke hook
(which exercises the drag/slider callbacks for real) and exits without opening a window. That is
how these scripts get checked without a display; the actual visual check is running them
normally.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import CheckButtons, Slider


def in_smoke_mode() -> bool:
    return bool(os.environ.get("CADK_SMOKE"))


class DraggableControlPoints:
    """Control points the user can grab and drag, with a callback fired on every move.

    Parameters
    ----------
    ax : the axes to draw on
    points : (N, 2) initial control point positions
    on_change : callable(points) -> None, invoked after every drag update
    pick_radius : how close (in screen pixels) a click must be to grab a point
    """

    def __init__(self, ax, points, on_change=None, pick_radius: float = 12.0,
                 show_polygon: bool = True, color: str = "0.35"):
        self.ax = ax
        self.points = np.array(points, dtype=float)
        self.on_change = on_change
        self.pick_radius = pick_radius
        self._dragging: int | None = None

        style = "o--" if show_polygon else "o"
        (self.artist,) = ax.plot(self.points[:, 0], self.points[:, 1], style,
                                  color=color, lw=1, ms=7, mfc="white", mew=1.6,
                                  label="control points", zorder=5)

        canvas = ax.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)

    # -- event handling --------------------------------------------------------------

    def _nearest_within_radius(self, event) -> int | None:
        """Index of the control point nearest the cursor, if inside pick_radius pixels.
        Distance is measured in display space so picking feels the same at any zoom level."""
        if event.x is None or event.y is None:
            return None
        display = self.ax.transData.transform(self.points)
        distances = np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)
        index = int(np.argmin(distances))
        return index if distances[index] <= self.pick_radius else None

    def _on_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self._dragging = self._nearest_within_radius(event)

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.points[self._dragging] = (event.xdata, event.ydata)
        self._refresh()

    def _on_release(self, event):
        self._dragging = None

    def _refresh(self):
        self.artist.set_data(self.points[:, 0], self.points[:, 1])
        if self.on_change is not None:
            self.on_change(self.points)
        self.ax.figure.canvas.draw_idle()

    # -- programmatic control (used by the smoke hooks) -------------------------------

    def set_point(self, index: int, xy) -> None:
        """Move a control point directly, firing the same callback a drag would."""
        self.points[index] = xy
        self._refresh()

    def simulate_drag(self, index: int, to_xy) -> None:
        """Synthesize a real press/motion/release sequence, so smoke runs exercise the actual
        picking and dragging handlers rather than bypassing them."""
        canvas = self.ax.figure.canvas
        start = self.ax.transData.transform(self.points[index])
        end = self.ax.transData.transform(np.asarray(to_xy, dtype=float))
        for name, (x, y) in (("button_press_event", start),
                              ("motion_notify_event", end),
                              ("button_release_event", end)):
            canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))


def add_slider(fig, rect, label, valmin, valmax, valinit, on_change=None, valfmt=None):
    """Place a slider on the figure. `rect` is [left, bottom, width, height] in figure coords."""
    ax = fig.add_axes(rect)
    kwargs = {"valfmt": valfmt} if valfmt else {}
    slider = Slider(ax, label, valmin, valmax, valinit=valinit, **kwargs)
    if on_change is not None:
        slider.on_changed(on_change)
    return slider


def add_checkbuttons(fig, rect, labels, states, on_change=None):
    ax = fig.add_axes(rect)
    buttons = CheckButtons(ax, labels, states)
    if on_change is not None:
        buttons.on_clicked(on_change)
    return buttons


def run(fig, smoke_hook=None) -> None:
    """Show the figure interactively, or -- under CADK_SMOKE=1 -- run the smoke hook and exit.

    Call this instead of plt.show() at the end of every script.
    """
    if in_smoke_mode():
        if smoke_hook is not None:
            smoke_hook()
        fig.canvas.draw()
        plt.close(fig)
        print(f"smoke ok: {os.path.basename(__import__('sys').argv[0])}")
    else:
        plt.show()
