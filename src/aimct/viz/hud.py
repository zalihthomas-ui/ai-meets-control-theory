r"""Shared telemetry overlay for replay and sandbox views.

A monospace read-out pinned to a corner of the axes plus, optionally, a thin
progress bar along the bottom.  The content comes from
:meth:`aimct.viz.artists.SystemArtist.hud_lines`, so every system reports itself
in its own natural units without the animator knowing anything about it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TelemetryHUD"]


class TelemetryHUD:
    def __init__(self, ax, *, loc: str = "upper left", title: str | None = None,
                 progress: bool = False):
        from matplotlib.lines import Line2D

        self.ax = ax
        self.title = title
        x, ha = (0.025, "left") if "left" in loc else (0.975, "right")
        y, va = (0.975, "top") if "upper" in loc else (0.03, "bottom")
        self._text = ax.text(
            x, y, "", transform=ax.transAxes, ha=ha, va=va,
            family="monospace", fontsize=9.5, zorder=20,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#CFCFCF",
                      alpha=0.92),
        )
        self._bar = None
        if progress:
            # both lines live in axes-fraction coordinates
            ax.add_line(Line2D([0.04, 0.96], [0.02, 0.02], transform=ax.transAxes,
                               lw=3, color="#E4E4E4", zorder=19,
                               solid_capstyle="round"))
            self._bar = Line2D([0.04, 0.04], [0.02, 0.02], transform=ax.transAxes,
                               lw=3, color="#333333", zorder=20,
                               solid_capstyle="round")
            ax.add_line(self._bar)

    @property
    def artists(self):
        return [a for a in (self._text, self._bar) if a is not None]

    def update(self, lines, *, frac: float | None = None,
               controller: str | None = None):
        head = []
        if self.title:
            head.append(self.title)
        if controller:
            head.append(f"[ {controller} ]")
        self._text.set_text("\n".join(head + ([""] if head else []) + list(lines)))
        if self._bar is not None and frac is not None:
            self._bar.set_xdata([0.04, 0.04 + 0.92 * float(np.clip(frac, 0, 1))])
