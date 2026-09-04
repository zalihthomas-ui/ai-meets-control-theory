"""``aimct.dev`` — authoring-time tools. Currently one: a design preview for a
new :class:`~aimct.systems.base.DynamicalSystem` (see :mod:`aimct.dev.preview`).
Kept separate from :mod:`aimct.viz` (the runtime replay / sandbox story)."""

from .preview import DesignReport, build_report, load_system, preview_once, render, watch

__all__ = [
    "DesignReport",
    "build_report",
    "render",
    "load_system",
    "preview_once",
    "watch",
]
