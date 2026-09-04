r"""``aimct.viz`` — one visual language for every system in the library.

Every model in :mod:`aimct.systems` shares one interface, so every benchmark
shares one harness.  This package completes the symmetry for *pictures*:

* :func:`animate` replays any simulated run (a :class:`aimct.simulate.Trajectory`
  or a :class:`~aimct.benchmarks.tracking.TrackingResult`) as an animation, with
  the reference drawn alongside and a telemetry HUD in the system's own units.
* :class:`Sandbox` is the real-time, interactive counterpart: hold a set-point
  while *you* inject disturbances from sliders and keys and switch controllers
  on the fly.

Both are driven by a single :class:`~aimct.viz.artists.SystemArtist` contract,
so teaching the library to draw a new 2-D system is one small class plus a
:func:`~aimct.viz.artists.register_artist` call.

    from aimct.viz import animate
    animate(trajectory, system, ref=reference).save("run.gif")
"""

from __future__ import annotations

from .artists import (
    SystemArtist,
    get_artist,
    has_artist,
    register_artist,
)
from .replay import Replay, animate
from .sandbox import Disturbance, Sandbox

__all__ = [
    "animate",
    "Replay",
    "Sandbox",
    "Disturbance",
    "SystemArtist",
    "get_artist",
    "has_artist",
    "register_artist",
]
