r"""Replay a simulated run as an animation.

``animate`` takes the output of :func:`aimct.simulate.simulate` (or a
:class:`~aimct.benchmarks.tracking.TrackingResult`) plus the system it was run
on, and plays it back: the mechanism moving, the reference it was chasing drawn
alongside, a fading trail, and a telemetry HUD in the system's own units.

    from aimct.simulate import simulate
    from aimct.viz import animate

    tr = simulate(arm, controller, x0=x0, dt=0.01, t_final=6.0)
    rep = animate(tr, arm, ref=joint_target, title="computed torque")
    rep.save("arm.gif")          # or .show() in an interactive session

The point is that this is *one* function for every 2-D system in the library —
the same call animates a pendulum, a cart-pole, a quadrotor or a mobile robot,
because each system carries its own :class:`~aimct.viz.artists.SystemArtist`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artists import get_artist
from .hud import TelemetryHUD

__all__ = ["animate", "Replay"]


class Replay:
    """Handle around a :class:`matplotlib.animation.FuncAnimation`."""

    def __init__(self, anim, fig, ax):
        self.anim, self.fig, self.ax = anim, fig, ax

    def save(self, path, *, dpi: int = 110, fps: int | None = None):
        """Write the animation to ``path`` (``.gif`` or ``.mp4``).

        ``.mp4`` needs ffmpeg on PATH; without it, fall back to a ``.gif`` next
        to the requested path and report where it went.
        """
        from matplotlib.animation import FFMpegWriter, PillowWriter, writers

        path = Path(path)
        fps = fps or int(round(1000.0 / self.anim._interval))
        if path.suffix.lower() == ".mp4" and writers.is_available("ffmpeg"):
            self.anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
        else:
            if path.suffix.lower() != ".gif":
                path = path.with_suffix(".gif")
            self.anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        return path

    def show(self):
        """Open the animation in an interactive Matplotlib window (blocks until
        closed). Returns ``self``."""
        import matplotlib.pyplot as plt

        plt.show()
        return self

    def to_html(self) -> str:
        """The animation as a self-contained JS/HTML string (for embedding).
        Notebooks display it automatically via ``_repr_html_``."""
        return self.anim.to_jshtml()

    def _repr_html_(self):                      # notebook display
        return self.anim.to_jshtml()


# ----------------------------------------------------------------------
def _as_series(traj, controller):
    """Normalise the many accepted inputs to ``(t, X, U)`` arrays."""
    # TrackingResult / ComparisonResult: dict of named Trajectories
    trajs = getattr(traj, "trajectories", None)
    if isinstance(trajs, dict) and trajs:
        if controller is None:
            if len(trajs) == 1:
                controller = next(iter(trajs))
            else:
                raise ValueError(
                    "several controllers in the result; pass controller=<name> "
                    f"({', '.join(trajs)})")
        traj = trajs[controller]
    t = np.asarray(traj.t, float)
    X = np.asarray(traj.x, float)
    U = np.asarray(traj.u, float) if getattr(traj, "u", None) is not None else None
    return t, X, U, controller


def _resample(t, X, U, frame_t):
    """Linear-interp the state, zero-order-hold the input, at ``frame_t``."""
    Xf = np.column_stack([np.interp(frame_t, t, X[:, j]) for j in range(X.shape[1])])
    if U is None:
        return Xf, None
    idx = np.clip(np.searchsorted(t, frame_t, side="right") - 1, 0, len(t) - 1)
    return Xf, np.atleast_2d(U)[idx]


def _ref_at(ref, ts, T):
    """Evaluate a reference spec at frame times -> ``(len(ts), k)`` or ``None``."""
    if ref is None:
        return None
    if hasattr(ref, "pos"):                                   # aimct.trajectories
        return np.array([np.asarray(ref.pos(min(s, getattr(ref, "duration", T))),
                                    float) for s in ts])
    if callable(ref):
        return np.array([np.asarray(ref(s), float).ravel() for s in ts])
    ref = np.asarray(ref, float)
    if ref.ndim == 1:
        return np.tile(ref, (len(ts), 1))
    # array aligned to the *original* time grid -> interp column-wise
    if ref.shape[0] != len(ts):
        raise ValueError("ref array length does not match the number of frames; "
                         "pass an aimct.trajectories.Trajectory or a callable")
    return ref


def animate(traj, system, *, ref=None, target=None, path=None, controller=None,
            trail=True, fps: int = 30, speed: float = 1.0, title: str | None = None,
            hud: bool = True, figsize=(7.2, 7.2), accent: str | None = None,
            interval_ms: int | None = None) -> Replay:
    """Animate ``traj`` (a run on ``system``).

    Parameters
    ----------
    traj : a :class:`aimct.simulate.Trajectory`, or a
        :class:`~aimct.benchmarks.tracking.TrackingResult` (then ``controller``
        selects which run and ``ref`` defaults to its trajectory).
    system : the model the run used — supplies the artist.
    ref : reference to draw + score against — an ``aimct.trajectories``
        object, a callable ``t -> pos``, a length-``n`` vector, or an array
        with one row per frame.
    target : a fixed goal marker (arm tip set-point, hover point) if there is no
        moving ``ref``.
    path : an ``(N, 2)`` polyline to trace under a robot / arm.
    trail : draw the fading breadcrumb trail.
    fps, speed : playback frame rate and time multiplier (``speed=2`` → 2×).
    """
    import matplotlib
    if matplotlib.get_backend().lower() == "agg":
        pass
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from ..plot_style import set_aimct_style

    t, X, U, controller = _as_series(traj, controller)
    T = float(t[-1])
    dt_frame = max(speed / fps, 1e-6)
    frame_t = np.arange(0.0, T + 1e-9, dt_frame)
    Xf, Uf = _resample(t, X, U, frame_t)

    # a TrackingResult carries its own reference
    if ref is None:
        ref = getattr(traj, "trajectory", None)
    R = _ref_at(ref, frame_t, T)

    art = get_artist(system, accent=accent)
    set_aimct_style()
    fig, ax = plt.subplots(figsize=figsize)
    (xlim, ylim) = art.bounds(X)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if art.aspect_equal:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    head = title or f"{art.label}"
    if controller:
        head = f"{head}  —  {controller}"
    ax.set_title(head, fontweight="bold")

    art.build(ax)
    overlay = TelemetryHUD(ax, title=art.label if hud else None,
                           progress=hud) if hud else None

    pos_hist: list = []

    def _pos(xrow):
        return np.asarray(art.position(xrow), float)

    def frame(i):
        x_i = Xf[i]
        u_i = Uf[i] if Uf is not None else None
        aux = {}
        if R is not None:
            aux["ref"] = R[i][:2] if R[i].ndim else R[i]
        if target is not None:
            aux["target"] = np.asarray(target, float)
        if path is not None:
            aux["path"] = np.asarray(path, float)
        if trail:
            pos_hist.append(_pos(x_i))
            aux["trail"] = np.array(pos_hist[-240:])
        art.draw(x_i, u_i, float(frame_t[i]), aux)
        if overlay is not None:
            overlay.update(art.hud_lines(x_i, u_i, float(frame_t[i]), aux),
                           frac=i / max(len(frame_t) - 1, 1),
                           controller=controller)
        return art._artists + (overlay.artists if overlay else [])

    interval = interval_ms or int(round(1000.0 / fps))
    anim = FuncAnimation(fig, frame, frames=len(frame_t), interval=interval,
                         blit=False, cache_frame_data=False)
    fig.tight_layout()
    return Replay(anim, fig, ax)
