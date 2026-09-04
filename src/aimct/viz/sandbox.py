r"""Real-time interactive sandbox — the live counterpart of :func:`animate`.

A :class:`Sandbox` runs a system forward in wall-clock time while you perturb it:
slider-driven steady disturbances, hot-key impulses, and a controller you can
switch on the fly.  It reuses the system's :class:`~aimct.viz.artists.
SystemArtist` for drawing and the shared :class:`~aimct.viz.hud.TelemetryHUD`
for the read-out, so a new sandbox is just *(system, controllers, disturbance)*.

    from aimct.viz import Sandbox, Disturbance

    box = Sandbox(arm, {"PD+grav": pd, "adaptive": adaptive},
                  x0=x0, target=np.array([0.4, 0.2]),
                  disturbance=Disturbance(
                      sliders=[("payload [kg]", 0.0, 0.5, 0.0)],
                      hotkeys=[("p", "poke", lambda s: s.kick([0, 0, 3.0, -3.0]))],
                      on_slider=lambda s, name, v: setattr(s.system, "payload", v)))
    box.run()                     # opens the window
    box.headless(steps=500)       # or: no GUI, print steady-state error
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from ..simulate import rk4_step
from .artists import get_artist
from .hud import TelemetryHUD

__all__ = ["Sandbox", "Disturbance"]


class Disturbance:
    """What a sandbox lets you perturb.

    Parameters
    ----------
    sliders : sequence of ``(name, lo, hi, init)`` — continuous knobs whose
        current values are handed to ``xdot_extra`` and to ``on_slider``.
    hotkeys : sequence of ``(key, label, fn)`` — ``fn(sandbox)`` runs on that
        key press (e.g. ``sandbox.kick(...)`` for an impulse).
    xdot_extra : ``f(t, x, u, knobs) -> (n_states,)`` added to the state
        derivative every physics sub-step (a steady external force/disturbance).
    on_slider : ``f(sandbox, name, value)`` when a slider moves (e.g. write a
        system attribute).  Also called once at start-up with the init values.
    help_text : one-line hint shown under the plot.
    """

    def __init__(self, *, sliders: Sequence = (), hotkeys: Sequence = (),
                 xdot_extra: Callable | None = None,
                 on_slider: Callable | None = None, help_text: str = ""):
        self.sliders = list(sliders)
        self.hotkeys = list(hotkeys)
        self.xdot_extra = xdot_extra
        self.on_slider = on_slider
        self.help_text = help_text


class Sandbox:
    """Drive a system in real time under a switchable controller + disturbances.

    Parameters
    ----------
    system : an :mod:`aimct.systems` model with a registered artist.
    controllers : ``{label: controller}`` — each needs ``update(x, dt)`` and
        ``reset()``.  Switch with number keys ``1..N`` or the radio buttons.
    x0 : initial state.
    target : a fixed goal (drawn + scored).  A plain ``np.ndarray`` is mutated
        in place by :meth:`nudge_target`, so controller closures can capture it.
    ref : a moving reference (an ``aimct.trajectories`` object); overrides
        ``target`` for scoring when present.
    dt : control period.  ``substeps`` RK4 steps of ``dt/substeps`` per tick.
    disturbance : a :class:`Disturbance` (or ``None``).
    path : an ``(N, 2)`` polyline to draw under the system.
    """

    def __init__(self, system, controllers: Mapping[str, object], *, x0,
                 target=None, ref=None, dt: float = 0.02, substeps: int = 6,
                 disturbance: Disturbance | None = None, title: str | None = None,
                 path=None, accent: str | None = None, on_click=None,
                 on_step=None, aux_extra=None):
        self.system = system
        self.controllers = dict(controllers)
        self.names = list(self.controllers)
        self.active = self.names[0]
        self.x0 = np.asarray(x0, float).copy()
        self.x = self.x0.copy()
        self.target = None if target is None else np.asarray(target, float)
        self._target0 = None if self.target is None else self.target.copy()
        self.ref = ref
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.dist = disturbance or Disturbance()
        self.path = None if path is None else np.asarray(path, float)
        self.title = title
        self.accent = accent
        self.on_click = on_click
        self.on_step = on_step        # f(sandbox) called each tick before step()
        self.aux_extra = aux_extra    # f(sandbox) -> dict merged into the draw aux
        self.t = 0.0
        self.knobs = {name: init for (name, _, _, init) in self.dist.sliders}
        self._trail: list = []
        for name, v in self.knobs.items():
            if self.dist.on_slider:
                self.dist.on_slider(self, name, v)
        self.controllers[self.active].reset()

    # -- disturbance API used by hotkey callbacks -------------------------
    def kick(self, delta) -> None:
        """Add an instantaneous impulse to the state (a poke / shove / gust)."""
        self.x = self.x + np.asarray(delta, float)

    def nudge_target(self, delta) -> None:
        if self.target is not None:
            self.target += np.asarray(delta, float)

    def set_target(self, xy) -> None:
        if self.target is not None:
            self.target[:len(xy)] = np.asarray(xy, float)[:len(self.target)]

    def set_controller(self, name: str) -> None:
        if name in self.controllers and name != self.active:
            self.active = name
            self.controllers[name].reset()

    def reset(self) -> None:
        self.x = self.x0.copy()
        self.t = 0.0
        self._trail.clear()
        if self._target0 is not None:
            self.target[...] = self._target0
        self.controllers[self.active].reset()

    # -- one control tick ------------------------------------------------
    def _ref_now(self):
        if self.ref is not None and hasattr(self.ref, "pos"):
            return np.asarray(self.ref.pos(self.t), float)
        return self.target

    def step(self):
        if self.on_step is not None:
            self.on_step(self)
        ctrl = self.controllers[self.active]
        u = np.atleast_1d(np.asarray(ctrl.update(self.x, self.dt), float))
        extra = self.dist.xdot_extra
        h = self.dt / self.substeps

        def f(t, x, uu):
            xd = self.system.dynamics(t, x, uu)
            if extra is not None:
                xd = xd + np.asarray(extra(t, x, uu, self.knobs), float)
            return xd

        for _ in range(self.substeps):
            self.x = rk4_step(f, self.t, self.x, u, h)
            self.t += h
        return u

    # -- headless -------------------------------------------------------
    def headless(self, steps: int = 500, *, quiet: bool = False):
        """Run without a GUI; return the mean tip/pose error over the last 25 %."""
        art = get_artist(self.system)
        errs = []
        for k in range(steps):
            u = self.step()
            ref = self._ref_now()
            if ref is not None:
                errs.append(float(np.linalg.norm(
                    art.position(self.x) - np.asarray(ref, float).ravel()[:2])))
        tail = errs[int(0.75 * len(errs)):] if errs else [float("nan")]
        res = {"controller": self.active, "steps": steps,
               "final_state": self.x.copy(),
               "mean_err_tail_mm": float(np.mean(tail) * 1e3)}
        if not quiet:
            print(f"[{self.active}] {steps} steps  "
                  f"tail error {res['mean_err_tail_mm']:.1f} mm  "
                  f"state {np.round(self.x, 3)}")
        return res

    # -- interactive --------------------------------------------------------
    def run(self):                                       # pragma: no cover (GUI)
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import RadioButtons, Slider

        from ..plot_style import set_aimct_style

        set_aimct_style()
        art = get_artist(self.system, accent=self.accent)
        fig = plt.figure(figsize=(9.6, 8.2))
        ax = fig.add_axes([0.06, 0.30, 0.66, 0.66])
        (xlim, ylim) = art.bounds(np.atleast_2d(self.x0))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        if art.aspect_equal:
            ax.set_aspect("equal", adjustable="box")
        ax.set_title(self.title or art.label, fontweight="bold")
        art.build(ax)
        hud = TelemetryHUD(ax, title=art.label)

        # controller radio
        rax = fig.add_axes([0.76, 0.55, 0.21, 0.30])
        rax.set_title("controller", fontsize=9)
        radio = RadioButtons(rax, self.names, active=0)
        radio.on_clicked(self.set_controller)

        # sliders
        sliders = []
        for i, (name, lo, hi, init) in enumerate(self.dist.sliders):
            sax = fig.add_axes([0.12, 0.20 - 0.055 * i, 0.52, 0.03])
            s = Slider(sax, name, lo, hi, valinit=init)

            def _cb(v, nm=name):
                self.knobs[nm] = v
                if self.dist.on_slider:
                    self.dist.on_slider(self, nm, v)

            s.on_changed(_cb)
            sliders.append(s)
        self._sliders = sliders                          # keep refs alive

        hint = self.dist.help_text or "keys: r reset"
        keymap = {k: fn for (k, _lbl, fn) in self.dist.hotkeys}
        legend = "   ".join(f"{k}:{lbl}" for (k, lbl, _f) in self.dist.hotkeys)
        fig.text(0.06, 0.035, (legend + "   " if legend else "")
                 + "1..%d:controller   r:reset   " % len(self.names) + hint,
                 fontsize=8.5, family="monospace", color="#555555")

        def on_key(ev):
            if ev.key == "r":
                self.reset()
            elif ev.key in keymap:
                keymap[ev.key](self)
            elif ev.key and ev.key.isdigit():
                j = int(ev.key) - 1
                if 0 <= j < len(self.names):
                    self.set_controller(self.names[j])
                    radio.set_active(j)

        fig.canvas.mpl_connect("key_press_event", on_key)

        if self.on_click is not None:
            def _click(ev):
                if ev.inaxes is ax and ev.xdata is not None:
                    self.on_click(self, float(ev.xdata), float(ev.ydata))
            fig.canvas.mpl_connect("button_press_event", _click)

        def frame(_):
            u = self.step()
            ref = self._ref_now()
            self._trail.append(art.position(self.x))
            aux = {"trail": np.array(self._trail[-240:])}
            if ref is not None:
                aux["ref"] = np.asarray(ref, float).ravel()
            if self.target is not None:
                aux["target"] = self.target
            if self.path is not None:
                aux["path"] = self.path
            if self.aux_extra is not None:
                aux.update(self.aux_extra(self) or {})
            art.draw(self.x, u, self.t, aux)
            hud.update(art.hud_lines(self.x, u, self.t, aux),
                       controller=self.active)
            return art._artists

        self._anim = FuncAnimation(fig, frame, interval=33, blit=False,
                                   cache_frame_data=False)
        plt.show()
        return self
