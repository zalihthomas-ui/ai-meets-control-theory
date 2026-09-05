r"""System artists — how to *draw* a dynamical system's state.

Every model in :mod:`aimct.systems` shares one interface, so every benchmark
shares one harness (:func:`aimct.simulate.simulate`, :func:`~aimct.benchmarks.
compare`, :func:`~aimct.benchmarks.track_trajectory`).  This module completes the
symmetry for pictures: a :class:`SystemArtist` says once how to render a state,
and both the replay animator (:mod:`aimct.viz.replay`) and the interactive
sandbox (:mod:`aimct.viz.sandbox`) consume it unchanged.

Adding a view for a new 2-D system is one class:

    class MyArtist(SystemArtist):
        def bounds(self, states=None): ...
        def build(self, ax): ...
        def draw(self, x, u=None, t=0.0, aux=None): ...

    register_artist(MySystem, MyArtist)

``aux`` is an optional per-frame dict the caller fills in — ``ref`` (reference
state or position), ``target`` (a commanded goal, e.g. an arm tip set-point),
``path`` (an ``(N, 2)`` polyline to trace), ``trail`` (past positions),
``controller`` (name) — an artist uses what it recognises and ignores the rest.
"""

from __future__ import annotations

import numpy as np

from ..plot_style import PALETTE

__all__ = [
    "SystemArtist",
    "PendulumArtist",
    "CartPoleArtist",
    "TwoLinkArmArtist",
    "DiffDriveArtist",
    "PlanarQuadrotorArtist",
    "register_artist",
    "get_artist",
    "has_artist",
]

_REGISTRY: dict[type, type] = {}

_INK = "#2B2B2B"        # near-black for structure
_GHOST = "#9AA0A6"      # muted grey for reference / envelope


def register_artist(system_cls: type, artist_cls: type) -> None:
    """Associate a :class:`SystemArtist` subclass with a system class."""
    _REGISTRY[system_cls] = artist_cls


def has_artist(system) -> bool:
    cls = system if isinstance(system, type) else type(system)
    return any(issubclass(cls, k) for k in _REGISTRY)


def get_artist(system, **kw) -> "SystemArtist":
    """Instantiate the artist registered for ``system`` (an instance).

    Raises ``LookupError`` listing the drawable systems if there is none.
    """
    cls = type(system)
    for k, v in _REGISTRY.items():
        if issubclass(cls, k):
            return v(system, **kw)
    names = ", ".join(sorted(k.__name__ for k in _REGISTRY))
    raise LookupError(
        f"no aimct.viz artist for {cls.__name__}; drawable systems: {names}"
    )


# ======================================================================
# base
# ======================================================================
class SystemArtist:
    """Render one system's state onto a Matplotlib ``Axes``.

    Subclasses implement :meth:`bounds`, :meth:`build` and :meth:`draw`.
    :meth:`hud_lines` supplies the telemetry read-out and may be overridden;
    :meth:`position` returns the point that traces the breadcrumb trail.
    """

    aspect_equal: bool = True
    label: str = "system"

    def __init__(self, system, *, accent: str | None = None):
        self.system = system
        self.accent = accent or PALETTE["lqr"]
        self._artists: list = []

    # -- geometry / lifecycle -------------------------------------------
    def bounds(self, states: np.ndarray | None = None):
        """World extent as ``((xmin, xmax), (ymin, ymax))``.  ``states`` is the
        whole run's ``(N, n_states)`` array when available, so the frame can be
        sized to the actual motion."""
        raise NotImplementedError

    def build(self, ax) -> list:
        """Create the persistent Matplotlib artists on ``ax`` once; return them
        (the animator keeps the list to redraw each frame)."""
        raise NotImplementedError

    def draw(self, x, u=None, t: float = 0.0, aux: dict | None = None) -> None:
        """Move the artists built by :meth:`build` to match state ``x`` (input
        ``u``, time ``t``, optional per-frame ``aux`` dict)."""
        raise NotImplementedError

    def position(self, x) -> np.ndarray:
        """The ``(x, y)`` point that traces the trail. Default: first two state
        components (correct for the robots and the quadrotor)."""
        return np.asarray(x, float)[:2]

    # -- telemetry ----------------------------------------------------------
    def hud_lines(self, x, u=None, t: float = 0.0, aux: dict | None = None):
        """Lines for the telemetry overlay, in the system's own units.  The
        default is time + raw state + ``|u|``; subclasses override with
        labelled, unit-bearing rows."""
        x = np.asarray(x, float)
        rows = [f"t   = {t:6.2f} s",
                "x   = [" + ", ".join(f"{v:+.3f}" for v in x) + "]"]
        if u is not None:
            u = np.atleast_1d(np.asarray(u, float))
            rows.append("u   = [" + ", ".join(f"{v:+.3f}" for v in u) + "]")
        return rows

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _rot(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])

    def _remember(self, *artists):
        self._artists.extend(a for a in artists if a is not None)
        return artists

    def _new_trail(self, ax, color=None):
        from matplotlib.lines import Line2D

        ln = Line2D([], [], lw=2.0, color=color or self.accent, alpha=0.4,
                    zorder=2, solid_capstyle="round")
        ax.add_line(ln)
        return ln

    @staticmethod
    def _set_trail(line, trail):
        if trail is not None and len(trail):
            trail = np.asarray(trail, float)
            line.set_data(trail[:, 0], trail[:, 1])


# ======================================================================
# pendulum  -  state [theta, omega],  theta = 0 hangs straight down
# ======================================================================
class PendulumArtist(SystemArtist):
    label = "pendulum"

    def bounds(self, states=None):
        r = 1.3 * self.system.L
        return (-r, r), (-r, r)

    def build(self, ax):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle

        L = self.system.L
        self.pivot_mount = Line2D([-0.16 * L, 0.16 * L], [0, 0], lw=5,
                                  color=_INK, solid_capstyle="round", zorder=3)
        self.rod = Line2D([0, 0], [0, -L], lw=5, color=self.accent,
                          solid_capstyle="round", zorder=5)
        self.bob = Circle((0, -L), 0.10 * L, color=self.accent, ec=_INK,
                          lw=1.2, zorder=6)
        self.pivot = Circle((0, 0), 0.035 * L, color=_INK, zorder=7)
        self.torque = Line2D([], [], lw=3, color=PALETTE["rl"], alpha=0.9,
                             zorder=4, solid_capstyle="round")
        self.trail = self._new_trail(ax)
        for ln in (self.pivot_mount, self.rod, self.torque):
            ax.add_line(ln)
        ax.add_patch(self.bob)
        ax.add_patch(self.pivot)
        return self._remember(self.trail, self.pivot_mount, self.rod,
                              self.torque, self.bob, self.pivot)

    def position(self, x):
        L = self.system.L
        th = float(np.asarray(x, float)[0])
        return np.array([L * np.sin(th), -L * np.cos(th)])

    def draw(self, x, u=None, t=0.0, aux=None):
        L = self.system.L
        th = float(np.asarray(x, float)[0])
        tip = self.position(x)
        self.rod.set_data([0, tip[0]], [0, tip[1]])
        self.bob.center = tuple(tip)
        if u is not None:
            tau = float(np.clip(np.atleast_1d(u)[0], -6, 6))
            a = np.linspace(0, tau * 0.22, 18)
            rr = 0.40 * L
            self.torque.set_data(rr * np.sin(a), -rr * np.cos(a) - 0.0)
        self._set_trail(self.trail, (aux or {}).get("trail"))

    def hud_lines(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        dev = np.degrees((x[0] + np.pi) % (2 * np.pi) - np.pi)
        rows = [f"t      = {t:6.2f} s",
                f"angle  = {dev:+7.1f} deg from down",
                f"rate   = {np.degrees(x[1]):+7.1f} deg/s"]
        if u is not None:
            rows.append(f"torque = {float(np.atleast_1d(u)[0]):+7.3f} N.m")
        return rows


# ======================================================================
# cart-pole  -  state [x, xdot, theta, thetadot],  theta = 0 upright
# ======================================================================
class CartPoleArtist(SystemArtist):
    label = "cart-pole"

    def bounds(self, states=None):
        pole = 2.0 * self.system.l
        xr = (np.abs(states[:, 0]).max() + 0.7) if (states is not None and len(states)) else 2.4
        span = max(xr, pole + 0.5)
        return (-span, span), (-0.55, pole + 0.65)

    def build(self, ax):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyBboxPatch

        self._cw, self._ch = 0.44, 0.24
        self.rail = Line2D([-1e3, 1e3], [-self._ch / 2, -self._ch / 2], lw=2.0,
                           color=_GHOST, zorder=1)
        self.cart = FancyBboxPatch((-self._cw / 2, -self._ch / 2), self._cw,
                                   self._ch, boxstyle="round,pad=0,rounding_size=0.05",
                                   facecolor=_INK, edgecolor="black", zorder=4)
        self.wheelL = Circle((-self._cw * 0.3, -self._ch / 2), 0.045,
                             color="#555", zorder=5)
        self.wheelR = Circle((self._cw * 0.3, -self._ch / 2), 0.045,
                             color="#555", zorder=5)
        self.pole = Line2D([], [], lw=6, color=self.accent,
                           solid_capstyle="round", zorder=6)
        self.hinge = Circle((0, 0), 0.028, color="#FFFFFF", ec=_INK, zorder=7)
        self.tipmass = Circle((0, 0), 0.05, color=self.accent, ec=_INK,
                              lw=1.0, zorder=7)
        self.force = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                                 color=PALETTE["saturation"]),
                                 zorder=6)
        self.trail = self._new_trail(ax)
        ax.add_line(self.rail)
        ax.add_line(self.pole)
        for pch in (self.cart, self.wheelL, self.wheelR, self.hinge, self.tipmass):
            ax.add_patch(pch)
        return self._remember(self.trail, self.rail, self.cart, self.wheelL,
                              self.wheelR, self.pole, self.hinge, self.tipmass)

    def position(self, x):
        return np.array([float(np.asarray(x, float)[0]), 0.0])

    def draw(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        px, th = x[0], x[2]
        pole = 2.0 * self.system.l
        self.cart.set_x(px - self._cw / 2)
        self.wheelL.center = (px - self._cw * 0.3, -self._ch / 2)
        self.wheelR.center = (px + self._cw * 0.3, -self._ch / 2)
        top = (px, self._ch / 2)
        tip = (px + pole * np.sin(th), self._ch / 2 + pole * np.cos(th))
        self.pole.set_data([top[0], tip[0]], [top[1], tip[1]])
        self.hinge.center = top
        self.tipmass.center = tip
        if u is not None:
            f = float(np.atleast_1d(u)[0])
            L = float(np.clip(f * 0.05, -1.1, 1.1))
            y = -self._ch / 2 - 0.16
            if abs(L) < 1e-3:
                self.force.set_position((px, y)); self.force.xy = (px, y)
            else:
                self.force.set_position((px, y)); self.force.xy = (px + L, y)
        self._set_trail(self.trail, (aux or {}).get("trail"))

    def hud_lines(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        rows = [f"t     = {t:6.2f} s",
                f"cart  = {x[0]:+7.3f} m   ({x[1]:+.2f} m/s)",
                f"pole  = {np.degrees(x[2]):+7.1f} deg  ({np.degrees(x[3]):+.0f} deg/s)"]
        if u is not None:
            rows.append(f"force = {float(np.atleast_1d(u)[0]):+7.2f} N")
        return rows


# ======================================================================
# two-link arm  -  state [q1, q2, dq1, dq2]
# ======================================================================
class TwoLinkArmArtist(SystemArtist):
    label = "2-link arm"

    def bounds(self, states=None):
        r = 1.15 * (self.system.l1 + self.system.l2)
        return (-r, r), (-r, r)

    def build(self, ax):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle

        s = self.system
        reach = s.l1 + s.l2
        c1 = self.accent
        c2 = _lighten(self.accent, 0.35)
        self.envelope = Circle((0, 0), reach, fill=False, ls=(0, (2, 3)),
                               lw=1.0, ec=_GHOST, alpha=0.7, zorder=1)
        self.goal_path = Line2D([], [], lw=1.3, ls="--", color=_GHOST,
                                alpha=0.9, zorder=2)
        self.trail = self._new_trail(ax, color=c1)
        self.link1 = Line2D([], [], lw=9, color=c1, solid_capstyle="round",
                            zorder=4)
        self.link2 = Line2D([], [], lw=7, color=c2, solid_capstyle="round",
                            zorder=4)
        self.base = Circle((0, 0), 0.055 * reach, color=_INK, zorder=6)
        self.elbow = Circle((0, 0), 0.033 * reach, color="#FFFFFF", ec=_INK,
                            lw=1.4, zorder=6)
        self.tip = Circle((0, 0), 0.040 * reach, color=c1, ec=_INK, lw=1.2,
                          zorder=7)
        self.payload = Circle((0, 0), 0.0, color=PALETTE["saturation"],
                              alpha=0.5, ec=PALETTE["saturation"], zorder=5)
        self.target = Line2D([], [], marker="P", ms=15, mew=1.5,
                             mfc=PALETTE["hybrid"], mec=_INK, ls="none",
                             zorder=3)
        for a in (self.envelope, self.base, self.elbow, self.tip, self.payload):
            ax.add_patch(a)
        for a in (self.goal_path, self.link1, self.link2, self.target):
            ax.add_line(a)
        return self._remember(self.trail, self.goal_path, self.link1,
                              self.link2, self.base, self.elbow, self.tip,
                              self.payload, self.target)

    def _fk(self, q):
        s = self.system
        e = np.array([s.l1 * np.cos(q[0]), s.l1 * np.sin(q[0])])
        w = e + np.array([s.l2 * np.cos(q[0] + q[1]),
                          s.l2 * np.sin(q[0] + q[1])])
        return e, w

    def position(self, x):
        return self._fk(np.asarray(x, float)[:2])[1]

    def draw(self, x, u=None, t=0.0, aux=None):
        q = np.asarray(x, float)[:2]
        e, w = self._fk(q)
        self.link1.set_data([0, e[0]], [0, e[1]])
        self.link2.set_data([e[0], w[0]], [e[1], w[1]])
        self.elbow.center = tuple(e)
        self.tip.center = tuple(w)
        reach = self.system.l1 + self.system.l2
        mp = getattr(self.system, "payload", 0.0)
        self.payload.center = tuple(w)
        self.payload.set_radius(0.0 if mp <= 0 else (0.035 + 0.11 * mp) * reach)
        aux = aux or {}
        tgt = aux.get("target", aux.get("ref"))
        if tgt is not None:
            tgt = np.asarray(tgt, float).ravel()[:2]
            self.target.set_data([tgt[0]], [tgt[1]])
        path = aux.get("path")
        if path is not None and len(path):
            path = np.asarray(path, float)
            self.goal_path.set_data(path[:, 0], path[:, 1])
        self._set_trail(self.trail, aux.get("trail"))

    def hud_lines(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        _, w = self._fk(x[:2])
        rows = [f"t    = {t:6.2f} s",
                f"q    = [{np.degrees(x[0]):+6.1f}, {np.degrees(x[1]):+6.1f}] deg",
                f"tip  = [{w[0]:+.3f}, {w[1]:+.3f}] m"]
        aux = aux or {}
        tgt = aux.get("target", aux.get("ref"))
        if tgt is not None:
            tgt = np.asarray(tgt, float).ravel()[:2]
            rows.append(f"err  = {np.linalg.norm(w - tgt) * 1e3:6.1f} mm")
        mp = getattr(self.system, "payload", 0.0)
        if mp:
            rows.append(f"load = {mp:.2f} kg  (ctrl model: 0.00)")
        if u is not None:
            u = np.atleast_1d(np.asarray(u, float))
            rows.append(f"tau  = [{u[0]:+.2f}, {u[1]:+.2f}] N.m")
        return rows


# ======================================================================
# differential-drive robot  -  state [x, y, theta, v, omega]
# ======================================================================
class DiffDriveArtist(SystemArtist):
    label = "diff-drive robot"

    def bounds(self, states=None):
        if states is not None and len(states):
            xs, ys = states[:, 0], states[:, 1]
            pad = 0.2 + 0.08 * (np.ptp(xs) + np.ptp(ys)) + self.system.wheel_base
            b = ((xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad))
        else:
            b = ((-1, 1), (-1, 1))
        self._span = max(b[0][1] - b[0][0], b[1][1] - b[1][0])
        return b

    def build(self, ax):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Polygon

        span = getattr(self, "_span", 2.0)
        # readable chassis: scaled to the view but never larger than ~9 % of it
        L = float(np.clip(self.system.wheel_base * 1.4, 0.05 * span, 0.09 * span))
        W = L * 0.78
        self._body0 = np.array([[-L * 0.55, -W / 2], [L * 0.30, -W / 2],
                                [L * 0.62, 0.0], [L * 0.30, W / 2],
                                [-L * 0.55, W / 2]])
        self._wheel0 = np.array([[-L * 0.18, 0], [L * 0.18, 0]])
        self._Lc = L

        self.path = Line2D([], [], lw=1.6, ls="--", color=_GHOST, alpha=0.9,
                           zorder=1)
        self.start = Line2D([], [], marker="o", ms=7, mfc="#FFFFFF", mec=_INK,
                            ls="none", zorder=2)
        self.goal = Line2D([], [], marker="*", ms=15, mfc=PALETTE["hybrid"],
                           mec=_INK, ls="none", zorder=2)
        self.trail = self._new_trail(ax)
        self.body = Polygon(self._body0, closed=True, facecolor=self.accent,
                            edgecolor=_INK, lw=1.4, alpha=0.92, zorder=4)
        self.wheelL = Line2D([], [], lw=5, color=_INK, solid_capstyle="round",
                             zorder=5)
        self.wheelR = Line2D([], [], lw=5, color=_INK, solid_capstyle="round",
                             zorder=5)
        self.lookahead = Line2D([], [], marker="o", ms=8, mfc=PALETTE["mpc"],
                                mec=_INK, ls="none", zorder=6)
        for ln in (self.path, self.start, self.goal, self.wheelL, self.wheelR,
                   self.lookahead):
            ax.add_line(ln)
        ax.add_patch(self.body)
        return self._remember(self.path, self.start, self.goal, self.trail,
                              self.body, self.wheelL, self.wheelR,
                              self.lookahead)

    def draw(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        p, th = x[:2], x[2]
        R = self._rot(th)
        self.body.set_xy((self._body0 @ R.T) + p)
        for wheel, sgn in ((self.wheelL, -1), (self.wheelR, +1)):
            off = np.array([0.0, sgn * self._Lc * 0.5])   # ± body-y (track)
            seg = (self._wheel0 + off) @ R.T + p
            wheel.set_data(seg[:, 0], seg[:, 1])
        aux = aux or {}
        path = aux.get("path")
        if path is not None and len(path):
            path = np.asarray(path, float)
            self.path.set_data(path[:, 0], path[:, 1])
            self.start.set_data([path[0, 0]], [path[0, 1]])
            self.goal.set_data([path[-1, 0]], [path[-1, 1]])
        self._set_trail(self.trail, aux.get("trail"))
        la = aux.get("lookahead")
        if la is not None:
            la = np.asarray(la, float)
            self.lookahead.set_data([la[0]], [la[1]])

    def hud_lines(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        rows = [f"t     = {t:6.2f} s",
                f"pose  = [{x[0]:+.3f}, {x[1]:+.3f}] m  @ {np.degrees(x[2]):+.0f} deg",
                f"speed = {x[3]:+.3f} m/s   yaw = {np.degrees(x[4]):+.0f} deg/s"]
        aux = aux or {}
        if "cross_track_mm" in aux:
            rows.append(f"x-track = {aux['cross_track_mm']:6.1f} mm")
        if u is not None:
            u = np.atleast_1d(np.asarray(u, float))
            rows.append(f"cmd   = v {u[0]:+.3f}  w {u[1]:+.2f}")
        return rows


# ======================================================================
# planar quadrotor  -  state [x, z, theta, xdot, zdot, thetadot]
# ======================================================================
class PlanarQuadrotorArtist(SystemArtist):
    label = "planar quadrotor"

    def bounds(self, states=None):
        if states is not None and len(states):
            xs, zs = states[:, 0], states[:, 1]
            pad = 0.3 + 0.35 * max(np.ptp(xs), np.ptp(zs), 0.25)
            return (xs.min() - pad, xs.max() + pad), (zs.min() - pad, zs.max() + pad)
        return (-1.2, 1.2), (0.0, 2.0)

    def build(self, ax):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Ellipse

        self._span = max(9 * self.system.l, 0.30)
        s = self._span
        self.trail = self._new_trail(ax)
        self.target = Line2D([], [], marker="P", ms=14, mfc=PALETTE["hybrid"],
                             mec=_INK, ls="none", zorder=2)
        self.arm = Line2D([], [], lw=5, color=_INK, solid_capstyle="round",
                          zorder=4)
        self.hull = Line2D([], [], lw=9, color=self.accent,
                           solid_capstyle="round", zorder=4)
        self.propL = Ellipse((0, 0), 0.42 * s, 0.09 * s, fc=_INK, alpha=0.55,
                             zorder=5)
        self.propR = Ellipse((0, 0), 0.42 * s, 0.09 * s, fc=_INK, alpha=0.55,
                             zorder=5)
        self.thrustL = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle="-|>", lw=2.6,
                                                   color=PALETTE["rl"]), zorder=6)
        self.thrustR = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle="-|>", lw=2.6,
                                                   color=PALETTE["rl"]), zorder=6)
        for ln in (self.target, self.arm, self.hull):
            ax.add_line(ln)
        ax.add_patch(self.propL)
        ax.add_patch(self.propR)
        return self._remember(self.trail, self.target, self.arm, self.hull,
                              self.propL, self.propR)

    def draw(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        p, th = x[:2], x[2]
        R = self._rot(th)
        half = self._span / 2
        left = p + R @ np.array([-half, 0.0])
        right = p + R @ np.array([half, 0.0])
        self.arm.set_data([left[0], right[0]], [left[1], right[1]])
        self.hull.set_data([p[0] - 0.16 * self._span * np.cos(th),
                            p[0] + 0.16 * self._span * np.cos(th)],
                           [p[1] - 0.16 * self._span * np.sin(th),
                            p[1] + 0.16 * self._span * np.sin(th)])
        up = R @ np.array([0.0, 1.0])
        for prop, base in ((self.propL, left), (self.propR, right)):
            prop.center = tuple(base + up * 0.03 * self._span)
            prop.angle = np.degrees(th)
        if u is not None:
            u = np.atleast_1d(np.asarray(u, float))
            tmax = getattr(self.system, "thrust_max", max(float(u.max()), 1e-6))
            for arrow, base, mag in ((self.thrustL, left, u[0]),
                                     (self.thrustR, right,
                                      u[1] if u.size > 1 else u[0])):
                ln = 1.1 * self._span * float(np.clip(mag / (tmax + 1e-9), 0, 1.4))
                tip = base + up * ln
                arrow.set_position((base[0], base[1]))
                arrow.xy = (tip[0], tip[1])
        aux = aux or {}
        ref = aux.get("target", aux.get("ref"))
        if ref is not None:
            ref = np.asarray(ref, float).ravel()[:2]
            self.target.set_data([ref[0]], [ref[1]])
        self._set_trail(self.trail, aux.get("trail"))

    def hud_lines(self, x, u=None, t=0.0, aux=None):
        x = np.asarray(x, float)
        rows = [f"t     = {t:6.2f} s",
                f"pos   = [{x[0]:+.3f}, {x[1]:+.3f}] m",
                f"pitch = {np.degrees(x[2]):+7.1f} deg"]
        aux = aux or {}
        ref = aux.get("target", aux.get("ref"))
        if ref is not None:
            ref = np.asarray(ref, float).ravel()[:2]
            rows.append(f"err   = {np.linalg.norm(x[:2] - ref) * 1e3:6.1f} mm")
        if u is not None:
            u = np.atleast_1d(np.asarray(u, float))
            rows.append("thr   = [" + ", ".join(f"{v:.3f}" for v in u) + "] N")
        return rows


# ======================================================================
def _lighten(hex_color: str, amount: float) -> str:
    """Blend ``hex_color`` toward white by ``amount`` in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _register_defaults() -> None:
    from .. import systems as S

    pairs = [
        ("Pendulum", PendulumArtist),
        ("CartPole", CartPoleArtist),
        ("TwoLinkArm", TwoLinkArmArtist),
        ("DifferentialDriveRobot", DiffDriveArtist),
        ("PlanarQuadrotor", PlanarQuadrotorArtist),
    ]
    for name, artist in pairs:
        cls = getattr(S, name, None)
        if isinstance(cls, type):
            register_artist(cls, artist)


_register_defaults()
