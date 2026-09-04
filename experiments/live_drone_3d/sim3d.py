"""3-D live drone-vs-wind sandbox - physics + control engine.

A real-time :class:`aimct.systems.Quadrotor3D` (Crazyflie 2.0) holding a hover
point while YOU drive a 3-D wind vector. Switch controllers on the fly and watch
which ones droop, which recover, and which reject a steady wind outright - the
3-D counterpart of ``experiments/live_drone/live.py``.

The engine is **renderer-agnostic**: each frame it produces a :class:`Frame`
(drone pose, per-rotor thrusts, wind, HUD text). A minimal matplotlib-3D
renderer ships here so the sandbox runs standalone; a richer renderer (a real
drone mesh, WebGL) plugs in by implementing :class:`Renderer` - see
``experiments/live_drone_3d/RENDERER_SPEC.md``.

    python experiments/live_drone_3d/sim3d.py            # matplotlib-3D
    python experiments/live_drone_3d/sim3d.py --headless # physics smoke check
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np

from scipy.linalg import solve_continuous_are

from aimct.systems import Quadrotor3D, rotation_matrix


def _lqr_gain(A, B, Q, R):
    """LQR gain via scipy's CARE - the from-scratch Hamiltonian solver is
    fragile on this 12-state, wildly-scaled problem; this is a viz tool."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)

PHYS_DT = 0.002
SUBSTEPS = 8                       # -> ~62 Hz control / render
HOVER = np.array([0.0, 0.0, 1.5])  # target x, y, z
GUST = 0.06                        # N impulse


# ------------------------------------------------------------------ controllers
def _bryson_QR():
    q = 1.0 / np.array([0.15, 0.15, 0.15, 0.6, 0.6, 0.6,
                        0.30, 0.30, 0.40, 4.0, 4.0, 4.0]) ** 2
    r = 1.0 / np.array([0.20, 3.0e-3, 3.0e-3, 3.0e-3]) ** 2
    return np.diag(q), np.diag(r)


def build_controllers(quad: Quadrotor3D):
    A, B = quad.linearize()
    Q, R = _bryson_QR()
    K = _lqr_gain(A, B, Q, R)
    xr = np.concatenate([HOVER, np.zeros(9)])

    def lqr(x, integ):
        return quad.u_hover - K @ (x - xr)

    # integral-augmented LQR on the 3 position channels -> nulls steady wind
    Ci = np.zeros((3, 12)); Ci[0, 0] = Ci[1, 1] = Ci[2, 2] = 1.0
    Aa = np.block([[A, np.zeros((12, 3))], [Ci, np.zeros((3, 3))]])
    Ba = np.vstack([B, np.zeros((3, 4))])
    Qa = np.diag(np.concatenate([np.diag(Q), [6.0, 6.0, 6.0]]))
    Ka = _lqr_gain(Aa, Ba, Qa, R)
    Kx, Ki = Ka[:, :12], Ka[:, 12:]

    def lqi(x, integ):
        return quad.u_hover - Kx @ (x - xr) - Ki @ integ

    Ksoft = _lqr_gain(A, B, np.diag(np.diag(Q) * 0.04), R * 25)

    def lqr_soft(x, integ):
        return quad.u_hover - Ksoft @ (x - xr)

    return {"LQR (stiff)": lqr, "LQR + integral (wind-adaptive)": lqi,
            "LQR (soft)": lqr_soft}


# ------------------------------------------------------------------ mixer (viz)
def rotor_thrusts(quad: Quadrotor3D, u):
    """Split ``[f, tau_x, tau_y, tau_z]`` into 4 non-negative per-rotor thrusts
    for an X-configuration (front-right, back-right, back-left, front-left)."""
    f, tx, ty, tz = u
    d = quad.arm / np.sqrt(2.0)
    T = np.array([
        f / 4 - ty / (4 * d) + tx / (4 * d) - tz / 4e-2,
        f / 4 - ty / (4 * d) - tx / (4 * d) + tz / 4e-2,
        f / 4 + ty / (4 * d) - tx / (4 * d) - tz / 4e-2,
        f / 4 + ty / (4 * d) + tx / (4 * d) + tz / 4e-2,
    ])
    return np.clip(T, 0.0, None)


# ------------------------------------------------------------------ engine
class Frame(NamedTuple):
    pos: np.ndarray            # (3,) world position
    R: np.ndarray              # (3, 3) body -> world rotation
    rotors: np.ndarray         # (4,) per-rotor thrust for arrow lengths
    wind: np.ndarray           # (3,) current wind force
    trail: np.ndarray          # (N, 3) recent positions
    hud: str
    tumbling: bool


class Engine:
    def __init__(self, ctrl_name: str | None = None):
        self.quad = Quadrotor3D()
        self.controllers = build_controllers(self.quad)
        self.ctrl = ctrl_name or next(iter(self.controllers))
        self.reset()
        self.steady_wind = np.zeros(3)
        self.gust = np.zeros(3)
        self.gust_t = 0.0
        self._trail: list = []

    def reset(self):
        self.x = np.concatenate([HOVER, np.zeros(9)])
        self.x[6:9] = [0.15, -0.1, 0.0]          # a small initial tilt
        self.integ = np.zeros(3)
        self.u = self.quad.u_hover.copy()
        self._trail = []

    def set_controller(self, name):
        if name in self.controllers:
            self.ctrl = name
            self.integ[:] = 0.0

    def add_gust(self, vec):
        self.gust = np.asarray(vec, float)
        self.gust_t = 0.30

    def _wind(self):
        w = self.steady_wind.copy()
        if self.gust_t > 0:
            w = w + self.gust
        return w

    def step_frame(self) -> Frame:
        ctl = self.controllers[self.ctrl]
        q = self.quad
        for _ in range(SUBSTEPS):
            w = self._wind()
            self.u = ctl(self.x, self.integ)
            x = self.x

            def f(xs):
                xd = q.dynamics(0.0, xs, self.u)
                xd[3:6] += w / q.m                # wind as a world-frame force
                return xd

            k1 = f(x); k2 = f(x + .5 * PHYS_DT * k1)
            k3 = f(x + .5 * PHYS_DT * k2); k4 = f(x + PHYS_DT * k3)
            self.x = x + PHYS_DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.integ += (self.x[:3] - HOVER) * PHYS_DT
            self.integ = np.clip(self.integ, -1.5, 1.5)
            if self.gust_t > 0:
                self.gust_t -= PHYS_DT

        p = self.x[:3]
        R = rotation_matrix(*self.x[6:9])
        self._trail.append(p.copy()); self._trail[:] = self._trail[-240:]
        err = float(np.linalg.norm(p - HOVER))
        att = np.degrees(self.x[6:9])
        tumbling = abs(self.x[6]) > 1.2 or abs(self.x[7]) > 1.2 or p[2] < 0.1
        hud = (f" controller : {self.ctrl}\n"
               f" wind       : ({self._wind()[0]:+.2f}, {self._wind()[1]:+.2f}, "
               f"{self._wind()[2]:+.2f}) N  {'GUST' if self.gust_t > 0 else ''}\n"
               f" pos error  : {err * 1e3:6.1f} mm\n"
               f" attitude   : r{att[0]:+5.1f} p{att[1]:+5.1f} y{att[2]:+5.1f} deg\n"
               f" status     : {'TUMBLING - reset' if tumbling else 'holding'}")
        return Frame(p, R, rotor_thrusts(q, self.u), self._wind(),
                     np.array(self._trail), hud, tumbling)


# ------------------------------------------------------------------ headless
def _headless(steps: int = 700) -> int:
    for name in build_controllers(Quadrotor3D()):
        e = Engine(name)
        for k in range(steps):
            e.steady_wind = np.array([0.03, -0.02, 0.0]) if k > 120 else np.zeros(3)
            if 250 < k < 275:
                e.add_gust([0.05, 0.0, 0.03])
            fr = e.step_frame()
        err = float(np.linalg.norm(e.x[:3] - HOVER))
        assert np.all(np.isfinite(e.x)) and err < 1.0, f"{name}: lost hover ({err:.2f} m)"
        print(f"  {name:<32s} steady-state error {err * 1e3:6.1f} mm")
    print("headless check OK")
    return 0


# ------------------------------------------------------------------ mpl renderer
def _run_matplotlib() -> int:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider
        from matplotlib.animation import FuncAnimation
    except Exception as exc:                       # pragma: no cover
        print(f"matplotlib GUI unavailable: {exc}")
        return 1
    if matplotlib.get_backend().lower() == "agg":
        print("No interactive backend. Run in a desktop session.")
        return 1

    eng = Engine()
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(0, 3))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.scatter(*HOVER, marker="+", s=120, color="#888")
    (trail_ln,) = ax.plot([], [], [], "-", lw=0.8, color="#56B4E9", alpha=0.7)
    arms = [ax.plot([], [], [], "-", lw=3, color="#0072B2")[0] for _ in range(2)]
    rotor_ln = [ax.plot([], [], [], "-", lw=2, color="#D55E00")[0] for _ in range(4)]
    wind_q = [None]
    hud = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, va="top",
                    family="monospace", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    names = list(eng.controllers)
    plt.subplots_adjust(bottom=0.24)
    s_wx = Slider(fig.add_axes([0.12, 0.14, 0.3, 0.03]), "wind x", -0.08, 0.08, 0)
    s_wy = Slider(fig.add_axes([0.12, 0.10, 0.3, 0.03]), "wind y", -0.08, 0.08, 0)
    s_wz = Slider(fig.add_axes([0.12, 0.06, 0.3, 0.03]), "wind z", -0.06, 0.06, 0)
    radio = RadioButtons(fig.add_axes([0.55, 0.04, 0.28, 0.13]), names, active=1)
    b_gust = Button(fig.add_axes([0.86, 0.11, 0.11, 0.05]), "gust")
    b_reset = Button(fig.add_axes([0.86, 0.04, 0.11, 0.05]), "reset")

    def sync_wind(_=None):
        eng.steady_wind = np.array([s_wx.val, s_wy.val, s_wz.val])
    for s in (s_wx, s_wy, s_wz):
        s.on_changed(sync_wind)
    radio.on_clicked(eng.set_controller)
    b_gust.on_clicked(lambda _e: eng.add_gust(
        np.random.default_rng().uniform(-1, 1, 3) * [GUST, GUST, GUST * 0.5]))
    b_reset.on_clicked(lambda _e: eng.reset())

    def on_key(ev):
        step = {"left": [-GUST, 0, 0], "right": [GUST, 0, 0],
                "down": [0, -GUST, 0], "up": [0, GUST, 0],
                "pageup": [0, 0, GUST], "pagedown": [0, 0, -GUST]}.get(ev.key)
        if step:
            eng.add_gust(step)
        elif ev.key in ("r", "R"):
            eng.reset()
    fig.canvas.mpl_connect("key_press_event", on_key)

    L = eng.quad.arm * 5

    def draw(_i):
        fr = eng.step_frame()
        p, R = fr.pos, fr.R
        ex, ey = R[:, 0] * L, R[:, 1] * L
        arms[0].set_data_3d(*np.array([p - ex, p + ex]).T)
        arms[1].set_data_3d(*np.array([p - ey, p + ey]).T)
        up = R[:, 2]
        for ln, tip, T in zip(rotor_ln, [p + ex, p - ex, p + ey, p - ey], fr.rotors):
            ln.set_data_3d(*np.array([tip, tip + up * (0.4 + 8 * T)]).T)
        if fr.trail.size:
            trail_ln.set_data_3d(fr.trail[:, 0], fr.trail[:, 1], fr.trail[:, 2])
        if wind_q[0] is not None:
            wind_q[0].remove()
        w = fr.wind
        wind_q[0] = ax.quiver(*(p + [0, 0, 0.4]), *(w * 6), color="#009E73", lw=2)
        hud.set_text(fr.hud)
        return arms + rotor_ln + [trail_ln, hud]

    _anim = FuncAnimation(fig, draw, interval=16, blit=False, cache_frame_data=False)
    fig._anim = _anim
    plt.show()
    return 0


def main() -> int:
    if "--headless" in sys.argv:
        return _headless()
    return _run_matplotlib()


if __name__ == "__main__":
    sys.exit(main())
