"""Live interactive drone-balancing sandbox.

Real-time planar quadrotor (Crazyflie 2.0) holding a hover point. YOU drive the
wind - a steady slider plus gust buttons / arrow keys / mouse drag - and watch
the controller fight to stay on target. Switch controllers on the fly and see
which ones droop, which recover, and which reject steady wind outright.

    python experiments/live_drone/live.py
    python -m aimct live                       # same thing

Controls
--------
  * Wind slider           steady horizontal wind [N]
  * "gust <<" / "gust >>"  ~0.3 s hard gust
  * Left / Right arrows    gust left / right      Up / Down: vertical gust
  * mouse drag on the sky  push the drone toward the drag
  * radio buttons          switch controller live
  * "reset" / R            recentre the drone
  * space                  clear the wind
"""

from __future__ import annotations

import sys

import numpy as np

from aimct.controllers import LQR
from aimct.systems import PlanarQuadrotor

PHYS_DT = 0.002          # s   physics step
SUBSTEPS = 8             #     physics steps per rendered frame  -> ~62 FPS
TRAIL = 220
HOVER = np.array([0.0, 1.0])     # target x, z


class LiveQuad(PlanarQuadrotor):
    """Quadrotor with a live external wind force ``[fx, fz]`` (N)."""

    def __init__(self):
        super().__init__()
        self.wind = np.zeros(2)

    def dynamics(self, t, x, u):
        xd = super().dynamics(t, x, u)
        xd[3] += self.wind[0] / self.m
        xd[4] += self.wind[1] / self.m
        return xd


def _bryson():
    qx = 1.0 / np.array([0.10, 0.10, 0.20, 0.5, 0.5, 3.0]) ** 2
    ru = 1.0 / np.array([0.15, 0.15]) ** 2
    return np.diag(qx), np.diag(ru)


def build_controllers(q: LiveQuad):
    A, B = q.linearize()
    Q, R = _bryson()
    K = LQR(A, B, Q, R).K

    def lqr(x, integ):
        xr = np.array([HOVER[0], HOVER[1], 0, 0, 0, 0])
        return q.u_hover - K @ (x - xr)

    # integral-augmented LQR ("LQI"): rejects *steady* wind (zero droop)
    Ci = np.zeros((2, 6)); Ci[0, 0] = 1.0; Ci[1, 1] = 1.0
    Aa = np.block([[A, np.zeros((6, 2))], [Ci, np.zeros((2, 2))]])
    Ba = np.vstack([B, np.zeros((2, 2))])
    Qa = np.diag(np.concatenate([np.diag(Q), [6.0, 6.0]]))
    Ka = LQR(Aa, Ba, Qa, R).K
    Kx, Ki = Ka[:, :6], Ka[:, 6:]

    def lqi(x, integ):
        xr = np.array([HOVER[0], HOVER[1], 0, 0, 0, 0])
        return q.u_hover - Kx @ (x - xr) - Ki @ integ

    # deliberately soft LQR -> visibly droops in wind, for contrast
    Ksoft = LQR(A, B, np.diag(np.diag(Q) * 0.05), R * 20).K

    def lqr_soft(x, integ):
        xr = np.array([HOVER[0], HOVER[1], 0, 0, 0, 0])
        return q.u_hover - Ksoft @ (x - xr)

    return {
        "LQR (stiff)": lqr,
        "LQR + integral\n(wind-adaptive)": lqi,
        "LQR (soft)": lqr_soft,
    }


def _headless_check(steps: int = 700) -> int:
    """Run the physics + every controller with no GUI - a smoke test.

    Scenario: a steady 0.03 N side wind for the whole run + a short 0.05 N gust.
    Expected: the integral controller drives the *steady* droop to ~0; the plain
    LQRs hold a finite offset; nobody tumbles.
    """
    q = LiveQuad()
    controllers = build_controllers(q)
    for name, ctl in controllers.items():
        x = np.array([HOVER[0], HOVER[1], 0, 0, 0, 0], dtype=float)
        integ = np.zeros(2)
        for k in range(steps):
            gust = 0.05 if 200 < k < 240 else 0.0
            q.wind = np.array([0.03 + gust, 0.0])
            u = np.clip(ctl(x, integ), 0.0, q.thrust_max)
            for _ in range(SUBSTEPS):
                k1 = q.dynamics(0, x, u)
                k2 = q.dynamics(0, x + 0.5 * PHYS_DT * k1, u)
                k3 = q.dynamics(0, x + 0.5 * PHYS_DT * k2, u)
                k4 = q.dynamics(0, x + PHYS_DT * k3, u)
                x = x + (PHYS_DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                integ += (x[:2] - HOVER) * PHYS_DT
                integ[:] = np.clip(integ, -0.5, 0.5)
        err = float(np.hypot(x[0] - HOVER[0], x[1] - HOVER[1]))
        assert np.all(np.isfinite(x)), f"{name}: diverged"
        assert abs(x[2]) < 1.2 and err < 0.6, f"{name}: lost the hover point ({err:.2f} m)"
        print(f"  {name.splitlines()[0]:<20s} steady-state error {err*1e3:6.1f} mm")
    print("headless check OK")
    return 0


def main() -> int:
    if "--headless" in sys.argv:
        return _headless_check()

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib GUI not available: {exc}")
        return 1
    if matplotlib.get_backend().lower() == "agg":
        print("No interactive matplotlib backend (Agg). Run in a normal desktop "
              "session; on Windows the Tk backend ships with Python.")
        return 1

    q = LiveQuad()
    controllers = build_controllers(q)
    names = list(controllers)
    state = {
        "x": np.array([HOVER[0], HOVER[1], 0, 0, 0, 0], dtype=float),
        "integ": np.zeros(2),
        "ctrl": names[1],
        "gust": np.zeros(2),
        "gust_t": 0.0,
        "steady_wind": 0.0,
        "err_hist": [],
        "trail": [],
        "u": q.u_hover.copy(),
    }

    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title("AI Meets Control Theory - live drone")
    ax = fig.add_axes([0.05, 0.28, 0.60, 0.68])
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(0.0, 2.2); ax.set_aspect("equal")
    ax.set_title("hold the hover point - you control the wind")
    ax.axhline(0.0, color="#8a6d3b", lw=3)                       # ground
    ax.plot(*HOVER, "+", ms=16, mew=2, color="#888")            # target
    (trail_ln,) = ax.plot([], [], "-", lw=1.0, color="#56B4E9", alpha=0.7)
    (body_ln,) = ax.plot([], [], "-", lw=4, color="#0072B2", solid_capstyle="round")
    (t1_ln,) = ax.plot([], [], "-", lw=3, color="#D55E00")
    (t2_ln,) = ax.plot([], [], "-", lw=3, color="#D55E00")
    wind_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=2))
    hud = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
                  family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    axp = fig.add_axes([0.72, 0.60, 0.25, 0.34])
    axp.set_title("position error [mm]"); axp.set_xlim(0, TRAIL); axp.set_ylim(0, 400)
    (err_ln,) = axp.plot([], [], color="#0072B2")
    axt = fig.add_axes([0.72, 0.16, 0.25, 0.34])
    axt.set_title("total thrust [mN]"); axt.set_xlim(0, TRAIL)
    axt.set_ylim(0, q.thrust_max * 2 * 1e3 * 1.05)
    axt.axhline(q.m * q.g * 1e3, color="#999", lw=1)
    axt.axhline(q.thrust_max * 2 * 1e3, color="#D62728", ls=":", lw=1)
    (thr_ln,) = axt.plot([], [], color="#D55E00")
    thr_hist: list = []

    s_wind = Slider(fig.add_axes([0.10, 0.16, 0.45, 0.03]), "steady wind [N]",
                    -0.08, 0.08, valinit=0.0)
    radio = RadioButtons(fig.add_axes([0.06, 0.02, 0.24, 0.11]), names, active=1)
    b_l = Button(fig.add_axes([0.34, 0.06, 0.11, 0.05]), "gust <<")
    b_r = Button(fig.add_axes([0.46, 0.06, 0.11, 0.05]), "gust >>")
    b_reset = Button(fig.add_axes([0.34, 0.005, 0.23, 0.05]), "reset")

    GUST = 0.10  # N

    def do_gust(dx):
        state["gust"] = np.array([dx, 0.0]); state["gust_t"] = 0.30

    b_l.on_clicked(lambda _e: do_gust(-GUST))
    b_r.on_clicked(lambda _e: do_gust(+GUST))
    radio.on_clicked(lambda label: state.update(ctrl=label, integ=np.zeros(2)))
    s_wind.on_changed(lambda v: state.update(steady_wind=v))

    def reset(_e=None):
        state["x"] = np.array([HOVER[0], HOVER[1], 0, 0, 0, 0], dtype=float)
        state["integ"] = np.zeros(2)
        state["trail"].clear(); state["err_hist"].clear(); thr_hist.clear()
    b_reset.on_clicked(reset)

    def on_key(ev):
        if ev.key == "left": do_gust(-GUST)
        elif ev.key == "right": do_gust(+GUST)
        elif ev.key == "up": state.update(gust=np.array([0.0, GUST])); state["gust_t"] = 0.3
        elif ev.key == "down": state.update(gust=np.array([0.0, -GUST])); state["gust_t"] = 0.3
        elif ev.key == " ": s_wind.set_val(0.0)
        elif ev.key in ("r", "R"): reset()
    fig.canvas.mpl_connect("key_press_event", on_key)

    drag = {"x0": None}

    def on_press(ev):
        if ev.inaxes is ax:
            drag["x0"] = np.array([ev.xdata, ev.ydata])

    def on_release(ev):
        if drag["x0"] is not None and ev.inaxes is ax:
            d = np.array([ev.xdata, ev.ydata]) - drag["x0"]
            state["gust"] = np.clip(d * 0.25, -0.2, 0.2); state["gust_t"] = 0.35
        drag["x0"] = None
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)

    def step_physics():
        ctl = controllers[state["ctrl"]]
        for _ in range(SUBSTEPS):
            wx = state["steady_wind"] + (state["gust"][0] if state["gust_t"] > 0 else 0.0)
            wz = (state["gust"][1] if state["gust_t"] > 0 else 0.0)
            q.wind = np.array([wx, wz])
            x = state["x"]
            u = np.clip(ctl(x, state["integ"]), 0.0, q.thrust_max)
            state["u"] = u
            # advance
            k1 = q.dynamics(0, x, u)
            k2 = q.dynamics(0, x + 0.5 * PHYS_DT * k1, u)
            k3 = q.dynamics(0, x + 0.5 * PHYS_DT * k2, u)
            k4 = q.dynamics(0, x + PHYS_DT * k3, u)
            state["x"] = x + (PHYS_DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            state["integ"] += (state["x"][:2] - HOVER) * PHYS_DT
            state["integ"] = np.clip(state["integ"], -2.0, 2.0)
            if state["gust_t"] > 0:
                state["gust_t"] -= PHYS_DT

    def draw(_frame):
        step_physics()
        x = state["x"]
        px, pz, th = x[0], x[1], x[2]
        L = q.l * 6                     # visually exaggerate the arm
        dx, dz = L * np.cos(th), L * np.sin(th)
        body_ln.set_data([px - dx, px + dx], [pz - dz, pz + dz])
        # thrust arrows perpendicular to the body, length ~ thrust
        nx, nz = -np.sin(th), np.cos(th)
        a = 2.0
        T1, T2 = state["u"]
        t1_ln.set_data([px + dx, px + dx + a * T1 * nx], [pz + dz, pz + dz + a * T1 * nz])
        t2_ln.set_data([px - dx, px - dx + a * T2 * nx], [pz - dz, pz - dz + a * T2 * nz])

        state["trail"].append((px, pz))
        state["trail"][:] = state["trail"][-TRAIL:]
        tr = np.array(state["trail"])
        trail_ln.set_data(tr[:, 0], tr[:, 1])

        w = state["steady_wind"] + (state["gust"][0] if state["gust_t"] > 0 else 0.0)
        wz = (state["gust"][1] if state["gust_t"] > 0 else 0.0)
        wind_arrow.set_position((px - np.sign(w) * 0.05 - w * 4, pz + 0.35))
        wind_arrow.xy = (px + w * 4, pz + 0.35 + wz * 4)

        err_mm = np.hypot(px - HOVER[0], pz - HOVER[1]) * 1e3
        state["err_hist"].append(err_mm)
        state["err_hist"][:] = state["err_hist"][-TRAIL:]
        err_ln.set_data(range(len(state["err_hist"])), state["err_hist"])
        thr_hist.append(state["u"].sum() * 1e3)
        thr_hist[:] = thr_hist[-TRAIL:]
        thr_ln.set_data(range(len(thr_hist)), thr_hist)

        rms = float(np.sqrt(np.mean(np.square(state["err_hist"][-120:] or [0]))))
        tumbling = abs(th) > 1.2 or pz < 0.1
        hud.set_text(
            f" controller : {state['ctrl'].splitlines()[0]}\n"
            f" wind       : {w:+.3f} N   gust {'ON ' if state['gust_t']>0 else 'off'}\n"
            f" pos error  : {err_mm:6.1f} mm   (rms {rms:5.1f})\n"
            f" pitch      : {np.degrees(th):+6.1f} deg\n"
            f" status     : {'TUMBLING - press reset' if tumbling else 'holding'}"
        )
        return body_ln, t1_ln, t2_ln, trail_ln, err_ln, thr_ln, hud

    from matplotlib.animation import FuncAnimation
    _anim = FuncAnimation(fig, draw, interval=16, blit=False, cache_frame_data=False)
    fig._live_anim = _anim          # keep a ref alive
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
