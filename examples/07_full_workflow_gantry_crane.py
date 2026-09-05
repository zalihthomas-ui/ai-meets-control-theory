r"""End-to-end aimct workflow on a hard problem it does not ship: gantry-crane
anti-sway.

A trolley on a rail carries a payload on a rigid cable.  Move the payload 3 m
and stop it *without residual swing*, under a force limit, keeping the sway
small in transit.  This is the opposite of cart-pole: the pendulum is
open-loop stable (hangs down), but every trolley move pumps energy into it.

The script walks the whole loop:
  1. define a new ``DynamicalSystem``          (subclass + Lagrangian dynamics)
  2. sanity-check it                            (aimct.dev preview report)
  3. linearise + check controllability
  4. design four controllers                    (LQR / LQR+input-shaper /
                                                 constrained MPC / iLQR-RTI)
  5. benchmark them honestly across 3 scenarios (nominal, wind gust,
                                                 cable-length model error)
  6. draw it                                    (a custom SystemArtist + animate)

Run:  python examples/07_full_workflow_gantry_crane.py
Outputs: examples/_out/crane_table.md, crane_scenarios.png, crane_animation.gif
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.controllers import LQR, ILQR, LinearMPC
from aimct.controllers.state_feedback import is_controllable
from aimct.simulate import simulate


def _scalar(v):
    return float(np.ravel(v)[0])


OUT = Path(__file__).parent / "_out"
OUT.mkdir(exist_ok=True)

G = 9.81
TARGET = 3.0                    # move the payload this far [m]
F_MAX = 40.0                    # trolley force limit [N]
SWAY_MAX = np.deg2rad(8.0)      # in-transit sway cap [rad]
DT, T_FINAL = 0.02, 12.0


# ======================================================================
# 1. a new system
# ======================================================================
from aimct.systems import DynamicalSystem


class GantryCrane(DynamicalSystem):
    r"""Planar trolley + rigid-cable payload.

    State ``[p, theta, p_dot, theta_dot]`` - trolley position (m), cable angle
    from vertical (rad, 0 = payload straight down), and their rates.
    Input ``[F]`` - horizontal force on the trolley (N).

    Euler-Lagrange (Spong-style), light trolley friction ``b`` and cable
    damping ``d``::

        (M+m) p'' + m L cos(th) th'' - m L sin(th) th'^2 = F - b p'
        m L th''  + m cos(th) p''   + m g sin(th)        = -d th'
    """

    n_states, n_inputs = 4, 1

    def __init__(self, M=5.0, m=2.0, L=2.0, b=8.0, d=0.05, g=G):
        self.M, self.m, self.L, self.b, self.d, self.g = M, m, L, b, d, g

    def dynamics(self, t, x, u):
        p, th, pd, thd = np.asarray(x, float)
        F = float(np.atleast_1d(u)[0])
        M, m, L, b, d, g = self.M, self.m, self.L, self.b, self.d, self.g
        s, c = np.sin(th), np.cos(th)
        mat = np.array([[M + m, m * L * c], [m * c, m * L]])
        rhs = np.array([F - b * pd + m * L * s * thd**2,
                        -d * thd - m * g * s])
        pdd, thdd = np.linalg.solve(mat, rhs)
        return np.array([pd, thd, pdd, thdd])


CRANE = GantryCrane()
X_EQ = np.array([TARGET, 0.0, 0.0, 0.0])
U_EQ = np.zeros(1)


# ======================================================================
# 2. sanity-check with the design-time preview report
# ======================================================================
def sanity_check():
    from aimct.dev import build_report

    rep = build_report(CRANE, x_eq=X_EQ, u_eq=U_EQ, dt=DT, t_final=6.0,
                       name="GantryCrane")
    print("-- aimct.dev preview --------------------------------------------")
    print(rep.summary())
    print()


# ======================================================================
# 3. linearise + controllability
# ======================================================================
A, B = CRANE.linearize(X_EQ, U_EQ)
print(f"open-loop poles about the target: {np.round(np.linalg.eigvals(A), 3)}")
print(f"controllable (rank 4): {is_controllable(A, B)}")
OMEGA_N = np.sqrt(G / CRANE.L)
print(f"payload natural frequency wn = {OMEGA_N:.3f} rad/s "
      f"(period {2 * np.pi / OMEGA_N:.2f} s)\n")


# ======================================================================
# 4. four controllers
# ======================================================================
# A shared minimum-jerk position reference: 0 -> TARGET over MOVE_T s, then
# hold. The feedback controllers track it; the raw ZV shaper works off the
# bare step (that is its whole point). A move slower than the pendulum period
# is already gentle - the controllers' job is to shave the rest.
MOVE_T = 5.0


def xref(t):
    """min-jerk reference state ``[p_ref, 0, p_dot_ref, 0]`` at time ``t``."""
    if t >= MOVE_T:
        return np.array([TARGET, 0.0, 0.0, 0.0])
    s = t / MOVE_T
    p = TARGET * (10 * s**3 - 15 * s**4 + 6 * s**5)
    pd = TARGET / MOVE_T * (30 * s**2 - 60 * s**3 + 30 * s**4)
    return np.array([p, 0.0, pd, 0.0])


# 4a. LQR tracking the min-jerk reference (moderate gains, heavy sway weight)
K_LQR = LQR(A, B, np.diag([12.0, 400.0, 2.0, 40.0]), np.array([[0.4]])).K


class LqrCtl:
    name = "LQR (tracks ref)"

    def reset(self):
        self._t = 0.0

    def update(self, x, dt):
        u = _scalar(-K_LQR @ (np.asarray(x) - xref(self._t)))
        self._t += dt
        return float(np.clip(u, -F_MAX, F_MAX))


# 4b. LQR + Zero-Vibration input shaper on the bare step reference. Pre-shapes
# the set-point so the commanded motion does not excite the pendulum mode:
# two impulses [1/(1+K), K/(1+K)] half a damped period apart,
# K = exp(-zeta*pi / sqrt(1 - zeta**2)).
class ShapedLqrCtl:
    name = "LQR + ZV shaper"

    def __init__(self, model_L):
        wn = np.sqrt(G / model_L)
        zeta = 0.05
        Kimp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
        self.amps = np.array([1.0 / (1 + Kimp), Kimp / (1 + Kimp)])
        self.delays = np.array([0.0, np.pi / (wn * np.sqrt(1 - zeta**2))])
        self.Ks = LQR(A, B, np.diag([10, 120, 2, 12.0]), np.array([[1.2]])).K

    def reset(self):
        self._t = 0.0

    def update(self, x, dt):
        tgt = TARGET * sum(a * (self._t >= dl)
                           for a, dl in zip(self.amps, self.delays))
        xr = np.array([tgt, 0.0, 0.0, 0.0])
        self._t += dt
        return _scalar(np.clip(-self.Ks @ (np.asarray(x) - xr), -F_MAX, F_MAX))


# 4c. Linear MPC previewing the min-jerk reference, HARD |theta| <= SWAY_MAX.
class MpcCtl:
    name = "MPC (|sway|<=8deg)"
    N = 60

    def __init__(self):
        self.mpc = LinearMPC(
            A, B, Q=np.diag([15, 60, 2, 8.0]), R=np.array([[0.3]]), N=self.N,
            x_ref=lambda t: np.array([xref(t + k * DT) for k in range(self.N)]),
            u_bounds=(-F_MAX, F_MAX),
            x_bounds=(np.array([-np.inf, -SWAY_MAX, -np.inf, -np.inf]),
                      np.array([np.inf, SWAY_MAX, np.inf, np.inf])))

    def reset(self):
        self.mpc.reset()

    def update(self, x, dt):
        return _scalar(np.clip(self.mpc.update(x, dt), -F_MAX, F_MAX))


# 4d. iLQR / RTI-NMPC on the true nonlinear dynamics, tracking the reference.
class IlqrCtl:
    name = "iLQR / RTI-NMPC"

    def __init__(self):
        self.c = ILQR.from_system(
            CRANE, DT, horizon=90,
            Q=np.diag([12, 200, 2, 20.0]), R=np.array([[0.3]]),
            Qf=np.diag([60, 400, 10, 40.0]),
            x_ref=xref, u_bounds=(-F_MAX, F_MAX), rti_iters=1, warm_iters=80)

    def reset(self):
        self.c.reset()

    def update(self, x, dt):
        return _scalar(np.clip(self.c.update(x, dt), -F_MAX, F_MAX))


def make_controllers():
    return {c.name: c for c in
            (LqrCtl(), ShapedLqrCtl(model_L=CRANE.L), MpcCtl(), IlqrCtl())}


# ======================================================================
# 5. honest benchmark across 3 scenarios
# ======================================================================
def payload_x(x):
    return x[:, 0] + CRANE.L * np.sin(x[:, 1])


def metrics(tr):
    t, X = tr.t, tr.x
    err = np.abs(payload_x(X) - TARGET)
    hit = np.where(err < 0.02)[0]
    t_settle = (float(t[hit[0]]) if hit.size and np.all(err[hit[0]:] < 0.03)
                else np.nan)
    tail = t >= (T_FINAL - 3.0)
    return {
        "t_settle_s": t_settle,
        "resid_sway_deg": float(np.rad2deg(np.sqrt(np.mean(X[tail, 1] ** 2)))),
        "peak_transit_sway_deg": float(np.rad2deg(np.max(np.abs(X[t < MOVE_T + 1, 1])))),
        "peak_force_N": float(np.max(np.abs(tr.u))),
        "ctrl_energy": float(np.trapezoid(tr.u.ravel() ** 2, t)),
        "sway_violations": int(np.sum(np.abs(X[:, 1]) > SWAY_MAX + 1e-3)),
    }


def gust(t):
    """A 0.6 s lateral wind pulse (25 N) on the trolley at t = 6 s."""
    return np.array([25.0]) if 6.0 <= t < 6.6 else np.array([0.0])


SCENARIOS = {
    "nominal": dict(sys=CRANE, dist=None),
    "wind gust": dict(sys=CRANE, dist=gust),
    "cable 40pct longer": dict(sys=GantryCrane(L=CRANE.L * 1.40), dist=None),
}


def run_benchmark():
    x0 = np.zeros(4)
    rows, trajs = [], {}
    for scen, cfg in SCENARIOS.items():
        for name, ctrl in make_controllers().items():
            ctrl.reset()
            tr = simulate(cfg["sys"], ctrl, x0=x0, dt=DT, t_final=T_FINAL,
                          u_bounds=(-F_MAX, F_MAX), input_disturbance=cfg["dist"])
            rows.append((scen, name, metrics(tr)))
            trajs[(scen, name)] = tr
    return rows, trajs


def write_table(rows):
    cols = ["t_settle_s", "resid_sway_deg", "peak_transit_sway_deg",
            "peak_force_N", "ctrl_energy", "sway_violations"]
    lines = ["# Gantry-crane anti-sway - 4 controllers x 3 scenarios", ""]
    cur = None
    for scen, name, m in rows:
        if scen != cur:
            cur = scen
            lines += ["", f"## {scen}", "",
                      "| controller | " + " | ".join(cols) + " |",
                      "| --- |" + " --- |" * len(cols)]
        lines.append("| " + name + " | " + " | ".join(
            "-" if (isinstance(m[c], float) and np.isnan(m[c])) else f"{m[c]:.4g}"
            for c in cols) + " |")
    (OUT / "crane_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ======================================================================
# 6. a custom SystemArtist + an animation
# ======================================================================
def register_crane_artist():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from aimct.viz.artists import SystemArtist, register_artist

    class CraneArtist(SystemArtist):
        label = "gantry crane"

        def bounds(self, states=None):
            return (-0.6, TARGET + 1.2), (-CRANE.L - 0.6, 0.9)

        def build(self, ax):
            self.rail = Line2D([-1e3, 1e3], [0, 0], lw=2, color="#999")
            self.trolley = Rectangle((-0.15, -0.06), 0.3, 0.12, fc="#2b2b2b")
            self.cable = Line2D([], [], lw=1.6, color="#555")
            self.payload = Rectangle((0, 0), 0.24, 0.24, fc=self.accent,
                                     ec="#2b2b2b")
            self.goal = Line2D([TARGET], [-CRANE.L], marker="x", ms=13, mew=2.5,
                               color="#56b4e9", ls="none")
            self.trail = self._new_trail(ax)
            for ln in (self.rail, self.cable, self.goal):
                ax.add_line(ln)
            ax.add_patch(self.trolley)
            ax.add_patch(self.payload)
            return self._remember(self.trail, self.rail, self.cable, self.goal,
                                  self.trolley, self.payload)

        def position(self, x):
            return np.array([x[0] + CRANE.L * np.sin(x[1]),
                             -CRANE.L * np.cos(x[1])])

        def draw(self, x, u=None, t=0.0, aux=None):
            p = x[0]
            self.trolley.set_xy((p - 0.15, -0.06))
            pay = self.position(x)
            self.cable.set_data([p, pay[0]], [0, pay[1]])
            self.payload.set_xy((pay[0] - 0.12, pay[1] - 0.12))
            self._set_trail(self.trail, (aux or {}).get("trail"))

        def hud_lines(self, x, u=None, t=0.0, aux=None):
            px = x[0] + CRANE.L * np.sin(x[1])
            r = [f"t      = {t:5.2f} s",
                 f"payload = {px:+.3f} m   (target {TARGET})",
                 f"sway   = {np.rad2deg(x[1]):+6.2f} deg"]
            if u is not None:
                r.append(f"force  = {float(np.atleast_1d(u)[0]):+6.1f} N")
            return r

    register_artist(GantryCrane, CraneArtist)


def make_animation(trajs):
    import matplotlib
    matplotlib.use("Agg")
    from aimct.viz import animate

    tr = trajs[("wind gust", "iLQR / RTI-NMPC")]
    animate(tr, CRANE, target=np.array([TARGET, -CRANE.L]),
            title="Gantry crane - iLQR/RTI rejecting a wind gust", fps=25,
            speed=1.4).save(OUT / "crane_animation.gif")
    print(f"\nsaved {OUT / 'crane_animation.gif'}")


def scenario_figure(trajs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aimct.plot_style import PALETTE, set_aimct_style

    set_aimct_style()
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    cyc = [PALETTE["lqr"], PALETTE["state_feedback"], PALETTE["mpc"],
           PALETTE["rl"]]
    for row, scen in enumerate(SCENARIOS):
        for i, name in enumerate(make_controllers()):
            tr = trajs[(scen, name)]
            axes[row, 0].plot(tr.t, payload_x(tr.x), color=cyc[i], lw=1.6,
                              label=name)
            axes[row, 1].plot(tr.t, np.rad2deg(tr.x[:, 1]), color=cyc[i], lw=1.6)
        axes[row, 0].axhline(TARGET, ls="--", color="#888", lw=1)
        for sgn in (1, -1):
            axes[row, 1].axhline(sgn * np.rad2deg(SWAY_MAX), ls=":",
                                 color=PALETTE["saturation"])
        axes[row, 0].set_ylabel(f"{scen}\npayload x [m]")
        axes[row, 1].set_ylabel("sway [deg]")
    axes[0, 0].set_title("(a) payload position")
    axes[0, 1].set_title("(b) cable sway")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    fig.suptitle("Gantry-crane anti-sway: 4 controllers, 3 scenarios",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "crane_scenarios.png", dpi=140)
    print(f"saved {OUT / 'crane_scenarios.png'}")


# ======================================================================
def main():
    sanity_check()
    register_crane_artist()
    rows, trajs = run_benchmark()
    write_table(rows)
    scenario_figure(trajs)
    make_animation(trajs)


if __name__ == "__main__":
    main()
