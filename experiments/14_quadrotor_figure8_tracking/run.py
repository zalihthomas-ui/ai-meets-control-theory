"""Experiment 14 - planar quadrotor (Crazyflie 2.0) flies a figure-8.

A real nano-drone model. Controllers are designed on the hover linearisation and
flown on the true nonlinear dynamics, tracking a lemniscate with analytic
feed-forward: LQR (+/- flatness feed-forward), a single-setpoint linear MPC, and
a reference-preview linear MPC that takes the whole trajectory over its horizon.

Run:  python experiments/14_quadrotor_figure8_tracking/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.controllers import LQR, LinearMPC
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.simulate import simulate
from aimct.systems import PlanarQuadrotor

HERE = Path(__file__).parent

DT, DURATION = 0.004, 12.0
A_X, B_Z, PERIOD, Z0 = 0.6, 0.35, 4.0, 1.0      # aggressive: ~3.7 m/s peak speed
W = 2.0 * np.pi / PERIOD
WIND = (5.0, 8.0, 0.030)                        # (t_on, t_off, lateral force [N])
# Bryson-scaled cost: Q_ii = 1/max_accept(x_i)^2, R_jj = 1/max_accept(du_j)^2.
# The input matrix B is very badly scaled (pitch-torque gain ~3300 vs thrust ~36),
# so an unscaled R gives a nonsensical LQR (a ~2000 rad/s pole); this fixes it.
_X_MAX = np.array([0.10, 0.10, 0.20, 0.5, 0.5, 3.0])     # m, m, rad, m/s, m/s, rad/s
_U_MAX = np.array([0.15, 0.15])                          # N deviation from hover
Q = np.diag(1.0 / _X_MAX**2)
R = np.diag(1.0 / _U_MAX**2)
N_MPC = 25
MPC_LOOKAHEAD = 12          # steps ahead to point the (single) MPC setpoint
N_PREVIEW = 40              # preview MPC horizon (40 * 4 ms = 160 ms of trajectory)
MPC_HZ_DIV = 4             # preview MPC re-solves every 4th step (1 kHz -> 250 Hz)

class WindyQuad(PlanarQuadrotor):
    """PlanarQuadrotor plus an external lateral wind force during ``[t_on, t_off]``."""

    def __init__(self, gust=WIND):
        super().__init__()
        self.gust = gust

    def dynamics(self, t, x, u):
        xdot = super().dynamics(t, x, u)
        t_on, t_off, fx = self.gust
        if t_on <= t < t_off:
            xdot[3] += fx / self.m
        return xdot


quad = WindyQuad()
G, M, IYY, L = quad.g, quad.m, quad.Iyy, quad.l
T_MAX = quad.thrust_max


def reference(t: float):
    """Return (x_ref_state[6], u_ref[2]) for the lemniscate + feed-forward."""
    s1, c1 = np.sin(W * t), np.cos(W * t)
    s2 = np.sin(2 * W * t)
    c2 = np.cos(2 * W * t)
    # x_r and its first four time derivatives (differential-flatness feed-forward)
    x = A_X * s1
    xd = A_X * W * c1
    xdd = -A_X * W**2 * s1
    xddd = -A_X * W**3 * c1
    xdddd = A_X * W**4 * s1
    z, zd, zdd = Z0 + B_Z * s2, 2 * B_Z * W * c2, -4 * B_Z * W**2 * s2
    th = -xdd / G                      # small-angle: xdd ~ -g*theta
    thd = -xddd / G
    thdd_ff = -xdddd / G               # pitch-accel feed-forward
    T_tot = M * (G + zdd)              # collective thrust for vertical accel
    delta = IYY * thdd_ff / L          # differential thrust for pitch accel
    u_ref = np.array([0.5 * T_tot + 0.5 * delta, 0.5 * T_tot - 0.5 * delta])
    x_ref = np.array([x, z, th, xd, zd, thd])
    return x_ref, np.clip(u_ref, 0.0, T_MAX)


class TrajectoryTracker:
    """Wrap an LQR / LinearMPC: refresh its x_ref (and, unless ``feedback_only``,
    its u_ref feed-forward) from the trajectory each step.

    ``mode``:
      "point"   - a single set-point, pointed ``lookahead`` steps ahead (LQR, and
                  the naive MPC baseline);
      "preview" - the whole reference trajectory over the MPC horizon: x_ref is an
                  (N, 6) array at prediction steps 1..N, u_ref an (N, 2) flatness
                  feed-forward at steps 0..N-1."""

    def __init__(self, inner, name, lookahead_steps=0, feedback_only=False,
                 mode="point", resolve_every=1):
        self.inner, self.name = inner, name
        self.lead = lookahead_steps * DT
        self.feedback_only = feedback_only
        self.mode = mode
        self.resolve_every = int(resolve_every)   # re-solve the QP every k steps
        self._t = 0.0
        self._k = 0
        self._held = None

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()
        self._t = 0.0
        self._k = 0
        self._held = None

    def update(self, x, dt):
        if self.mode == "preview":
            # the MPC's model is the hover linearisation, so it works in
            # deviation coordinates: feed the feed-forward as (u_flat - u_hover)
            # and add u_hover back to its output.
            if self._k % self.resolve_every == 0 or self._held is None:
                N = self.inner.N
                self.inner.x_ref = np.array(
                    [reference(self._t + (j + 1) * DT)[0] for j in range(N)])
                self.inner.u_ref = np.array(
                    [reference(self._t + j * DT)[1] - quad.u_hover for j in range(N)])
                du = np.asarray(self.inner.update(x, dt)).reshape(2)
                self._plan = self.inner.horizon_plan          # (N, 2) thrust deviations
                self._held = quad.u_hover + du
            else:
                self._held = quad.u_hover + self._plan[
                    min(self._k % self.resolve_every, len(self._plan) - 1)]
            u = self._held
        else:
            xr, ur = reference(self._t + self.lead)
            self.inner.x_ref = xr
            self.inner.u_ref = quad.u_hover if self.feedback_only else ur
            u = np.asarray(self.inner.update(x, dt)).reshape(2)
        self._t += dt
        self._k += 1
        return np.clip(u, 0.0, T_MAX)


def build_controllers():
    Aq, Bq = quad.linearize()
    return {
        "LQR + flatness feedforward":
            TrajectoryTracker(LQR(Aq, Bq, Q, R), "LQR + flatness feedforward"),
        "LQR feedback only":
            TrajectoryTracker(LQR(Aq, Bq, Q, R), "LQR feedback only",
                              feedback_only=True),
        "Linear MPC (single setpoint)":
            TrajectoryTracker(LinearMPC(Aq, Bq, Q=Q, R=R, N=N_MPC,
                                        u_bounds=(0.0, T_MAX)),
                              "Linear MPC (single setpoint)", MPC_LOOKAHEAD),
        # preview MPC runs unconstrained (flatness feed-forward keeps thrust near
        # hover, far inside [0, T_max]); thrust is clipped post-hoc like the LQR.
        "Linear MPC (preview)":
            TrajectoryTracker(LinearMPC(Aq, Bq, Q=Q, R=R, N=N_PREVIEW),
                              "Linear MPC (preview)", mode="preview",
                              resolve_every=MPC_HZ_DIV),
    }


def metrics(traj):
    t = traj.t
    ref = np.array([reference(tt)[0] for tt in t])
    pos_err = np.hypot(traj.x[:, 0] - ref[:, 0], traj.x[:, 1] - ref[:, 1])
    pitch_err = traj.x[:, 2] - ref[:, 2]
    du = traj.u - quad.u_hover
    energy = float(np.trapezoid(np.sum(du**2, axis=1), t))
    thrust_tot = traj.u.sum(axis=1)
    sat = float(np.mean((traj.u >= T_MAX - 1e-6).any(axis=1)) * 100.0)
    return {
        "rms_pos_err_mm": float(np.sqrt(np.mean(pos_err**2)) * 1e3),
        "max_pos_err_mm": float(np.max(pos_err) * 1e3),
        "rms_pitch_deg": float(np.degrees(np.sqrt(np.mean(pitch_err**2)))),
        "ctrl_energy": energy,
        "peak_thrust_N": float(np.max(thrust_tot)),
        "thrust_sat_pct": sat,
    }


def main():
    x0 = np.array([0.0, Z0, 0.0, 0.0, 0.0, 0.0])
    rows, trajs = {}, {}
    for name, ctrl in build_controllers().items():
        tr = simulate(quad, ctrl, x0=x0, dt=DT, t_final=DURATION,
                      u_bounds=(0.0, T_MAX))
        rows[name] = metrics(tr)
        trajs[name] = tr

    cols = ["rms_pos_err_mm", "max_pos_err_mm", "rms_pitch_deg", "ctrl_energy",
            "peak_thrust_N", "thrust_sat_pct"]
    lines = ["# Experiment 14 - quadrotor figure-8 tracking (Crazyflie 2.0)", "",
             "| controller | " + " | ".join(cols) + " |",
             "| --- |" + " --- |" * len(cols)]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3g}" for c in cols) + " |")
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import csv
    with open(HERE / "table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["controller", *cols])
        for name, m in rows.items():
            w.writerow([name, *(m[c] for c in cols)])

    _figure(trajs)
    print((HERE / "table.md").read_text())


def _figure(trajs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    t = next(iter(trajs.values())).t
    ref = np.array([reference(tt)[0] for tt in t])
    _cyc = [PALETTE["lqr"], PALETTE["state_feedback"], PALETTE["mpc"], PALETTE["rl"]]
    styles = {name: _cyc[i % len(_cyc)] for i, name in enumerate(trajs)}

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ref[:, 0], ref[:, 1], "--", color=PALETTE["reference"], lw=1.6,
                  label="reference")
    for name, tr in trajs.items():
        ax[0, 0].plot(tr.x[:, 0], tr.x[:, 1], color=styles[name], lw=1.8, label=name)
        pe = np.hypot(tr.x[:, 0] - ref[:, 0], tr.x[:, 1] - ref[:, 1]) * 1e3
        ax[0, 1].plot(t, pe, color=styles[name], lw=1.6, label=name)
        ax[1, 0].plot(t, np.degrees(tr.x[:, 2]), color=styles[name], lw=1.4, label=name)
        ax[1, 1].plot(t, tr.u.sum(axis=1) * 1e3, color=styles[name], lw=1.2, label=name)
    ax[1, 0].plot(t, np.degrees(ref[:, 2]), "--", color=PALETTE["reference"], lw=1.2)
    ax[1, 1].axhline(T_MAX * 2 * 1e3, ls=":", color=PALETTE["saturation"], lw=1.4,
                     label=r"$2\,T_{\max}$")
    ax[1, 1].axhline(quad.m * quad.g * 1e3, ls="-", color="#999", lw=1.0,
                     label="hover")

    ax[0, 0].set(title="(a) trajectory  x-z [m]", xlabel="x [m]", ylabel="z [m]")
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 1].set(title="(b) position error [mm]", xlabel="t [s]", ylabel="mm")
    ax[1, 0].set(title="(c) pitch angle [deg]", xlabel="t [s]", ylabel="deg")
    ax[1, 1].set(title="(d) total thrust [mN]", xlabel="t [s]", ylabel="mN")
    for a in ax.ravel():
        a.legend(fontsize=8)
    fig.suptitle("Exp 14 - Crazyflie 2.0 figure-8 tracking: LQR vs linear MPC",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
