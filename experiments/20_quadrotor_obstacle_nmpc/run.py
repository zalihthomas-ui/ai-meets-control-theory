"""Experiment 20 - obstacle-aware nonlinear MPC on the quadrotor.

Capstone step C9.2. A figure-8 with a keep-out disk on the path. LQR + flatness
feedforward is obstacle-blind; a cross-entropy sampling MPC - planning through
either the true dynamics or a learned grey-box model (RK4 hover linearisation +
residual MLP) - bends the drone around the disk while holding the trajectory.

Run:  python experiments/20_quadrotor_obstacle_nmpc/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.controllers import LQR, SamplingMPC
from aimct.controllers.lqr import solve_care
from aimct.ml import LearnedDynamics, system_step
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.simulate import simulate
from aimct.systems import PlanarQuadrotor

HERE = Path(__file__).parent
DT, DURATION, SEED = 0.02, 12.0, 0
AX, BZ, PERIOD, Z0 = 0.55, 0.30, 6.0, 1.0
W = 2 * np.pi / PERIOD
OBS_C = np.array([0.30, 1.16])
OBS_R, OBS_MARGIN = 0.16, 0.04

quad = PlanarQuadrotor()
G, M, IYY, L = quad.g, quad.m, quad.Iyy, quad.l
T_MAX = quad.thrust_max
UH = quad.u_hover
A_HOV, B_HOV = quad.linearize()

Q_TRK = np.diag([6.0, 6.0, 0.5, 0.2, 0.2, 0.05])
R_TRK = 40.0
W_OBS = 4.0e3
_K_LQR = LQR(A_HOV, B_HOV,
             np.diag(1 / np.array([.1, .1, .2, .5, .5, 3.]) ** 2),
             np.diag(1 / np.array([.15, .15]) ** 2)).K
P_CARE = solve_care(A_HOV, B_HOV, Q_TRK, np.diag([R_TRK, R_TRK]))


def reference(t):
    s1, c1 = np.sin(W * t), np.cos(W * t)
    s2, c2 = np.sin(2 * W * t), np.cos(2 * W * t)
    x, xd, xdd = AX * s1, AX * W * c1, -AX * W**2 * s1
    xddd, xdddd = -AX * W**3 * c1, AX * W**4 * s1
    z, zd, zdd = Z0 + BZ * s2, 2 * BZ * W * c2, -4 * BZ * W**2 * s2
    th, thd, thdd = -xdd / G, -xddd / G, -xdddd / G
    u_ref = np.array([0.5 * M * (G + zdd) + 0.5 * IYY * thdd / L,
                      0.5 * M * (G + zdd) - 0.5 * IYY * thdd / L])
    return np.array([x, z, th, xd, zd, thd]), np.clip(u_ref, 0.0, T_MAX)


def hover_rk4(X, U, dt=DT):
    X = np.atleast_2d(np.asarray(X, float))
    U = np.atleast_2d(np.asarray(U, float))
    f = lambda Xs: Xs @ A_HOV.T + (U - UH) @ B_HOV.T
    k1 = f(X); k2 = f(X + 0.5 * dt * k1); k3 = f(X + 0.5 * dt * k2); k4 = f(X + dt * k3)
    return X + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


TRUE_STEP = system_step(quad, DT)


def collect_flight_data(n_steps):
    tk = {"k": 0}

    def ctl(x, dt):
        xr, ur = reference(tk["k"] * DT)
        tk["k"] += 1
        return np.clip(ur - _K_LQR @ (np.asarray(x) - xr), 0.0, T_MAX)

    tr = simulate(quad, ctl, x0=reference(0)[0], dt=DT, t_final=n_steps * DT,
                  u_bounds=(0.0, T_MAX))
    return tr.x, tr.u[:-1]


class LqrFF:
    name = "LQR + flatness feedforward"

    def __init__(self):
        self._t = 0.0

    def reset(self):
        self._t = 0.0

    def update(self, x, dt):
        xr, ur = reference(self._t)
        self._t += dt
        return np.clip(ur - _K_LQR @ (np.asarray(x) - xr), 0.0, T_MAX)


def build_mpc(step_fn):
    """SamplingMPC with a trajectory-tracking + obstacle running cost that reads
    the planner's live control-step count (mpc.k) for the reference time."""
    mpc_ref = {}

    def rc(X, U, h):
        t = mpc_ref["mpc"].k * DT + h * DT
        xr, _ = reference(t)
        e = X - xr
        trk = np.einsum("bi,ij,bj->b", e, Q_TRK, e)
        eff = R_TRK * np.sum((U - UH) ** 2, axis=1)
        d2 = (X[:, 0] - OBS_C[0]) ** 2 + (X[:, 1] - OBS_C[1]) ** 2
        pen = W_OBS * np.maximum(0.0, (OBS_R + OBS_MARGIN) ** 2 - d2) ** 2
        return trk + eff + pen

    def term(X):
        t = mpc_ref["mpc"].k * DT + 20 * DT
        e = X - reference(t)[0]
        return np.einsum("bi,ij,bj->b", e, P_CARE, e)

    mpc = SamplingMPC(step_fn, rc, terminal_cost=term, horizon=20, n_samples=400,
                      n_elite=40, n_iter=3, u_dim=2, u_bounds=(0.0, T_MAX), seed=SEED)
    mpc_ref["mpc"] = mpc
    mpc.name = "sampling-mpc"
    return mpc


def metrics(tr):
    t = tr.t
    ref = np.array([reference(tt)[0] for tt in t])
    pos_err = np.hypot(tr.x[:, 0] - ref[:, 0], tr.x[:, 1] - ref[:, 1])
    clr = np.hypot(tr.x[:, 0] - OBS_C[0], tr.x[:, 1] - OBS_C[1]) - OBS_R
    du = tr.u - UH
    return {
        "rms_pos_err_mm": float(np.sqrt(np.mean(pos_err ** 2)) * 1e3),
        "min_clearance_mm": float(np.min(clr) * 1e3),
        "steps_in_keepout": int(np.sum(clr < 0.0)),
        "ctrl_energy": float(np.trapezoid(np.sum(du ** 2, axis=1), t)),
    }


def main():
    Xtr, Utr = collect_flight_data(3000)
    model = LearnedDynamics(6, 2, hidden=(48, 48), base_step=hover_rk4, seed=0)
    hist = model.fit(Xtr, Utr, epochs=350, lr=3e-3)
    pe = model.prediction_error(*collect_flight_data(600), horizon=20)

    ctrls = {
        "LQR + flatness feedforward": LqrFF(),
        "SamplingMPC (true model)": build_mpc(TRUE_STEP),
        "SamplingMPC (learned model)": build_mpc(model.step),
    }

    x0 = reference(0)[0]
    rows, trajs = {}, {}
    for name, c in ctrls.items():
        tr = simulate(quad, c, x0=x0, dt=DT, t_final=DURATION, u_bounds=(0.0, T_MAX))
        rows[name] = metrics(tr)
        trajs[name] = tr

    cols = ["rms_pos_err_mm", "min_clearance_mm", "steps_in_keepout", "ctrl_energy"]
    lines = ["# Experiment 20 - obstacle-aware nonlinear MPC (quadrotor)", "",
             f"Learned grey-box model 20-step prediction error: {pe:.3f}", "",
             "| controller | " + " | ".join(cols) + " |",
             "| --- |" + " --- |" * len(cols)]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.4g}" for c in cols) + " |")
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
    cyc = [PALETTE["lqr"], PALETTE["mpc"], PALETTE["state_feedback"]]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].add_patch(plt.Circle(OBS_C, OBS_R, color=PALETTE["saturation"], alpha=0.25))
    ax[0].add_patch(plt.Circle(OBS_C, OBS_R, color=PALETTE["saturation"], fill=False, lw=1.5))
    ax[0].plot(ref[:, 0], ref[:, 1], "--", color=PALETTE["reference"], lw=1.4, label="reference")
    for i, (name, tr) in enumerate(trajs.items()):
        ax[0].plot(tr.x[:, 0], tr.x[:, 1], color=cyc[i % 3], lw=1.8, label=name)
        pe = np.hypot(tr.x[:, 0] - ref[:, 0], tr.x[:, 1] - ref[:, 1]) * 1e3
        ax[1].plot(t, pe, color=cyc[i % 3], lw=1.4, label=name)
    ax[0].set(title="(a) trajectory (keep-out disk in red)", xlabel="x [m]", ylabel="z [m]")
    ax[0].set_aspect("equal", "box"); ax[0].legend(fontsize=8)
    ax[1].set(title="(b) position error [mm]", xlabel="t [s]", ylabel="mm")
    ax[1].legend(fontsize=8)
    fig.suptitle("Exp 20 - obstacle-aware sampling MPC vs obstacle-blind LQR "
                 "(Crazyflie 2.0)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
