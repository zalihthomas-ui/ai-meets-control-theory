"""Experiment 24 - gradient vs sampling for online nonlinear MPC.

Closes the Exp-21 remark that the cross-entropy sampling MPC is "loose and
slow". Same two nonlinear plants, same horizon and cost, two receding-horizon
planners:

* **Sampling MPC (CEM)** - derivative-free: each step it refines a Gaussian over
  action sequences by keeping the elite rollouts (Exp 10 / 20 / 21).
* **iLQR / RTI-NMPC** - gradient-based: a regularised backward Riccati sweep +
  line-searched forward rollout. The first step runs a full solve; every later
  step runs a single iteration from the shifted warm start (Diehl's real-time
  iteration) and applies ``u0 + K0 (x - x0)``.

Tasks
-----
1. **Cart-pole swing-up** - a genuine nonlinear OCP solved *online*: from
   hanging, drive the pole upright and hold it, input-limited.
2. **Quadrotor figure-8 tracking** - the Exp-14 lemniscate on the Crazyflie
   2.0, no obstacle.

We score tracking / task error, control effort, and - the point of the
experiment - the **per-step wall-clock latency**, since an online NMPC has a
hard real-time budget.

Run:   python experiments/24_ilqr_vs_sampling_mpc/run.py
       AIMCT_EXP_FULL=1 python experiments/24_ilqr_vs_sampling_mpc/run.py   # committed artifacts
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from aimct.controllers import ILQR, SamplingMPC
from aimct.controllers.lqr import solve_care
from aimct.ml.planning import system_step
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.simulate import simulate
from aimct.systems import CartPole, PlanarQuadrotor

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"
SEED = 0
DT = 0.02


# ======================================================================
# timing wrapper - counts every update() call's wall-clock
# ======================================================================
class Timed:
    def __init__(self, ctrl):
        self.c = ctrl
        self.name = getattr(ctrl, "name", type(ctrl).__name__)
        self.dt_ms: list[float] = []

    def reset(self):
        self.dt_ms.clear()
        self.c.reset()

    def update(self, x, dt):
        t0 = time.perf_counter()
        u = self.c.update(x, dt)
        self.dt_ms.append((time.perf_counter() - t0) * 1e3)
        return u

    def stats(self):
        a = np.asarray(self.dt_ms[1:] or self.dt_ms)   # drop the cold first call
        return float(np.median(a)), float(np.percentile(a, 95)), float(self.dt_ms[0])


# ======================================================================
# Task 1 - cart-pole swing-up (online receding horizon)
# ======================================================================
def swingup_task():
    cp = CartPole()
    H = 60                       # fixed: FULL scales the CEM population, not the task
    x_up = np.zeros(4)
    Q = np.diag([1.0, 0.5, 4.0, 0.2])
    Qf = np.diag([25.0, 5.0, 120.0, 12.0])
    R = np.array([[0.03]])
    F_MAX = 20.0
    x0 = np.array([0.0, 0.0, np.pi, 0.0])
    T_FINAL = 4.0

    step = system_step(cp, DT)

    def running_cost(X, U, h):
        th = X[:, 2]
        e = np.stack([X[:, 0], X[:, 1], np.arctan2(np.sin(th), np.cos(th)), X[:, 3]], 1)
        return np.einsum("bi,ij,bj->b", e, Q, e) + R[0, 0] * U[:, 0] ** 2

    def terminal_cost(X):
        th = X[:, 2]
        e = np.stack([X[:, 0], X[:, 1], np.arctan2(np.sin(th), np.cos(th)), X[:, 3]], 1)
        return np.einsum("bi,ij,bj->b", e, Qf, e)

    cem = SamplingMPC(step, running_cost, terminal_cost=terminal_cost, horizon=H,
                      n_samples=800 if FULL else 500, n_elite=60 if FULL else 40,
                      n_iter=5 if FULL else 4, u_dim=1, u_bounds=(-F_MAX, F_MAX),
                      seed=SEED)
    cem.name = "Sampling MPC (CEM)"

    ilqr = ILQR.from_system(cp, DT, horizon=H, Q=Q, R=R, Qf=Qf,
                            x_ref=x_up, u_bounds=(-F_MAX, F_MAX),
                            warm_iters=400 if FULL else 250, rti_iters=1,
                            max_iter=400)
    ilqr.name = "iLQR / RTI-NMPC"

    out = {}
    for ctrl in (Timed(cem), Timed(ilqr)):
        ctrl.reset()
        tr = simulate(cp, ctrl, x0=x0, dt=DT, t_final=T_FINAL, u_bounds=(-F_MAX, F_MAX))
        th_err = np.abs(np.arctan2(np.sin(tr.x[:, 2]), np.cos(tr.x[:, 2])))
        upright = np.where(th_err < 0.15)[0]
        t_up = float(tr.t[upright[0]]) if upright.size else np.nan
        held = bool(upright.size and np.all(th_err[upright[0]:] < 0.35))
        med, p95, cold = ctrl.stats()
        out[ctrl.name] = dict(
            trajectory=tr,
            time_to_upright_s=t_up,
            held_upright=held,
            final_angle_deg=float(np.degrees(th_err[-1])),
            peak_force_N=float(np.max(np.abs(tr.u))),
            rms_force_N=float(np.sqrt(np.mean(tr.u ** 2))),
            lat_median_ms=med, lat_p95_ms=p95, lat_cold_ms=cold,
        )
    return out, dict(t_final=T_FINAL, F_MAX=F_MAX, H=H)


# ======================================================================
# Task 2 - quadrotor figure-8 tracking
# ======================================================================
def figure8_task():
    quad = PlanarQuadrotor()
    G, M, IYY, L = quad.g, quad.m, quad.Iyy, quad.l
    T_MAX = quad.thrust_max
    UH = quad.u_hover
    AX, BZ, PERIOD, Z0 = 0.55, 0.30, 6.0, 1.0
    W = 2 * np.pi / PERIOD
    H = 20
    T_FINAL = 12.0

    Q = np.diag([6.0, 6.0, 0.5, 0.2, 0.2, 0.05])
    R = np.diag([40.0, 40.0])
    # terminal cost = infinite-horizon cost-to-go of the hover linearisation
    # (both planners get the same Qf, so the short horizon is not the variable)
    A_hov, B_hov = quad.linearize()
    Qf = solve_care(A_hov, B_hov, Q, R)

    def reference(t):
        s1, c1 = np.sin(W * t), np.cos(W * t)
        s2, c2 = np.sin(2 * W * t), np.cos(2 * W * t)
        x, xd, xdd = AX * s1, AX * W * c1, -AX * W ** 2 * s1
        xddd, xdddd = -AX * W ** 3 * c1, AX * W ** 4 * s1
        z, zd, zdd = Z0 + BZ * s2, 2 * BZ * W * c2, -4 * BZ * W ** 2 * s2
        th, thd, thdd = -xdd / G, -xddd / G, -xdddd / G
        uref = np.array([0.5 * M * (G + zdd) + 0.5 * IYY * thdd / L,
                         0.5 * M * (G + zdd) - 0.5 * IYY * thdd / L])
        return (np.array([x, z, th, xd, zd, thd]), np.clip(uref, 0.0, T_MAX))

    step = system_step(quad, DT)

    def running_cost(X, U, h):
        # SamplingMPC increments .k each control step; h is the horizon offset
        t = cem.k * DT + h * DT
        e = X - reference(t)[0]
        return (np.einsum("bi,ij,bj->b", e, Q, e)
                + np.einsum("bi,ij,bj->b", U - UH, R, U - UH))

    def terminal_cost(X):
        e = X - reference(cem.k * DT + H * DT)[0]
        return np.einsum("bi,ij,bj->b", e, Qf, e)

    cem = SamplingMPC(step, running_cost, terminal_cost=terminal_cost, horizon=H,
                      n_samples=500 if FULL else 300, n_elite=50 if FULL else 30,
                      n_iter=4 if FULL else 3, u_dim=2, u_bounds=(0.0, T_MAX),
                      seed=SEED)
    cem.name = "Sampling MPC (CEM)"

    ilqr = ILQR.from_system(
        quad, DT, horizon=H, Q=Q, R=R, Qf=Qf, u_bounds=(0.0, T_MAX),
        x_ref=lambda t: reference(t)[0], u_ref=lambda t: reference(t)[1],
        warm_iters=60, rti_iters=1, max_iter=60)
    ilqr.name = "iLQR / RTI-NMPC"

    x0 = reference(0.0)[0]
    ts = np.arange(0.0, T_FINAL + DT, DT)
    ref = np.array([reference(t)[0] for t in ts])

    out = {}
    for ctrl in (Timed(cem), Timed(ilqr)):
        ctrl.reset()
        tr = simulate(quad, ctrl, x0=x0, dt=DT, t_final=T_FINAL, u_bounds=(0.0, T_MAX))
        n = min(len(tr.t), len(ref))
        pos_err = np.hypot(tr.x[:n, 0] - ref[:n, 0], tr.x[:n, 1] - ref[:n, 1])
        du = tr.u - UH
        med, p95, cold = ctrl.stats()
        out[ctrl.name] = dict(
            trajectory=tr, reference=ref,
            rms_pos_err_mm=float(np.sqrt(np.mean(pos_err ** 2)) * 1e3),
            max_pos_err_mm=float(np.max(pos_err) * 1e3),
            ctrl_energy=float(np.trapezoid(np.sum(du ** 2, axis=1), tr.t)),
            lat_median_ms=med, lat_p95_ms=p95, lat_cold_ms=cold,
        )
    return out, dict(t_final=T_FINAL, H=H, real_time_budget_ms=DT * 1e3)


# ======================================================================
# report
# ======================================================================
def main():
    su, su_meta = swingup_task()
    f8, f8_meta = figure8_task()

    lines = ["# Experiment 24 - iLQR / RTI-NMPC vs Sampling MPC (CEM)", ""]

    lines += ["## Task 1 - cart-pole swing-up (online receding horizon)", "",
              f"horizon {su_meta['H']} steps ({su_meta['H'] * DT:.2f} s), "
              f"|F| <= {su_meta['F_MAX']:.0f} N, {su_meta['t_final']:.0f} s run", ""]
    c1 = ["time_to_upright_s", "held_upright", "final_angle_deg", "peak_force_N",
          "rms_force_N", "lat_median_ms", "lat_p95_ms", "lat_cold_ms"]
    lines += ["| controller | " + " | ".join(c1) + " |",
              "| --- |" + " --- |" * len(c1)]
    for name, m in su.items():
        lines.append("| " + name + " | " + " | ".join(
            (str(m[c]) if c == "held_upright" else f"{m[c]:.4g}") for c in c1) + " |")

    lines += ["", "## Task 2 - quadrotor figure-8 tracking (Crazyflie 2.0)", "",
              f"horizon {f8_meta['H']} steps, {f8_meta['t_final']:.0f} s, "
              f"real-time budget {f8_meta['real_time_budget_ms']:.0f} ms/step", ""]
    c2 = ["rms_pos_err_mm", "max_pos_err_mm", "ctrl_energy",
          "lat_median_ms", "lat_p95_ms", "lat_cold_ms"]
    lines += ["| controller | " + " | ".join(c2) + " |",
              "| --- |" + " --- |" * len(c2)]
    for name, m in f8.items():
        lines.append("| " + name + " | " + " | ".join(f"{m[c]:.4g}" for c in c2) + " |")

    lines += ["", "_median / p95 latency exclude the cold first solve, reported "
              "separately as `lat_cold_ms`._", ""]

    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "controller", *c1, *c2])
        for name, m in su.items():
            w.writerow(["swingup", name, *(m[c] for c in c1), *([""] * len(c2))])
        for name, m in f8.items():
            w.writerow(["figure8", name, *([""] * len(c1)), *(m[c] for c in c2)])

    _figure(su, f8)
    print((HERE / "table.md").read_text())


def _figure(su, f8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cyc = [PALETTE["mpc"], PALETTE["lqr"]]

    # (a) swing-up: pole angle vs time
    for i, (name, m) in enumerate(su.items()):
        tr = m["trajectory"]
        th = np.degrees(np.arctan2(np.sin(tr.x[:, 2]), np.cos(tr.x[:, 2])))
        ax[0].plot(tr.t, th, color=cyc[i], lw=1.7, label=name)
    ax[0].axhline(0.0, ls="--", color=PALETTE["reference"], lw=1.0)
    ax[0].set(title="(a) cart-pole swing-up: pole angle", xlabel="t [s]",
              ylabel="angle from upright [deg]")
    ax[0].legend(fontsize=8)

    # (b) figure-8 path
    ref = next(iter(f8.values()))["reference"]
    ax[1].plot(ref[:, 0], ref[:, 1], "--", color=PALETTE["reference"], lw=1.3,
               label="reference")
    for i, (name, m) in enumerate(f8.items()):
        tr = m["trajectory"]
        ax[1].plot(tr.x[:, 0], tr.x[:, 1], color=cyc[i], lw=1.6, label=name)
    ax[1].set(title="(b) quadrotor figure-8", xlabel="x [m]", ylabel="z [m]")
    ax[1].set_aspect("equal", "box")
    ax[1].legend(fontsize=8)

    # (c) per-step latency (log) - both tasks
    labels, med, p95, colors = [], [], [], []
    for i, (name, m) in enumerate(su.items()):
        labels.append("swing-up\n" + name.split("(")[0].split("/")[0].strip())
        med.append(m["lat_median_ms"]); p95.append(m["lat_p95_ms"])
        colors.append(cyc[i])
    for i, (name, m) in enumerate(f8.items()):
        labels.append("figure-8\n" + name.split("(")[0].split("/")[0].strip())
        med.append(m["lat_median_ms"]); p95.append(m["lat_p95_ms"])
        colors.append(cyc[i])
    xs = np.arange(len(labels))
    ax[2].bar(xs, med, color=colors, alpha=0.85)
    ax[2].errorbar(xs, med, yerr=[np.zeros(len(med)), np.array(p95) - np.array(med)],
                   fmt="none", ecolor="0.3", capsize=3, lw=1.0)
    ax[2].axhline(DT * 1e3, ls=":", color=PALETTE["saturation"], lw=1.4,
                  label=f"{DT * 1e3:.0f} ms real-time budget")
    ax[2].set_yscale("log")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(labels, fontsize=7)
    ax[2].set(title="(c) per-step latency (bar=median, whisker=p95)",
              ylabel="ms / control step")
    ax[2].legend(fontsize=8)

    fig.suptitle("Exp 24 - gradient (iLQR/RTI) vs sampling (CEM) for online NMPC",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
