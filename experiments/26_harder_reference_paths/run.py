"""Experiment 26 - does the Exp-24 winner survive a harder path?

Experiment 24 found iLQR/RTI-NMPC beating sampling MPC (CEM) by ~2 orders of
magnitude on the Exp-14 lemniscate (smooth, low-curvature, no obstacle) at
comparable compute. That is the easiest reference a figure-8 tracker sees in
this repo. This experiment re-runs the *same two planners, same cost, same
horizon* on two harder tracks:

* **Lissajous 3:2** - sharp velocity reversals at the lobes (coprime frequency
  ratio; a strictly harder stress than the 2:1 figure-8).
* **Outward spiral** - monotonically *increasing* curvature demand as the
  radius grows (the mirror image of a tightening turn).

...alongside the lemniscate baseline, recomputed here so all three rows are
directly comparable in one table.

Run:   python experiments/26_harder_reference_paths/run.py
       AIMCT_EXP_FULL=1 python experiments/26_harder_reference_paths/run.py
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
from aimct.systems import PlanarQuadrotor
from aimct.trajectories import Lemniscate, Lissajous, Spiral

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"
SEED = 0
DT = 0.02
H = 20
Z0 = 1.0                          # hover-height offset added to every path's 2nd coord

quad = PlanarQuadrotor()
G, M, IYY, L, T_MAX, UH = quad.g, quad.m, quad.Iyy, quad.l, quad.thrust_max, quad.u_hover

Q = np.diag([6.0, 6.0, 0.5, 0.2, 0.2, 0.05])
R = np.diag([40.0, 40.0])
A_hov, B_hov = quad.linearize()
Qf = solve_care(A_hov, B_hov, Q, R)

PATHS = {
    "Lemniscate (baseline)": Lemniscate(A=0.55, B=0.30, period=6.0),
    "Lissajous 3:2": Lissajous(A=0.55, B=0.35, a=3, b=2, period=8.0),
    "Spiral": Spiral(r0=0.15, growth=0.035, w=1.0, duration=12.0),
}
T_FINAL = {"Lemniscate (baseline)": 12.0, "Lissajous 3:2": 8.0, "Spiral": 12.0}


# ======================================================================
# generic differential-flatness feed-forward for any 2-D Trajectory
# ======================================================================
def flat_reference(traj, t, *, eps=2e-4):
    """``traj(t) -> (pos, vel, acc)`` in the quad's (x, z-Z0) plane -> the full
    quad state + thrust feed-forward, by differential flatness. Jerk / snap
    (needed for the pitch feed-forward) are recovered by central-differencing
    the trajectory's own analytic acceleration - generic across any
    ``Trajectory`` subclass, no extra derivatives required of it."""
    pos, vel, acc = traj(t)
    _, _, acc_p = traj(t + eps)
    _, _, acc_m = traj(t - eps)
    jerk = (acc_p - acc_m) / (2 * eps)
    snap = (acc_p - 2 * acc + acc_m) / eps ** 2

    x, xd, xdd, xddd, xdddd = pos[0], vel[0], acc[0], jerk[0], snap[0]
    z, zd, zdd = pos[1] + Z0, vel[1], acc[1]
    th, thd, thdd = -xdd / G, -xddd / G, -xdddd / G
    u_ref = np.array([0.5 * M * (G + zdd) + 0.5 * IYY * thdd / L,
                      0.5 * M * (G + zdd) - 0.5 * IYY * thdd / L])
    x_ref = np.array([x, z, th, xd, zd, thd])
    return x_ref, np.clip(u_ref, 0.0, T_MAX)


# ======================================================================
# timing wrapper (same as Exp 24)
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
        a = np.asarray(self.dt_ms[1:] or self.dt_ms)
        return float(np.median(a)), float(np.percentile(a, 95)), float(self.dt_ms[0])


# ======================================================================
# one path -> both planners
# ======================================================================
def run_path(name, traj):
    step = system_step(quad, DT)

    def running_cost(X, U, h):
        t = cem.k * DT + h * DT
        e = X - flat_reference(traj, t)[0]
        return (np.einsum("bi,ij,bj->b", e, Q, e)
                + np.einsum("bi,ij,bj->b", U - UH, R, U - UH))

    def terminal_cost(X):
        e = X - flat_reference(traj, cem.k * DT + H * DT)[0]
        return np.einsum("bi,ij,bj->b", e, Qf, e)

    cem = SamplingMPC(step, running_cost, terminal_cost=terminal_cost, horizon=H,
                      n_samples=500 if FULL else 300, n_elite=50 if FULL else 30,
                      n_iter=4 if FULL else 3, u_dim=2, u_bounds=(0.0, T_MAX),
                      seed=SEED)
    cem.name = "Sampling MPC (CEM)"

    ilqr = ILQR.from_system(
        quad, DT, horizon=H, Q=Q, R=R, Qf=Qf, u_bounds=(0.0, T_MAX),
        x_ref=lambda t: flat_reference(traj, t)[0],
        u_ref=lambda t: flat_reference(traj, t)[1],
        warm_iters=60, rti_iters=1, max_iter=60)
    ilqr.name = "iLQR / RTI-NMPC"

    t_final = T_FINAL[name]
    x0 = flat_reference(traj, 0.0)[0]
    ts = np.arange(0.0, t_final + DT, DT)
    ref = np.array([flat_reference(traj, t)[0] for t in ts])

    out = {}
    for ctrl in (Timed(cem), Timed(ilqr)):
        ctrl.reset()
        tr = simulate(quad, ctrl, x0=x0, dt=DT, t_final=t_final, u_bounds=(0.0, T_MAX))
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
    return out


# ======================================================================
# report
# ======================================================================
def main():
    results = {name: run_path(name, traj) for name, traj in PATHS.items()}

    cols = ["rms_pos_err_mm", "max_pos_err_mm", "ctrl_energy",
           "lat_median_ms", "lat_p95_ms", "lat_cold_ms"]
    lines = ["# Experiment 26 - does the Exp-24 winner survive a harder path?", "",
            f"20-step / 0.4 s horizon, {DT * 1e3:.0f} ms real-time budget, "
            f"same Q/R/Qf as Exp 24 on every path.", ""]
    for name in PATHS:
        lines += [f"## {name}", "",
                  "| controller | " + " | ".join(cols) + " |",
                  "| --- |" + " --- |" * len(cols)]
        for cname, m in results[name].items():
            lines.append("| " + cname + " | " + " | ".join(
                f"{m[c]:.4g}" for c in cols) + " |")
        lines.append("")

    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "controller", *cols])
        for name in PATHS:
            for cname, m in results[name].items():
                w.writerow([name, cname, *(m[c] for c in cols)])

    _figure(results)
    print((HERE / "table.md").read_text())


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    names = list(PATHS)
    fig, ax = plt.subplots(1, len(names) + 1, figsize=(5.2 * (len(names) + 1), 4.8))
    cyc = [PALETTE["mpc"], PALETTE["lqr"]]

    for i, name in enumerate(names):
        ref = next(iter(results[name].values()))["reference"]
        ax[i].plot(ref[:, 0], ref[:, 1], "--", color=PALETTE["reference"], lw=1.3,
                  label="reference")
        for j, (cname, m) in enumerate(results[name].items()):
            tr = m["trajectory"]
            ax[i].plot(tr.x[:, 0], tr.x[:, 1], color=cyc[j], lw=1.5, label=cname)
        ax[i].set(title=name, xlabel="x [m]", ylabel="z [m]")
        ax[i].set_aspect("equal", "box")
        ax[i].legend(fontsize=7)

    # rms error bar chart across paths x controllers
    ap = ax[-1]
    x = np.arange(len(names))
    w = 0.35
    for j, cname in enumerate(("Sampling MPC (CEM)", "iLQR / RTI-NMPC")):
        vals = [results[n][cname]["rms_pos_err_mm"] for n in names]
        ap.bar(x + (j - 0.5) * w, vals, w, color=cyc[j], label=cname)
    ap.set_yscale("log")
    ap.set_xticks(x); ap.set_xticklabels([n.split(" (")[0] for n in names], fontsize=8)
    ap.set(title="RMS tracking error by path", ylabel="mm (log)")
    ap.legend(fontsize=8)

    fig.suptitle("Exp 26 - iLQR vs CEM: does the winner survive a harder path?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
