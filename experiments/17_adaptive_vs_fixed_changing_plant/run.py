"""Experiment 17 - adaptive vs fixed control when the plant drifts.

The plant is a mass-spring-damper whose stiffness k ramps 1 -> 5 over 40 s.
A fixed LQR is tuned once; MRAC adapts online to a reference model without ever
identifying the new plant; a gain-scheduled LQR is the "k is measurable" option.

Run:  python experiments/17_adaptive_vs_fixed_changing_plant/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.controllers import LQR, MRAC, GainScheduledLQR
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.simulate import simulate
from aimct.systems import MassSpringDamper

HERE = Path(__file__).parent

DT, T_FINAL = 0.002, 40.0
U_BOUNDS = (-60.0, 60.0)
K0, K1 = 1.0, 5.0
M, C = 1.0, 0.4
A_M = np.array([[0.0, 1.0], [-4.0, -2.8]])
B_M = np.array([[0.0], [4.0]])
R_CMD = np.array([1.0])
XM_SS = float((-np.linalg.solve(A_M, B_M @ R_CMD))[0])       # = 1.0


class DriftingMSD(MassSpringDamper):
    """Stiffness ramps K0 -> K1 linearly over the episode."""

    def dynamics(self, t, x, u):
        self.k = K0 + (K1 - K0) * min(t / T_FINAL, 1.0)
        return super().dynamics(t, x, u)


def msd_lin(k):
    return MassSpringDamper(m=M, c=C, k=k).linearize()


def build():
    A0, B = msd_lin(K0)
    Q, Rw = np.diag([60.0, 2.0]), np.array([[1.0]])

    lqr_nom = LQR(A0, B, Q, Rw)
    lqr_nom.x_ref = np.array([XM_SS, 0.0])

    lqr_wc = LQR(*msd_lin(K1), Q, Rw)
    lqr_wc.x_ref = np.array([XM_SS, 0.0])

    mrac = MRAC(A_M, B_M, B, A_nom=A0, gamma=80.0, Q=np.diag([6.0, 1.0]),
                u_bounds=U_BOUNDS)
    mrac.r = R_CMD

    gs = GainScheduledLQR(msd_lin, np.linspace(K0, K1, 9),
                          schedule_fn=lambda x: K0, Q=Q, R=Rw,
                          x_ref=[XM_SS, 0.0])
    # schedule on the *true* current stiffness (measurable case)
    gs.schedule_fn = lambda x, _c=[0.0]: _c[0]

    return {"LQR (nominal k=1)": lqr_nom, "LQR (worst-case k=5)": lqr_wc,
            "MRAC": mrac, "GainScheduled LQR": gs}, gs


def run_one(name, ctrl, gs):
    plant = DriftingMSD(m=M, c=C, k=K0)
    if name == "GainScheduled LQR":
        # feed the scheduler the live stiffness through a mutable cell
        cell = [K0]
        gs.schedule_fn = lambda x: cell[0]

        class Wrap:
            name = "gs"
            def reset(self): ctrl.reset()
            def update(self, x, dt):
                cell[0] = plant.k
                return ctrl.update(x, dt)
        c = Wrap()
    else:
        c = ctrl
    return simulate(plant, c, x0=[0.0, 0.0], dt=DT, t_final=T_FINAL,
                    u_bounds=U_BOUNDS)


def main():
    ctrls, gs = build()
    rows, trajs = {}, {}
    for name, ctrl in ctrls.items():
        tr = run_one(name, ctrl, gs)
        trajs[name] = tr
        t, y = tr.t, tr.x[:, 0]
        err = y - XM_SS
        early = (t >= 10) & (t < 18)      # k ~ 2.0-2.7, past MRAC's adaptation transient
        late = t >= 32                    # k ~ 4.2-5.0
        rows[name] = {
            "rms_err_early": float(np.sqrt(np.mean(err[early]**2))),
            "rms_err_late": float(np.sqrt(np.mean(err[late]**2))),
            "settle_err_final": float(abs(err[-1])),
            "ctrl_energy": float(np.trapezoid(tr.u[:, 0]**2, t)),
        }

    cols = list(next(iter(rows.values())))
    lines = ["# Experiment 17 - adaptive vs fixed control, drifting plant", "",
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
    cyc = [PALETTE["lqr"], PALETTE["state_feedback"], PALETTE["mpc"], PALETTE["rl"]]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].axhline(XM_SS, ls="--", color=PALETTE["reference"], lw=1.4, label="target")
    for i, (name, tr) in enumerate(trajs.items()):
        c = cyc[i % len(cyc)]
        ax[0, 0].plot(t, tr.x[:, 0], color=c, lw=1.4, label=name)
        ax[0, 1].plot(t, tr.x[:, 0] - XM_SS, color=c, lw=1.2, label=name)
        ax[1, 0].plot(t, tr.u[:, 0], color=c, lw=1.0, label=name)
    ax[1, 1].plot(t, K0 + (K1 - K0) * np.minimum(t / T_FINAL, 1.0),
                  color="#333", lw=2)
    ax[0, 0].set(title="(a) position  y(t)", xlabel="t [s]", ylabel="y")
    ax[0, 1].set(title="(b) tracking error  y - target", xlabel="t [s]", ylabel="err")
    ax[1, 0].set(title="(c) control  u(t)", xlabel="t [s]", ylabel="u [N]")
    ax[1, 1].set(title="(d) true stiffness k(t)", xlabel="t [s]", ylabel="k [N/m]")
    for a in ax.ravel()[:3]:
        a.legend(fontsize=8)
    fig.suptitle("Exp 17 - adaptive (MRAC) vs fixed LQR on a plant that drifts "
                 "(k: 1 -> 5)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
