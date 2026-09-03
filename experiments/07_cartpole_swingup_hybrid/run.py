"""Experiment 07 - Cart-pole swing-up from hanging + hybrid handoff to LQR.

From the stable downward rest (theta0 = pi) an energy-shaping swing-up
(`EnergyShapingSwingUp`, Spong partial feedback linearisation) pumps the
pendulum toward the upright separatrix; a hysteresis switch
(`HybridSwingUpLQR`) hands control to a balancing LQR once the state enters the
capture window, and hands back if the pole is lost.

We run three energy-pump gains k_E and compare the time/energy cost of the
swing-up, plus a figure showing the pendulum energy climbing to zero and the
mode switch.

Run:  python experiments/07_cartpole_swingup_hybrid/run.py
Outputs (next to this file): table.md, table.csv, metrics_full.csv,
  swingup_energy.png, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks import compare
from aimct.controllers import EnergyShapingSwingUp, HybridSwingUpLQR, LQR, wrap_angle
from aimct.systems import CartPole

HERE = Path(__file__).parent

X0 = np.array([0.0, 0.0, np.pi, 0.0])       # hanging, at rest
F_MAX = 20.0
DT, T_FINAL = 2e-3, 10.0
PUMP_GAINS = {"k_E = 6": 6.0, "k_E = 10": 10.0, "k_E = 14": 14.0}

# balancing LQR (reference "balanced" tuning)
_CP = CartPole()
_A, _B = _CP.linearize()
_LQR_QR = (np.diag([10.0, 1.0, 100.0, 10.0]), np.array([[0.1]]))


def build_controllers() -> dict[str, HybridSwingUpLQR]:
    out = {}
    for name, k in PUMP_GAINS.items():
        su = EnergyShapingSwingUp(_CP, k_energy=k, u_max=F_MAX)
        lqr = LQR(_A, _B, *_LQR_QR)
        out[name] = HybridSwingUpLQR(su, lqr)
    return out


def swingup_metrics(name: str, hyb: HybridSwingUpLQR, traj) -> dict:
    t = traj.t
    thw = np.array([wrap_angle(a) for a in traj.x[:, 2]])
    su = hyb.swingup
    E = np.array([su.pendulum_energy(x) for x in traj.x])

    switch = hyb.switch_steps
    t_capture = float(t[switch[0]]) if switch else float("inf")

    # settled = |wrap(theta)| stays < 2 deg for the rest of the run
    out_of_band = np.where(np.abs(thw) > np.radians(2.0))[0]
    t_settle = float(t[out_of_band[-1] + 1]) if out_of_band.size and out_of_band[-1] + 1 < len(t) else (
        0.0 if not out_of_band.size else float("inf"))

    dt = t[1] - t[0]
    return {
        "controller": name,
        "t_capture_s": round(t_capture, 3),
        "t_settle_s": round(t_settle, 3),
        "n_switches": len(switch),
        "control_energy": round(float(np.sum(traj.u[:, 0] ** 2) * dt), 2),
        "peak_force_N": round(float(np.max(np.abs(traj.u))), 2),
        "cart_excursion_m": round(float(np.max(np.abs(traj.x[:, 0]))), 3),
        "balanced": bool(abs(thw[-1]) < np.radians(2.0) and abs(traj.x[-1, 3]) < 0.2),
    }


def write_tables(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    (HERE / "table.csv").write_text(
        ",".join(cols) + "\n"
        + "\n".join(",".join(str(r[c]) for c in cols) for r in rows) + "\n",
        encoding="utf-8", newline="\n",
    )
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (HERE / "table.md").write_text(
        "# Experiment 07 - cart-pole swing-up + hybrid handoff\n\n"
        "From hanging (theta0 = pi), |F| <= 20 N. `t_capture` = first switch to "
        "LQR balance; `t_settle` = |wrap(theta)| < 2 deg thereafter.\n\n"
        + "\n".join([head, sep, *body]) + "\n",
        encoding="utf-8", newline="\n",
    )


def energy_figure(trajs: dict, hybs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aimct.plot_style import set_aimct_style

    set_aimct_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    (ax_th, ax_E), (ax_u, ax_x) = axes
    colors = ["#0072B2", "#D55E00", "#009E73"]

    for (name, traj), color in zip(trajs.items(), colors):
        hyb = hybs[name]
        t = traj.t
        thw = np.array([wrap_angle(a) for a in traj.x[:, 2]])
        E = np.array([hyb.swingup.pendulum_energy(x) for x in traj.x])
        ax_th.plot(t, np.degrees(thw), color=color, lw=1.8, label=name)
        ax_E.plot(t, E, color=color, lw=1.8, label=name)
        ax_u.plot(t, traj.u[:, 0], color=color, lw=1.3, label=name)
        ax_x.plot(t, traj.x[:, 0], color=color, lw=1.8, label=name)
        for s in hyb.switch_steps:
            ax_th.plot(t[s], np.degrees(thw[s]), "o", color=color, ms=7, zorder=5)
            ax_E.plot(t[s], E[s], "o", color=color, ms=7, zorder=5)

    ax_th.axhspan(-20, 20, color="#000000", alpha=0.06, label="capture window (+/-20 deg)")
    ax_th.set_title("(a) Pole angle wrap($\\theta$) - dot = handoff to LQR")
    ax_th.set_ylabel("angle [deg]")
    ax_E.axhline(0.0, color="#555555", ls="--", lw=1.2)
    ax_E.set_title("(b) Pendulum energy $E_p \\to 0$ (upright separatrix)")
    ax_E.set_ylabel("energy [J]")
    for ax in (ax_u,):
        ax.axhline(F_MAX, color="#D62728", ls=":", lw=1.3)
        ax.axhline(-F_MAX, color="#D62728", ls=":", lw=1.3)
    ax_u.set_title("(c) Cart force $u(t)$")
    ax_u.set_ylabel("force [N]")
    ax_x.set_title("(d) Cart position $x(t)$")
    ax_x.set_ylabel("position [m]")
    for ax in (ax_th, ax_E, ax_u, ax_x):
        ax.set_xlabel("time [s]")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Exp 07 - Cart-pole energy swing-up + hysteresis handoff to LQR",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "swingup_energy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    controllers = build_controllers()
    result = compare(
        _CP, controllers, x0=X0, dt=DT, t_final=T_FINAL, reference=0.0,
        output_index=2, u_bounds=(-F_MAX, F_MAX),
        title="Exp 07 - cart-pole swing-up + hybrid LQR (from hanging)",
    )

    rows = [swingup_metrics(n, controllers[n], result.trajectories[n])
            for n in result.names]
    write_tables(rows)
    (HERE / "metrics_full.csv").write_text(result.full_metrics_csv(),
                                           encoding="utf-8", newline="\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, _ = result.figure(state_label=r"Pole angle $\theta$ [rad]",
                           control_label=r"Cart force $u(t)$ [N]")
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)

    energy_figure(result.trajectories, controllers)

    print("wrote:", ", ".join(p.name for p in sorted(HERE.glob("*"))
                              if p.suffix in {".md", ".csv", ".png"}))
    print()
    print((HERE / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
