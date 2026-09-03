"""Experiment 03 - PID stabilizes an open-loop unstable second-order plant.

    y_ddot - omega0^2 y = u + d        state = [y, y_dot]

Open-loop poles at s = +/- omega0 (one in the right half plane). We compare
P-only, PD, PID (no anti-windup) and PID (+ conditional-integration anti-windup)
under a unit step reference and a step input disturbance at t = 5 s, with the
actuator saturated at +/- u_max.

Run:  python experiments/03_pid_stabilizes_unstable/run.py
Outputs (written next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aimct.benchmarks.metrics import compute_all_metrics
from aimct.controllers import PID
from aimct.plot_style import plot_benchmark_comparison
from aimct.simulate import simulate
from aimct.systems import LinearSystem

HERE = Path(__file__).parent

# --- configuration (mirrors config.yaml) ----------------------------------
OMEGA0 = 2.0
U_MAX = 10.0
SETPOINT = 1.0
D_MAG, D_T = 0.5, 5.0
WN, ZETA, KI, N = 4.0, 0.7071, 15.0, 10.0
DT, T_FINAL = 1e-3, 10.0

KP = WN**2 + OMEGA0**2          # 20.0
KD = 2.0 * ZETA * WN           # ~5.657
TAU_D = KD / (N * KP)          # ~0.0283


def make_plant() -> LinearSystem:
    A = np.array([[0.0, 1.0], [OMEGA0**2, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])          # measured output = y
    return LinearSystem(A, B, C)


def disturbance(t: float) -> np.ndarray:
    return np.array([D_MAG]) if t >= D_T else np.array([0.0])


def controllers() -> dict[str, PID]:
    return {
        "P-only": PID(kp=KP, setpoint=SETPOINT),
        "PD": PID(kp=KP, kd=KD, tau_d=TAU_D, setpoint=SETPOINT),
        "PID (no AW)": PID(kp=KP, ki=KI, kd=KD, tau_d=TAU_D, setpoint=SETPOINT),
        "PID + AW": PID(
            kp=KP, ki=KI, kd=KD, tau_d=TAU_D, setpoint=SETPOINT,
            output_limits=(-U_MAX, U_MAX),
        ),
    }


def main() -> None:
    plant = make_plant()
    x0 = np.array([0.0, 0.0])
    rows: list[dict] = []
    trajectories: dict[str, dict[str, np.ndarray]] = {}

    for name, ctrl in controllers().items():
        traj = simulate(
            plant, ctrl, x0=x0, dt=DT, t_final=T_FINAL,
            u_bounds=(-U_MAX, U_MAX), input_disturbance=disturbance,
        )
        y, u = traj.x[:, 0], traj.u[:, 0]
        m = compute_all_metrics(traj.t, y, u, target=SETPOINT, u_limit=U_MAX)
        m["controller"] = name
        rows.append(m)
        trajectories[name] = {"state": traj.x, "input": u}

    _write_tables(rows)
    _write_figure(traj.t, trajectories)
    print(f"wrote {HERE/'table.md'}, {HERE/'table.csv'}, {HERE/'figure.png'}")
    print(open(HERE / "table.md").read())


def _write_tables(rows: list[dict]) -> None:
    cols = [
        "controller", "rise_time", "settling_time", "peak_overshoot_pct",
        "steady_state_error", "iae", "itae", "control_energy", "peak_control",
        "saturation_pct",
    ]
    with open(HERE / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    def fmt(v):
        if isinstance(v, float):
            return "inf" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    (HERE / "table.md").write_text(
        "# Experiment 03 - PID vs unstable second-order plant\n\n"
        + "\n".join(lines) + "\n"
    )


def _write_figure(t: np.ndarray, trajectories: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.full_like(t, SETPOINT)
    fig, axes = plot_benchmark_comparison(
        t, ref, trajectories,
        title="Exp 03 - Stabilizing an unstable plant with PID (step + disturbance @ 5 s)",
        state_label=r"Output $y(t)$ [rad]",
        control_label=r"Control torque $u(t)$ [N$\cdot$m]",
        u_limits=(-U_MAX, U_MAX),
    )
    # P-only diverges under saturation on this RHP plant; clamp the shared axes
    # so the stabilising controllers stay legible (P-only leaves the frame).
    axes = np.asarray(axes).ravel()
    axes[0].set_ylim(-0.5, 2.6)          # output tracking
    axes[2].set_ylim(-1.6, 1.2)          # tracking error
    axes[3].set_xlim(-0.5, 2.6)          # phase portrait: y
    axes[3].set_ylim(-8.0, 8.0)          # phase portrait: y_dot
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
