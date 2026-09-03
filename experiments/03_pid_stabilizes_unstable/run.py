"""Experiment 03 - PID stabilizes an open-loop unstable second-order plant.

    y_ddot - omega0^2 y = u + d        state = [y, y_dot]

Open-loop poles at s = +/- omega0 (one in the right half plane). We compare
P-only, PD, PID (no anti-windup) and PID (+ conditional-integration anti-windup)
under a unit step reference and a step input disturbance at t = 5 s, with the
actuator saturated at +/- u_max.

Run:  python experiments/03_pid_stabilizes_unstable/run.py
Outputs (written next to this file): table.md, table.csv, metrics_full.csv, figure.png

Everything below the config block is delegated to
``aimct.benchmarks.compare`` - the standard comparison harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks import compare
from aimct.controllers import PID
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
    result = compare(
        make_plant(),
        controllers(),
        x0=np.zeros(2),
        dt=DT,
        t_final=T_FINAL,
        reference=SETPOINT,
        disturbance=disturbance,
        u_bounds=(-U_MAX, U_MAX),
        title="Exp 03 - Stabilizing an unstable plant with PID (step + disturbance @ 5 s)",
    )

    (HERE / "table.md").write_text(
        "# Experiment 03 - PID vs unstable second-order plant\n\n"
        + result.to_markdown()
        + "\n"
        + result.summary(),
        encoding="utf-8", newline="\n",
    )
    (HERE / "table.csv").write_text(result.to_csv(), encoding="utf-8", newline="\n")
    (HERE / "metrics_full.csv").write_text(
        result.full_metrics_csv(), encoding="utf-8", newline="\n"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = result.figure(
        state_label=r"Output $y(t)$ [rad]",
        control_label=r"Control torque $u(t)$ [N$\cdot$m]",
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

    print(f"wrote {HERE/'table.md'}, {HERE/'table.csv'}, "
          f"{HERE/'metrics_full.csv'}, {HERE/'figure.png'}")
    print((HERE / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
