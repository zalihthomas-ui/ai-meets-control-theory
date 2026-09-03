"""Experiment 06 - LQG vs full-state LQR under measurement noise (cart-pole).

Four controllers balance the nonlinear cart-pole from theta0 = 0.08 rad:

  * LQR (full state, clean) - the unattainable ideal: sees all 4 true states.
  * LQG                     - Kalman filter (2 noisy encoders -> x_hat) + same K.
  * Luenberger + K          - fast pole-placed observer + same K, no noise model.
  * LQG (overconfident)     - Kalman filter told the sensors are 10x better than
                              they are (V too small) - over-trusts measurements.

Every controller is scored on the pole angle under the *same* noise realisation.
Reference: docs/references/observers-kalman-reference.md.

Run:  python experiments/06_lqg_vs_lqr_measurement_noise/run.py
Outputs (next to this file): table.md, table.csv, metrics_full.csv, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks import compare
from aimct.controllers import LQR, ObserverFeedback
from aimct.systems import CartPole

HERE = Path(__file__).parent

X0 = np.array([0.0, 0.0, 0.08, 0.0])
F_MAX = 20.0
DT, T_FINAL = 0.002, 5.0
SEED = 0

C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
SIGMA = np.array([0.005, 0.0087])                       # [m, rad]
Q = np.diag([10.0, 1.0, 100.0, 10.0])
R = np.array([[0.1]])
W = np.diag([1e-4, 1e-3, 1e-4, 1e-3])
V = np.diag(SIGMA**2)
OBS_POLES = [-20.0, -22.0, -24.0, -26.0]

# one shared noise realisation, indexed by step -> every controller sees it
_n_steps = int(round(T_FINAL / DT)) + 2
_NOISE = np.random.default_rng(SEED).normal(0.0, 1.0, size=(_n_steps, 2)) * SIGMA


def noisy_encoders(t, x, u):
    k = min(int(round(t / DT)), _n_steps - 1)
    return C @ x + _NOISE[k]


def clean_state(t, x, u):
    return x


def build_controllers(plant: CartPole):
    A, B = plant.linearize()
    K = LQR(A, B, Q, R).K

    lqg = ObserverFeedback.lqg(A, B, C, Q, R, W, V)
    luen = ObserverFeedback.luenberger(A, B, C, K, observer_poles=OBS_POLES)
    lqg_overconf = ObserverFeedback.lqg(A, B, C, Q, R, W, V / 100.0)

    ctrls = {
        "LQR (full state, clean)": LQR(A, B, Q, R),
        "LQG": lqg,
        "Luenberger + K": luen,
        "LQG (overconfident)": lqg_overconf,
    }
    meas = {
        "LQR (full state, clean)": clean_state,
        "LQG": noisy_encoders,
        "Luenberger + K": noisy_encoders,
        "LQG (overconfident)": noisy_encoders,
    }
    return ctrls, meas


def main() -> None:
    plant = CartPole()
    ctrls, meas = build_controllers(plant)

    result = compare(
        plant, ctrls,
        x0=X0, dt=DT, t_final=T_FINAL,
        reference=0.0,
        u_bounds=(-F_MAX, F_MAX),
        output_index=2, deriv_index=3,
        measurement_fns=meas,
        title=f"Exp 06 - LQG vs full-state LQR under encoder noise "
              f"(sigma_theta = {SIGMA[1]*1e3:.1f} mrad)",
    )

    (HERE / "table.md").write_text(
        "# Experiment 06 - LQG vs full-state LQR under measurement noise\n\n"
        + result.to_markdown() + "\n" + result.summary() + "\n",
        encoding="utf-8",
    )
    (HERE / "table.csv").write_text(result.to_csv(), encoding="utf-8")
    (HERE / "metrics_full.csv").write_text(result.full_metrics_csv(), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = result.figure(
        state_label=r"Pole angle $\theta(t)$ [rad]",
        control_label=r"Cart force $u(t)$ [N]",
    )
    axes = np.asarray(axes).ravel()
    axes[0].set_ylim(-0.12, 0.12)
    axes[2].set_ylim(-0.12, 0.12)
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)

    print(f"wrote table.md, table.csv, metrics_full.csv, figure.png in {HERE}")
    print((HERE / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
