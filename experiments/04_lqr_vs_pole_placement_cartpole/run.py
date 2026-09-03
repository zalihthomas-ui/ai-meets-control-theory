"""Experiment 04 - LQR vs pole placement vs PID on the nonlinear cart-pole.

Balance the pole from theta0 = 0.1 rad. Four controllers, identical conditions:

  * pole placement  - Ackermann, poles set to LQR-set-1's closed-loop eigenvalues
  * LQR (balanced)  - Q = diag(10, 1, 100, 10),  R = 0.1
  * LQR (soft)      - Q = diag(1, 0.1, 10, 1),    R = 1.0
  * PID (angle only)- single loop on theta; nothing controls the cart position

The linear controllers are designed on CartPole.linearize() (upright) and run on
the true nonlinear dynamics. Reference: docs/references/cartpole-lqr-reference.md.

Run:  python experiments/04_lqr_vs_pole_placement_cartpole/run.py
Outputs (next to this file): table.md, table.csv, metrics_full.csv, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks import compare
from aimct.controllers import LQR, PID, StateFeedback
from aimct.controllers.state_feedback import place_poles
from aimct.systems import CartPole

HERE = Path(__file__).parent

X0 = np.array([0.0, 0.0, 0.10, 0.0])
F_MAX = 20.0
X_RAIL = 2.4
DT, T_FINAL = 0.002, 6.0

Q_BAL = np.diag([10.0, 1.0, 100.0, 10.0])
R_BAL = np.array([[0.1]])
Q_SOFT = np.diag([1.0, 0.1, 10.0, 1.0])
R_SOFT = np.array([[1.0]])
PLACE_POLES = np.array([-15.6206, -3.1855, -1.3114 + 1.0795j, -1.3114 - 1.0795j])
PID_KP, PID_KD, PID_TAU_D = -180.0, -22.0, 0.02   # negated: B[3] < 0 for the pole loop


def build_controllers(plant: CartPole):
    A, B = plant.linearize()  # about upright, u = 0
    K_place = place_poles(A, B, PLACE_POLES)
    lqr_bal = LQR(A, B, Q_BAL, R_BAL)
    lqr_soft = LQR(A, B, Q_SOFT, R_SOFT)

    ctrls = {
        "Pole placement": StateFeedback(K_place),
        "LQR (balanced)": lqr_bal,
        "LQR (soft)": lqr_soft,
        "PID (angle only)": PID(kp=PID_KP, kd=PID_KD, tau_d=PID_TAU_D, setpoint=0.0),
    }
    gains = {
        "Pole placement": K_place.ravel(),
        "LQR (balanced)": lqr_bal.K.ravel(),
        "LQR (soft)": lqr_soft.K.ravel(),
    }
    return ctrls, gains


def main() -> None:
    plant = CartPole()
    ctrls, gains = build_controllers(plant)

    result = compare(
        plant,
        ctrls,
        x0=X0,
        dt=DT,
        t_final=T_FINAL,
        reference=0.0,                       # regulate pole angle to upright
        u_bounds=(-F_MAX, F_MAX),
        output_index=2,                      # theta
        deriv_index=3,                       # thetadot
        measurement_fns={"PID (angle only)": lambda t, x, u: x[[2]]},
        title="Exp 04 - Cart-pole balance: LQR vs pole placement vs PID "
              f"(theta0 = {X0[2]:.2f} rad)",
    )

    # extra context: peak cart excursion vs the physical rail
    rail_notes = []
    for name, traj in result.trajectories.items():
        cart_peak = float(np.max(np.abs(traj.x[:, 0])))
        off_rail = cart_peak > X_RAIL
        rail_notes.append(
            f"- **{name}**: peak |cart x| = {cart_peak:.3f} m "
            f"({'OFF RAIL (>' + str(X_RAIL) + ' m)' if off_rail else 'on rail'})"
        )

    gain_lines = ["", "## Designed feedback gains  K = [x, xdot, theta, thetadot]", ""]
    for name, k in gains.items():
        gain_lines.append(f"- **{name}**: [{', '.join(f'{v:.3f}' for v in k)}]")

    (HERE / "table.md").write_text(
        "# Experiment 04 - Cart-pole: LQR vs pole placement vs PID\n\n"
        + result.to_markdown() + "\n"
        + result.summary() + "\n"
        + "\n## Peak cart excursion (rail limit +/- 2.4 m)\n\n"
        + "\n".join(rail_notes) + "\n"
        + "\n".join(gain_lines) + "\n",
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
    axes[0].set_ylim(-0.25, 0.25)
    axes[2].set_ylim(-0.25, 0.25)
    axes[1].set_xlim(0.0, 1.5)   # all the control action is in the first ~1 s
    axes[0].annotate(
        "pole placement coincides with LQR (balanced)",
        xy=(0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8, style="italic", color="#444444",
    )
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)

    print(f"wrote table.md, table.csv, metrics_full.csv, figure.png in {HERE}")
    print((HERE / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
