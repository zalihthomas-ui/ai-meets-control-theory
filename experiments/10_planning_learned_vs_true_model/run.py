"""Experiment 10 - MPC planning with a LEARNED model vs the TRUE model.

Fit a small from-scratch MLP one-step model of the cart-pole from data, then run
a cross-entropy-method sampling MPC that plans through it. Compare against the
same planner on the true (batched analytic) model and against a plain LQR, all
balancing the true nonlinear cart-pole from theta0 = 0.2 rad.

Run:  python experiments/10_planning_learned_vs_true_model/run.py
Outputs (next to this file): table.md, table.csv, metrics_full.csv, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks import compare
from aimct.controllers import LQR, SamplingMPC
from aimct.controllers.lqr import solve_care
from aimct.ml import LearnedDynamics, system_step
from aimct.simulate import simulate
from aimct.systems import CartPole

HERE = Path(__file__).parent

DT, T_FINAL, SEED = 0.02, 5.0, 0
F_MAX = 20.0
X0 = np.array([0.0, 0.0, 0.20, 0.0])

Q_COST = np.diag([1.0, 0.1, 20.0, 1.0])
R_COST = 0.02
Q_LQR = np.diag([10.0, 1.0, 100.0, 10.0])
R_LQR = np.array([[0.1]])

# CEM: short horizon made stabilising by a CARE terminal cost x' P x
H, N_SAMPLES, N_ELITE, N_ITER = 30, 600, 60, 4


def collect_data(plant, n_steps, seed):
    A0, B0 = plant.linearize()
    K_mild = LQR(A0, B0, np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[1.0]])).K
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-8.0, 8.0, size=(n_steps // 8 + 1, 1))
    prbs = np.repeat(raw, 8, axis=0)[:n_steps]
    k = {"i": 0}

    def ctrl(y, _dt):
        i = min(k["i"], n_steps - 1)
        k["i"] += 1
        return (-K_mild @ y + prbs[i]).ravel()

    tr = simulate(plant, ctrl, x0=rng.normal(scale=0.05, size=4),
                  dt=DT, t_final=n_steps * DT, u_bounds=(-F_MAX, F_MAX))
    return tr.x, tr.u[:-1]


def running_cost(X, U):
    return np.einsum("bi,ij,bj->b", X, Q_COST, X) + R_COST * U[:, 0] ** 2


def make_terminal_cost(A, B):
    """x' P x with P from the CARE(A, B, Q_cost, R_cost) - the infinite-horizon
    cost-to-go, so a short CEM horizon still 'sees' beyond its end."""
    P = solve_care(A, B, Q_COST, np.array([[R_COST]]))
    return lambda X: np.einsum("bi,ij,bj->b", X, P, X)


def main() -> None:
    plant = CartPole()
    A_true, B_true = plant.linearize()

    # --- learn a one-step model ------------------------------------------------
    Xtr, Utr = collect_data(plant, 4000, SEED)
    Xte, Ute = collect_data(plant, 1000, SEED + 99)
    model = LearnedDynamics(4, 1, hidden=(64, 64), seed=0)
    hist = model.fit(Xtr, Utr, epochs=600, lr=3e-3, batch_size=256)
    pe1 = model.prediction_error(Xte, Ute, horizon=1)
    pe30 = model.prediction_error(Xte, Ute, horizon=30)

    true_step = system_step(plant, DT)
    learned_step = model.step
    term_true = make_terminal_cost(A_true, B_true)
    # terminal cost for the learned planner uses a model identified from the SAME
    # data (least-squares), so it never touches the true equations of motion
    from aimct.sysid import least_squares_id, to_continuous
    A_id, B_id = to_continuous(*least_squares_id(Xtr, Utr), DT)
    term_learned = make_terminal_cost(A_id, B_id)

    def mk(step, term):
        return SamplingMPC(step, running_cost, terminal_cost=term, horizon=H,
                           n_samples=N_SAMPLES, n_elite=N_ELITE, n_iter=N_ITER,
                           u_dim=1, u_bounds=(-F_MAX, F_MAX), seed=0)

    controllers = {
        "LQR (true model)": LQR(A_true, B_true, Q_LQR, R_LQR),
        "SamplingMPC (true model)": mk(true_step, term_true),
        "SamplingMPC (learned model)": mk(learned_step, term_learned),
    }

    result = compare(
        plant, controllers,
        x0=X0, dt=DT, t_final=T_FINAL,
        reference=0.0, u_bounds=(-F_MAX, F_MAX),
        output_index=2, deriv_index=3,
        title="Exp 10 - CEM-MPC: learned model vs true model vs LQR (cart-pole balance)",
    )

    note = [
        "",
        "## Learned model",
        "",
        f"- residual MLP `[5, 64, 64, 4]`, {model.net.n_params()} params, "
        f"trained on 4000 steps ({4000*DT:.0f} s); final train MSE {hist[-1]:.2e}",
        f"- held-out prediction error: 1-step {pe1:.2e}, 30-step {pe30:.2e}",
    ]
    (HERE / "table.md").write_text(
        "# Experiment 10 - planning with a learned model vs the true model\n\n"
        + result.to_markdown() + "\n" + result.summary() + "\n"
        + "\n".join(note) + "\n",
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
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)

    print(f"wrote outputs in {HERE}")
    print((HERE / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
