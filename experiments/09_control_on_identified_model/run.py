"""Experiment 09 - design an LQR on a data-identified cart-pole model.

No equations of motion: excite the true nonlinear cart-pole in *closed loop*
(a mild stabiliser + PRBS, which keeps the data near upright), identify a linear
(A, B) from the noisy rollout, design an LQR on the *identified* model, and run
that LQR on the true nonlinear plant.

For each data length we run several noise realisations (Monte-Carlo) so the
trend - and the closed-loop-identification bias - is not a single fluke. The
4-panel figure shows the median run per data length against the exact-model LQR.

Run:  python experiments/09_control_on_identified_model/run.py
Outputs (next to this file): table.md, table.csv, metrics_full.csv, figure.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks import compare
from aimct.benchmarks.metrics import rmse, settling_time
from aimct.controllers import LQR
from aimct.simulate import simulate
from aimct.sysid import least_squares_id, model_mismatch, to_continuous
from aimct.systems import CartPole

HERE = Path(__file__).parent

DT = 0.002
F_MAX = 20.0
DATA_LENGTHS = [300, 1500, 12000]
N_SEEDS = 6
PRBS_AMP, PRBS_HOLD = 8.0, 10
MEAS_NOISE = np.array([3e-3, 6e-3, 3e-3, 6e-3])   # sensor noise std [m, m/s, rad, rad/s]

Q_DATA = np.diag([1.0, 1.0, 10.0, 1.0])
R_DATA = np.array([[1.0]])
Q_TASK = np.diag([10.0, 1.0, 100.0, 10.0])
R_TASK = np.array([[0.1]])
X0_TASK = np.array([0.0, 0.0, 0.10, 0.0])
T_FINAL = 6.0


def collect_closed_loop_data(plant, n_steps, rng):
    A0, B0 = plant.linearize()
    K_mild = LQR(A0, B0, Q_DATA, R_DATA).K
    raw = rng.uniform(-PRBS_AMP, PRBS_AMP, size=(n_steps // PRBS_HOLD + 1, 1))
    prbs = np.repeat(raw, PRBS_HOLD, axis=0)[:n_steps]
    k = {"i": 0}

    def ctrl(y, _dt):
        i = min(k["i"], n_steps - 1)
        k["i"] += 1
        return (-K_mild @ y + prbs[i]).ravel()

    traj = simulate(plant, ctrl, x0=rng.normal(scale=0.05, size=4),
                    dt=DT, t_final=n_steps * DT, u_bounds=(-F_MAX, F_MAX))
    return traj.x + rng.normal(scale=MEAS_NOISE, size=traj.x.shape), traj.u[:-1]


def eval_on_true_plant(plant, K):
    """Closed-loop angle RMSE + settle time of u = -K x on the true nonlinear plant."""
    from aimct.controllers import StateFeedback
    sf = StateFeedback(K)
    traj = simulate(plant, sf, x0=X0_TASK, dt=DT, t_final=T_FINAL, u_bounds=(-F_MAX, F_MAX))
    if traj.diverged or not np.all(np.isfinite(traj.x)):
        return np.inf, np.inf, traj
    th = traj.x[:, 2]
    return (rmse(traj.t, th, 0.0), settling_time(traj.t, th, 0.0), traj)


def main() -> None:
    plant = CartPole()
    A_true, B_true = plant.linearize()
    K_true = LQR(A_true, B_true, Q_TASK, R_TASK).K

    rows = []
    median_models = {}   # data_len -> K of the median-RMSE run, for the figure
    for n in DATA_LENGTHS:
        recs = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(1000 * s + n)
            X, U = collect_closed_loop_data(plant, n, rng)
            A_id, B_id = to_continuous(*least_squares_id(X, U), DT)
            mm = model_mismatch(A_true, B_true, A_id, B_id)
            try:
                K_id = LQR(A_id, B_id, Q_TASK, R_TASK, check_controllable=False).K
                r, ts, _ = eval_on_true_plant(plant, K_id)
            except Exception:
                K_id, r, ts = None, np.inf, np.inf
            recs.append((mm["A_rel_fro"],
                         float(np.linalg.norm((K_id if K_id is not None else 0*K_true)
                                              - K_true) / np.linalg.norm(K_true)),
                         r, ts, K_id))
        A_err = np.median([x[0] for x in recs])
        K_err = np.median([x[1] for x in recs])
        rmses = np.array([x[2] for x in recs])
        n_stable = int(np.sum(np.isfinite(rmses) & (rmses < 1.0)))
        rmse_med = float(np.median(rmses[np.isfinite(rmses)])) if np.any(np.isfinite(rmses)) else np.inf
        rows.append((n, A_err, K_err, rmse_med, n_stable, N_SEEDS))
        # median run for the figure (among stable ones if possible)
        order = sorted(recs, key=lambda x: (not np.isfinite(x[2]), x[2]))
        median_models[n] = order[len(order) // 2][4]

    controllers = {"LQR (true model)": LQR(A_true, B_true, Q_TASK, R_TASK)}
    from aimct.controllers import StateFeedback
    for n in DATA_LENGTHS:
        K = median_models[n]
        if K is not None:
            controllers[f"LQR (identified, {n} steps)"] = StateFeedback(K)

    result = compare(
        plant, controllers,
        x0=X0_TASK, dt=DT, t_final=T_FINAL,
        reference=0.0, u_bounds=(-F_MAX, F_MAX),
        output_index=2, deriv_index=3,
        title="Exp 09 - LQR on a data-identified cart-pole model vs the true model",
    )

    id_tbl = [
        "", f"## Identification vs data length "
        f"(closed-loop LS, sensor noise std {MEAS_NOISE.tolist()}, "
        f"{N_SEEDS} seeds, medians)", "",
        "| data [steps] | A rel-Fro err | K_id error (rel) | closed-loop RMSE(theta) | stable runs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for n, ae, ke, rm, ns, tot in rows:
        rm_s = "inf" if not np.isfinite(rm) else f"{rm:.4f}"
        id_tbl.append(f"| {n} ({n*DT:.0f} s) | {ae:.2e} | {ke:.3f} | {rm_s} | {ns}/{tot} |")

    (HERE / "table.md").write_text(
        "# Experiment 09 - control on an identified model\n\n"
        + result.to_markdown() + "\n" + result.summary() + "\n"
        + "\n".join(id_tbl) + "\n",
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
    axes[0].set_ylim(-0.3, 0.3)
    axes[2].set_ylim(-0.3, 0.3)
    axes[3].set_xlim(-0.3, 0.3)      # phase portrait: the 300-step run spins the
    axes[3].set_ylim(-1.2, 1.2)      # pole to ~60 rad and leaves the frame
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)

    import sys
    sys.stdout.reconfigure(errors="replace")
    print(f"wrote outputs in {HERE}")
    print((HERE / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
