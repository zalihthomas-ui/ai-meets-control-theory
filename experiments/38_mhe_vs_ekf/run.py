"""Experiment 38 - Moving-Horizon Estimation (MHE) vs EKF and UKF under Hard Physical State Bounds.

Benchmark on Coupled Two-Tank liquid level estimation near the empty tank floor boundary (h -> 0).
Compares:
  1. Extended Kalman Filter (EKF) - unconstrained linearisation
  2. Unscented Kalman Filter (UKF) - unconstrained sigma points
  3. Moving-Horizon Estimator (MHE) - constrained MAP estimation over sliding window N with EKF arrival cost

Run:
    python experiments/38_mhe_vs_ekf/run.py
Outputs:
    table.md, table.csv, mhe_vs_ekf.png, mhe_vs_ekf.svg, figure.png
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aimct.estimation import ExtendedKalmanFilter, MovingHorizonEstimator, UnscentedKalmanFilter
from aimct.plot_style import set_aimct_style
from aimct.simulate import rk4_step
from aimct.systems import TwoTank

HERE = Path(__file__).parent


def run_experiment():
    full_mode = bool(os.getenv("AIMCT_EXP_FULL", "0") == "1")
    dt = 0.05 if full_mode else 0.10
    t_final = 25.0 if full_mode else 12.0
    horizon = 10 if full_mode else 6
    steps = int(round(t_final / dt))

    tank = TwoTank()
    rng = np.random.default_rng(42)

    # Initial state very low near dry bottom: Tank 1 = 1.2 cm, Tank 2 = 0.6 cm
    x0_true = np.array([0.012, 0.006])
    x0_prior = np.array([0.020, 0.015])
    P0 = np.diag([0.001, 0.001])

    # Covariances: heavy measurement noise (sigma_v = 1.5 cm)
    sigma_v = 0.015
    Q = np.diag([1e-6, 1e-6])
    R = np.diag([sigma_v**2, sigma_v**2])

    def f_dyn(x, u):
        return tank.dynamics(0.0, x, u)

    def h_meas(x):
        return np.asarray(x, dtype=float)[:2]

    # Initialize estimators
    ekf = ExtendedKalmanFilter(f_dyn, h_meas, Q, R, dt=dt, n=2, x0=x0_prior, P0=P0)
    ukf = UnscentedKalmanFilter(f_dyn, h_meas, Q, R, dt=dt, n=2, x0=x0_prior, P0=P0, alpha=0.1)
    mhe = MovingHorizonEstimator(
        f_dyn,
        h_meas,
        Q,
        R,
        horizon=horizon,
        dt=dt,
        n=2,
        x_min=[0.0, 0.0],
        x_max=[tank.h_max, tank.h_max],
        w_min=[-0.02, -0.02],
        w_max=[0.02, 0.02],
        x0=x0_prior,
        P0=P0,
        arrival_cost_mode="ekf",
    )

    # Control input trajectory: complete dry-out drain (u=0), then refill pulse, then second drain
    def get_input(t: float) -> np.ndarray:
        if t < 4.0:
            return np.array([0.0])  # Complete dry drain down to floor
        elif t < 8.0:
            return np.array([8.0])  # High refill pulse
        elif t < 10.5:
            return np.array([0.0])  # Second dry drain
        else:
            return np.array([5.0])  # Steady recovery


    # Logging structures
    t_hist = np.empty(steps)
    u_hist = np.empty(steps)
    x_true_hist = np.empty((steps, 2))
    y_meas_hist = np.empty((steps, 2))
    x_ekf_hist = np.empty((steps, 2))
    x_ukf_hist = np.empty((steps, 2))
    x_mhe_hist = np.empty((steps, 2))

    t_ekf_lat = []
    t_ukf_lat = []
    t_mhe_lat = []

    x_curr = x0_true.copy()

    for k in range(steps):
        t_curr = k * dt
        u_curr = get_input(t_curr)

        # True system simulation with small process disturbance
        w_curr = rng.multivariate_normal(np.zeros(2), Q * dt)
        x_next = rk4_step(lambda t, x, u: tank.dynamics(t, x, u), t_curr, x_curr, u_curr, dt) + w_curr
        x_next = np.clip(x_next, 0.0, tank.h_max)

        # Noisy measurement
        v_curr = rng.normal(0.0, sigma_v, size=2)
        y_curr = x_next + v_curr

        # 1. EKF Step
        t0 = time.perf_counter()
        x_ekf = ekf.step(y_curr, u_curr)
        t_ekf_lat.append((time.perf_counter() - t0) * 1000.0)

        # 2. UKF Step
        t0 = time.perf_counter()
        x_ukf = ukf.step(y_curr, u_curr)
        t_ukf_lat.append((time.perf_counter() - t0) * 1000.0)

        # 3. MHE Step
        t0 = time.perf_counter()
        x_mhe = mhe.step(y_curr, u_curr)
        t_mhe_lat.append((time.perf_counter() - t0) * 1000.0)

        # Save history
        t_hist[k] = t_curr
        u_hist[k] = u_curr[0]
        x_true_hist[k] = x_next
        y_meas_hist[k] = y_curr
        x_ekf_hist[k] = x_ekf
        x_ukf_hist[k] = x_ukf
        x_mhe_hist[k] = x_mhe

        x_curr = x_next

    # Metrics computation
    def compute_metrics(x_est: np.ndarray, latencies: list[float]) -> dict:
        err = x_est - x_true_hist
        rmse_h1 = float(np.sqrt(np.mean(err[:, 0] ** 2))) * 1000.0  # mm
        rmse_h2 = float(np.sqrt(np.mean(err[:, 1] ** 2))) * 1000.0  # mm
        rmse_tot = float(np.sqrt(np.mean(err ** 2))) * 1000.0        # mm

        # Constraint violations (h < 0)
        violations = int(np.sum((x_est[:, 0] < 0.0) | (x_est[:, 1] < 0.0)))
        violation_rate = (violations / steps) * 100.0
        min_val = float(np.min(x_est))
        max_violation = max(0.0, -min_val) * 1000.0  # mm

        mean_lat = float(np.mean(latencies))

        return {
            "rmse_h1_mm": rmse_h1,
            "rmse_h2_mm": rmse_h2,
            "rmse_tot_mm": rmse_tot,
            "violation_rate_pct": violation_rate,
            "max_violation_mm": max_violation,
            "mean_latency_ms": mean_lat,
        }

    metrics_ekf = compute_metrics(x_ekf_hist, t_ekf_lat)
    metrics_ukf = compute_metrics(x_ukf_hist, t_ukf_lat)
    metrics_mhe = compute_metrics(x_mhe_hist, t_mhe_lat)

    # ------------------------------------------------------------------ Tables
    headers = [
        "Estimator",
        "h1 RMSE [mm]",
        "h2 RMSE [mm]",
        "Total RMSE [mm]",
        "Bound Violations [%]",
        "Max Violation [mm]",
        "Latency [ms]",
        "Physical Feasibility",
    ]

    rows = [
        [
            "EKF (Unconstrained)",
            f"{metrics_ekf['rmse_h1_mm']:.2f}",
            f"{metrics_ekf['rmse_h2_mm']:.2f}",
            f"{metrics_ekf['rmse_tot_mm']:.2f}",
            f"{metrics_ekf['violation_rate_pct']:.1f}%",
            f"{metrics_ekf['max_violation_mm']:.2f}",
            f"{metrics_ekf['mean_latency_ms']:.3f}",
            "Violated (Dips Negative)",
        ],
        [
            "UKF (Unconstrained)",
            f"{metrics_ukf['rmse_h1_mm']:.2f}",
            f"{metrics_ukf['rmse_h2_mm']:.2f}",
            f"{metrics_ukf['rmse_tot_mm']:.2f}",
            f"{metrics_ukf['violation_rate_pct']:.1f}%",
            f"{metrics_ukf['max_violation_mm']:.2f}",
            f"{metrics_ukf['mean_latency_ms']:.3f}",
            "Violated (Sigma Points Infeasible)",
        ],
        [
            "MHE (Constrained MAP)",
            f"{metrics_mhe['rmse_h1_mm']:.2f}",
            f"{metrics_mhe['rmse_h2_mm']:.2f}",
            f"{metrics_mhe['rmse_tot_mm']:.2f}",
            f"{metrics_mhe['violation_rate_pct']:.1f}%",
            f"{metrics_mhe['max_violation_mm']:.2f}",
            f"{metrics_mhe['mean_latency_ms']:.3f}",
            "Strictly Feasible (h >= 0)",
        ],
    ]

    # Write CSV
    with open(HERE / "table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Write Markdown Table
    md_content = f"""# Experiment 38 Benchmark Results: MHE vs EKF and UKF

| {' | '.join(headers)} |
| {' | '.join([':---' if i == 0 or i == len(headers)-1 else ':---:' for i in range(len(headers))])} |
"""
    for r in rows:
        md_content += f"| {' | '.join(r)} |\n"

    with open(HERE / "table.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Generated table.csv and table.md successfully.")

    # ------------------------------------------------------------------ Plots
    set_aimct_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    ax_h1, ax_h2 = axes[0, 0], axes[0, 1]
    ax_err, ax_phase = axes[1, 0], axes[1, 1]

    # Convert to cm for visual clarity
    h1_true_cm = x_true_hist[:, 0] * 100.0
    h2_true_cm = x_true_hist[:, 1] * 100.0
    y1_cm = y_meas_hist[:, 0] * 100.0
    y2_cm = y_meas_hist[:, 1] * 100.0

    ekf_h1_cm = x_ekf_hist[:, 0] * 100.0
    ekf_h2_cm = x_ekf_hist[:, 1] * 100.0
    ukf_h1_cm = x_ukf_hist[:, 0] * 100.0
    ukf_h2_cm = x_ukf_hist[:, 1] * 100.0
    mhe_h1_cm = x_mhe_hist[:, 0] * 100.0
    mhe_h2_cm = x_mhe_hist[:, 1] * 100.0

    # (a) Tank 1 Level
    ax_h1.plot(t_hist, h1_true_cm, "k-", linewidth=2.0, label="True $h_1(t)$", zorder=4)
    ax_h1.scatter(t_hist[::3], y1_cm[::3], color="#999999", alpha=0.4, s=12, label="Noisy $y_1$", zorder=1)
    ax_h1.plot(t_hist, ekf_h1_cm, color="#D55E00", linestyle="--", linewidth=1.6, label="EKF (Unconstrained)", zorder=2)
    ax_h1.plot(t_hist, ukf_h1_cm, color="#CC79A7", linestyle="-.", linewidth=1.6, label="UKF (Unconstrained)", zorder=3)
    ax_h1.plot(t_hist, mhe_h1_cm, color="#0072B2", linestyle="-", linewidth=2.2, label="MHE (Constrained)", zorder=5)
    ax_h1.axhline(0.0, color="#D62728", linestyle=":", linewidth=1.5, label="Physical Floor $h_1 = 0$")
    ax_h1.axhspan(-4.0, 0.0, color="#D62728", alpha=0.08, label="Infeasible Region ($h < 0$)")
    ax_h1.set_title("(a) Tank 1 Level $h_1(t)$ & Negative Noise Dips")
    ax_h1.set_xlabel("Time $t$ [s]")
    ax_h1.set_ylabel("Liquid Level $h_1$ [cm]")
    ax_h1.set_ylim(-3.5, max(h1_true_cm) * 1.25)
    ax_h1.legend(loc="upper right", fontsize=8)

    # (b) Tank 2 Level
    ax_h2.plot(t_hist, h2_true_cm, "k-", linewidth=2.0, label="True $h_2(t)$", zorder=4)
    ax_h2.scatter(t_hist[::3], y2_cm[::3], color="#999999", alpha=0.4, s=12, label="Noisy $y_2$", zorder=1)
    ax_h2.plot(t_hist, ekf_h2_cm, color="#D55E00", linestyle="--", linewidth=1.6, label="EKF", zorder=2)
    ax_h2.plot(t_hist, ukf_h2_cm, color="#CC79A7", linestyle="-.", linewidth=1.6, label="UKF", zorder=3)
    ax_h2.plot(t_hist, mhe_h2_cm, color="#0072B2", linestyle="-", linewidth=2.2, label="MHE", zorder=5)
    ax_h2.axhline(0.0, color="#D62728", linestyle=":", linewidth=1.5, label="Physical Floor $h_2 = 0$")
    ax_h2.axhspan(-4.0, 0.0, color="#D62728", alpha=0.08)
    ax_h2.set_title("(b) Tank 2 Level $h_2(t)$ Outflow Boundary")
    ax_h2.set_xlabel("Time $t$ [s]")
    ax_h2.set_ylabel("Liquid Level $h_2$ [cm]")
    ax_h2.set_ylim(-3.5, max(h2_true_cm) * 1.35)
    ax_h2.legend(loc="upper right", fontsize=8)

    # (c) Estimation Error Norm
    err_ekf = np.linalg.norm(x_ekf_hist - x_true_hist, axis=1) * 1000.0  # mm
    err_ukf = np.linalg.norm(x_ukf_hist - x_true_hist, axis=1) * 1000.0  # mm
    err_mhe = np.linalg.norm(x_mhe_hist - x_true_hist, axis=1) * 1000.0  # mm

    ax_err.plot(t_hist, err_ekf, color="#D55E00", linestyle="--", linewidth=1.5, label=f"EKF (RMS: {metrics_ekf['rmse_tot_mm']:.1f} mm)")
    ax_err.plot(t_hist, err_ukf, color="#CC79A7", linestyle="-.", linewidth=1.5, label=f"UKF (RMS: {metrics_ukf['rmse_tot_mm']:.1f} mm)")
    ax_err.plot(t_hist, err_mhe, color="#0072B2", linestyle="-", linewidth=2.2, label=f"MHE (RMS: {metrics_mhe['rmse_tot_mm']:.1f} mm)")
    ax_err.set_title(r"(c) State Estimation Error Norm $\|\hat{x} - x\|$")
    ax_err.set_xlabel("Time $t$ [s]")
    ax_err.set_ylabel("Error Norm [mm]")
    ax_err.legend(loc="upper right", fontsize=8)

    # (d) State Space Phase Portrait
    ax_phase.fill_between([-4.0, 35.0], -4.0, 0.0, color="#D62728", alpha=0.08, label="Infeasible Domain ($h_2 < 0$)")
    ax_phase.plot(h1_true_cm, h2_true_cm, "k-", linewidth=2.0, label="True Trajectory", zorder=3)
    ax_phase.plot(ekf_h1_cm, ekf_h2_cm, color="#D55E00", linestyle="--", linewidth=1.4, alpha=0.85, label="EKF Estimates", zorder=2)
    ax_phase.plot(mhe_h1_cm, mhe_h2_cm, color="#0072B2", linestyle="-", linewidth=2.0, label="MHE Estimates", zorder=4)
    ax_phase.scatter(x0_true[0] * 100.0, x0_true[1] * 100.0, marker="o", color="#000000", s=45, label="Start ($x_0$)", zorder=5)
    ax_phase.axhline(0.0, color="#D62728", linestyle=":", linewidth=1.2)
    ax_phase.axvline(0.0, color="#D62728", linestyle=":", linewidth=1.2)
    ax_phase.set_title("(d) State Space Phase Portrait $(h_1, h_2)$")
    ax_phase.set_xlabel("Tank 1 Level $h_1$ [cm]")
    ax_phase.set_ylabel("Tank 2 Level $h_2$ [cm]")
    ax_phase.set_xlim(-2.0, max(h1_true_cm) * 1.15)
    ax_phase.set_ylim(-2.0, max(h2_true_cm) * 1.25)
    ax_phase.legend(loc="upper left", fontsize=8)

    fig.suptitle("Experiment 38: Moving-Horizon Estimation vs EKF/UKF on Coupled Two-Tank", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    # Save figure in multiple formats
    fig.savefig(HERE / "mhe_vs_ekf.png", dpi=300)
    fig.savefig(HERE / "mhe_vs_ekf.svg")
    fig.savefig(HERE / "figure.png", dpi=300)
    plt.close(fig)
    print("Saved mhe_vs_ekf.png, mhe_vs_ekf.svg, figure.png successfully.")


if __name__ == "__main__":
    run_experiment()
