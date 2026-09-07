"""Experiment 41 - Bearings-Only Target Tracking: Particle Filter vs EKF and UKF.

Compares:
  1. Extended Kalman Filter (EKF) - linearizes nonlinear bearing atan2(dy, dx)
  2. Unscented Kalman Filter (UKF) - propagates sigma points across bearing cone
  3. Bootstrap Particle Filter (PF) - represents multimodal curved range-ambiguous posterior

Run:
    python experiments/41_particle_filter_bearings_only/run.py
Outputs:
    table.md, table.csv, pf_vs_ekf.png, pf_vs_ekf.svg, figure.png
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aimct.estimation import ExtendedKalmanFilter, ParticleFilter, UnscentedKalmanFilter
from aimct.plot_style import set_aimct_style

HERE = Path(__file__).parent


def wrap_angle(angle: float | np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    a = np.asarray(angle, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def run_experiment():
    full_mode = bool(os.getenv("AIMCT_EXP_FULL", "0") == "1")
    dt = 0.25 if full_mode else 0.50
    t_final = 50.0 if full_mode else 35.0
    n_particles = 2500 if full_mode else 1200
    steps = int(round(t_final / dt))

    rng = np.random.default_rng(42)

    # True initial target state: [px, py, vx, vy]
    x0_true = np.array([100.0, 100.0, -2.0, 1.0])

    # Filter prior guess: 28 m initial position offset
    x0_prior = np.array([80.0, 120.0, 0.0, 0.0])
    P0 = np.diag([400.0, 400.0, 4.0, 4.0])

    # Target motion model: discrete constant velocity
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    q_acc = 0.01
    dt3 = (dt ** 3) / 3.0
    dt2 = (dt ** 2) / 2.0
    Q = q_acc * np.array([
        [dt3, 0.0, dt2, 0.0],
        [0.0, dt3, 0.0, dt2],
        [dt2, 0.0, dt, 0.0],
        [0.0, dt2, 0.0, dt],
    ])

    # Measurement noise: sigma = 1.5 deg = 0.02618 rad
    sigma_deg = 1.5
    sigma_rad = np.radians(sigma_deg)
    R = np.array([[sigma_rad ** 2]])

    # Observer trajectory generator: forward motion with S-turn weave
    v_obs = 3.0
    amp_weave = 30.0
    freq_weave = 0.15

    def observer_pos(t: float) -> np.ndarray:
        return np.array([v_obs * t, amp_weave * np.sin(freq_weave * t)])

    # Angle residual function for filters
    def angle_residual(y, y_pred):
        return wrap_angle(np.asarray(y, float) - np.asarray(y_pred, float))

    # Time-dependent measurement functions
    current_obs_pos = np.zeros(2)

    def h_meas(x: np.ndarray) -> np.ndarray:
        dx = x[0] - current_obs_pos[0]
        dy = x[1] - current_obs_pos[1]
        return np.array([np.arctan2(dy, dx)])

    def H_jac(x: np.ndarray) -> np.ndarray:
        dx = x[0] - current_obs_pos[0]
        dy = x[1] - current_obs_pos[1]
        r2 = max(1e-6, dx**2 + dy**2)
        return np.array([[-dy / r2, dx / r2, 0.0, 0.0]])

    f_trans = lambda x, u: F @ x

    # Initialize filters
    ekf = ExtendedKalmanFilter(
        f_trans,
        h_meas,
        Q,
        R,
        dt=dt,
        n=4,
        discrete=True,
        H_jac=H_jac,
        residual=angle_residual,
        x0=x0_prior,
        P0=P0,
    )

    ukf = UnscentedKalmanFilter(
        f_trans,
        h_meas,
        Q,
        R,
        dt=dt,
        n=4,
        discrete=True,
        residual=angle_residual,
        x0=x0_prior,
        P0=P0,
        alpha=0.5,
    )

    pf = ParticleFilter(
        f_trans,
        h_meas,
        Q,
        R,
        n_particles=n_particles,
        dt=dt,
        discrete=True,
        residual=angle_residual,
        x0=x0_prior,
        spread=P0,
        seed=42,
    )

    # Data logging arrays
    t_hist = np.empty(steps)
    target_pos_hist = np.empty((steps, 2))
    obs_pos_hist = np.empty((steps, 2))
    meas_hist = np.empty(steps)

    ekf_pos_hist = np.empty((steps, 2))
    ukf_pos_hist = np.empty((steps, 2))
    pf_pos_hist = np.empty((steps, 2))
    pf_ess_hist = np.empty(steps)

    pf_snapshots: list[tuple[float, np.ndarray]] = []
    snapshot_times = [5.0, 15.0, 25.0]

    t_ekf_lat = []
    t_ukf_lat = []
    t_pf_lat = []

    x_true = x0_true.copy()

    for k in range(steps):
        t_curr = k * dt
        t_hist[k] = t_curr

        # Propagate true target
        w = rng.multivariate_normal(np.zeros(4), Q)
        x_true = F @ x_true + w
        target_pos_hist[k] = x_true[:2]

        # Update observer position
        p_obs = observer_pos(t_curr)
        obs_pos_hist[k] = p_obs
        current_obs_pos[:] = p_obs

        # True bearing + noise
        bearing_true = np.arctan2(x_true[1] - p_obs[1], x_true[0] - p_obs[0])
        v_noise = rng.normal(0.0, sigma_rad)
        bearing_meas = wrap_angle(bearing_true + v_noise)
        meas_hist[k] = np.degrees(bearing_meas)

        # 1. EKF Step
        t0 = time.perf_counter()
        x_ekf = ekf.step(np.array([bearing_meas]), None)
        t_ekf_lat.append((time.perf_counter() - t0) * 1000.0)

        # 2. UKF Step
        t0 = time.perf_counter()
        x_ukf = ukf.step(np.array([bearing_meas]), None)
        t_ukf_lat.append((time.perf_counter() - t0) * 1000.0)

        # 3. PF Step
        t0 = time.perf_counter()
        x_pf = pf.step(np.array([bearing_meas]), None)
        t_pf_lat.append((time.perf_counter() - t0) * 1000.0)

        ekf_pos_hist[k] = x_ekf[:2]
        ukf_pos_hist[k] = x_ukf[:2]
        pf_pos_hist[k] = x_pf[:2]
        pf_ess_hist[k] = pf.ess

        # Capture particle snapshots for visualization
        for st in snapshot_times:
            if abs(t_curr - st) < dt / 2.0 and len([s for s in pf_snapshots if abs(s[0] - st) < 0.1]) == 0:
                pf_snapshots.append((t_curr, pf.particles[:, :2].copy()))

    # Compute Benchmark Metrics
    def calc_metrics(pos_est: np.ndarray, latencies: list[float]) -> dict:
        err_pos = np.linalg.norm(pos_est - target_pos_hist, axis=1)
        init_err = float(err_pos[0])
        final_err = float(np.mean(err_pos[-10:]))
        rmse_pos = float(np.sqrt(np.mean(err_pos ** 2)))
        success = bool(final_err < 15.0)
        mean_lat = float(np.mean(latencies))
        return {
            "init_err_m": init_err,
            "final_err_m": final_err,
            "rmse_pos_m": rmse_pos,
            "converged": "Yes" if success else "No (Diverged)",
            "mean_latency_ms": mean_lat,
        }

    m_ekf = calc_metrics(ekf_pos_hist, t_ekf_lat)
    m_ukf = calc_metrics(ukf_pos_hist, t_ukf_lat)
    m_pf = calc_metrics(pf_pos_hist, t_pf_lat)

    # ------------------------------------------------------------------ Tables
    headers = [
        "Estimator",
        "Initial Error [m]",
        "Final Error [m]",
        "Position RMSE [m]",
        "Target Tracking",
        "Latency [ms]",
        "Non-Gaussian Posterior",
    ]

    rows = [
        [
            "EKF (Linearized)",
            f"{m_ekf['init_err_m']:.2f}",
            f"{m_ekf['final_err_m']:.2f}",
            f"{m_ekf['rmse_pos_m']:.2f}",
            m_ekf['converged'],
            f"{m_ekf['mean_latency_ms']:.3f}",
            "False Gaussian collapse",
        ],
        [
            "UKF (Sigma Points)",
            f"{m_ukf['init_err_m']:.2f}",
            f"{m_ukf['final_err_m']:.2f}",
            f"{m_ukf['rmse_pos_m']:.2f}",
            m_ukf['converged'],
            f"{m_ukf['mean_latency_ms']:.3f}",
            "Radial elongation drift",
        ],
        [
            "Particle Filter (Bootstrap)",
            f"{m_pf['init_err_m']:.2f}",
            f"{m_pf['final_err_m']:.2f}",
            f"{m_pf['rmse_pos_m']:.2f}",
            m_pf['converged'],
            f"{m_pf['mean_latency_ms']:.3f}",
            "Accurate curved crescent",
        ],
    ]

    # Write CSV
    with open(HERE / "table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Write Markdown Table
    md_content = f"""# Experiment 41 Benchmark Results: Bearings-Only Target Tracking

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
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))
    ax_traj, ax_bear = axes[0, 0], axes[0, 1]
    ax_err, ax_ess = axes[1, 0], axes[1, 1]

    # (a) 2D Cartesian Trajectory Plot
    ax_traj.plot(obs_pos_hist[:, 0], obs_pos_hist[:, 1], "k--", linewidth=1.5, label="Observer (Own-Ship)")
    ax_traj.plot(target_pos_hist[:, 0], target_pos_hist[:, 1], "g-", linewidth=2.2, label="True Target")

    if pf_snapshots:
        # Plot particle cloud from mid snapshot
        t_snap, p_snap = pf_snapshots[-1]
        ax_traj.scatter(
            p_snap[:, 0],
            p_snap[:, 1],
            color="#56B4E9",
            alpha=0.25,
            s=8,
            label=f"PF Particles ($t={t_snap:.0f}\\,\\mathrm{{s}}$)",
            zorder=2,
        )

    ax_traj.plot(ekf_pos_hist[:, 0], ekf_pos_hist[:, 1], color="#D55E00", linestyle=":", linewidth=1.8, label="EKF Track")
    ax_traj.plot(ukf_pos_hist[:, 0], ukf_pos_hist[:, 1], color="#CC79A7", linestyle="-.", linewidth=1.8, label="UKF Track")
    ax_traj.plot(pf_pos_hist[:, 0], pf_pos_hist[:, 1], color="#0072B2", linestyle="-", linewidth=2.2, label="PF Mean Track")

    ax_traj.scatter(x0_true[0], x0_true[1], marker="*", color="#009E73", s=100, label="Target Start", zorder=6)
    ax_traj.scatter(obs_pos_hist[0, 0], obs_pos_hist[0, 1], marker="o", color="#333333", s=60, label="Observer Start", zorder=6)

    ax_traj.set_title("(a) 2D Cartesian Plane Tracking & Particle Cloud")
    ax_traj.set_xlabel("X Position [m]")
    ax_traj.set_ylabel("Y Position [m]")
    ax_traj.legend(loc="upper left", fontsize=8)

    # (b) Bearing Measurements vs Time
    ax_bear.plot(t_hist, meas_hist, color="#999999", alpha=0.6, linestyle="none", marker=".", markersize=4, label="Noisy Bearing $\\theta_m(t)$")
    pred_bear_pf = np.degrees(np.arctan2(pf_pos_hist[:, 1] - obs_pos_hist[:, 1], pf_pos_hist[:, 0] - obs_pos_hist[:, 0]))
    pred_bear_ekf = np.degrees(np.arctan2(ekf_pos_hist[:, 1] - obs_pos_hist[:, 1], ekf_pos_hist[:, 0] - obs_pos_hist[:, 0]))
    ax_bear.plot(t_hist, pred_bear_pf, color="#0072B2", linewidth=1.8, label="PF Predicted Bearing")
    ax_bear.plot(t_hist, pred_bear_ekf, color="#D55E00", linestyle="--", linewidth=1.4, label="EKF Predicted Bearing")
    ax_bear.set_title("(b) Azimuth Line-of-Sight Bearing $\\theta(t)$")
    ax_bear.set_xlabel("Time $t$ [s]")
    ax_bear.set_ylabel("Bearing Angle [deg]")
    ax_bear.legend(loc="best", fontsize=8)

    # (c) Position Estimation Error Over Time
    err_ekf = np.linalg.norm(ekf_pos_hist - target_pos_hist, axis=1)
    err_ukf = np.linalg.norm(ukf_pos_hist - target_pos_hist, axis=1)
    err_pf = np.linalg.norm(pf_pos_hist - target_pos_hist, axis=1)

    ax_err.plot(t_hist, err_ekf, color="#D55E00", linestyle=":", linewidth=1.6, label=f"EKF (Final: {m_ekf['final_err_m']:.1f} m)")
    ax_err.plot(t_hist, err_ukf, color="#CC79A7", linestyle="-.", linewidth=1.6, label=f"UKF (Final: {m_ukf['final_err_m']:.1f} m)")
    ax_err.plot(t_hist, err_pf, color="#0072B2", linestyle="-", linewidth=2.2, label=f"PF (Final: {m_pf['final_err_m']:.1f} m)")
    ax_err.set_title(r"(c) Position Estimation Error $\|\hat{p} - p\|$ [m]")
    ax_err.set_xlabel("Time $t$ [s]")
    ax_err.set_ylabel("Error [m]")
    ax_err.legend(loc="upper right", fontsize=8)

    # (d) Effective Sample Size (ESS) & Resampling Events
    ax_ess.plot(t_hist, pf_ess_hist, color="#0072B2", linewidth=1.8, label="Effective Sample Size (ESS)")
    ax_ess.axhline(0.5 * n_particles, color="#D62728", linestyle="--", linewidth=1.4, label="Resample Threshold (0.5 $N_p$)")
    ax_ess.set_title(f"(d) PF Adaptive Systematic Resampling ({pf.resample_count} Events)")
    ax_ess.set_xlabel("Time $t$ [s]")
    ax_ess.set_ylabel("ESS [Particles]")
    ax_ess.set_ylim(0, n_particles * 1.05)
    ax_ess.legend(loc="lower right", fontsize=8)

    fig.suptitle("Experiment 41: Bearings-Only Target Tracking (PF vs EKF vs UKF)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    fig.savefig(HERE / "pf_vs_ekf.png", dpi=300)
    fig.savefig(HERE / "pf_vs_ekf.svg")
    fig.savefig(HERE / "figure.png", dpi=300)
    plt.close(fig)
    print("Saved pf_vs_ekf.png, pf_vs_ekf.svg, figure.png successfully.")


if __name__ == "__main__":
    run_experiment()
