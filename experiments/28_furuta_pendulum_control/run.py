"""
Experiment 28: Furuta Pendulum Underactuated Control & Swing-Up Benchmark.

Compares LQR, Linear MPC, and Hybrid Energy-Shaping Swing-Up on the
canonical Quanser QUBE-Servo 2 Rotary Inverted Pendulum.
"""

from __future__ import annotations

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aimct.benchmarks.metrics import (
    control_energy,
    peak_control,
    peak_overshoot,
    rise_time,
    rmse,
    saturation_duty_cycle,
    settling_time,
    steady_state_error,
)
from aimct.controllers import LQR, LinearMPC, StateFeedback
from aimct.plot_style import set_aimct_style
from aimct.simulate import simulate
from aimct.systems.furuta_pendulum import FurutaPendulum


class FurutaHybridSwingUp:
    """Åström-Furuta Energy-Shaping Swing-Up with Hysteresis LQR Catch."""

    def __init__(
        self,
        system: FurutaPendulum,
        K_lqr: np.ndarray,
        *,
        k_energy: float = 300.0,
        kp_arm: float = 4.0,
        kd_arm: float = 1.2,
        capture_angle: float = 0.28,
        capture_rate: float = 2.5,
        release_angle: float = 0.50,
        u_max: float = 0.15,
    ) -> None:
        self.sys = system
        self.K = np.asarray(K_lqr, dtype=float)
        self.k_energy = float(k_energy)
        self.kp_arm = float(kp_arm)
        self.kd_arm = float(kd_arm)
        self.capture_angle = float(capture_angle)
        self.capture_rate = float(capture_rate)
        self.release_angle = float(release_angle)
        self.u_max = float(u_max)
        self.reset()

    def reset(self) -> None:
        self.mode = "swingup"
        self.switch_steps: list[int] = []

    def update(self, measurement: np.ndarray, dt: float) -> float:
        th, al, th_d, al_d = measurement
        al_w = float(np.arctan2(np.sin(al), np.cos(al)))

        if self.mode == "swingup":
            if abs(al_w) <= self.capture_angle and abs(al_d) <= self.capture_rate:
                self.mode = "balance"
                self.switch_steps.append(1)
        elif abs(al_w) > self.release_angle:
            self.mode = "swingup"

        if self.mode == "balance":
            xw = np.array([th, al_w, th_d, al_d])
            u = float(-(self.K @ xw)[0])
        else:
            E = self.sys.pendulum_energy(measurement)
            s = float(np.sign(al_d * np.cos(al)) or 1.0)
            a_des = self.k_energy * E * s - self.kp_arm * th - self.kd_arm * th_d
            u = self.sys.J_t * a_des + self.sys.Dr * th_d

        return float(np.clip(u, -self.u_max, self.u_max))


def main() -> None:
    set_aimct_style()
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    furuta = FurutaPendulum()
    A, B = furuta.linearize()

    # Bryson-calibrated weights for QUBE-Servo 2 RIP
    Q = np.diag([2.0, 10.0, 0.2, 0.5])
    R = np.array([[20.0]])
    u_lim = 0.15

    lqr = LQR(A, B, Q, R)
    ctrl_lqr = StateFeedback(lqr.K, x_ref=np.zeros(4))
    ctrl_mpc = LinearMPC(A, B, Q=Q, R=R, N=20, u_bounds=(-u_lim, u_lim), x_ref=np.zeros(4))
    ctrl_hybrid = FurutaHybridSwingUp(furuta, lqr.K)

    # -------------------------------------------------------------------------
    # Case A: Upright Stabilization Benchmark (alpha_0 = 0.05 rad / 2.9 deg)
    # -------------------------------------------------------------------------
    dt_a = 0.001
    t_final_a = 3.5
    x0_a = np.array([0.0, 0.05, 0.0, 0.0])

    traj_lqr = simulate(furuta, ctrl_lqr, x0=x0_a, dt=dt_a, t_final=t_final_a, u_bounds=(-u_lim, u_lim))
    traj_mpc = simulate(furuta, ctrl_mpc, x0=x0_a, dt=dt_a, t_final=t_final_a, u_bounds=(-u_lim, u_lim))

    # Metrics evaluation on Case A (output is pendulum angle alpha)
    results_rows = []
    for name, traj in [("LQR", traj_lqr), ("Linear MPC", traj_mpc)]:
        t = traj.t
        y = traj.x[:, 1]  # pendulum angle
        u = traj.u[:, 0]
        results_rows.append({
            "Controller": name,
            "Task": "Upright Regulation",
            "Rise $t_r$ [s]": f"{rise_time(t, y, 0.0):.3f}",
            "Settling $t_s$ [s]": f"{settling_time(t, y, 0.0, band=0.02):.3f}",
            "Overshoot $M_p$ [%]": f"{peak_overshoot(t, y, 0.0):.1f}",
            "Steady error $e_{ss}$ [rad]": f"{steady_state_error(t, y, 0.0):.2e}",
            "RMSE [rad]": f"{rmse(t, y, 0.0):.4f}",
            "Energy $E_u$ [N²m²s]": f"{control_energy(t, u):.4f}",
            "Peak Torque [N·m]": f"{peak_control(t, u):.4f}",
            "Saturation [%]": f"{saturation_duty_cycle(t, u, u_lim):.1f}",
            "Status": "Stable" if not traj.diverged else "Diverged",
        })

    # -------------------------------------------------------------------------
    # Case B: Full Underactuated Swing-Up & Balance (alpha_0 = pi rad)
    # -------------------------------------------------------------------------
    dt_b = 0.001
    t_final_b = 6.0
    x0_b = np.array([0.0, np.pi, 0.0, 0.01])  # Small perturbation from hanging rest

    traj_swing = simulate(furuta, ctrl_hybrid, x0=x0_b, dt=dt_b, t_final=t_final_b, u_bounds=(-u_lim, u_lim))
    al_wrapped = np.arctan2(np.sin(traj_swing.x[:, 1]), np.cos(traj_swing.x[:, 1]))
    u_swing = traj_swing.u[:, 0]

    results_rows.append({
        "Controller": "Hybrid Swing-Up + LQR",
        "Task": "Full Swing-Up (180° -> 0°)",
        "Rise $t_r$ [s]": f"{rise_time(traj_swing.t, al_wrapped, 0.0):.3f}",
        "Settling $t_s$ [s]": f"{settling_time(traj_swing.t, al_wrapped, 0.0, band=0.05):.3f}",
        "Overshoot $M_p$ [%]": f"{peak_overshoot(traj_swing.t, al_wrapped, 0.0):.1f}",
        "Steady error $e_{ss}$ [rad]": f"{steady_state_error(traj_swing.t, al_wrapped, 0.0):.2e}",
        "RMSE [rad]": f"{rmse(traj_swing.t, al_wrapped, 0.0):.4f}",
        "Energy $E_u$ [N²m²s]": f"{control_energy(traj_swing.t, u_swing):.4f}",
        "Peak Torque [N·m]": f"{peak_control(traj_swing.t, u_swing):.4f}",
        "Saturation [%]": f"{saturation_duty_cycle(traj_swing.t, u_swing, u_lim):.1f}",
        "Status": "Stable" if not traj_swing.diverged else "Diverged",
    })

    # Write CSV
    csv_file = outdir / "table.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    # Write Markdown
    headers = list(results_rows[0].keys())
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))]) + " |",
    ]
    for row in results_rows:
        md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    table_md = "\n".join(md_lines) + "\n"
    (outdir / "table.md").write_text(table_md, encoding="utf-8")

    print("=== Experiment 28 Benchmark Results ===")
    print(table_md)

    # -------------------------------------------------------------------------
    # Figure 1: Case A Upright Stabilization Comparison
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    t_a = traj_lqr.t

    # (a) Pendulum angle alpha(t)
    axes[0, 0].plot(t_a, traj_lqr.x[:, 1] * 180 / np.pi, label="LQR", color="#1f77b4", lw=2)
    axes[0, 0].plot(t_a, traj_mpc.x[:, 1] * 180 / np.pi, label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[0, 0].axhline(0, color="#333", ls=":", lw=1)
    axes[0, 0].set_ylabel("Pendulum Angle $\\alpha$ [deg]")
    axes[0, 0].set_title("(a) Pendulum Deflection", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (b) Rotary arm angle theta(t)
    axes[0, 1].plot(t_a, traj_lqr.x[:, 0] * 180 / np.pi, label="LQR", color="#1f77b4", lw=2)
    axes[0, 1].plot(t_a, traj_mpc.x[:, 0] * 180 / np.pi, label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[0, 1].axhline(0, color="#333", ls=":", lw=1)
    axes[0, 1].set_ylabel("Rotary Arm $\\theta$ [deg]")
    axes[0, 1].set_title("(b) Rotary Arm Translation", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (c) Motor Torque tau(t)
    axes[1, 0].plot(t_a, traj_lqr.u[:, 0], label="LQR", color="#1f77b4", lw=1.8)
    axes[1, 0].plot(t_a, traj_mpc.u[:, 0], label="Linear MPC", color="#2ca02c", lw=1.8, ls="--")
    axes[1, 0].axhline(u_lim, color="#d62728", ls="--", lw=1, label="Motor Limit $\\pm 0.15$ N$\\cdot$m")
    axes[1, 0].axhline(-u_lim, color="#d62728", ls="--", lw=1)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Motor Torque $\\tau$ [N$\\cdot$m]")
    axes[1, 0].set_title("(c) Actuator Effort", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Phase portrait alpha vs alpha_dot
    axes[1, 1].plot(traj_lqr.x[:, 1], traj_lqr.x[:, 3], label="LQR", color="#1f77b4", lw=1.8)
    axes[1, 1].plot(traj_mpc.x[:, 1], traj_mpc.x[:, 3], label="Linear MPC", color="#2ca02c", lw=1.8, ls="--")
    axes[1, 1].plot([0], [0], "r*", ms=10, label="Upright Equilibrium")
    axes[1, 1].set_xlabel("Pendulum Angle $\\alpha$ [rad]")
    axes[1, 1].set_ylabel("Angular Velocity $\\dot{\\alpha}$ [rad/s]")
    axes[1, 1].set_title("(d) Pendulum Phase Portrait", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 28: Furuta Pendulum Upright Regulation Benchmark", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "furuta_benchmark.png", dpi=300)
    fig.savefig(outdir / "furuta_benchmark.svg")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Case B Full Energy Swing-Up and Catch
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 7))
    t_b = traj_swing.t

    # (a) Pendulum angle swing-up
    ax2[0, 0].plot(t_b, al_wrapped * 180 / np.pi, color="#9467bd", lw=2, label="Pendulum $\\alpha$")
    ax2[0, 0].axhline(0, color="#333", ls=":", lw=1)
    ax2[0, 0].set_ylabel("Angle $\\alpha$ [deg]")
    ax2[0, 0].set_title("(a) Pendulum Swing-Up (180° -> 0°)", fontweight="bold")
    ax2[0, 0].legend()
    ax2[0, 0].grid(True, alpha=0.3)

    # (b) Rotary arm angle
    ax2[0, 1].plot(t_b, traj_swing.x[:, 0] * 180 / np.pi, color="#ff7f0e", lw=2, label="Rotary Arm $\\theta$")
    ax2[0, 1].axhline(0, color="#333", ls=":", lw=1)
    ax2[0, 1].set_ylabel("Arm Angle $\\theta$ [deg]")
    ax2[0, 1].set_title("(b) Rotary Arm Pumping Motion", fontweight="bold")
    ax2[0, 1].legend()
    ax2[0, 1].grid(True, alpha=0.3)

    # (c) Swing-up motor torque
    ax2[1, 0].plot(t_b, u_swing, color="#e377c2", lw=1.8, label="Torque $\\tau$")
    ax2[1, 0].axhline(u_lim, color="#d62728", ls="--", lw=1, label="Limit $\\pm 0.15$ N$\\cdot$m")
    ax2[1, 0].axhline(-u_lim, color="#d62728", ls="--", lw=1)
    ax2[1, 0].set_xlabel("Time [s]")
    ax2[1, 0].set_ylabel("Torque $\\tau$ [N$\\cdot$m]")
    ax2[1, 0].set_title("(c) Control Action", fontweight="bold")
    ax2[1, 0].legend()
    ax2[1, 0].grid(True, alpha=0.3)

    # (d) Relative Pendulum Energy E(t) -> 0
    E_t = [furuta.pendulum_energy(x) for x in traj_swing.x]
    ax2[1, 1].plot(t_b, E_t, color="#17becf", lw=2, label="Energy $E(\\alpha, \\dot{\\alpha})$")
    ax2[1, 1].axhline(0.0, color="#2ca02c", ls="--", lw=1.5, label="Upright Separatrix ($E=0$)")
    ax2[1, 1].axhline(-2.0 * furuta.mp * furuta.g * furuta.lp, color="#d62728", ls=":", label="Hanging Energy ($E_0$)")
    ax2[1, 1].set_xlabel("Time [s]")
    ax2[1, 1].set_ylabel("Mechanical Energy [J]")
    ax2[1, 1].set_title("(d) Energy Pumping & Catch", fontweight="bold")
    ax2[1, 1].legend()
    ax2[1, 1].grid(True, alpha=0.3)

    fig2.suptitle("Experiment 28: Åström-Furuta Energy Swing-Up & Hysteresis LQR Catch", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(outdir / "furuta_swingup.png", dpi=300)
    fig2.savefig(outdir / "furuta_swingup.svg")
    plt.close(fig2)

    print(f"Artifacts successfully written to {outdir}")


if __name__ == "__main__":
    main()
