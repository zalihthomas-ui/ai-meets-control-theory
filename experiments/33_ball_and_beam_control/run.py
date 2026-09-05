"""
Experiment 33: Ball and Beam Underactuated Regulation & Control Benchmark.

Compares Cascade PID, Partial Feedback Linearization (PFL), Multivariable LQR,
and Constrained Linear MPC on ball position setpoint regulation and relative degree 4 dynamics.
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
from aimct.systems.ball_and_beam import BallAndBeam


class CascadePID:
    """Outer Loop Position PID + Inner Loop Beam Angle PD."""

    def __init__(
        self,
        *,
        kp_pos: float = 1.5,
        kd_pos: float = 0.8,
        kp_ang: float = 8.0,
        kd_ang: float = 1.5,
        r_target: float = 0.10,
        tau_max: float = 1.50,
    ) -> None:
        self.kp_pos = float(kp_pos)
        self.kd_pos = float(kd_pos)
        self.kp_ang = float(kp_ang)
        self.kd_ang = float(kd_ang)
        self.r_target = float(r_target)
        self.tau_max = float(tau_max)

    def reset(self) -> None:
        pass

    def update(self, measurement: np.ndarray, dt: float) -> float:
        r, r_d, th, th_d = measurement
        e_r = self.r_target - r
        th_cmd = float(np.clip(-(self.kp_pos * e_r - self.kd_pos * r_d), -0.40, 0.40))
        e_th = th_cmd - th
        tau = self.kp_ang * e_th - self.kd_ang * th_d
        return float(np.clip(tau, -self.tau_max, self.tau_max))


class PFLController:
    """Partial Feedback Linearization with gravity cancellation."""

    def __init__(
        self,
        system: BallAndBeam,
        *,
        kp: float = 8.0,
        kd: float = 4.0,
        kp_inner: float = 15.0,
        kd_inner: float = 2.0,
        r_target: float = 0.10,
        tau_max: float = 1.50,
    ) -> None:
        self.sys = system
        self.kp = float(kp)
        self.kd = float(kd)
        self.kp_inner = float(kp_inner)
        self.kd_inner = float(kd_inner)
        self.r_target = float(r_target)
        self.tau_max = float(tau_max)

    def reset(self) -> None:
        pass

    def update(self, measurement: np.ndarray, dt: float) -> float:
        r, r_d, th, th_d = measurement
        a_des = -self.kp * (r - self.r_target) - self.kd * r_d
        arg = float(np.clip(-a_des / ((5.0 / 7.0) * self.sys.g), -0.99, 0.99))
        th_des = np.arcsin(arg)
        tau_grav = self.sys.m * self.sys.g * r * np.cos(th)
        tau = tau_grav + self.kp_inner * (th_des - th) - self.kd_inner * th_d
        return float(np.clip(tau, -self.tau_max, self.tau_max))


class BallAndBeamLQR:
    """Multivariable LQR with state-dependent feedforward gravity equilibrium."""

    def __init__(
        self,
        K: np.ndarray,
        *,
        m: float = 0.064,
        g: float = 9.81,
        r_target: float = 0.10,
        tau_max: float = 1.50,
    ) -> None:
        self.K = np.asarray(K, dtype=float)
        self.m = float(m)
        self.g = float(g)
        self.r_target = float(r_target)
        self.tau_max = float(tau_max)

    def reset(self) -> None:
        pass

    def update(self, measurement: np.ndarray, dt: float) -> float:
        x_ref = np.array([self.r_target, 0.0, 0.0, 0.0])
        tau_eq = self.m * self.g * self.r_target
        u = tau_eq - self.K @ (measurement - x_ref)
        return float(np.clip(u[0], -self.tau_max, self.tau_max))


def main() -> None:
    set_aimct_style()
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    bb = BallAndBeam()
    A, B = bb.linearize()

    Q = np.diag([20.0, 2.0, 5.0, 0.5])
    R = np.array([[1.0]])
    u_lim = bb.tau_max

    lqr = LQR(A, B, Q, R)
    ctrl_cas = CascadePID(r_target=0.10, tau_max=u_lim)
    ctrl_pfl = PFLController(bb, r_target=0.10, tau_max=u_lim)
    ctrl_lqr = BallAndBeamLQR(lqr.K, m=bb.m, g=bb.g, r_target=0.10, tau_max=u_lim)
    ctrl_mpc = LinearMPC(
        A,
        B,
        Q=Q,
        R=R,
        N=20,
        u_bounds=(-u_lim, u_lim),
        x_ref=np.array([0.10, 0.0, 0.0, 0.0]),
        u_ref=np.array([bb.m * bb.g * 0.10]),
    )

    # -------------------------------------------------------------------------
    # 1. Step Response Benchmark (-0.10 m -> +0.10 m)
    # -------------------------------------------------------------------------
    dt = 0.002
    t_final = 5.0
    x0 = np.array([-0.10, 0.0, 0.0, 0.0])

    traj_cas = simulate(bb, ctrl_cas, x0=x0, dt=dt, t_final=t_final, u_bounds=(-u_lim, u_lim))
    traj_pfl = simulate(bb, ctrl_pfl, x0=x0, dt=dt, t_final=t_final, u_bounds=(-u_lim, u_lim))
    traj_lqr = simulate(bb, ctrl_lqr, x0=x0, dt=dt, t_final=t_final, u_bounds=(-u_lim, u_lim))
    traj_mpc = simulate(bb, ctrl_mpc, x0=x0, dt=dt, t_final=t_final, u_bounds=(-u_lim, u_lim))

    controllers = [
        ("Cascade PID", traj_cas),
        ("PFL (Nonlinear)", traj_pfl),
        ("Multivariable LQR", traj_lqr),
        ("Linear MPC", traj_mpc),
    ]
    results_rows = []

    for name, traj in controllers:
        t = traj.t
        y = traj.x[:, 0]  # Ball position
        u = traj.u[:, 0]  # Motor torque
        results_rows.append({
            "Controller": name,
            "Rise $t_r$ [s]": f"{rise_time(t, y, 0.10):.3f}",
            "Settling $t_s$ [s]": f"{settling_time(t, y, 0.10, band=0.02):.3f}",
            "Overshoot $M_p$ [%]": f"{peak_overshoot(t, y, 0.10):.1f}",
            "Steady error $e_{ss}$ [cm]": f"{steady_state_error(t, y, 0.10) * 100:.2e}",
            "RMSE [cm]": f"{rmse(t, y, 0.10) * 100:.2f}",
            "Energy $E_u$ [$\\text{N}^2\\cdot\\text{m}^2\\cdot\\text{s}$]": f"{control_energy(t, u):.4f}",
            "Peak Torque $|\\tau|_{\\max}$ [N$\\cdot$m]": f"{peak_control(t, u):.3f}",
            "Saturation [%]": f"{saturation_duty_cycle(t, u, u_lim):.1f}",
            "Status": "Stable" if not traj.diverged else "Diverged",
        })

    # Write CSV & Markdown
    csv_file = outdir / "table.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    headers = list(results_rows[0].keys())
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))]) + " |",
    ]
    for row in results_rows:
        md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    table_md = "\n".join(md_lines) + "\n"
    (outdir / "table.md").write_text(table_md, encoding="utf-8")

    print("=== Experiment 33 Benchmark Results ===")
    print(table_md)

    # -------------------------------------------------------------------------
    # Figure 1: 4-Panel Step Transition Comparison
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    t_grid = traj_lqr.t

    # (a) Ball Position r(t)
    axes[0, 0].plot(t_grid, traj_cas.x[:, 0] * 100, label="Cascade PID", color="#d62728", lw=1.8)
    axes[0, 0].plot(t_grid, traj_pfl.x[:, 0] * 100, label="PFL", color="#9467bd", lw=1.8)
    axes[0, 0].plot(t_grid, traj_lqr.x[:, 0] * 100, label="LQR", color="#1f77b4", lw=2)
    axes[0, 0].plot(t_grid, traj_mpc.x[:, 0] * 100, label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[0, 0].axhline(10.0, color="#333", ls=":", lw=1.5, label="Target $r^* = 10$ cm")
    axes[0, 0].set_ylabel("Ball Position $r$ [cm]")
    axes[0, 0].set_title("(a) Ball Position Response", fontweight="bold")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # (b) Beam Angle theta(t)
    axes[0, 1].plot(t_grid, traj_cas.x[:, 2] * 180 / np.pi, label="Cascade PID", color="#d62728", lw=1.8)
    axes[0, 1].plot(t_grid, traj_pfl.x[:, 2] * 180 / np.pi, label="PFL", color="#9467bd", lw=1.8)
    axes[0, 1].plot(t_grid, traj_lqr.x[:, 2] * 180 / np.pi, label="LQR", color="#1f77b4", lw=2)
    axes[0, 1].plot(t_grid, traj_mpc.x[:, 2] * 180 / np.pi, label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[0, 1].axhline(bb.theta_max * 180 / np.pi, color="#d62728", ls="--", lw=1, label="Limit $\\pm 25.8^\\circ$")
    axes[0, 1].axhline(-bb.theta_max * 180 / np.pi, color="#d62728", ls="--", lw=1)
    axes[0, 1].set_ylabel("Beam Tilt $\\theta$ [deg]")
    axes[0, 1].set_title("(b) Beam Tilt Angle", fontweight="bold")
    axes[0, 1].legend(loc="lower right", fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # (c) Motor Torque tau(t)
    axes[1, 0].plot(t_grid, traj_cas.u[:, 0], label="Cascade PID", color="#d62728", lw=1.8)
    axes[1, 0].plot(t_grid, traj_pfl.u[:, 0], label="PFL", color="#9467bd", lw=1.8)
    axes[1, 0].plot(t_grid, traj_lqr.u[:, 0], label="LQR", color="#1f77b4", lw=2)
    axes[1, 0].plot(t_grid, traj_mpc.u[:, 0], label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[1, 0].axhline(u_lim, color="#d62728", ls="--", lw=1, label="Torque Limit $\\pm 1.5$ N$\\cdot$m")
    axes[1, 0].axhline(-u_lim, color="#d62728", ls="--", lw=1)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Motor Torque $\\tau$ [N$\\cdot$m]")
    axes[1, 0].set_title("(c) Actuator Effort", fontweight="bold")
    axes[1, 0].legend(loc="lower right", fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Phase Portrait r vs r_dot
    axes[1, 1].plot(traj_cas.x[:, 0] * 100, traj_cas.x[:, 1] * 100, label="Cascade PID", color="#d62728", lw=1.8)
    axes[1, 1].plot(traj_pfl.x[:, 0] * 100, traj_pfl.x[:, 1] * 100, label="PFL", color="#9467bd", lw=1.8)
    axes[1, 1].plot(traj_lqr.x[:, 0] * 100, traj_lqr.x[:, 1] * 100, label="LQR", color="#1f77b4", lw=2)
    axes[1, 1].plot(traj_mpc.x[:, 0] * 100, traj_mpc.x[:, 1] * 100, label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[1, 1].plot([10.0], [0.0], "r*", ms=10, label="Target $(10, 0)$")
    axes[1, 1].set_xlabel("Ball Position $r$ [cm]")
    axes[1, 1].set_ylabel("Ball Velocity $\\dot{r}$ [cm/s]")
    axes[1, 1].set_title("(d) Ball Phase Portrait", fontweight="bold")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 33: Ball and Beam Underactuated Benchmark", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "ball_and_beam_benchmark.png", dpi=300)
    fig.savefig(outdir / "ball_and_beam_benchmark.svg")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Multi-Step Tracking (-10cm -> +10cm -> -5cm -> 0cm)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    dt_multi = 0.005
    t_multi = np.arange(0.0, 16.0, dt_multi)

    target_prof = np.piecewise(
        t_multi,
        [t_multi < 4.0, (t_multi >= 4.0) & (t_multi < 8.0), (t_multi >= 8.0) & (t_multi < 12.0), t_multi >= 12.0],
        [-0.10, 0.10, -0.05, 0.0],
    )

    x_curr_lqr = np.array([-0.10, 0.0, 0.0, 0.0])
    x_curr_mpc = np.array([-0.10, 0.0, 0.0, 0.0])

    r_hist_lqr, r_hist_mpc = [], []
    u_hist_lqr, u_hist_mpc = [], []

    for k, t_val in enumerate(t_multi):
        r_tgt = target_prof[k]

        # LQR update
        ctrl_lqr.r_target = r_tgt
        u_l = ctrl_lqr.update(x_curr_lqr, dt_multi)
        x_curr_lqr = x_curr_lqr + dt_multi * bb.dynamics(t_val, x_curr_lqr, [u_l])
        r_hist_lqr.append(x_curr_lqr[0])
        u_hist_lqr.append(u_l)

        # MPC update
        mpc_multi = LinearMPC(
            A,
            B,
            Q=Q,
            R=R,
            N=15,
            u_bounds=(-u_lim, u_lim),
            x_ref=np.array([r_tgt, 0.0, 0.0, 0.0]),
            u_ref=np.array([bb.m * bb.g * r_tgt]),
        )
        u_m = mpc_multi.update(x_curr_mpc, dt_multi)
        u_m_val = float(u_m) if np.ndim(u_m) == 0 else float(u_m[0])
        x_curr_mpc = x_curr_mpc + dt_multi * bb.dynamics(t_val, x_curr_mpc, [u_m_val])
        r_hist_mpc.append(x_curr_mpc[0])
        u_hist_mpc.append(u_m_val)

    ax2[0].plot(t_multi, target_prof * 100, "k--", lw=1.5, label="Target $r^*(t)$")
    ax2[0].plot(t_multi, np.array(r_hist_lqr) * 100, color="#1f77b4", lw=2, label="Multivariable LQR")
    ax2[0].plot(t_multi, np.array(r_hist_mpc) * 100, color="#2ca02c", lw=2, label="Linear MPC")
    ax2[0].set_ylabel("Ball Position $r$ [cm]")
    ax2[0].set_title("Multi-Step Dynamic Setpoint Tracking", fontweight="bold")
    ax2[0].legend(loc="upper right")
    ax2[0].grid(True, alpha=0.3)

    ax2[1].plot(t_multi, u_hist_lqr, color="#1f77b4", lw=1.8, label="LQR Torque")
    ax2[1].plot(t_multi, u_hist_mpc, color="#2ca02c", lw=1.8, label="MPC Torque")
    ax2[1].axhline(u_lim, color="#d62728", ls="--", lw=1, label="Limit (1.5 N·m)")
    ax2[1].axhline(-u_lim, color="#d62728", ls="--", lw=1)
    ax2[1].set_xlabel("Time [s]")
    ax2[1].set_ylabel("Torque $\\tau$ [N$\\cdot$m]")
    ax2[1].legend(loc="upper right")
    ax2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(outdir / "ball_and_beam_setpoint.png", dpi=300)
    fig2.savefig(outdir / "ball_and_beam_setpoint.svg")
    plt.close(fig2)

    print(f"Artifacts successfully written to {outdir}")


if __name__ == "__main__":
    main()
