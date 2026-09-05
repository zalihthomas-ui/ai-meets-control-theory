"""
Experiment 30: Coupled Two-Tank Level Regulation & Interactive Capacity Benchmark.

Compares SISO PI, Multivariable LQR, and Linear MPC on liquid level setpoint
regulation and interactive capacity handling on the canonical Quanser Coupled Tanks.
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
from aimct.systems.two_tank import TwoTank


class TankPI:
    """SISO PI Controller on Tank 2 liquid level with anti-windup clamping."""

    def __init__(
        self,
        *,
        kp: float = 40.0,
        ki: float = 1.2,
        u_ff: float = 7.68,
        setpoint: float = 0.15,
        u_min: float = 0.0,
        u_max: float = 12.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.u_ff = float(u_ff)
        self.setpoint = float(setpoint)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0

    def update(self, measurement: np.ndarray, dt: float) -> float:
        h2 = measurement[1] if hasattr(measurement, "__len__") else measurement
        err = self.setpoint - h2
        self.integral += err * dt
        u_raw = self.u_ff + self.kp * err + self.ki * self.integral
        u_sat = float(np.clip(u_raw, self.u_min, self.u_max))
        if u_raw != u_sat and np.sign(err) == np.sign(u_raw - self.u_ff):
            self.integral -= err * dt  # Anti-windup clamping
        return u_sat


class TankLQR:
    """Multivariable State Feedback LQR with operating point feedforward."""

    def __init__(
        self,
        K: np.ndarray,
        x_ref: np.ndarray,
        u_ref: np.ndarray,
        *,
        u_min: float = 0.0,
        u_max: float = 12.0,
    ) -> None:
        self.K = np.asarray(K, dtype=float)
        self.x_ref = np.asarray(x_ref, dtype=float)
        self.u_ref = np.asarray(u_ref, dtype=float)
        self.u_min = float(u_min)
        self.u_max = float(u_max)

    def reset(self) -> None:
        pass

    def update(self, measurement: np.ndarray, dt: float) -> float:
        u = self.u_ref - self.K @ (measurement - self.x_ref)
        return float(np.clip(u[0], self.u_min, self.u_max))


def main() -> None:
    set_aimct_style()
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    tank = TwoTank()
    x_eq0, u_eq0 = tank.steady_state_operating_point(0.10)
    x_eq1, u_eq1 = tank.steady_state_operating_point(0.15)
    A, B = tank.linearize(x_eq=x_eq1, u_eq=u_eq1)

    Q = np.diag([2.0, 50.0])
    R = np.array([[1.0]])
    u_lim = tank.v_max

    lqr = LQR(A, B, Q, R)
    ctrl_pi = TankPI(kp=40.0, ki=1.2, u_ff=float(u_eq1[0]), setpoint=0.15, u_max=u_lim)
    ctrl_lqr = TankLQR(lqr.K, x_eq1, u_eq1, u_max=u_lim)
    ctrl_mpc = LinearMPC(A, B, Q=Q, R=R, N=20, u_bounds=(0.0, u_lim), x_ref=x_eq1, u_ref=u_eq1)

    # -------------------------------------------------------------------------
    # 1. Step Response Benchmark (0.10 m -> 0.15 m)
    # -------------------------------------------------------------------------
    dt = 0.1
    t_final = 75.0
    x0 = x_eq0.copy()

    traj_pi = simulate(tank, ctrl_pi, x0=x0, dt=dt, t_final=t_final, u_bounds=(0.0, u_lim))
    traj_lqr = simulate(tank, ctrl_lqr, x0=x0, dt=dt, t_final=t_final, u_bounds=(0.0, u_lim))
    traj_mpc = simulate(tank, ctrl_mpc, x0=x0, dt=dt, t_final=t_final, u_bounds=(0.0, u_lim))

    controllers = [("SISO PI", traj_pi), ("Multivariable LQR", traj_lqr), ("Linear MPC", traj_mpc)]
    results_rows = []

    for name, traj in controllers:
        t = traj.t
        y = traj.x[:, 1]  # Tank 2 level
        u = traj.u[:, 0]  # Pump voltage
        results_rows.append({
            "Controller": name,
            "Rise $t_r$ [s]": f"{rise_time(t, y, 0.15):.2f}",
            "Settling $t_s$ [s]": f"{settling_time(t, y, 0.15, band=0.02):.2f}",
            "Overshoot $M_p$ [%]": f"{peak_overshoot(t, y, 0.15):.1f}",
            "Steady error $e_{ss}$ [cm]": f"{steady_state_error(t, y, 0.15) * 100:.2f}",
            "RMSE [cm]": f"{rmse(t, y, 0.15) * 100:.2f}",
            "Energy $E_u$ [$\\text{V}^2\\cdot\\text{s}$]": f"{control_energy(t, u):.1f}",
            "Peak Voltage $V_{\\max}$ [V]": f"{peak_control(t, u):.2f}",
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

    print("=== Experiment 30 Benchmark Results ===")
    print(table_md)

    # -------------------------------------------------------------------------
    # Figure 1: 4-Panel Step Transition Comparison
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    t_grid = traj_pi.t

    # (a) Tank 2 level h2(t)
    axes[0, 0].plot(t_grid, traj_pi.x[:, 1] * 100, label="SISO PI", color="#d62728", lw=2)
    axes[0, 0].plot(t_grid, traj_lqr.x[:, 1] * 100, label="Multivariable LQR", color="#1f77b4", lw=2)
    axes[0, 0].plot(t_grid, traj_mpc.x[:, 1] * 100, label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[0, 0].axhline(15.0, color="#333", ls=":", lw=1.5, label="Setpoint $h_2^* = 15$ cm")
    axes[0, 0].set_ylabel("Tank 2 Level $h_2$ [cm]")
    axes[0, 0].set_title("(a) Process Variable (Tank 2 Level)", fontweight="bold")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)

    # (b) Tank 1 level h1(t)
    axes[0, 1].plot(t_grid, traj_pi.x[:, 0] * 100, label="SISO PI", color="#d62728", lw=2)
    axes[0, 1].plot(t_grid, traj_lqr.x[:, 0] * 100, label="Multivariable LQR", color="#1f77b4", lw=2)
    axes[0, 1].plot(t_grid, traj_mpc.x[:, 0] * 100, label="Linear MPC", color="#2ca02c", lw=2, ls="--")
    axes[0, 1].axhline(30.0, color="#d62728", ls="--", lw=1, label="Overflow Limit $30$ cm")
    axes[0, 1].set_ylabel("Tank 1 Level $h_1$ [cm]")
    axes[0, 1].set_title("(b) Intermediate Capacity (Tank 1 Level)", fontweight="bold")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)

    # (c) Pump Input Voltage V_p(t)
    axes[1, 0].plot(t_grid, traj_pi.u[:, 0], label="SISO PI", color="#d62728", lw=1.8)
    axes[1, 0].plot(t_grid, traj_lqr.u[:, 0], label="Multivariable LQR", color="#1f77b4", lw=1.8)
    axes[1, 0].plot(t_grid, traj_mpc.u[:, 0], label="Linear MPC", color="#2ca02c", lw=1.8, ls="--")
    axes[1, 0].axhline(u_lim, color="#d62728", ls="--", lw=1, label="Max Voltage $12$ V")
    axes[1, 0].axhline(0.0, color="#333", ls=":", lw=1)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Pump Voltage $V_p$ [V]")
    axes[1, 0].set_title("(c) Control Action", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Phase Portrait h1 vs h2
    axes[1, 1].plot(traj_pi.x[:, 0] * 100, traj_pi.x[:, 1] * 100, label="SISO PI", color="#d62728", lw=1.8)
    axes[1, 1].plot(traj_lqr.x[:, 0] * 100, traj_lqr.x[:, 1] * 100, label="Multivariable LQR", color="#1f77b4", lw=1.8)
    axes[1, 1].plot(traj_mpc.x[:, 0] * 100, traj_mpc.x[:, 1] * 100, label="Linear MPC", color="#2ca02c", lw=1.8, ls="--")
    axes[1, 1].plot([x_eq0[0] * 100], [x_eq0[1] * 100], "ko", ms=6, label="Initial State")
    axes[1, 1].plot([x_eq1[0] * 100], [x_eq1[1] * 100], "r*", ms=10, label="Target Equilibrium")
    axes[1, 1].set_xlabel("Tank 1 Level $h_1$ [cm]")
    axes[1, 1].set_ylabel("Tank 2 Level $h_2$ [cm]")
    axes[1, 1].set_title("(d) Hydraulic State Space Phase Portrait", fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 30: Coupled Two-Tank Level Regulation Benchmark", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "two_tank_benchmark.png", dpi=300)
    fig.savefig(outdir / "two_tank_benchmark.svg")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Multi-Step Setpoint Tracking Trajectory (10cm -> 15cm -> 8cm)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    dt_multi = 0.25
    t_multi = np.arange(0.0, 160.0, dt_multi)

    target_prof = np.piecewise(
        t_multi,
        [t_multi < 60.0, (t_multi >= 60.0) & (t_multi < 110.0), t_multi >= 110.0],
        [0.10, 0.15, 0.08],
    )

    x_curr_pi = x_eq0.copy()
    x_curr_mpc = x_eq0.copy()
    pi_multi = TankPI(kp=40.0, ki=1.2, u_ff=float(u_eq0[0]), setpoint=0.10, u_max=u_lim)

    h2_hist_pi, h2_hist_mpc = [], []
    u_hist_pi, u_hist_mpc = [], []

    for k, t_val in enumerate(t_multi):
        r_target = target_prof[k]
        x_target, u_target = tank.steady_state_operating_point(r_target)

        # PI update
        pi_multi.setpoint = r_target
        pi_multi.u_ff = float(u_target[0])
        u_p = pi_multi.update(x_curr_pi, dt_multi)
        x_curr_pi = x_curr_pi + dt_multi * tank.dynamics(t_val, x_curr_pi, [u_p])
        h2_hist_pi.append(x_curr_pi[1])
        u_hist_pi.append(u_p)

        # MPC update
        mpc_multi = LinearMPC(A, B, Q=Q, R=R, N=15, u_bounds=(0.0, u_lim), x_ref=x_target, u_ref=u_target)
        u_m = mpc_multi.update(x_curr_mpc, dt_multi)
        u_m_val = float(u_m) if np.ndim(u_m) == 0 else float(u_m[0])
        x_curr_mpc = x_curr_mpc + dt_multi * tank.dynamics(t_val, x_curr_mpc, [u_m_val])
        h2_hist_mpc.append(x_curr_mpc[1])
        u_hist_mpc.append(u_m_val)

    ax2[0].plot(t_multi, target_prof * 100, "k--", lw=1.5, label="Target $h_2^*(t)$")
    ax2[0].plot(t_multi, np.array(h2_hist_pi) * 100, color="#d62728", lw=2, label="SISO PI")
    ax2[0].plot(t_multi, np.array(h2_hist_mpc) * 100, color="#2ca02c", lw=2, label="Linear MPC")
    ax2[0].set_ylabel("Tank 2 Level $h_2$ [cm]")
    ax2[0].set_title("Multi-Step Dynamic Setpoint Tracking", fontweight="bold")
    ax2[0].legend(loc="upper right")
    ax2[0].grid(True, alpha=0.3)

    ax2[1].plot(t_multi, u_hist_pi, color="#d62728", lw=1.8, label="PI Voltage")
    ax2[1].plot(t_multi, u_hist_mpc, color="#2ca02c", lw=1.8, label="MPC Voltage")
    ax2[1].axhline(u_lim, color="#d62728", ls="--", lw=1, label="Max Voltage (12V)")
    ax2[1].axhline(0.0, color="#333", ls=":", lw=1)
    ax2[1].set_xlabel("Time [s]")
    ax2[1].set_ylabel("Pump Voltage $V_p$ [V]")
    ax2[1].legend(loc="upper right")
    ax2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(outdir / "two_tank_setpoint.png", dpi=300)
    fig2.savefig(outdir / "two_tank_setpoint.svg")
    plt.close(fig2)

    print(f"Artifacts successfully written to {outdir}")


if __name__ == "__main__":
    main()
