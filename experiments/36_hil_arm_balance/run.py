"""Experiment 36: Double-Inverted Pendulum Arm Balance through HIL Emulator.

Benchmarks LQR (Stiff), LQR + Integral (LQI), and LQR (Soft) on the two-link arm
upright balance problem under a realistic 1 kHz Hardware-in-the-Loop (HIL) emulator:
- 12-bit magnetic encoder quantization (AS5600 standard, 4096 bins over [-pi, pi])
- 8 ms round-trip transport delay (InProcess + delay line buffer)
- 80 N*m/s torque slew rate limit (Dynamixel XM430 standard)
- Actuator saturation (+-15 / +-10 N*m)
- Sensor measurement noise and sample timing jitter
"""

from __future__ import annotations

import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aimct.systems import TwoLinkArm
from aimct.controllers.lqr import solve_care
from aimct.hil import PlantEmulator, RealTimeLoop

EXP_DIR = pathlib.Path(__file__).parent
FULL_RUN = os.environ.get("AIMCT_EXP_FULL", "0") == "1"

# --- System & Controllers ---------------------------------------------------
arm = TwoLinkArm()
x_eq = np.array([np.pi / 2.0, 0.0, 0.0, 0.0])  # upright standing
u_eq = arm.G(x_eq[:2])                         # zero equilibrium torque
A, B = arm.linearize(x_eq=x_eq, u_eq=u_eq)

Q = np.diag([8.0, 8.0, 1.0, 1.0])
R = np.diag([1.0, 1.0])
P = solve_care(A, B, Q, R)
K_stiff = np.linalg.solve(R, B.T @ P)

# Integral-augmented LQR
Ci = np.zeros((2, 4))
Ci[0, 0] = Ci[1, 1] = 1.0
Aa = np.block([[A, np.zeros((4, 2))], [Ci, np.zeros((2, 2))]])
Ba = np.vstack([B, np.zeros((2, 2))])
Qa = np.diag(np.concatenate([np.diag(Q), [3.0, 3.0]]))
Pa = solve_care(Aa, Ba, Qa, R)
Ka = np.linalg.solve(R, Ba.T @ Pa)
Kx, Ki = Ka[:, :4], Ka[:, 4:]

# Soft LQR (45% gain scaling)
K_soft = 0.45 * K_stiff


class LqrStiff:
    name = "LQR (Stiff)"
    def update(self, y, dt):
        return u_eq - K_stiff @ (y - x_eq)
    def reset(self): pass


class LqrIntegral:
    name = "LQR + Integral (LQI)"
    def __init__(self):
        self.integ = np.zeros(2)
    def update(self, y, dt):
        self.integ += (y[:2] - x_eq[:2]) * dt
        return u_eq - Kx @ (y - x_eq) - Ki @ self.integ
    def reset(self):
        self.integ = np.zeros(2)


class LqrSoft:
    name = "LQR (Soft)"
    def update(self, y, dt):
        return u_eq - K_soft @ (y - x_eq)
    def reset(self): pass


# --- Simulation Parameters --------------------------------------------------
RATE_HZ = 1000.0  # 1 kHz loop
DT = 1.0 / RATE_HZ
DURATION = 4.0
STEPS = int(DURATION * RATE_HZ)
time_arr = np.linspace(0.0, DURATION, STEPS)

# Initial perturbation: 0.12 rad (~6.9 deg) link 1, -0.06 rad (~-3.4 deg) link 2
x0 = x_eq + np.array([0.12, -0.06, 0.0, 0.0])


def run_hil_simulation(controller, delay_ms: float = 8.0, with_wind: bool = True):
    emu = PlantEmulator(
        arm,
        quantization_bits=12,
        quantize_channels=[0, 1],
        quantize_ranges=[(-np.pi, np.pi), (-np.pi, np.pi)],
        delay_s=delay_ms * 1e-3,
        slew_rate_max=[80.0, 80.0],
        u_min=[-15.0, -10.0],
        u_max=[15.0, 10.0],
        sensor_noise_std=0.0005,
        seed=42,
    )

    loop = RealTimeLoop(
        rate_hz=RATE_HZ,
        controller=controller,
        plant=emu,
        simulated_time=True,
    )

    if hasattr(controller, "reset"): controller.reset()
    y_meas = emu.reset(x0)

    hist_t = np.zeros(STEPS)
    hist_x = np.zeros((STEPS, 4))
    hist_y = np.zeros((STEPS, 4))
    hist_u = np.zeros((STEPS, 2))
    diverged = False

    for k in range(STEPS):
        t = k * DT
        hist_t[k] = t
        hist_x[k] = emu.x.copy()
        hist_y[k] = y_meas.copy()

        u_cmd = controller.update(y_meas, DT)
        u_cmd = np.atleast_1d(np.asarray(u_cmd, float))
        hist_u[k] = u_cmd.copy()

        # External wind torque disturbance pulse during t in [1.5, 2.5] s
        if with_wind and 1.5 <= t <= 2.5:
            u_cmd_with_dist = u_cmd + np.array([1.5, 0.8])
        else:
            u_cmd_with_dist = u_cmd

        y_meas = emu.step(u_cmd_with_dist, DT)

        # Fall condition: |theta1 - pi/2| > 1.3 rad
        if abs(emu.x[0] - x_eq[0]) > 1.3 or not np.all(np.isfinite(emu.x)):
            diverged = True
            hist_t = hist_t[:k+1]
            hist_x = hist_x[:k+1]
            hist_y = hist_y[:k+1]
            hist_u = hist_u[:k+1]
            break

    return {
        "t": hist_t,
        "x": hist_x,
        "y": hist_y,
        "u": hist_u,
        "diverged": diverged,
    }


# --- 1. Nominal 8 ms HIL Benchmark -----------------------------------------
controllers = {
    "LQR (Stiff)": LqrStiff(),
    "LQR + Integral (LQI)": LqrIntegral(),
    "LQR (Soft)": LqrSoft(),
}

nom_results = {}
for name, ctl in controllers.items():
    nom_results[name] = run_hil_simulation(ctl, delay_ms=8.0, with_wind=True)

# --- 2. Delay Margin Sweep -------------------------------------------------
delay_grid = np.arange(0, 102, 2 if FULL_RUN else 4)  # 0 to 100 ms
delay_margins = {}
delay_sweep_records = {name: [] for name in controllers}

for name, ctl in controllers.items():
    max_stable = 0.0
    for d_ms in delay_grid:
        res = run_hil_simulation(ctl, delay_ms=d_ms, with_wind=False)
        fell = res["diverged"] or len(res["t"]) < STEPS
        # Max error during last 1 second
        if not fell:
            tail_err = np.max(np.abs(res["x"][-1000:, :2] - x_eq[:2]))
            if tail_err > 0.3:
                fell = True
        delay_sweep_records[name].append((d_ms, fell))
        if not fell:
            max_stable = d_ms
    delay_margins[name] = max_stable

# --- 3. Compute Metrics ----------------------------------------------------
metrics = []
for name, ctl in controllers.items():
    res = nom_results[name]
    xs = res["x"]
    us = res["u"]
    ts = res["t"]
    fell = res["diverged"]

    if not fell and len(ts) == STEPS:
        # Angular errors in degrees
        th1_err_deg = np.rad2deg(xs[:, 0] - x_eq[0])
        th2_err_deg = np.rad2deg(xs[:, 1] - x_eq[1])
        max_tip_deg = np.max(np.abs(th1_err_deg))
        # Settling time for initial poke (before wind at 1.5s, within 0.5 deg = 0.0087 rad)
        tol_rad = np.deg2rad(0.5)
        settled_idx = None
        for i in range(1500):
            if np.all(np.abs(xs[i:1500, :2] - x_eq[:2]) < tol_rad):
                settled_idx = i
                break
        t_settle = settled_idx * DT if settled_idx is not None else 1.5
        # Steady error during wind (2.0 to 2.5 s -> indices 2000:2500)
        ess_wind_deg = np.mean(np.abs(th1_err_deg[2000:2500]))
        # Post-wind steady error (3.5 to 4.0 s -> indices 3500:4000)
        ess_post_deg = np.mean(np.abs(th1_err_deg[3500:4000]))
        energy = float(np.sum(us**2) * DT)
        peak_tau = float(np.max(np.abs(us)))
        # Slew rate saturation check
        slew_applied = np.abs(np.diff(us, axis=0)) / DT
        slew_sat_pct = float(np.mean(slew_applied >= 79.5) * 100.0)
        status = "Stable"
    else:
        max_tip_deg = 90.0
        t_settle = 4.0
        ess_wind_deg = 90.0
        ess_post_deg = 90.0
        energy = float(np.sum(us**2) * DT) if len(us) > 0 else 0.0
        peak_tau = float(np.max(np.abs(us))) if len(us) > 0 else 15.0
        slew_sat_pct = 100.0
        status = "Fallen"

    metrics.append({
        "controller": name,
        "delay_margin_ms": delay_margins[name],
        "settling_s": t_settle,
        "max_tip_deg": max_tip_deg,
        "ess_wind_deg": ess_wind_deg,
        "ess_post_deg": ess_post_deg,
        "energy": energy,
        "peak_tau": peak_tau,
        "slew_sat_pct": slew_sat_pct,
        "status": status,
    })

# --- 4. Export Markdown & CSV Tables ---------------------------------------
md_lines = [
    "| Controller | Delay Margin $\\tau_{\\max}$ [ms] | Poke Settling $t_s$ [s] | Max Tip $|\\theta_1 - \\pi/2|_{\\max}$ [deg] | Wind Bias $e_{\\text{wind}}$ [deg] | Post-Wind $e_{ss}$ [deg] | Energy $E_u$ [$\\text{N}^2\\cdot\\text{m}^2\\cdot\\text{s}$] | Peak Torque $|\\tau|_{\\max}$ [$\\text{N}\\cdot\\text{m}$] | Slew Sat [%] | Status |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
]

for m in metrics:
    md_lines.append(
        f"| {m["controller"]} | {m["delay_margin_ms"]:.0f} | {m["settling_s"]:.3f} | {m["max_tip_deg"]:.2f} | "
        f"{m["ess_wind_deg"]:.2f} | {m["ess_post_deg"]:.3f} | {m["energy"]:.2f} | {m["peak_tau"]:.2f} | "
        f"{m["slew_sat_pct"]:.1f} | **{m["status"]}** |"
    )

table_md = "\n".join(md_lines) + "\n"
(EXP_DIR / "table.md").write_text(table_md, encoding="utf-8")

csv_lines = [
    "Controller,Delay_Margin_ms,Poke_Settling_s,Max_Tip_deg,Wind_Bias_deg,Post_Wind_Ess_deg,Energy_N2m2s,Peak_Torque_Nm,Slew_Sat_pct,Status",
]
for m in metrics:
    csv_lines.append(
        f"{m["controller"]},{m["delay_margin_ms"]:.1f},{m["settling_s"]:.4f},{m["max_tip_deg"]:.3f},"
        f"{m["ess_wind_deg"]:.3f},{m["ess_post_deg"]:.4f},{m["energy"]:.4f},{m["peak_tau"]:.3f},"
        f"{m["slew_sat_pct"]:.2f},{m["status"]}"
    )
table_csv = "\n".join(csv_lines) + "\n"
(EXP_DIR / "table.csv").write_text(table_csv, encoding="utf-8")

print("=== Experiment 36 Benchmark Results ===")
print(table_md)

# --- 5. High-Resolution Visualizations -------------------------------------
# Figure 1: 4-Panel Trajectories under 8 ms HIL Emulator
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
colors = {"LQR (Stiff)": "#3182CE", "LQR + Integral (LQI)": "#38A169", "LQR (Soft)": "#E53E3E"}
styles = {"LQR (Stiff)": "-", "LQR + Integral (LQI)": "--", "LQR (Soft)": ":"}

# Panel 1: Link 1 Angle theta1(t)
ax1 = axs[0]
for name, res in nom_results.items():
    ax1.plot(res["t"], np.rad2deg(res["x"][:, 0] - x_eq[0]), label=name, color=colors[name], linestyle=styles[name], linewidth=2.0)
ax1.axvspan(1.5, 2.5, color="#EDF2F7", alpha=0.8, label="Wind Torque Pulse (1.5 N*m)")
ax1.axhline(0.0, color="black", linestyle="--", alpha=0.4)
ax1.set_ylabel("Link 1 Dev $\\theta_1 - \\pi/2$ [deg]", fontweight="bold")
ax1.set_title("Double-Inverted Pendulum HIL Balance (1 kHz, 12-Bit Encoders, 8 ms Delay)", fontweight="bold")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Link 2 Relative Angle theta2(t)
ax2 = axs[1]
for name, res in nom_results.items():
    ax2.plot(res["t"], np.rad2deg(res["x"][:, 1] - x_eq[1]), label=name, color=colors[name], linestyle=styles[name], linewidth=2.0)
ax2.axvspan(1.5, 2.5, color="#EDF2F7", alpha=0.8)
ax2.axhline(0.0, color="black", linestyle="--", alpha=0.4)
ax2.set_ylabel("Link 2 Angle $\\theta_2$ [deg]", fontweight="bold")
ax2.grid(True, alpha=0.3)

# Panel 3: Joint 1 Control Torque tau_1(t)
ax3 = axs[2]
for name, res in nom_results.items():
    ax3.plot(res["t"], res["u"][:, 0], label=name, color=colors[name], linestyle=styles[name], linewidth=1.8)
ax3.axvspan(1.5, 2.5, color="#EDF2F7", alpha=0.8)
ax3.axhline(15.0, color="red", linestyle=":", alpha=0.5, label="Torque Limit (+-15 N*m)")
ax3.axhline(-15.0, color="red", linestyle=":", alpha=0.5)
ax3.set_ylabel("Joint 1 Torque $\\tau_1$ [N$\\cdot$m]", fontweight="bold")
ax3.grid(True, alpha=0.3)

# Panel 4: Joint 2 Control Torque tau_2(t)
ax4 = axs[3]
for name, res in nom_results.items():
    ax4.plot(res["t"], res["u"][:, 1], label=name, color=colors[name], linestyle=styles[name], linewidth=1.8)
ax4.axvspan(1.5, 2.5, color="#EDF2F7", alpha=0.8)
ax4.axhline(10.0, color="red", linestyle=":", alpha=0.5, label="Torque Limit (+-10 N*m)")
ax4.axhline(-10.0, color="red", linestyle=":", alpha=0.5)
ax4.set_xlabel("Time [s]", fontweight="bold")
ax4.set_ylabel("Joint 2 Torque $\\tau_2$ [N$\\cdot$m]", fontweight="bold")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(EXP_DIR / "hil_arm_balance.png", dpi=300)
fig.savefig(EXP_DIR / "hil_arm_balance.svg")
plt.close(fig)

# Figure 2: Delay Margin Stability Boundary
fig2, ax_del = plt.subplots(figsize=(8, 5))
for name in controllers:
    d_vals = [d for d, fell in delay_sweep_records[name]]
    status_vals = [0 if fell else 1 for d, fell in delay_sweep_records[name]]
    ax_del.plot(d_vals, status_vals, marker="o", label=f"{name} (Max Margin: {delay_margins[name]:.0f} ms)", color=colors[name], linewidth=2.0)

ax_del.axvline(8.0, color="purple", linestyle="--", label="Nominal HIL Latency (8 ms)")
ax_del.set_xlabel("Round-Trip Transport Delay $\\tau_{\\text{delay}}$ [ms]", fontweight="bold")
ax_del.set_ylabel("Closed-Loop Stability [1=Stable, 0=Fallen]", fontweight="bold")
ax_del.set_title("HIL Transport Latency Margin & Stability Boundaries (TwoLinkArm)", fontweight="bold")
ax_del.set_yticks([0, 1])
ax_del.set_yticklabels(["Unstable / Fallen", "Stable Balance"])
ax_del.legend(loc="lower left", fontsize=9)
ax_del.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(EXP_DIR / "hil_delay_margin.png", dpi=300)
fig2.savefig(EXP_DIR / "hil_delay_margin.svg")
plt.close(fig2)

print("Artifacts generated successfully in", EXP_DIR)
