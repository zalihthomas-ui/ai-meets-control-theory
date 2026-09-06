"""Experiment 34: Disturbance Observer (DOB) Wind Rejection Benchmark.

Evaluates Disturbance Observer (DOB) against Nominal LQR, Integral-Augmented LQR
(LQI), and Model Reference Adaptive Control (MRAC) on the underactuated
PlanarQuadrotor under matched (vertical thrust, pitch torque) and unmatched
(horizontal lateral force) atmospheric wind disturbances.
"""

from __future__ import annotations

import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aimct.systems.quadrotor import PlanarQuadrotor
from aimct.controllers.lqr import LQR
from aimct.controllers.disturbance_observer import DisturbanceObserver
from aimct.controllers.adaptive import solve_lyapunov

# --- Configuration & Plant Setup ---------------------------------------------
EXP_DIR = pathlib.Path(__file__).parent
quad = PlanarQuadrotor()
A, B = quad.linearize()
m, g, Iyy, arm, cd = quad.m, quad.g, quad.Iyy, quad.l, quad.cd

# Bryson LQR state & input weights
qx = 1.0 / np.array([0.10, 0.10, 0.20, 0.5, 0.5, 3.0]) ** 2
ru = 1.0 / np.array([0.15, 0.15]) ** 2
Q = np.diag(qx)
R = np.diag(ru)
x_target = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

# 1. Base LQR
base_lqr = LQR(A, B, Q, R, x_ref=x_target, u_ref=quad.u_hover)
K = base_lqr.K

# 2. Integral LQR (LQI) - integrates x and z tracking errors
Ci = np.zeros((2, 6))
Ci[0, 0] = 1.0
Ci[1, 1] = 1.0
Aa = np.block([[A, np.zeros((6, 2))], [Ci, np.zeros((2, 2))]])
Ba = np.vstack([B, np.zeros((2, 2))])
Qa = np.diag(np.concatenate([qx, [30.0, 30.0]]))
Ka = LQR(Aa, Ba, Qa, R).K
Kx, Ki = Ka[:, :6], Ka[:, 6:]

# 3. MRAC - baseline LQR + direct adaptive cancellation on matched channels
Am = A - B @ K
Qm = np.eye(6) * 1.0
Pm = solve_lyapunov(Am, Qm)
Gamma = np.eye(7) * 0.05
sigma_mrac = 0.01

# 4. DOB + LQR (Disturbance Observer)
dob_lqr = DisturbanceObserver(
    base_controller=base_lqr,
    plant=quad,
    cutoff_freq=12.0,
    damping=1.0,
    filter_order=2,
)

# --- Wind Disturbance Profile ------------------------------------------------
DT = 0.002
T_FINAL = 10.0
STEPS = int(T_FINAL / DT)
time = np.linspace(0.0, T_FINAL, STEPS)


def get_wind(t: float):
    """Return [Fx_wind (N), Fz_wind (N), tau_wind (N*m)]."""
    if t < 2.0:
        return 0.0, 0.0, 0.0
    elif t < 6.0:
        # Steady wind phase
        return 0.030, -0.015, 5.0e-5
    elif t < 8.0:
        # Severe dynamic gust phase
        t_g = t - 6.0
        fx = 0.060 + 0.015 * np.sin(2.0 * np.pi * 2.0 * t_g)
        fz = -0.030
        tau = 1.0e-4 * np.sin(2.0 * np.pi * 2.0 * t_g)
        return fx, fz, tau
    else:
        # Recovery phase
        return 0.0, 0.0, 0.0


# --- Simulation Loop ---------------------------------------------------------
def simulate_controller(name: str):
    x = x_target.copy()
    hist_x = []
    hist_u = []
    hist_dhat = []
    hist_dtrue = []

    integ = np.zeros(2)
    theta_hat = np.zeros((7, 2))
    base_lqr.reset()
    dob_lqr.reset()

    for t in time:
        fx_w, fz_w, tau_w = get_wind(t)
        hist_dtrue.append([fx_w / m, fz_w / m, tau_w / Iyy])

        if name == "Nominal LQR":
            u = base_lqr.update(x, DT)
        elif name == "Integral LQR (LQI)":
            integ += (x[:2] - x_target[:2]) * DT
            u = quad.u_hover - Kx @ (x - x_target) - Ki @ integ
        elif name == "MRAC (Adaptive)":
            e = x - x_target
            phi = np.concatenate([e, [1.0]])
            th_dot = Gamma @ (np.outer(phi, e @ Pm @ B) - sigma_mrac * theta_hat)
            theta_hat += th_dot * DT
            u_adapt = theta_hat.T @ phi
            u = base_lqr.update(x, DT) - u_adapt
        elif name == "DOB + LQR":
            u = dob_lqr.update(x, DT)
            hist_dhat.append(dob_lqr.d_hat.copy())

        u = np.clip(u, 0.0, quad.thrust_max)
        hist_x.append(x.copy())
        hist_u.append(u.copy())

        # Forward dynamics with wind disturbance
        xdot = quad.dynamics(t, x, u)
        xdot[3] += fx_w / m
        xdot[4] += fz_w / m
        xdot[5] += tau_w / Iyy

        x = x + xdot * DT

    return {
        "x": np.array(hist_x),
        "u": np.array(hist_u),
        "d_hat": np.array(hist_dhat) if hist_dhat else None,
        "d_true": np.array(hist_dtrue),
    }


controllers = ["Nominal LQR", "Integral LQR (LQI)", "MRAC (Adaptive)", "DOB + LQR"]
results = {}
for name in controllers:
    results[name] = simulate_controller(name)

# --- Compute Benchmark Metrics -----------------------------------------------
metrics = []

for name in controllers:
    res = results[name]
    xs = res["x"]
    us = res["u"]

    x_pos = xs[:, 0]
    z_pos = xs[:, 1]
    th_deg = np.rad2deg(xs[:, 2])

    x_err = x_pos - x_target[0]
    z_err = z_pos - x_target[1]

    # Unmatched x channel
    rmse_x = np.sqrt(np.mean(x_err**2)) * 100.0          # cm
    max_drift_x = np.max(np.abs(x_err)) * 100.0           # cm
    # Steady error in [5.0, 6.0] s (indices 2500:3000)
    ess_x = np.mean(np.abs(x_err[2500:3000])) * 100.0     # cm

    # Settling time after steady wind onset (t >= 2.0 s, find time where |x_err| stays < 2.0 cm)
    tol_m = 0.02
    settled_idx = None
    for i in range(1000, 3000):  # during steady wind [2, 6] s
        if np.all(np.abs(x_err[i:3000]) < tol_m):
            settled_idx = i
            break
    if settled_idx is not None:
        ts_x = (settled_idx - 1000) * DT
    else:
        ts_x = 4.0  # did not settle within steady window

    # Matched z channel
    rmse_z = np.sqrt(np.mean(z_err**2)) * 100.0          # cm
    max_drift_z = np.max(np.abs(z_err)) * 100.0           # cm
    ess_z = np.mean(np.abs(z_err[2500:3000])) * 100.0     # cm

    # Matched theta channel
    rmse_th = np.sqrt(np.mean(th_deg**2))                 # deg
    max_th = np.max(np.abs(th_deg))                       # deg

    # Control effort
    energy = float(np.sum(us**2) * DT)
    peak_thrust = float(np.max(us))
    sat_pct = float(np.mean(us >= (quad.thrust_max - 1e-4)) * 100.0)

    metrics.append({
        "controller": name,
        "rmse_x": rmse_x,
        "max_drift_x": max_drift_x,
        "ess_x": ess_x,
        "ts_x": ts_x,
        "rmse_z": rmse_z,
        "max_drift_z": max_drift_z,
        "ess_z": ess_z,
        "rmse_th": rmse_th,
        "max_th": max_th,
        "energy": energy,
        "peak_thrust": peak_thrust,
        "sat_pct": sat_pct,
    })

# --- Export Markdown & CSV Tables --------------------------------------------
md_lines = [
    r"| Controller | Unmatched $x$ RMSE [cm] | $x$ Max Drift [cm] | $x$ $e_{ss}$ [cm] | $x$ $t_s$ [s] | Matched $z$ RMSE [cm] | $z$ Max Drift [cm] | $\theta$ RMSE [deg] | Energy $E_u$ [$\text{N}^2\cdot\text{s}$] | Peak Thrust $T_{\max}$ [N] | Sat [%] |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
]

for m_dict in metrics:
    md_lines.append(
        f"| {m_dict['controller']} | {m_dict['rmse_x']:.2f} | {m_dict['max_drift_x']:.2f} | {m_dict['ess_x']:.2f} | {m_dict['ts_x']:.2f} | "
        f"{m_dict['rmse_z']:.2f} | {m_dict['max_drift_z']:.2f} | {m_dict['rmse_th']:.2f} | "
        f"{m_dict['energy']:.4f} | {m_dict['peak_thrust']:.4f} | {m_dict['sat_pct']:.1f} |"
    )

table_md = "\n".join(md_lines) + "\n"
(EXP_DIR / "table.md").write_text(table_md, encoding="utf-8")

csv_lines = [
    "Controller,x_RMSE_cm,x_MaxDrift_cm,x_Ess_cm,x_ts_s,z_RMSE_cm,z_MaxDrift_cm,th_RMSE_deg,Energy_N2s,Peak_Thrust_N,Saturation_pct",
]
for m_dict in metrics:
    csv_lines.append(
        f"{m_dict['controller']},{m_dict['rmse_x']:.4f},{m_dict['max_drift_x']:.4f},{m_dict['ess_x']:.4f},{m_dict['ts_x']:.4f},"
        f"{m_dict['rmse_z']:.4f},{m_dict['max_drift_z']:.4f},{m_dict['rmse_th']:.4f},"
        f"{m_dict['energy']:.6f},{m_dict['peak_thrust']:.6f},{m_dict['sat_pct']:.2f}"
    )

table_csv = "\n".join(csv_lines) + "\n"
(EXP_DIR / "table.csv").write_text(table_csv, encoding="utf-8")

print("=== Benchmark Results ===")
print(table_md)

# --- High-Resolution Visualizations ------------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

colors = {
    "Nominal LQR": "#4A5568",          # Slate gray
    "Integral LQR (LQI)": "#3182CE",   # Blue
    "MRAC (Adaptive)": "#DD6B20",      # Orange
    "DOB + LQR": "#E53E3E",            # Vivid red
}

styles = {
    "Nominal LQR": ":",
    "Integral LQR (LQI)": "--",
    "MRAC (Adaptive)": "-.",
    "DOB + LQR": "-",
}

# 1. Unmatched Horizontal Position x(t)
ax_x = axs[0]
for name in controllers:
    ax_x.plot(time, results[name]["x"][:, 0] * 100.0, label=name, color=colors[name], linestyle=styles[name], linewidth=2.0)

ax_x.axhline(0.0, color="black", linestyle="--", alpha=0.4, label="Target (x=0)")
ax_x.axvspan(2.0, 6.0, color="#EDF2F7", alpha=0.7, label="Steady Wind (0.03 N)")
ax_x.axvspan(6.0, 8.0, color="#FEB2B2", alpha=0.4, label="Dynamic Gust (0.06 N)")
ax_x.set_ylabel("Horizontal $x$ [cm]", fontweight="bold")
ax_x.set_title("Unmatched Horizontal Position $x(t)$ (Virtual Tilt Reallocation vs Integrator Lag)", fontweight="bold")
ax_x.legend(loc="lower left", ncol=3, fontsize=9)
ax_x.grid(True, alpha=0.3)

# 2. Matched Vertical Position z(t)
ax_z = axs[1]
for name in controllers:
    ax_z.plot(time, results[name]["x"][:, 1], label=name, color=colors[name], linestyle=styles[name], linewidth=2.0)

ax_z.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="Target (z=1.0m)")
ax_z.axvspan(2.0, 6.0, color="#EDF2F7", alpha=0.7)
ax_z.axvspan(6.0, 8.0, color="#FEB2B2", alpha=0.4)
ax_z.set_ylabel("Altitude $z$ [m]", fontweight="bold")
ax_z.set_title("Matched Altitude $z(t)$ (Direct Thrust Compensation)", fontweight="bold")
ax_z.grid(True, alpha=0.3)

# 3. Pitch Angle theta(t)
ax_th = axs[2]
for name in controllers:
    ax_th.plot(time, np.rad2deg(results[name]["x"][:, 2]), label=name, color=colors[name], linestyle=styles[name], linewidth=2.0)

ax_th.axvspan(2.0, 6.0, color="#EDF2F7", alpha=0.7)
ax_th.axvspan(6.0, 8.0, color="#FEB2B2", alpha=0.4)
ax_th.set_ylabel("Pitch $\\theta$ [deg]", fontweight="bold")
ax_th.set_title("Attitude Pitch Angle $\\theta(t)$ (Wind-Opposing Tilt Dynamics)", fontweight="bold")
ax_th.grid(True, alpha=0.3)

# 4. DOB Disturbance Estimation Fidelity
ax_d = axs[3]
d_true = results["DOB + LQR"]["d_true"]
d_hat = results["DOB + LQR"]["d_hat"]

ax_d.plot(time, d_true[:, 0], color="#718096", linestyle="--", linewidth=2.0, label="True Lateral $F_{w,x}/m$")
ax_d.plot(time, d_hat[:, 0], color="#E53E3E", linewidth=2.0, label="DOB Estimate $\\hat{d}_x$")
ax_d.plot(time, d_true[:, 1], color="#A0AEC0", linestyle=":", linewidth=1.5, label="True Vertical $F_{w,z}/m$")
ax_d.plot(time, d_hat[:, 1], color="#3182CE", linestyle="-.", linewidth=1.5, label="DOB Estimate $\\hat{d}_z$")

ax_d.axvspan(2.0, 6.0, color="#EDF2F7", alpha=0.7)
ax_d.axvspan(6.0, 8.0, color="#FEB2B2", alpha=0.4)
ax_d.set_xlabel("Time [s]", fontweight="bold")
ax_d.set_ylabel("Disturbance Accel [m/s$^2$]", fontweight="bold")
ax_d.set_title("DOB Disturbance Reconstructed Acceleration ($\\omega_Q = 12\\text{ rad/s}$ Q-Filter)", fontweight="bold")
ax_d.legend(loc="upper right", ncol=2, fontsize=9)
ax_d.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(EXP_DIR / "dob_wind_rejection.png", dpi=300)
fig.savefig(EXP_DIR / "dob_wind_rejection.svg")
plt.close(fig)

print("Artifacts generated successfully in", EXP_DIR)
