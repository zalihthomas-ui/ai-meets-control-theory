"""Experiment 15 - Output-feedback quadrotor figure-8 tracking with an EKF.

The Crazyflie-2.0 planar quadrotor flies the same lemniscate as Experiment 14,
but now the controller does **not** see the true state.  Sensors give noisy
position + attitude ``[x, z, theta]`` and a gyro ``theta_dot`` - the two
translational velocities are never measured.  Three ways to close the loop:

  * LQR (full state)      - the unattainable ideal: perfect state feedback.
  * LQR + EKF             - an extended Kalman filter fuses the 4 noisy channels
                            with the nonlinear model to reconstruct all 6 states.
  * LQR + finite-diff vel - the naive alternative: difference the noisy position
                            channels to get velocity.

Run:  python experiments/15_quadrotor_ekf_output_feedback/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aimct.controllers import LQR
from aimct.estimation import ExtendedKalmanFilter
from aimct.plot_style import set_aimct_style
from aimct.simulate import rk4_step, simulate
from aimct.systems import PlanarQuadrotor

HERE = Path(__file__).parent

DT, DURATION = 0.004, 12.0
A_X, B_Z, PERIOD, Z0 = 0.5, 0.3, 8.0, 1.0
W = 2.0 * np.pi / PERIOD
SEED = 0

# Bryson-scaled LQR cost (same as Experiment 14)
_X_MAX = np.array([0.10, 0.10, 0.20, 0.5, 0.5, 3.0])
_U_MAX = np.array([0.15, 0.15])
Q = np.diag(1.0 / _X_MAX**2)
R = np.diag(1.0 / _U_MAX**2)

quad = PlanarQuadrotor()
G, M, IYY, L = quad.g, quad.m, quad.Iyy, quad.l
T_MAX = quad.thrust_max

# sensor model: y = [x, z, theta, theta_dot] + noise
MEAS_IDX = np.array([0, 1, 2, 5])
MEAS_SIGMA = np.array([3e-3, 3e-3, np.radians(0.3), 1e-2])   # m, m, rad, rad/s


def reference(t: float):
    """(x_ref_state[6], u_ref[2]) for the lemniscate with flatness feed-forward."""
    s1, c1 = np.sin(W * t), np.cos(W * t)
    s2, c2 = np.sin(2 * W * t), np.cos(2 * W * t)
    x = A_X * s1
    xd = A_X * W * c1
    xdd = -A_X * W**2 * s1
    xddd = -A_X * W**3 * c1
    xdddd = A_X * W**4 * s1
    z, zd, zdd = Z0 + B_Z * s2, 2 * B_Z * W * c2, -4 * B_Z * W**2 * s2
    th, thd, thdd_ff = -xdd / G, -xddd / G, -xdddd / G
    T_tot = M * (G + zdd)
    delta = IYY * thdd_ff / L
    u_ref = np.clip(np.array([0.5 * T_tot + 0.5 * delta, 0.5 * T_tot - 0.5 * delta]),
                    0.0, T_MAX)
    return np.array([x, z, th, xd, zd, thd]), u_ref


# ---------------------------------------------------------------- estimators

class _EKFEstimator:
    def __init__(self, x0):
        H = np.zeros((4, 6))
        H[np.arange(4), MEAS_IDX] = 1.0
        self.ekf = ExtendedKalmanFilter(
            f=lambda x, u: quad.dynamics(0.0, x, u),
            h=lambda x: x[MEAS_IDX],
            Q=np.diag([1e-9, 1e-9, 1e-9, 1e-5, 1e-5, 1e-4]),
            R=np.diag(MEAS_SIGMA**2),
            dt=DT, n=6, H_jac=lambda x: H,
            x0=x0, P0=np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2]),
        )

    def reset(self):
        self.ekf.reset()

    def estimate(self, y, u_prev):
        return self.ekf.step(y, u_prev)


class _FiniteDiffEstimator:
    """Difference the noisy position/attitude channels for velocity (1-pole LPF)."""
    def __init__(self, x0):
        self._x0 = np.asarray(x0, float)
        self.reset()

    def reset(self):
        self._prev = self._x0[MEAS_IDX[:3]].copy()
        self._vel = self._x0[3:].copy()

    def estimate(self, y, u_prev):
        raw = (y[:3] - self._prev) / DT
        self._vel = 0.6 * self._vel + 0.4 * raw          # mild low-pass
        self._prev = y[:3].copy()
        return np.array([y[0], y[1], y[2], self._vel[0], self._vel[1], y[3]])


# ------------------------------------------------------------------ tracker

class OutputFeedbackTracker:
    def __init__(self, K, estimator=None):
        self.K = K
        self.estimator = estimator
        self.reset()

    def reset(self):
        self._t = 0.0
        self._u_prev = quad.u_hover.copy()
        if self.estimator is not None:
            self.estimator.reset()
        self.x_hat_log: list[np.ndarray] = []

    def update(self, measurement, dt):
        y = np.asarray(measurement, float)
        if self.estimator is None:
            x_hat = y                                    # full-state feedback
        else:
            x_hat = np.asarray(self.estimator.estimate(y, self._u_prev), float)
        self.x_hat_log.append(x_hat.copy())
        xr, ur = reference(self._t)
        u = np.clip(ur - self.K @ (x_hat - xr), 0.0, T_MAX)
        self._u_prev = u
        self._t += dt
        return u


# ------------------------------------------------------------------- driver

def run_case(estimator_factory, noise):
    x0 = np.array([0.0, Z0, 0.0, 0.0, 0.0, 0.0])
    K = LQR(*quad.linearize(), Q, R).K

    if estimator_factory is None:                        # ideal full-state feedback
        tracker = OutputFeedbackTracker(K, None)
        traj = simulate(quad, tracker, x0=x0, dt=DT, t_final=DURATION,
                        u_bounds=(0.0, T_MAX))           # no measurement_fn -> true state
        return traj, None

    tracker = OutputFeedbackTracker(K, estimator_factory(x0 + 1e-3))

    def measure(t, x, u):
        k = min(int(round(t / DT)), len(noise) - 1)
        return x[MEAS_IDX] + noise[k]

    traj = simulate(quad, tracker, x0=x0, dt=DT, t_final=DURATION,
                    u_bounds=(0.0, T_MAX), measurement_fn=measure)
    return traj, tracker


def metrics(traj):
    ref = np.array([reference(tt)[0] for tt in traj.t])
    pos_err = np.hypot(traj.x[:, 0] - ref[:, 0], traj.x[:, 1] - ref[:, 1])
    du = traj.u - quad.u_hover
    return {
        "rms_pos_err_mm": round(float(np.sqrt(np.mean(pos_err**2)) * 1e3), 2),
        "max_pos_err_mm": round(float(np.max(pos_err) * 1e3), 2),
        "rms_pitch_deg": round(float(np.degrees(np.sqrt(np.mean((traj.x[:, 2] - ref[:, 2])**2)))), 3),
        "ctrl_energy": round(float(np.trapezoid(np.sum(du**2, axis=1), traj.t)), 4),
    }


def vel_est_error(traj, tracker):
    if tracker is None or not tracker.x_hat_log:
        return None
    xh = np.array(tracker.x_hat_log)
    n = min(len(xh), len(traj.x) - 1)
    return round(float(np.sqrt(np.mean((xh[:n, 3:5] - traj.x[:n, 3:5])**2))), 4)


def main():
    rng = np.random.default_rng(SEED)
    n_steps = int(round(DURATION / DT)) + 2
    noise = rng.normal(0.0, 1.0, size=(n_steps, 4)) * MEAS_SIGMA

    cases = [
        ("LQR (full state)", None),
        ("LQR + EKF", _EKFEstimator),
        ("LQR + finite-diff vel", _FiniteDiffEstimator),
    ]
    rows, trajs, trackers = {}, {}, {}
    for name, fac in cases:
        tr, trk = run_case(fac, noise)
        m = metrics(tr)
        v = vel_est_error(tr, trk)
        if v is not None:
            m["rms_vel_est_err"] = v
        rows[name], trajs[name], trackers[name] = m, tr, trk

    _write_tables(rows)
    _figure(trajs, trackers)
    print((HERE / "table.md").read_text(encoding="utf-8"))


def _write_tables(rows):
    cols = ["rms_pos_err_mm", "max_pos_err_mm", "rms_pitch_deg", "ctrl_energy",
            "rms_vel_est_err"]
    lines = ["# Experiment 15 - output-feedback quadrotor tracking (EKF)", "",
             "Noisy [x, z, theta] + gyro; velocities unmeasured. "
             f"sigma = {MEAS_SIGMA.tolist()} (m, m, rad, rad/s).", "",
             "| controller | " + " | ".join(cols) + " |",
             "| --- |" + " --- |" * len(cols)]
    for name, m in rows.items():
        lines.append(f"| {name} | "
                     + " | ".join(str(m.get(c, "--")) for c in cols) + " |")
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with open(HERE / "table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", *cols])
        for name, m in rows.items():
            w.writerow([name, *(m.get(c, "") for c in cols)])


def _figure(trajs, trackers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(12.0, 9.0))
    tref = np.arange(0, DURATION, DT)
    ref = np.array([reference(tt)[0] for tt in tref])
    colors = {"LQR (full state)": "#999999", "LQR + EKF": "#0072B2",
              "LQR + finite-diff vel": "#D55E00"}

    a.plot(ref[:, 0], ref[:, 1], "k--", lw=1.2, label="reference")
    for name, tr in trajs.items():
        a.plot(tr.x[:, 0], tr.x[:, 1], color=colors[name], lw=1.6, label=name)
    a.set_title("(a) Flight path (x-z plane)"); a.set_xlabel("x [m]"); a.set_ylabel("z [m]")
    a.set_aspect("equal", "box"); a.legend(fontsize=8)

    for name, tr in trajs.items():
        r = np.array([reference(tt)[0] for tt in tr.t])
        b.plot(tr.t, np.hypot(tr.x[:, 0] - r[:, 0], tr.x[:, 1] - r[:, 1]) * 1e3,
               color=colors[name], lw=1.4, label=name)
    b.set_title("(b) Position tracking error"); b.set_xlabel("t [s]")
    b.set_ylabel("|pos err| [mm]"); b.legend(fontsize=8)

    ekf_tr, ekf_trk = trajs["LQR + EKF"], trackers["LQR + EKF"]
    xh = np.array(ekf_trk.x_hat_log)
    n = min(len(xh), len(ekf_tr.x))
    c.plot(ekf_tr.t[:n], ekf_tr.x[:n, 3], color="#000000", lw=1.6, label="true xdot")
    c.plot(ekf_tr.t[:n], xh[:n, 3], color="#0072B2", lw=1.2, ls="--", label="EKF xdot")
    c.plot(ekf_tr.t[:n], ekf_tr.x[:n, 4], color="#555555", lw=1.6, label="true zdot")
    c.plot(ekf_tr.t[:n], xh[:n, 4], color="#009E73", lw=1.2, ls="--", label="EKF zdot")
    c.set_title("(c) EKF reconstructs the unmeasured velocities")
    c.set_xlabel("t [s]"); c.set_ylabel("velocity [m/s]"); c.legend(fontsize=8)

    for name, tr in trajs.items():
        d.plot(tr.t, np.degrees(tr.x[:, 2]), color=colors[name], lw=1.2, label=name)
    d.set_title("(d) Pitch angle"); d.set_xlabel("t [s]"); d.set_ylabel("theta [deg]")
    d.legend(fontsize=8)

    fig.suptitle("Exp 15 - output-feedback quadrotor tracking: EKF vs naive vs ideal",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
