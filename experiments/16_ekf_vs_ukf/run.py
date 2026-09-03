"""Experiment 16 - EKF vs UKF: when does linearisation cost you the estimate?

Both filters fuse a noisy nonlinear measurement of the pendulum with the model.
Two regimes:

  * Hard: h(x) = [sin theta, cos theta], a hopeless initial guess (~pi off) and a
    large prior. The EKF linearises h once about the (wrong) estimate and stays
    trapped in that basin; the UKF's spread sigma points sample the true
    measurement map and pull the estimate to the right angle.
  * Mild: h(x) = [sin theta, theta_dot], a good initial guess and a small prior.
    The nonlinearity over the covariance is negligible - EKF and UKF agree to a
    few percent, so the EKF's cheaper single evaluation is the right call.

Run:  python experiments/16_ekf_vs_ukf/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aimct.estimation import ExtendedKalmanFilter, UnscentedKalmanFilter
from aimct.plot_style import set_aimct_style
from aimct.simulate import rk4_step
from aimct.systems import Pendulum

HERE = Path(__file__).parent
DT = 0.02
_P = Pendulum()


def _f(x, u):
    return _P.dynamics(0.0, x, u)


def _true_step(x, u):
    return rk4_step(lambda t, a, b: _P.dynamics(0.0, a, b), 0.0, x, u, DT)


def run_case(name, h, R, x0_true, x0_est, P0, meas_every, meas_sigma,
             ekf_kw=None, ukf_kw=None, steps=400, seed=0):
    Q = np.diag([1e-5, 1e-3])
    kw = dict(dt=DT, n=2, x0=np.asarray(x0_est, float), P0=np.diag(P0))
    ekf = ExtendedKalmanFilter(_f, h, Q, R, **kw, **(ekf_kw or {}))
    ukf = UnscentedKalmanFilter(_f, h, Q, R, **kw, **(ukf_kw or {}))

    rng = np.random.default_rng(seed)
    xt = np.asarray(x0_true, float)
    log = {"t": [], "xt": [], "ekf": [], "ukf": [],
           "ekf_ptr": [], "ukf_ptr": [], "ekf_e": [], "ukf_e": []}
    for k in range(steps):
        u = np.array([0.4 * np.sin(0.05 * k)])
        xt = _true_step(xt, u)
        if k % meas_every == 0:
            y = np.atleast_1d(h(xt)) + rng.normal(0.0, meas_sigma, size=np.atleast_1d(h(xt)).size)
            ekf.step(y, u)
            ukf.step(y, u)
        else:
            ekf.predict(u)
            ukf.predict(u)
        log["t"].append(k * DT)
        log["xt"].append(xt.copy())
        log["ekf"].append(ekf.x_hat.copy())
        log["ukf"].append(ukf.x_hat.copy())
        log["ekf_ptr"].append(np.trace(ekf.P))
        log["ukf_ptr"].append(np.trace(ukf.P))
        log["ekf_e"].append(np.linalg.norm(ekf.x_hat - xt))
        log["ukf_e"].append(np.linalg.norm(ukf.x_hat - xt))
    for k in log:
        log[k] = np.array(log[k])

    def summ(err):
        return {
            "rms_err": round(float(np.sqrt(np.mean(err**2))), 3),
            "final_err": round(float(np.mean(err[-50:])), 3),
            "recovered": bool(np.mean(err[-50:]) < 0.3),
        }

    return name, {"EKF": summ(log["ekf_e"]), "UKF": summ(log["ukf_e"])}, log


def main():
    cases = [
        run_case(
            "hard: sin/cos, bad init",
            lambda x: np.array([np.sin(x[0]), np.cos(x[0])]),
            np.diag([1e-2, 1e-2]),
            x0_true=[3.1, 2.0], x0_est=[0.0, 0.0], P0=[15.0, 15.0],
            meas_every=5, meas_sigma=0.1,
            ukf_kw=dict(alpha=1.0, beta=2.0, kappa=0.0), seed=4,
        ),
        run_case(
            "mild: sin/rate, good init",
            lambda x: np.array([np.sin(x[0]), x[1]]),
            np.diag([1e-3, 1e-3]),
            x0_true=[3.0, 0.4], x0_est=[2.75, 0.0], P0=[0.2, 0.2],
            meas_every=1, meas_sigma=np.array([0.03, 0.03]),
            ukf_kw=dict(alpha=0.5), seed=1,
        ),
    ]

    rows = []
    for name, res, _ in cases:
        for filt in ("EKF", "UKF"):
            rows.append({"case": name, "filter": filt, **res[filt]})
    _write_tables(rows)
    _figure(cases)
    print((HERE / "table.md").read_text(encoding="utf-8"))


def _write_tables(rows):
    cols = ["case", "filter", "rms_err", "final_err", "recovered"]
    (HERE / "table.csv").write_text(
        ",".join(cols) + "\n"
        + "\n".join(",".join(str(r[c]) for c in cols) for r in rows) + "\n",
        encoding="utf-8", newline="\n")
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (HERE / "table.md").write_text(
        "# Experiment 16 - EKF vs UKF on a nonlinear pendulum measurement\n\n"
        "`recovered` = mean state-estimate error over the final 50 steps < 0.3.\n\n"
        + "\n".join([head, sep, *body]) + "\n",
        encoding="utf-8", newline="\n")


def _figure(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    (_, _, hard), (_, _, mild) = cases
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(12.0, 8.5))

    a.plot(hard["t"], hard["xt"][:, 0], "k-", lw=1.8, label="true theta")
    a.plot(hard["t"], hard["ekf"][:, 0], color="#D55E00", lw=1.4, label="EKF")
    a.plot(hard["t"], hard["ukf"][:, 0], color="#0072B2", lw=1.4, label="UKF")
    a.set_title("(a) Hard case: angle estimate"); a.set_xlabel("t [s]")
    a.set_ylabel("theta [rad]"); a.legend(fontsize=8)

    b.semilogy(hard["t"], hard["ekf_e"], color="#D55E00", lw=1.4, label="EKF")
    b.semilogy(hard["t"], hard["ukf_e"], color="#0072B2", lw=1.4, label="UKF")
    b.set_title("(b) Hard case: estimate error"); b.set_xlabel("t [s]")
    b.set_ylabel("|x_hat - x| (log)"); b.legend(fontsize=8)

    c.plot(hard["t"], hard["ekf_ptr"], color="#D55E00", lw=1.4, label="EKF tr(P)")
    c.plot(hard["t"], hard["ukf_ptr"], color="#0072B2", lw=1.4, label="UKF tr(P)")
    c.set_title("(c) Hard case: reported uncertainty tr(P)")
    c.set_xlabel("t [s]"); c.set_ylabel("tr(P)"); c.set_yscale("log"); c.legend(fontsize=8)

    d.plot(mild["t"], mild["ekf_e"], color="#D55E00", lw=1.4, label="EKF")
    d.plot(mild["t"], mild["ukf_e"], color="#0072B2", lw=1.2, ls="--", label="UKF")
    d.set_title("(d) Mild case: EKF and UKF agree"); d.set_xlabel("t [s]")
    d.set_ylabel("|x_hat - x|"); d.legend(fontsize=8)

    fig.suptitle("Exp 16 - EKF vs UKF: linearisation vs sigma points",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
