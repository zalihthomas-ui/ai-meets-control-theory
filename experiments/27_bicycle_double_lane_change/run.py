"""Experiment 27 - a double lane-change at speed on the dynamic bicycle model.

Four controllers steer the *dynamic* single-track vehicle (lateral tire
forces, linear tire by default) through an ISO-3888-style double lane change
at 25 m/s (~90 km/h): shift one lane width left, then back, over a short
distance - hard enough that lateral tire force, not just kinematics, matters.

* **Stanley** - classic heading + cross-track steering law, model-free.
* **LQR** - linearised about the constant-speed cruise (the *true* dynamic
  model, including tire stiffness).
* **Kinematic MPC** - `LinearMPC` on the *kinematic* bicycle model (no tire
  slip - `Ydot = vx sin(psi)`, `psidot = vx tan(delta) / L`), driving the real
  dynamic vehicle. Tests whether ignoring lateral tire dynamics costs
  anything at this speed - the Experiment-02 linearisation-validity question,
  one level up.
* **RL policy** - PPO, trained directly on the task (see README for whether
  it bootstraps or needs a behaviour-cloned start, as Experiment 21's did).

Longitudinal speed is held to the cruise target by an identical simple P
controller on `ax` in every entry; the comparison is purely about the
*steering* law.

Run:   python experiments/27_bicycle_double_lane_change/run.py
       AIMCT_EXP_FULL=1 python experiments/27_bicycle_double_lane_change/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aimct.controllers import LQR, LinearMPC
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.simulate import simulate
from aimct.systems import BicycleVehicle

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

# --------------------------------------------------------------- the manoeuvre

VX0 = 25.0                    # cruise speed [m/s] (~90 km/h)
DY, X1, X2, WC = 3.5, 50.0, 120.0, 15.0   # lane width, shift centres, sharpness
T_FINAL = 8.0
DT = 0.01
KP_V = 0.5                    # longitudinal speed-hold gain (shared by every entry)


def ref_at_X(X):
    """Lane-change lateral offset + slope at longitudinal distance X."""
    z1, z2 = (X - X1) / WC, (X - X2) / WC
    Y = 0.5 * DY * (np.tanh(z1) + 1.0) - 0.5 * DY * (np.tanh(z2) + 1.0)
    dY = (0.5 * DY / WC * (1.0 - np.tanh(z1) ** 2)
          - 0.5 * DY / WC * (1.0 - np.tanh(z2) ** 2))
    return Y, dY


def ref_at_t(t):
    """Same reference, parametrised by time at the cruise speed - X(t) = vx0 t."""
    return ref_at_X(VX0 * t)


veh = BicycleVehicle()
X0 = np.array([0.0, 0.0, 0.0, VX0, 0.0, 0.0])
U_BOUNDS = (np.array([-veh.delta_max, -veh.ax_max]),
           np.array([veh.delta_max, veh.ax_max]))


def _ax_hold(vx):
    return KP_V * (VX0 - vx)


# --------------------------------------------------------------- controllers

class Stanley:
    name = "Stanley"
    K_E, K_PSI = 0.5, 1.2

    def reset(self):
        pass

    def update(self, x, dt):
        X, Y, psi, vx, vy, r = x
        Yref, dYref = ref_at_X(X)
        psi_ref = np.arctan(dYref)
        e, psi_e = Y - Yref, psi - psi_ref
        delta = -(self.K_PSI * psi_e + np.arctan2(self.K_E * e, max(vx, 1.0)))
        return np.array([delta, _ax_hold(vx)])


class BicycleLQR:
    name = "LQR"

    def __init__(self):
        A, B = veh.linearize(X0)
        # X gets a tiny (nonzero) weight for CARE detectability - it is a free
        # integrator at this equilibrium and is not otherwise interesting to track
        Q = np.diag([0.01, 30.0, 8.0, 1.0, 1.0, 1.0])
        R = np.diag([80.0, 1.0])
        self.K = LQR(A, B, Q, R).K

    def reset(self):
        pass

    def update(self, x, dt):
        X, Y, psi, vx, vy, r = x
        Yref, dYref = ref_at_X(X)
        psi_ref = np.arctan(dYref)
        x_ref = np.array([X, Yref, psi_ref, VX0, 0.0, 0.0])
        u = -self.K @ (np.asarray(x) - x_ref)
        u[1] = _ax_hold(vx)          # speed-hold handled uniformly, not by the LQR
        return u


class KinematicMPC:
    """LinearMPC on the *kinematic* bicycle (no tire slip), steering the true
    dynamic vehicle - do you need lateral tire dynamics in the plan, or does
    a simpler model suffice at this speed?"""

    name = "Kinematic MPC"

    def __init__(self, N=30):
        L = veh.a + veh.b
        A = np.array([[0.0, VX0], [0.0, 0.0]])          # state [Y, psi], input [delta]
        B = np.array([[0.0], [VX0 / L]])
        self.mpc = LinearMPC(A, B, Q=np.diag([30.0, 8.0]), R=np.array([[80.0]]),
                             N=N, u_bounds=(-veh.delta_max, veh.delta_max))
        self.N, self.dt_plan = N, DT

    def reset(self):
        self.mpc.reset()

    def update(self, x, dt):
        X, Y, psi, vx, vy, r = x
        xref = np.zeros((self.N, 2))
        for j in range(self.N):
            Yr, dYr = ref_at_X(X + VX0 * (j + 1) * self.dt_plan)
            xref[j] = [Yr, np.arctan(dYr)]
        self.mpc.x_ref = xref
        delta = float(np.asarray(self.mpc.update(np.array([Y, psi]), dt)).reshape(1)[0])
        return np.array([delta, _ax_hold(vx)])


# ------------------------------------------------------------------- scoring

def run_one(ctrl, plant=None):
    """Simulate ``ctrl`` on ``plant`` (default: the nominal design model
    ``veh``). Every controller was designed / trained against ``veh``'s linear
    tire; passing a different plant (Part B) tests generalisation - the
    controller never sees the swap."""
    plant = veh if plant is None else plant
    ctrl.reset()
    tr = simulate(plant, ctrl, x0=X0, dt=DT, t_final=T_FINAL, u_bounds=U_BOUNDS)
    Yref = np.array([ref_at_X(X)[0] for X in tr.x[:, 0]])
    err = tr.x[:, 1] - Yref
    du = tr.u - np.array([0.0, 0.0])
    return dict(
        trajectory=tr, reference_Y=Yref,
        rms_err_mm=float(np.sqrt(np.mean(err ** 2)) * 1e3),
        max_err_mm=float(np.max(np.abs(err)) * 1e3),
        final_err_mm=float(abs(err[-1]) * 1e3),
        peak_delta_deg=float(np.degrees(np.max(np.abs(tr.u[:, 0])))),
        ctrl_energy=float(np.trapezoid(np.sum(du ** 2, axis=1), tr.t)),
        diverged=tr.diverged,
    )


def build(rl_policy=None):
    c = {"Stanley": Stanley(), "LQR": BicycleLQR(), "Kinematic MPC": KinematicMPC()}
    if rl_policy is not None:
        c["RL (PPO)"] = rl_policy
    return c


STRESS_WC = 6.0                        # Part B: a much sharper lane change
STRESS_TIRE = dict(tire_model="pacejka", mu=0.6)   # ...on a wet, grippy-limited road


# ------------------------------------------------------------------------ main

def main():
    global WC

    rl_policy = None
    try:
        from train_policy import load_or_train_policy
        rl_policy = load_or_train_policy(veh, ref_at_X, X0, U_BOUNDS, full=FULL)
    except Exception as exc:                          # pragma: no cover
        print(f"RL entry skipped: {exc}")

    # Part A: the nominal manoeuvre, linear tire (every controller's own
    # design/training assumption)
    controllers_a = build(rl_policy)
    results_a = {name: run_one(ctrl) for name, ctrl in controllers_a.items()}

    # Part B: a sharper lane change on a low-mu Pacejka plant - none of the
    # controllers are told the plant changed; same instances, new test
    WC = STRESS_WC
    stress_veh = BicycleVehicle(**STRESS_TIRE)
    controllers_b = build(rl_policy)
    for ctrl in controllers_b.values():                # rebuild any internal state
        if hasattr(ctrl, "reset"):
            ctrl.reset()
    results_b = {name: run_one(ctrl, plant=stress_veh) for name, ctrl in controllers_b.items()}
    WC = 15.0                                          # restore for anything after

    cols = ["rms_err_mm", "max_err_mm", "final_err_mm", "peak_delta_deg", "ctrl_energy"]
    lines = ["# Experiment 27 - double lane change on the dynamic bicycle model", ""]

    lines += [f"## Part A - nominal ({VX0:.0f} m/s, {DY:.1f} m shift over "
             f"~{X2 - X1:.0f} m, linear tire, {T_FINAL:.0f} s)", "",
             "| controller | " + " | ".join(cols) + " | status |",
             "| --- |" + " --- |" * (len(cols) + 1)]
    for name, m in results_a.items():
        status = "Diverged" if m["diverged"] else "OK"
        lines.append("| " + name + " | " + " | ".join(f"{m[c]:.4g}" for c in cols)
                     + f" | {status} |")

    lines += ["", f"## Part B - aggressive ({STRESS_WC:.0f} m sharpness, "
             f"Pacejka tire, mu={STRESS_TIRE['mu']:.1f}, controllers unaware "
             f"of the swap)", "",
             "| controller | " + " | ".join(cols) + " | status |",
             "| --- |" + " --- |" * (len(cols) + 1)]
    for name, m in results_b.items():
        status = "Diverged" if m["diverged"] else "OK"
        lines.append("| " + name + " | " + " | ".join(f"{m[c]:.4g}" for c in cols)
                     + f" | {status} |")
    lines.append("")
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "controller", *cols, "status"])
        for name, m in results_a.items():
            w.writerow(["A_nominal", name, *(m[c] for c in cols),
                       "Diverged" if m["diverged"] else "OK"])
        for name, m in results_b.items():
            w.writerow(["B_aggressive", name, *(m[c] for c in cols),
                       "Diverged" if m["diverged"] else "OK"])

    _figure(results_a, results_b)
    print((HERE / "table.md").read_text())


def _figure(results_a, results_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    cyc = [PALETTE["state_feedback"], PALETTE["lqr"], PALETTE["mpc"], PALETTE["rl"]]

    for row, (results, title) in enumerate(
            [(results_a, "Part A - nominal (linear tire)"),
             (results_b, "Part B - aggressive (Pacejka, mu=0.6)")]):
        ap, ab = ax[row, 0], ax[row, 1]
        ref = next(iter(results.values()))["reference_Y"]     # sampled at the WC in force
        ref_X = next(iter(results.values()))["trajectory"].x[:, 0]
        ap.plot(ref_X, ref, "--", color=PALETTE["reference"], lw=1.4, label="lane centre")
        for i, (name, m) in enumerate(results.items()):
            tr = m["trajectory"]
            ap.plot(tr.x[:, 0], tr.x[:, 1], color=cyc[i % len(cyc)], lw=1.6, label=name)
        ap.set(title=f"({'a' if row == 0 else 'c'}) {title}", xlabel="X [m]", ylabel="Y [m]")
        ap.legend(fontsize=7)

        labels = list(results)
        x = np.arange(len(labels))
        rms = [results[n]["rms_err_mm"] for n in labels]
        ab.bar(x, rms, color=[cyc[i % len(cyc)] for i in range(len(labels))])
        ab.set_xticks(x); ab.set_xticklabels(labels, fontsize=8)
        ab.set(title=f"({'b' if row == 0 else 'd'}) RMS lateral error", ylabel="mm")

    fig.suptitle("Exp 27 - dynamic bicycle model: double lane change at 25 m/s",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
