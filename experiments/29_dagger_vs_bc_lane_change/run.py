"""Experiment 29 - does DAgger recover where behaviour cloning drove off the road?

Experiment 27 Part B showed a behaviour-cloned LQR lane-keeper failing
catastrophically (5223 mm RMS, off the road) the moment the tyre model
changed under it: cloning LQR's input->output map did not clone its
self-correcting feedback structure. DAgger (Ross et al. 2011) is the textbook
fix - roll the *learner*, have the expert relabel every state the learner
actually visits under deployment conditions, aggregate, refit, repeat. This
experiment re-runs Exp 27's Part B (sharp lane change, low-mu Pacejka tyre)
with:

* the three classical controllers, unchanged (Stanley, LQR, kinematic MPC),
* **plain BC** - a pure behaviour clone of the LQR expert (no PPO), via
  ``aimct.rl.imitation.BehaviorCloning``,
* **DAgger** - the same clone, then N rounds of the LQR expert relabelling the
  states the student visits *on the Pacejka plant*.

Run:   python experiments/29_dagger_vs_bc_lane_change/run.py
       AIMCT_EXP_FULL=1 python experiments/29_dagger_vs_bc_lane_change/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from aimct.rl.imitation import BehaviorCloning, dagger
from aimct.simulate import rk4_step, simulate
from aimct.systems import BicycleVehicle

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

# reuse Experiment 27's reference, controllers, plant config and scorer
sys.path.insert(0, str(HERE.parent / "27_bicycle_double_lane_change"))
import run as exp27          # noqa: E402
from train_policy import _lqr_lane_change_expert, _obs, _OBS_SCALE   # noqa: E402

VX0, DT, T_FINAL = exp27.VX0, exp27.DT, exp27.T_FINAL
X0, U_BOUNDS = exp27.X0, exp27.U_BOUNDS
STRESS_TIRE = exp27.STRESS_TIRE
STRESS_WC = exp27.STRESS_WC

_expert_fn = _lqr_lane_change_expert(BicycleVehicle(), X0, VX0)


def _expert(x):
    return np.clip(_expert_fn(x, exp27.ref_at_X), U_BOUNDS[0], U_BOUNDS[1])


def _observe(x):
    return _obs(x, exp27.ref_at_X, VX0, _OBS_SCALE)


class _PolicyController:
    def __init__(self, bc, name):
        self.bc, self.name = bc, name

    def reset(self):
        pass

    def update(self, x, dt):
        return np.clip(self.bc.act(_observe(np.asarray(x, float))),
                       U_BOUNDS[0], U_BOUNDS[1])


# --------------------------------------------------------------- BC + DAgger

def _bc_dataset(plant, n_steps, seed):
    """Roll the LQR expert on ``plant`` (with small action noise), record
    ``(obs, expert action)`` at every visited state - the plain-BC training set."""
    rng = np.random.default_rng(seed)
    X, U = [], []
    x = X0.copy()
    t = 0.0
    for _ in range(n_steps):
        u_star = _expert(x)
        X.append(_observe(x))
        U.append(u_star)
        u_take = np.clip(u_star + rng.normal(0.0, [0.01, 0.05], 2),
                         U_BOUNDS[0], U_BOUNDS[1])
        x = rk4_step(plant.dynamics, t, x, u_take, DT)
        t += DT
        if t > T_FINAL or abs(x[1] - exp27.ref_at_X(x[0])[0]) > 5.0:
            x, t = X0.copy(), 0.0
    return np.asarray(X), np.asarray(U)


def _rollout_states(plant):
    def roll(act_fn):
        x = X0.copy()
        xs = [x.copy()]
        t = 0.0
        n = int(round(T_FINAL / DT))
        for _ in range(n):
            u = np.clip(np.asarray(act_fn(x), float), U_BOUNDS[0], U_BOUNDS[1])
            x = rk4_step(plant.dynamics, t, x, u, DT)
            t += DT
            xs.append(x.copy())
            if abs(x[1]) > 60.0:               # ran off - stop recording garbage
                break
        return np.asarray(xs)
    return roll


TRAIN_WC = 15.0                            # BC is cloned on the *gentle* manoeuvre only


def build_bc_and_dagger(stress_plant):
    """Both policies are behaviour-cloned from the LQR expert on the **gentle**
    (nominal-tyre) manoeuvre - so neither has seen a hard steer. DAgger then
    lets the expert relabel the states the student visits on the **aggressive
    Pacejka** plant; plain BC never gets that correction."""
    nominal = BicycleVehicle()
    keep = exp27.WC
    exp27.WC = TRAIN_WC
    Xbc, Ubc = _bc_dataset(nominal, 40_000 if FULL else 15_000, seed=0)
    exp27.WC = keep

    bc = BehaviorCloning(Xbc.shape[1], Ubc.shape[1],
                         act_low=U_BOUNDS[0], act_high=U_BOUNDS[1], seed=0)
    bc.fit(Xbc, Ubc, epochs=300 if FULL else 200, verbose=False)

    dbc = BehaviorCloning(Xbc.shape[1], Ubc.shape[1],
                          act_low=U_BOUNDS[0], act_high=U_BOUNDS[1], seed=0)
    dbc.fit(Xbc, Ubc, epochs=300 if FULL else 200, verbose=False)
    # DAgger rollouts + eval happen with exp27.WC already set to the sharp value
    dagger(dbc, rollout_states=_rollout_states(stress_plant),
           expert=_expert, observe=_observe,
           iterations=8 if FULL else 5,
           fit_kwargs=dict(epochs=150 if FULL else 120), verbose=True)
    return bc, dbc


# ------------------------------------------------------------------------ main

def main():
    global_WC = exp27.WC
    exp27.WC = STRESS_WC                       # the sharp Part-B lane change
    stress_veh = BicycleVehicle(**STRESS_TIRE)

    bc, dbc = build_bc_and_dagger(stress_veh)

    controllers = {
        "Stanley": exp27.Stanley(),
        "LQR": exp27.BicycleLQR(),
        "Kinematic MPC": exp27.KinematicMPC(),
        "Plain BC": _PolicyController(bc, "Plain BC"),
        "DAgger": _PolicyController(dbc, "DAgger"),
    }
    results = {name: exp27.run_one(ctrl, plant=stress_veh)
               for name, ctrl in controllers.items()}
    exp27.WC = global_WC

    cols = ["rms_err_mm", "max_err_mm", "final_err_mm", "peak_delta_deg", "ctrl_energy"]
    lines = ["# Experiment 29 - DAgger vs behaviour cloning on the Exp-27 Part-B "
            "lane change", "",
            f"aggressive double lane change ({STRESS_WC:.0f} m sharpness), "
            f"Pacejka tyre mu={STRESS_TIRE['mu']:.1f}, {VX0:.0f} m/s.", "",
            "| controller | " + " | ".join(cols) + " | status |",
            "| --- |" + " --- |" * (len(cols) + 1)]
    for name, m in results.items():
        status = "Diverged" if m["diverged"] else "OK"
        lines.append("| " + name + " | " + " | ".join(f"{m[c]:.4g}" for c in cols)
                     + f" | {status} |")
    lines.append("")
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["controller", *cols, "status"])
        for name, m in results.items():
            w.writerow([name, *(m[c] for c in cols),
                       "Diverged" if m["diverged"] else "OK"])

    _figure(results)
    print((HERE / "table.md").read_text())


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aimct.plot_style import PALETTE, set_aimct_style

    set_aimct_style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    cyc = [PALETTE["state_feedback"], PALETTE["lqr"], PALETTE["mpc"],
          PALETTE["rl"], PALETTE["hybrid"]]
    ref = next(iter(results.values()))["reference_Y"]
    ref_X = next(iter(results.values()))["trajectory"].x[:, 0]
    ax[0].plot(ref_X, ref, "--", color=PALETTE["reference"], lw=1.4, label="lane centre")
    for i, (name, m) in enumerate(results.items()):
        tr = m["trajectory"]
        ax[0].plot(tr.x[:, 0], tr.x[:, 1], color=cyc[i % len(cyc)], lw=1.6, label=name)
    ax[0].set(title="(a) Part-B lane change (Pacejka, mu=0.6)", xlabel="X [m]", ylabel="Y [m]")
    ax[0].legend(fontsize=8)

    labels = list(results)
    x = np.arange(len(labels))
    rms = [results[n]["rms_err_mm"] for n in labels]
    ax[1].bar(x, rms, color=[cyc[i % len(cyc)] for i in range(len(labels))])
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8, rotation=12)
    ax[1].set(title="(b) RMS lateral error", ylabel="mm")
    ax[1].set_yscale("log")

    fig.suptitle("Exp 29 - DAgger vs plain BC when the plant shifts under the policy",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
