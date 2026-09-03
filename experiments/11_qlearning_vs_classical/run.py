"""Experiment 11 - Tabular Q-learning vs classical control on the pendulum.

Two head-to-heads on the same nonlinear ``Pendulum`` (torque |u| <= 4):

  * Swing-up (from hanging): tabular Q-learning vs an energy-shaping law that
    pumps mechanical energy toward the upright separatrix and hands off to LQR.
  * Balance (from near upright): tabular Q-learning vs the LQR itself.

Scored on: does it reach / hold upright, control effort, **environment samples
consumed to get there**, and the size of the resulting controller
(interpretability).

Run:  python experiments/11_qlearning_vs_classical/run.py            # quick
      AIMCT_EXP_FULL=1 python .../run.py                             # full training
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from aimct.controllers import LQR
from aimct.rl.env import make, wrap_to_pi
from aimct.rl.tabular import Discretizer, QLearning, evaluate, train
from aimct.simulate import simulate
from aimct.systems import Pendulum

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

U_MAX = 4.0
DT = 0.05
_P = Pendulum()
_J = _P.m * _P.L**2
_A, _B = _P.linearize()                                   # about upright
_K = LQR(_A, _B, np.diag([10.0, 1.0]), [[0.5]]).K         # balance gain

SWINGUP_EPISODES = 1500 if FULL else 250
BALANCE_EPISODES = 900 if FULL else 200


# --------------------------------------------------------------- classical

def _energy(theta, omega):
    return 0.5 * _J * omega**2 + _P.m * _P.g * _P.L * (-np.cos(theta) - 1.0)


def energy_swingup_controller(k_energy: float = 1.5):
    """Energy pumping (u = -k_E * sign(omega) * E) with an LQR catch near upright."""
    def law(x, dt):
        theta, omega = x
        err = wrap_to_pi(theta - np.pi)
        if abs(err) < 0.4 and abs(omega) < 3.0:
            return float(np.clip(-(_K @ np.array([err, omega]))[0], -U_MAX, U_MAX))
        s = np.sign(omega) or 1.0
        return float(np.clip(-k_energy * s * _energy(theta, omega), -U_MAX, U_MAX))
    law.reset = lambda: None
    return law


def lqr_balance_controller():
    def law(x, dt):
        err = wrap_to_pi(x[0] - np.pi)
        return float(np.clip(-(_K @ np.array([err, x[1]]))[0], -U_MAX, U_MAX))
    law.reset = lambda: None
    return law


# ---------------------------------------------------------------- rollout

class _shim:
    """Minimal (t, x, u) stand-in for metrics()/plotting from an RL rollout."""
    def __init__(self, states, actions):
        self.x = np.asarray(states, float)
        self.t = np.arange(len(self.x)) * DT
        a = np.asarray(actions, float).reshape(-1, 1)
        self.u = a if len(a) == len(self.x) else np.zeros((len(self.x), 1))


def rollout(controller, x0, t_final):
    traj = simulate(_P, controller, x0=np.asarray(x0, float), dt=DT,
                    t_final=t_final, u_bounds=(-U_MAX, U_MAX))
    err = np.abs(wrap_to_pi(traj.x[:, 0] - np.pi))
    return traj, err


def metrics(name, kind, traj, err, samples, n_params):
    up = np.where(err < 0.2)[0]
    held = float(np.mean(err[-20:])) < 0.15
    return {
        "controller": name,
        "task": kind,
        "min_err_rad": round(float(err.min()), 3),
        "held_upright": bool(held),
        "t_upright_s": round(float(traj.t[up[0]]), 2) if up.size else float("inf"),
        "final_err_rad": round(float(np.mean(err[-20:])), 3),
        "control_energy": round(float(np.sum(traj.u[:, 0] ** 2) * DT), 1),
        "train_samples": int(samples),
        "n_params": int(n_params),
    }


# ------------------------------------------------------------------- main

def main() -> None:
    rows = []

    # ---- swing-up: energy shaping ---------------------------------------
    es = energy_swingup_controller()
    tr, er = rollout(es, [0.0, 0.0], 15.0)
    rows.append(metrics("energy-shaping + LQR", "swing-up", tr, er, 0, 3))
    es_traj = tr

    # ---- swing-up: tabular Q-learning ---------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make("pendulum-swingup", max_steps=300)
        disc_s = Discretizer([-1, -1, -10], [1, 1, 10], [15, 15, 25], -U_MAX, U_MAX, 11)
        ag_s = QLearning(disc_s.n_states, disc_s.n_actions, alpha=0.25, gamma=0.99,
                         epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9975, seed=0)
        res_s = train(env, ag_s, disc_s, episodes=SWINGUP_EPISODES, seed=0)
        ev_s = evaluate(env, ag_s, disc_s, episodes=10)
    q_swing_traj = ev_s["states"]
    q_err = np.abs(wrap_to_pi(q_swing_traj[:, 0] - np.pi))
    q_swing_shim = _shim(q_swing_traj, ev_s["actions"])
    rows.append(metrics("tabular Q-learning", "swing-up", q_swing_shim, q_err,
                        SWINGUP_EPISODES * 300, ag_s.Q.size))

    # ---- balance: LQR -------------------------------------------------
    x0_bal = np.array([np.pi + 0.25, 0.4])
    tr, er = rollout(lqr_balance_controller(), x0_bal, 6.0)
    rows.append(metrics("LQR", "balance", tr, er, 0, 2))
    lqr_bal_traj = tr

    # ---- balance: tabular Q-learning near upright -------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env_b = make("pendulum-swingup", max_steps=200,
                     x0=np.array([np.pi, 0.0]),
                     reset_noise=np.array([0.25, 0.4]))
        disc_b = Discretizer([-1, -1, -6], [1, 1, 6], [15, 15, 21], -U_MAX, U_MAX, 11)
        ag_b = QLearning(disc_b.n_states, disc_b.n_actions, alpha=0.3, gamma=0.99,
                         epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.995, seed=1)
        res_b = train(env_b, ag_b, disc_b, episodes=BALANCE_EPISODES, seed=1)
        # deterministic eval from the same x0 the LQR faced
        env_b.reset(seed=0)
        env_b._x = x0_bal.copy()
        states, actions = [], []
        for _ in range(120):
            s = disc_b.encode(env_b._obs())
            states.append(env_b.state)
            a = ag_b.act(s, greedy=True)
            actions.append(float(disc_b.actions[a]))
            _, _, term, trunc, _ = env_b.step(disc_b.action(a))
            if term or trunc:
                break
    q_bal_shim = _shim(states, actions)
    q_bal_traj = q_bal_shim.x
    q_bal_err = np.abs(wrap_to_pi(q_bal_traj[:, 0] - np.pi))
    rows.append(metrics("tabular Q-learning", "balance", q_bal_shim, q_bal_err,
                        BALANCE_EPISODES * 200, ag_b.Q.size))

    _write_tables(rows)
    _figure(es_traj, q_swing_traj, res_s.returns, lqr_bal_traj, q_bal_traj,
            ag_b, disc_b)

    print("wrote:", ", ".join(p.name for p in sorted(HERE.glob("*"))
                              if p.suffix in {".md", ".csv", ".png"}))
    print()
    print((HERE / "table.md").read_text(encoding="utf-8"))


def _write_tables(rows):
    cols = list(rows[0].keys())
    (HERE / "table.csv").write_text(
        ",".join(cols) + "\n"
        + "\n".join(",".join(str(r[c]) for c in cols) for r in rows) + "\n",
        encoding="utf-8", newline="\n")
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (HERE / "table.md").write_text(
        "# Experiment 11 - tabular Q-learning vs classical control (pendulum)\n\n"
        f"Torque |u| <= {U_MAX:g}. `train_samples` = env steps consumed to learn "
        "the policy (0 for the model-based classical laws). `n_params` = size of "
        "the resulting controller.\n\n"
        + "\n".join([head, sep, *body]) + "\n",
        encoding="utf-8", newline="\n")


def _figure(es_traj, q_swing, returns, lqr_bal, q_bal, ag_b, disc_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aimct.plot_style import set_aimct_style

    set_aimct_style()
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(12.0, 8.5))

    a.plot(es_traj.t, np.degrees(wrap_to_pi(es_traj.x[:, 0] - np.pi)),
           color="#009E73", lw=2.0, label="energy-shaping + LQR")
    a.plot(np.arange(len(q_swing)) * DT,
           np.degrees(wrap_to_pi(q_swing[:, 0] - np.pi)),
           color="#0072B2", lw=2.0, label="tabular Q-learning")
    a.axhline(0, color="#555555", ls="--", lw=1.0)
    a.set_title("(a) Swing-up: angle to upright"); a.set_xlabel("time [s]")
    a.set_ylabel("angle [deg]"); a.legend(fontsize=8)

    b.plot(returns, color="#0072B2", lw=1.0)
    b.set_title("(b) Q-learning swing-up: return per episode")
    b.set_xlabel("episode"); b.set_ylabel("episode return")

    c.plot(lqr_bal.t, np.degrees(wrap_to_pi(lqr_bal.x[:, 0] - np.pi)),
           color="#D55E00", lw=2.0, label="LQR")
    c.plot(np.arange(len(q_bal)) * DT,
           np.degrees(wrap_to_pi(q_bal[:, 0] - np.pi)),
           color="#0072B2", lw=2.0, label="tabular Q-learning")
    c.axhline(0, color="#555555", ls="--", lw=1.0)
    c.set_title("(c) Balance from a tilt: angle to upright"); c.set_xlabel("time [s]")
    c.set_ylabel("angle [deg]"); c.legend(fontsize=8)

    # greedy balance policy on a (cos=1 slice) sin-omega grid
    b0, b1, b2 = disc_b.bins
    pol = ag_b.greedy_policy().reshape(b0, b1, b2)
    grid = disc_b.actions[pol[b0 // 2]]            # cos ~ upright slice: [sin, omega]
    im = d.imshow(grid.T, origin="lower", aspect="auto", cmap="coolwarm",
                  extent=[-1, 1, disc_b.low[2], disc_b.high[2]])
    d.set_title("(d) Q-learning balance: greedy torque\n(sin theta vs omega, near upright)")
    d.set_xlabel(r"$\sin\theta$"); d.set_ylabel(r"$\omega$ [rad/s]")
    fig.colorbar(im, ax=d, label="torque [N m]")

    fig.suptitle("Exp 11 - tabular Q-learning vs classical control (pendulum)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
