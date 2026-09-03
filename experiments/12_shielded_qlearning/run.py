"""Experiment 12 - Shielded tabular Q-learning on the pendulum.

The Q-learning swing-up agent from Experiment 11 gets the pole *up* but its
coarse grid limit-cycles near vertical (|theta - pi| ~ 1.4 rad, never held) - it
whips through vertical at 3-5 rad/s, far outside any bounded-torque LQR basin.

Wrap it in a safety shield: the RL policy drives while it is far from upright;
once it brings the state within `angle_handoff` of vertical, a classical
finisher (energy shaping + LQR) takes over and completes / holds the swing-up.

  is_safe(x) = |wrap(theta - pi)| > angle_handoff        # True -> RL drives

Run:  python experiments/12_shielded_qlearning/run.py            # quick
      AIMCT_EXP_FULL=1 python .../run.py                         # full training
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from aimct.controllers import LQR, wrap_angle
from aimct.hybrid import ShieldedController
from aimct.rl import Discretizer, GreedyPolicy, QLearning, make, train
from aimct.simulate import simulate
from aimct.systems import Pendulum

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

U_MAX, DT = 4.0, 0.05
EPISODES = 1500 if FULL else 250
ANGLE_HANDOFF = 1.0                       # rad from upright: RL -> classical finisher

_P = Pendulum()
_J = _P.m * _P.L**2
_A, _B = _P.linearize()
_K = LQR(_A, _B, np.diag([50.0, 2.0]), [[0.5]]).K


def _obs(x):
    return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])


def _energy(theta, omega):
    return 0.5 * _J * omega**2 + _P.m * _P.g * _P.L * (-np.cos(theta) - 1.0)


def energy_lqr_finisher(x, dt):
    """Energy pumping with an LQR catch near upright (the Exp-11 classical law)."""
    theta, omega = x
    err = wrap_angle(theta - np.pi)
    if abs(err) < 0.4 and abs(omega) < 3.0:
        return float(np.clip(-(_K @ np.array([err, omega]))[0], -U_MAX, U_MAX))
    s = np.sign(omega) or 1.0
    return float(np.clip(-1.5 * s * _energy(theta, omega), -U_MAX, U_MAX))
energy_lqr_finisher.reset = lambda: None


def train_agent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make("pendulum-swingup", max_steps=300)
        disc = Discretizer([-1, -1, -10], [1, 1, 10], [15, 15, 25], -U_MAX, U_MAX, 11)
        agent = QLearning(disc.n_states, disc.n_actions, alpha=0.25, gamma=0.99,
                          epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9975, seed=0)
        res = train(env, agent, disc, episodes=EPISODES, seed=0)
    return agent, disc, res.returns


def rollout(controller, t_final=15.0):
    traj = simulate(_P, controller, x0=np.array([0.0, 0.0]), dt=DT,
                    t_final=t_final, u_bounds=(-U_MAX, U_MAX))
    err = np.abs(wrap_angle(traj.x[:, 0] - np.pi))
    return traj, err


def summarise(name, traj, err, rate):
    reached = np.where(err < 0.2)[0]
    return {
        "controller": name,
        "min_err_rad": round(float(err.min()), 3),
        "final_err_rad": round(float(np.mean(err[-40:])), 3),
        "held_upright": bool(np.mean(err[-40:]) < 0.1),
        "t_upright_s": round(float(traj.t[reached[0]]), 2) if reached.size else float("inf"),
        "control_energy": round(float(np.sum(traj.u[:, 0] ** 2) * DT), 1),
        "shield_active_frac": round(float(rate), 3),
    }


def main() -> None:
    agent, disc, returns = train_agent()
    policy = GreedyPolicy(agent, disc, obs_fn=_obs)

    raw_traj, raw_err = rollout(policy)
    rows = [summarise("tabular Q-learning (raw)", raw_traj, raw_err, 0.0)]

    shield = ShieldedController(
        GreedyPolicy(agent, disc, obs_fn=_obs), energy_lqr_finisher,
        is_safe=lambda x: abs(wrap_angle(x[0] - np.pi)) > ANGLE_HANDOFF,
    )
    sh_traj, sh_err = rollout(shield)
    rows.append(summarise("shielded (RL + classical finisher)", sh_traj, sh_err,
                          shield.intervention_rate))

    _write_tables(rows)
    _figure(raw_traj, raw_err, sh_traj, sh_err, shield, returns)

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
        "# Experiment 12 - shielded tabular Q-learning (pendulum)\n\n"
        f"Torque |u| <= {U_MAX:g}. Shield hands off RL -> classical finisher when "
        f"|wrap(theta-pi)| <= {ANGLE_HANDOFF} rad. `shield_active_frac` = fraction "
        "of steps the classical fallback drove.\n\n"
        + "\n".join([head, sep, *body]) + "\n",
        encoding="utf-8", newline="\n")


def _figure(raw_traj, raw_err, sh_traj, sh_err, shield, returns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aimct.plot_style import set_aimct_style

    set_aimct_style()
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(12.0, 8.5))

    a.plot(raw_traj.t, np.degrees(raw_err), color="#0072B2", lw=2.0,
           label="Q-learning (raw)")
    a.plot(sh_traj.t, np.degrees(sh_err), color="#009E73", lw=2.0,
           label="shielded (RL + LQR)")
    a.axhspan(0, np.degrees(ANGLE_HANDOFF), color="#000000", alpha=0.06,
              label="classical-finisher region")
    a.set_title("(a) Angle to upright"); a.set_xlabel("time [s]")
    a.set_ylabel("|theta - pi| [deg]"); a.legend(fontsize=8)

    mode = np.array([0 if m else 1 for m in shield.intervention_log])  # 1 = fallback
    b.step(sh_traj.t[:len(mode)], mode, color="#D55E00", lw=1.5, where="post")
    b.set_yticks([0, 1]); b.set_yticklabels(["RL policy", "classical finisher"])
    b.set_title("(b) Shielded run: active controller"); b.set_xlabel("time [s]")

    c.plot(raw_traj.t, raw_traj.u[:, 0], color="#0072B2", lw=1.2, label="raw")
    c.plot(sh_traj.t, sh_traj.u[:, 0], color="#009E73", lw=1.2, label="shielded")
    c.set_title("(c) Torque"); c.set_xlabel("time [s]"); c.set_ylabel("u [N m]")
    c.legend(fontsize=8)

    for tr, col, lab in ((raw_traj, "#0072B2", "raw"), (sh_traj, "#009E73", "shielded")):
        d.plot(np.degrees(wrap_angle(tr.x[:, 0] - np.pi)), tr.x[:, 1],
               color=col, lw=1.5, label=lab)
    d.scatter([0], [0], marker="*", color="k", s=120, zorder=5)
    d.set_title("(d) Phase portrait near upright"); d.set_xlabel("theta - pi [deg]")
    d.set_ylabel("omega [rad/s]"); d.legend(fontsize=8)

    fig.suptitle("Exp 12 - shielded tabular Q-learning: RL swings up, LQR holds",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
