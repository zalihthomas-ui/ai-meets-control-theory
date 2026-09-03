"""Experiment 18 - the RL zoo vs LQR on cart-pole balance.

Puts the analytic LQR next to tabular Q-learning and from-scratch DQN / PPO
(and Stable-Baselines3 PPO as a library cross-check). Same task, same greedy
evaluation. The columns that matter: environment steps consumed, wall-clock,
and controller size - versus a controller that needs none of it.

Run:  python experiments/18_rl_zoo_vs_lqr/run.py         (CI budget, ~90 s)
      AIMCT_EXP_FULL=1 python .../run.py                  (committed artifacts)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from aimct.controllers import LQR
from aimct.rl import DQN, Discretizer, QLearning, make
from aimct.rl import train as q_train
from aimct.rl.ppo import PPO
from aimct.simulate import rk4_step
from aimct.systems import CartPole

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"
SEED = 0
MAX_STEPS = 200
Q_LQR = np.diag([10.0, 1.0, 100.0, 10.0])
R_LQR = np.array([[0.1]])
EVAL_SEEDS = list(range(20))


def make_env():
    return make("cartpole-balance", max_steps=MAX_STEPS)


def eval_policy(policy, n=len(EVAL_SEEDS)):
    """policy: obs -> action array. Returns (mean_return, mean_len)."""
    env = make_env()
    rets, lens = [], []
    for s in EVAL_SEEDS[:n]:
        obs, _ = env.reset(seed=s)
        done, ep_r, t = False, 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(np.asarray(policy(obs), float).reshape(1))
            ep_r += r; t += 1
            done = term or trunc
        rets.append(ep_r); lens.append(t)
    return float(np.mean(rets)), float(np.mean(lens))


def run_lqr():
    A, B = CartPole().linearize()
    K = LQR(A, B, Q_LQR, R_LQR).K
    lo, hi = -20.0, 20.0
    pol = lambda obs: np.clip(-K @ np.asarray(obs, float), lo, hi)
    mr, ml = eval_policy(pol)
    return dict(agent="LQR (analytic)", ret=mr, held=ml, env_steps=0,
                seconds=0.0, params=K.size)


def run_tabular():
    env = make_env()
    disc = Discretizer([-2.4, -3.0, -0.8, -3.5], [2.4, 3.0, 0.8, 3.5],
                       [15, 15, 15, 15], -20.0, 20.0, 7)
    ag = QLearning(disc.n_states, disc.n_actions, alpha=0.2, gamma=0.99,
                   epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, seed=SEED)
    episodes = 1500 if FULL else 400
    t0 = time.time()
    res = q_train(env, ag, disc, episodes=episodes, seed=SEED)
    secs = time.time() - t0
    pol = lambda obs: np.array([disc.action(int(np.argmax(ag.Q[disc.encode(obs)])))])
    mr, ml = eval_policy(pol)
    return dict(agent="Tabular Q-learning", ret=mr, held=ml,
                env_steps=int(res.env_steps) if hasattr(res, "env_steps")
                else int(np.sum([200] * episodes)),
                seconds=secs, params=int(ag.Q.size))


def run_dqn():
    env = make_env()
    ag = DQN(env, n_actions=5, hidden=(64, 64), seed=SEED, lr=1e-3, batch_size=64,
             warmup=500, target_tau=0.02, eps_decay_steps=6000)
    episodes = 500 if FULL else 150
    t0 = time.time()
    res = ag.train(episodes=episodes)
    secs = time.time() - t0
    mr, ml = eval_policy(lambda o: ag.greedy_action(o))
    return dict(agent="DQN (scratch)", ret=mr, held=ml,
                env_steps=int(ag._step), seconds=secs,
                params=ag.q.net.n_params(), curve=res.returns)


def run_ppo():
    env = make_env()
    p = PPO(env, hidden=(64, 64), seed=SEED, rollout_steps=2000, epochs=10,
            minibatch=64, lr_pi=3e-4, lr_v=1e-3, ent_coef=0.0)
    iters = 120 if FULL else 30
    t0 = time.time()
    res = p.train(iterations=iters)
    secs = time.time() - t0
    mr, ml = eval_policy(lambda o: p.pi.greedy(o))
    return dict(agent="PPO (scratch)", ret=mr, held=ml,
                env_steps=iters * 2000, seconds=secs,
                params=p.pi.net.n_params(), curve=res.returns)


def run_sb3():
    try:
        from stable_baselines3 import PPO as SB3PPO
    except Exception:
        return None
    env = make_env()
    steps = 120_000 if FULL else 40_000
    t0 = time.time()
    model = SB3PPO("MlpPolicy", env, seed=SEED, verbose=0)
    model.learn(total_timesteps=steps)
    secs = time.time() - t0
    pol = lambda o: model.predict(np.asarray(o, float), deterministic=True)[0]
    mr, ml = eval_policy(pol)
    nparam = sum(p.numel() for p in model.policy.parameters())
    return dict(agent="PPO (Stable-Baselines3)", ret=mr, held=ml,
                env_steps=steps, seconds=secs, params=int(nparam))


def main():
    rows = [run_lqr(), run_tabular(), run_dqn(), run_ppo()]
    sb3 = run_sb3()
    if sb3:
        rows.append(sb3)

    cols = ["agent", "ret", "held", "env_steps", "seconds", "params"]
    hdr = ["agent", "greedy return", "held /200", "env steps", "train s", "params"]
    lines = [f"# Experiment 18 - RL zoo vs LQR (cart-pole balance{' [FULL]' if FULL else ''})",
             "", "| " + " | ".join(hdr) + " |", "| --- |" + " --- |" * 5]
    for r in rows:
        lines.append("| " + " | ".join(
            r["agent"] if c == "agent" else
            (f"{r[c]:.1f}" if c in ("ret", "held", "seconds") else f"{int(r[c]):,}")
            for c in cols) + " |")
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import csv
    with open(HERE / "table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in rows:
            w.writerow([r[c] for c in cols])

    _figure(rows)
    print((HERE / "table.md").read_text())


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aimct.plot_style import PALETTE, set_aimct_style

    set_aimct_style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = [r["agent"].split(" (")[0] for r in rows]
    steps = [max(r["env_steps"], 1) for r in rows]
    ax[0].bar(names, steps, color=PALETTE["lqr"])
    ax[0].set_yscale("symlog")
    ax[0].set_title("(a) environment steps to train")
    ax[0].set_ylabel("steps (symlog)")
    ax[0].tick_params(axis="x", rotation=25)

    for r in rows:
        if "curve" in r and len(r["curve"]):
            c = r["curve"]
            ax[1].plot(np.linspace(0, r["env_steps"], len(c)),
                       np.convolve(c, np.ones(5) / 5, "same"),
                       label=r["agent"].split(" (")[0])
    ax[1].axhline(rows[0]["ret"], ls="--", color=PALETTE["reference"],
                  label="LQR greedy return")
    ax[1].set(title="(b) learning curve", xlabel="environment steps",
              ylabel="episode return (smoothed)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Exp 18 - model-free RL vs the analytic LQR on cart-pole balance",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
