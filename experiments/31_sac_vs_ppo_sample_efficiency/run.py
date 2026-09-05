"""Experiment 31 - SAC vs PPO: the sample-efficiency axis.

PPO is on-policy: every gradient step throws its rollouts away. SAC is
off-policy: a replay buffer lets it reuse every transition many times, and
that is supposed to buy a large factor in *environment steps* to reach a
given return. Here both learn the pendulum swing-up from scratch; the
classical energy-shaping + LQR-catch hybrid (Experiments 07 / 11), which
needs zero environment steps, is the reference line.

The x-axis is **environment steps**, not wall-clock or gradient steps - the
quantity that costs money on real hardware.

Run:   python experiments/31_sac_vs_ppo_sample_efficiency/run.py
       AIMCT_EXP_FULL=1 python experiments/31_sac_vs_ppo_sample_efficiency/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aimct.plot_style import PALETTE, set_aimct_style
from aimct.rl import PPO, SAC, make
from aimct.systems import Pendulum

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"
SEED = 0

TASK = dict(max_steps=250)
TOTAL_ENV_STEPS = 60_000 if FULL else 20_000
EVAL_EVERY = 4_000 if FULL else 2_000
EVAL_EPISODES = 10 if FULL else 5


# ---------------------------------------------------- classical reference

class EnergySwingUpLQR:
    """Åström energy pump to the upright separatrix, LQR catch near the top."""

    name = "Energy + LQR hybrid"

    def __init__(self):
        from aimct.controllers import LQR, wrap_angle

        self._wrap = wrap_angle
        p = Pendulum()
        self.m, self.L, self.g, self.b = p.m, p.L, p.g, p.b
        self.J = self.m * self.L ** 2
        A = np.array([[0.0, 1.0],
                      [self.g / self.L, -self.b / self.J]])       # about upright
        B = np.array([[0.0], [1.0 / self.J]])
        self.K = LQR(A, B, np.diag([10.0, 1.0]), np.array([[0.5]])).K
        self.u_max = 4.0

    def reset(self):
        pass

    def update(self, x, dt):
        theta, omega = float(x[0]), float(x[1])
        err = self._wrap(theta - np.pi)
        if abs(err) < 0.5 and abs(omega) < 4.0:
            return np.clip(-(self.K @ np.array([err, omega])), -self.u_max, self.u_max)
        E = 0.5 * self.J * omega ** 2 + self.m * self.g * self.L * (1.0 - np.cos(theta))
        E_des = 2.0 * self.m * self.g * self.L
        u = 0.9 * (E_des - E) * np.sign(omega * np.cos(theta) + 1e-9)
        return np.clip(np.array([u]), -self.u_max, self.u_max)


def _hybrid_return(episodes=20, seed=321):
    env = make("pendulum-swingup", **TASK)
    ctrl = EnergySwingUpLQR()
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False
        while not done:
            # the env obs is (cos, sin, omega); the hybrid wants (theta, omega)
            theta = np.arctan2(obs[1], obs[0])
            u = ctrl.update(np.array([theta, obs[2]]), env.dt)
            obs, r, term, trunc, _ = env.step(u)
            total += r
            done = term or trunc
    return total / episodes


# ----------------------------------------------------------- learners

def _run_sac():
    env = make("pendulum-swingup", **TASK)
    ag = SAC(env, hidden=(128, 128), seed=SEED, warmup=1000, batch_size=128,
             buffer_size=50_000, gamma=0.99, tau=0.005)
    res = ag.train(total_steps=TOTAL_ENV_STEPS, eval_every=EVAL_EVERY,
                   eval_episodes=EVAL_EPISODES, verbose=True)
    return res.steps, res.returns


def _run_ppo():
    env = make("pendulum-swingup", **TASK)
    rollout = 2048
    ppo = PPO(env, hidden=(64, 64), seed=SEED, gamma=0.99, lam=0.95,
             rollout_steps=rollout, ent_coef=1e-3, lr_pi=3e-4)
    steps, returns = [], []
    iters_per_eval = max(1, EVAL_EVERY // rollout)
    done_steps = 0
    while done_steps < TOTAL_ENV_STEPS:
        ppo.train(iterations=iters_per_eval)
        done_steps += iters_per_eval * rollout
        ret = float(ppo.evaluate(episodes=EVAL_EPISODES)["mean_return"])
        steps.append(done_steps); returns.append(ret)
        print(f"  PPO  step {done_steps:6d}  eval_return {ret:8.2f}")
    return np.array(steps), np.array(returns)


# ------------------------------------------------------------------------ main

def _steps_to(threshold, steps, returns):
    hit = np.where(returns >= threshold)[0]
    return int(steps[hit[0]]) if hit.size else None


def main():
    hybrid = _hybrid_return()
    print(f"classical energy+LQR hybrid: mean return {hybrid:.2f} (0 env steps)")

    sac_s, sac_r = _run_sac()
    ppo_s, ppo_r = _run_ppo()

    thr = hybrid - 150.0        # "within striking distance of the classical law"
    rows = [
        ("SAC", sac_r[-1], _steps_to(thr, sac_s, sac_r)),
        ("PPO", ppo_r[-1], _steps_to(thr, ppo_s, ppo_r)),
        ("Energy + LQR hybrid", hybrid, 0),
    ]
    lines = ["# Experiment 31 - SAC vs PPO: sample efficiency on pendulum swing-up",
            "",
            f"pendulum swing-up, {TOTAL_ENV_STEPS} env-step budget, "
            f"greedy eval every {EVAL_EVERY}.", "",
            f"'steps to threshold' = env steps to first reach return >= "
            f"{thr:.0f} (the hybrid's return minus 150).", "",
            "| method | final eval return | steps to threshold |",
            "| --- | --- | --- |"]
    for name, fin, s2t in rows:
        lines.append(f"| {name} | {fin:.1f} | "
                     f"{'n/a' if s2t is None else ('0 (no training)' if s2t == 0 else s2t)} |")
    lines.append("")
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "final_return", "steps_to_threshold"])
        for name, fin, s2t in rows:
            w.writerow([name, f"{fin:.3f}", "" if s2t is None else s2t])
        w.writerow([])
        w.writerow(["curve", "env_steps", "eval_return"])
        for s, r in zip(sac_s, sac_r):
            w.writerow(["SAC", int(s), f"{r:.3f}"])
        for s, r in zip(ppo_s, ppo_r):
            w.writerow(["PPO", int(s), f"{r:.3f}"])

    _figure(sac_s, sac_r, ppo_s, ppo_r, hybrid)
    print((HERE / "table.md").read_text())


def _figure(sac_s, sac_r, ppo_s, ppo_r, hybrid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(sac_s, sac_r, "-o", color=PALETTE["rl"], lw=1.8, ms=4, label="SAC (off-policy)")
    ax.plot(ppo_s, ppo_r, "-s", color=PALETTE["mpc"], lw=1.8, ms=4, label="PPO (on-policy)")
    ax.axhline(hybrid, ls="--", color=PALETTE["hybrid"], lw=1.6,
               label="energy + LQR hybrid (0 steps)")
    ax.set(title="Exp 31 - SAC vs PPO: return vs environment steps",
           xlabel="environment steps", ylabel="mean greedy return")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
