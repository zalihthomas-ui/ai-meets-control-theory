"""Experiment 19 - Intelligent Control Challenge leaderboard.

Submits five controllers - PID, LQR, linear MPC, a tabular Q-learning policy and
an energy-shaping + LQR hybrid - to the ICC engine and prints the leaderboard:
composite score and status per track.  Every submission sees only the ICC
``spec`` (state/action dims, limits, dt, target), never the plant.

Run:  python experiments/19_icc_leaderboard/run.py            # 2 tracks, quick
      AIMCT_EXP_FULL=1 python .../run.py                       # all tracks + Track 3

Outputs (next to this file): leaderboard.md, leaderboard.csv, figure.png
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from aimct.benchmarks.challenge import Challenge, ChallengeController
from aimct.controllers import LQR, LinearMPC, wrap_angle
from aimct.rl import Discretizer, QLearning, make, train
from aimct.systems import CartPole, DCMotor, MassSpringDamper, Pendulum

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

TRACKS = (["track1-msd", "track1-dcmotor", "track2-pendulum", "track2-cartpole"]
          if FULL else ["track1-msd", "track2-pendulum"])

# each submission builds its OWN model of the (named) plant from the spec dims
_MODELS = {
    2: lambda: MassSpringDamper(m=1.0, c=0.4, k=1.0),   # will be right for track1-msd
}


def _plant_for(spec):
    """A submission's best guess at the plant, keyed off the public spec only."""
    sd = spec["state_dim"]
    tgt0 = float(np.asarray(spec["target_state"], float)[0])
    if sd == 4:
        return CartPole()                                     # cart-pole swing-up
    if abs(tgt0 - np.pi) < 0.1:
        return Pendulum()                                     # swing-up to upright
    if spec["t_final"] < 4.0:
        return DCMotor().reduced()                            # short pi/2 slew
    return MassSpringDamper(m=1.0, c=0.4, k=1.0)              # 8 s unit step


# ------------------------------------------------------------- submissions

class PIDSubmission(ChallengeController):
    def __init__(self, spec):
        super().__init__(spec)
        self.kp, self.ki, self.kd = 30.0, 8.0, 6.0

    def reset(self, target):
        self._r = float(np.asarray(target, float)[0])   # regulate the first output
        self._i = 0.0
        self._prev = None

    def compute_action(self, obs, t):
        e = self._r - float(obs[0])
        self._i += e * self.dt
        d = 0.0 if self._prev is None else (e - self._prev) / self.dt
        self._prev = e
        return np.array([self.kp * e + self.ki * self._i + self.kd * d])


class _ModelLQR(ChallengeController):
    @staticmethod
    def _qr(sys):
        """Weights matched to the plant - the ill-scaled DC motor needs Bryson
        scaling, the cart-pole wants the angle weighted."""
        if sys.n_states == 4:
            return np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.1]])
        if "DCMotor" in type(sys).__name__:
            return np.diag([4.0, 2e-3]), np.array([[3e-2]])
        n = sys.n_states
        return np.eye(n), np.eye(1) * 0.1

    def __init__(self, spec):
        super().__init__(spec)
        self.sys = _plant_for(spec)
        A, B = self.sys.linearize()
        Q, R = self._qr(self.sys)
        self.K = LQR(A, B, Q, R).K
        self._A, self._Bpinv = A, np.linalg.pinv(B)
        self._upright = float(spec["target_state"][spec["state_dim"] // 2 - 1
                                                   if spec["state_dim"] == 4 else 0])

    def reset(self, target):
        self._xr = np.asarray(target, float)
        self._uff = -(self._Bpinv @ (self._A @ self._xr))

    def compute_action(self, obs, t):
        x = np.asarray(obs, float)
        oi = 2 if self.sys.n_states == 4 else 0
        if type(self.sys).__name__ == "Pendulum" or self.sys.n_states == 4:
            x = x.copy()
            x[oi] = self._xr[oi] + wrap_angle(x[oi] - self._xr[oi])   # angular wrap
        return self._uff - self.K @ (x - self._xr)


class MPCSubmission(_ModelLQR):
    def __init__(self, spec):
        super().__init__(spec)
        A, B = self.sys.linearize()
        Q, R = self._qr(self.sys)
        self.mpc = LinearMPC(A, B, Q=Q, R=R, N=20)

    def reset(self, target):
        super().reset(target)
        self.mpc.reset()
        self.mpc.x_ref = self._xr

    def compute_action(self, obs, t):
        x = np.asarray(obs, float)
        oi = 0 if self.sys.n_states == 2 else 2
        if type(self.sys).__name__ == "Pendulum" or self.sys.n_states == 4:
            x = x.copy()
            x[oi] = self._xr[oi] + wrap_angle(x[oi] - self._xr[oi])
        return np.atleast_1d(self.mpc.update(x, self.dt)) + self._uff


class EnergyHybridSubmission(ChallengeController):
    """Energy-shaping swing-up + an LQR catch - a classical hybrid."""

    def __init__(self, spec):
        super().__init__(spec)
        self.sys = _plant_for(spec)
        A, B = self.sys.linearize()
        self.n4 = self.sys.n_states == 4
        self.K = LQR(A, B, np.diag([1.0, 1.0, 10.0, 1.0] if self.n4 else [10.0, 1.0]),
                     [[0.1]]).K
        self._A, self._Bpinv = A, np.linalg.pinv(B)
        m = getattr(self.sys, "m", getattr(self.sys, "mp", 1.0))
        L = getattr(self.sys, "l", getattr(self.sys, "L", 1.0))
        self.J = m * L * L * (4.0 / 3.0 if self.n4 else 1.0)
        self.mgl = m * getattr(self.sys, "g", 9.81) * L
        self.oi = 2 if self.n4 else 0
        # only pump energy when there is actually something to swing up
        tgt0 = float(np.asarray(spec["target_state"], float)[self.oi])
        self._swingup = self.n4 or abs(tgt0 - np.pi) < 0.1

    def reset(self, target):
        self._xr = np.asarray(target, float)
        self._up = float(self._xr[self.oi])
        self._uff = -(self._Bpinv @ (self._A @ self._xr))

    def compute_action(self, obs, t):
        x = np.asarray(obs, float)
        th, om = x[self.oi], x[self.oi + 1]
        err = wrap_angle(th - self._up)
        if not self._swingup or (abs(err) < 0.5 and abs(om) < 3.5):
            xw = x.copy()
            xw[self.oi] = self._up + err
            uff = 0.0 if self._swingup else self._uff     # upright equ. needs ~0
            return uff - self.K @ (xw - self._xr)         # LQR catch / regulation
        E = 0.5 * self.J * om ** 2 + self.mgl * (np.cos(err) - 1.0)
        return np.array([-2.0 * (np.sign(om) or 1.0) * E])


def _train_q_policy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make("pendulum-swingup", max_steps=300)
        disc = Discretizer([-1, -1, -10], [1, 1, 10], [15, 15, 25], -4.0, 4.0, 11)
        ag = QLearning(disc.n_states, disc.n_actions, alpha=0.25, gamma=0.99,
                       epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9975, seed=0)
        train(env, ag, disc, episodes=1200 if FULL else 500, seed=0)
    return ag, disc


_Q_AGENT, _Q_DISC = _train_q_policy()


class TabularQSubmission(ChallengeController):
    """The Experiment-11 tabular Q-learning swing-up policy (trained only on the
    pendulum; it recognises that task from the spec and otherwise does nothing)."""

    def __init__(self, spec):
        super().__init__(spec)
        tgt0 = float(np.asarray(spec["target_state"], float)[0])
        self._known = spec["state_dim"] == 2 and abs(tgt0 - np.pi) < 0.1

    def reset(self, target):
        pass

    def compute_action(self, obs, t):
        if not self._known:                             # not its task -> abstain
            return np.zeros(self.action_dim)
        x = np.asarray(obs, float)
        o = np.array([np.cos(x[0]), np.sin(x[0]), x[1]])
        a = _Q_AGENT.act(_Q_DISC.encode(o), greedy=True)
        return np.array([_Q_DISC.actions[a]])


SUBMISSIONS = {
    "PID": PIDSubmission,
    "LQR": _ModelLQR,
    "Linear MPC": MPCSubmission,
    "Tabular Q": TabularQSubmission,
    "Energy+LQR hybrid": EnergyHybridSubmission,
}


# ------------------------------------------------------------------- run

def main():
    rows = {}
    for name, factory in SUBMISSIONS.items():
        rows[name] = {}
        for track in TRACKS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = Challenge(track).evaluate(factory, seed=0, quick=not FULL)
            rows[name][track] = (res.composite, res.status)
        if FULL:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r3 = Challenge("track2-pendulum").evaluate_robust(factory, seed=0,
                                                                  quick=True)
            rows[name]["track3-pendulum"] = (r3.composite, r3.status)

    cols = list(next(iter(rows.values())).keys())
    _write(rows, cols)
    _figure(rows, cols)
    print((HERE / "leaderboard.md").read_text(encoding="utf-8"))


def _write(rows, cols):
    head = "| submission | " + " | ".join(cols) + " |"
    sep = "| --- |" + " --- |" * len(cols)
    body = []
    for name, r in sorted(rows.items(),
                          key=lambda kv: -max(c for c, _ in kv[1].values())):
        cells = [f"{r[c][0]:.1f} ({r[c][1].replace('DQ_SAFETY', 'DQ')})" for c in cols]
        body.append(f"| {name} | " + " | ".join(cells) + " |")
    (HERE / "leaderboard.md").write_text(
        "# Experiment 19 - Intelligent Control Challenge leaderboard\n\n"
        "Composite score / 100 (status). Every submission sees only the ICC spec, "
        "never the plant.\n\n" + "\n".join([head, sep, *body]) + "\n",
        encoding="utf-8", newline="\n")
    import csv
    with open(HERE / "leaderboard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["submission", *[f"{c}_score" for c in cols],
                    *[f"{c}_status" for c in cols]])
        for name, r in rows.items():
            w.writerow([name, *[f"{r[c][0]:.2f}" for c in cols],
                        *[r[c][1] for c in cols]])


def _figure(rows, cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aimct.plot_style import set_aimct_style

    set_aimct_style()
    names = list(rows)
    x = np.arange(len(cols))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(cols)), 5))
    for i, name in enumerate(names):
        vals = [rows[name][c][0] for c in cols]
        ax.bar(x + i * w, vals, w, label=name)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(cols, rotation=15, ha="right")
    ax.set_ylabel("composite score / 100")
    ax.set_title("Exp 19 - ICC leaderboard: composite score by track")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
