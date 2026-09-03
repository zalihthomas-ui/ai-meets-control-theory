r"""Deep Q-Network — from scratch on the NumPy :class:`aimct.ml.MLP`.

A single Q-network ``Q_\theta(s)`` outputs one value per discrete action; a
slowly-tracked target network ``Q_{\theta^-}`` stabilises the bootstrap. The
loss on a replay minibatch is

.. math::
    \big(Q_\theta(s, a) - [\,r + \gamma\,(1-d)\max_{a'} Q_{\theta^-}(s', a')\,]\big)^2 ,

optimised by Adam over the MLP weights (gradient injected through
:meth:`aimct.ml.MLP.backprop`). Continuous ``ControlEnv`` actions are handled by
a uniform 1-D discretisation of the input box.

No torch. A Stable-Baselines3 DQN is used only as an external cross-check in the
experiments.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..ml import MLP

__all__ = ["QNetwork", "ReplayBuffer", "DQN", "dqn", "DQNResult"]


class QNetwork:
    """MLP ``obs_dim -> hidden... -> n_actions`` with a built-in Adam step."""

    def __init__(self, obs_dim, n_actions, hidden=(64, 64), seed=0):
        self.net = MLP([int(obs_dim), *hidden, int(n_actions)],
                       activation="relu", seed=seed)
        self._m = [np.zeros_like(w) for w in self.net.W] + \
                  [np.zeros_like(b) for b in self.net.b]
        self._v = [np.zeros_like(w) for w in self.net.W] + \
                  [np.zeros_like(b) for b in self.net.b]
        self._t = 0

    def q(self, obs):
        return self.net(np.atleast_2d(np.asarray(obs, dtype=float)))

    def clone_weights_from(self, other: "QNetwork"):
        self.net.W = [w.copy() for w in other.net.W]
        self.net.b = [b.copy() for b in other.net.b]

    def soft_update_from(self, other: "QNetwork", tau: float):
        for i in range(len(self.net.W)):
            self.net.W[i] = (1 - tau) * self.net.W[i] + tau * other.net.W[i]
            self.net.b[i] = (1 - tau) * self.net.b[i] + tau * other.net.b[i]

    def train_step(self, S, A_idx, target_q, lr):
        """One Adam step on the DQN loss for a minibatch. Returns the loss."""
        S = np.atleast_2d(np.asarray(S, dtype=float))
        A_idx = np.asarray(A_idx, dtype=int)
        target_q = np.asarray(target_q, dtype=float)
        B = S.shape[0]

        q_all, acts = self.net.forward(S, cache=True)
        rows = np.arange(B)
        q_sa = q_all[rows, A_idx]
        resid = q_sa - target_q
        loss = float(np.mean(resid ** 2))

        delta = np.zeros_like(q_all)
        delta[rows, A_idx] = (2.0 / B) * resid
        gW, gb = self.net.backprop(acts, delta)
        grads = list(gW) + list(gb)
        params = list(self.net.W) + list(self.net.b)

        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + eps)
        return loss


class ReplayBuffer:
    def __init__(self, capacity=50_000, seed=0):
        self.buf = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.buf)

    def add(self, s, a, r, s2, done):
        self.buf.append((np.asarray(s, float), int(a), float(r),
                         np.asarray(s2, float), bool(done)))

    def sample(self, batch_size):
        idx = self.rng.integers(0, len(self.buf), size=batch_size)
        S, A, R, S2, D = zip(*(self.buf[i] for i in idx))
        return (np.array(S), np.array(A), np.array(R), np.array(S2),
                np.array(D, dtype=float))


@dataclass
class DQNResult:
    returns: np.ndarray                       # per-episode return
    eval_return: float = 0.0
    losses: list = field(default_factory=list)


class DQN:
    """DQN agent over a uniformly discretised 1-D action box.

    Parameters
    ----------
    env : a Gymnasium env with a 1-D ``Box`` action space (e.g. ``ControlEnv``).
    n_actions : number of discrete action levels spanning the box.
    hidden, seed : Q-network shape / RNG seed.
    gamma, lr, batch_size : RL / optimiser hyper-parameters.
    buffer_size, warmup : replay capacity and steps of random play before learning.
    target_tau : Polyak factor for the target-net update (per gradient step).
    eps_start, eps_end, eps_decay_steps : linear epsilon schedule.
    """

    def __init__(self, env, *, n_actions=7, hidden=(64, 64), seed=0,
                 gamma=0.99, lr=1e-3, batch_size=64, buffer_size=50_000,
                 warmup=1_000, target_tau=0.01, train_every=1,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000):
        self.env = env
        lo = float(np.asarray(env.action_space.low).ravel()[0])
        hi = float(np.asarray(env.action_space.high).ravel()[0])
        self.actions = np.linspace(lo, hi, int(n_actions))
        self.obs_dim = int(np.asarray(env.observation_space.shape).prod())

        self.q = QNetwork(self.obs_dim, n_actions, hidden, seed)
        self.tgt = QNetwork(self.obs_dim, n_actions, hidden, seed)
        self.tgt.clone_weights_from(self.q)
        self.buf = ReplayBuffer(buffer_size, seed)
        self.rng = np.random.default_rng(seed)

        self.gamma, self.lr, self.batch_size = gamma, lr, batch_size
        self.warmup, self.target_tau, self.train_every = warmup, target_tau, train_every
        self.eps_start, self.eps_end, self.eps_decay_steps = \
            eps_start, eps_end, eps_decay_steps
        self._step = 0

    # ----------------------------------------------------------------- policy

    def epsilon(self):
        frac = min(1.0, self._step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act_index(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self.q.q(obs)[0]))

    def greedy_action(self, obs):
        return np.array([self.actions[int(np.argmax(self.q.q(obs)[0]))]])

    # ----------------------------------------------------------------- train

    def _learn(self):
        S, A, R, S2, D = self.buf.sample(self.batch_size)
        q_next = self.tgt.q(S2).max(axis=1)
        target = R + self.gamma * (1.0 - D) * q_next
        loss = self.q.train_step(S, A, target, self.lr)
        self.tgt.soft_update_from(self.q, self.target_tau)
        return loss

    def train(self, episodes=200, max_steps=None, verbose=False) -> DQNResult:
        rets, losses = [], []
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))
            done, ep_r, t = False, 0.0, 0
            while not done:
                eps = self.epsilon()
                ai = self.act_index(obs, eps)
                nobs, r, term, trunc, _ = self.env.step(np.array([self.actions[ai]]))
                self.buf.add(obs, ai, r, nobs, term)
                obs, ep_r, t = nobs, ep_r + r, t + 1
                self._step += 1
                if len(self.buf) >= self.warmup and self._step % self.train_every == 0:
                    losses.append(self._learn())
                done = term or trunc or (max_steps is not None and t >= max_steps)
            rets.append(ep_r)
            if verbose and (ep % max(1, episodes // 10) == 0 or ep == episodes - 1):
                print(f"  ep {ep:4d}  return {ep_r:8.1f}  eps {eps:.2f}")
        return DQNResult(returns=np.array(rets), losses=losses)

    def evaluate(self, episodes=10, seed=123):
        rng = np.random.default_rng(seed)
        rets, obs_hist = [], []
        for _ in range(episodes):
            obs, _ = self.env.reset(seed=int(rng.integers(1 << 31)))
            done, ep_r, hist = False, 0.0, [np.asarray(obs, float)]
            while not done:
                obs, r, term, trunc, _ = self.env.step(self.greedy_action(obs))
                ep_r += r
                hist.append(np.asarray(obs, float))
                done = term or trunc
            rets.append(ep_r)
            obs_hist.append(np.array(hist))
        return {"mean_return": float(np.mean(rets)), "returns": np.array(rets),
                "obs": obs_hist}


def dqn(env, **kwargs) -> DQN:
    """Convenience: build a :class:`DQN`. Call ``.train(...)`` / ``.evaluate()``."""
    return DQN(env, **kwargs)
