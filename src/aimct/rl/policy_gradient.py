r"""REINFORCE (Monte-Carlo policy gradient) with a Gaussian policy - from scratch.

The policy is a diagonal Gaussian whose mean is an :class:`aimct.ml.MLP` and
whose log-std is a free vector. For a trajectory :math:`\tau` with return-to-go
:math:`G_t`,

.. math::

    \nabla_\theta J \;\approx\;
    \frac{1}{B}\sum_{\tau}\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\,
    \big(G_t - b\big),

with a batch-mean baseline :math:`b` and (optionally) standardised advantages.
Gradients of :math:`\log\pi` w.r.t. the mean and log-std are analytic; the mean
gradient is backpropagated through the MLP via :meth:`aimct.ml.MLP.backprop`.

No torch. This is the reference implementation; a library path (Stable-Baselines3)
is used only for cross-checks in the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ml import MLP

__all__ = ["GaussianPolicy", "reinforce", "evaluate_policy", "ReinforceResult"]

_LOG2PI = float(np.log(2.0 * np.pi))


class GaussianPolicy:
    def __init__(self, obs_dim, act_dim, *, hidden=(64, 64), act_low=-1.0,
                 act_high=1.0, log_std_init=-0.5, seed=0) -> None:
        self.net = MLP([int(obs_dim), *hidden, int(act_dim)], activation="tanh", seed=seed)
        self.log_std = np.full(int(act_dim), float(log_std_init))
        self.act_dim = int(act_dim)
        self.act_low = np.broadcast_to(np.asarray(act_low, float), (self.act_dim,)).copy()
        self.act_high = np.broadcast_to(np.asarray(act_high, float), (self.act_dim,)).copy()

    # -- forward -------------------------------------------------------------

    def mean(self, obs):
        return self.net(np.atleast_2d(np.asarray(obs, dtype=float)))

    def act(self, obs, rng):
        mu = self.mean(obs)[0]
        a = mu + np.exp(self.log_std) * rng.standard_normal(self.act_dim)
        return np.clip(a, self.act_low, self.act_high)

    def greedy(self, obs):
        return np.clip(self.mean(obs)[0], self.act_low, self.act_high)

    # -- log prob + gradients ---------------------------------------------------

    def logp_and_grads(self, O, A, adv):
        """Batch ``O`` ``(B, obs)``, ``A`` ``(B, act)``, advantages ``adv`` ``(B,)``.

        Returns ``(mean_logp, gW, gb, g_log_std)`` where the parameter grads are
        for the loss ``L = -mean_b( logpi(a_b|s_b) * adv_b )``.
        """
        O = np.atleast_2d(np.asarray(O, dtype=float))
        A = np.atleast_2d(np.asarray(A, dtype=float))
        adv = np.asarray(adv, dtype=float).reshape(-1)
        B = O.shape[0]

        mu, acts = self.net.forward(O, cache=True)
        std = np.exp(self.log_std)
        z = (A - mu) / std
        logp = -0.5 * np.sum(z**2, axis=1) - np.sum(self.log_std) - 0.5 * self.act_dim * _LOG2PI

        # dL/dmu = -adv * (a-mu)/std^2 ;  dL/dlog_std = -adv * ((a-mu)^2/std^2 - 1)
        w = adv[:, None] / B
        dmu = -w * (z / std)
        gW, gb = self.net.backprop(acts, dmu)
        g_log_std = -np.sum(w * (z**2 - 1.0), axis=0)
        return float(np.mean(logp)), gW, gb, g_log_std

    # -- param vector view (for the optimiser) --------------------------------

    def params(self):
        """Live references to every trainable array (W's, b's, then log_std),
        so an optimiser can update them in place."""
        return [*self.net.W, *self.net.b, self.log_std]


@dataclass
class ReinforceResult:
    returns: np.ndarray                       # mean episode return per update
    final_eval: float = 0.0
    entropy: list = field(default_factory=list)


def _returns_to_go(rewards, gamma):
    G = np.zeros(len(rewards))
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        G[t] = acc
    return G


def reinforce(env, policy: GaussianPolicy, *, updates: int = 200,
              batch_episodes: int = 8, gamma: float = 0.99, lr: float = 1e-2,
              standardize_adv: bool = True, seed: int = 0,
              verbose: bool = False) -> ReinforceResult:
    rng = np.random.default_rng(seed)
    prm = policy.params()
    mvec = [np.zeros_like(p) for p in prm]
    vvec = [np.zeros_like(p) for p in prm]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t_adam = 0
    hist = []

    for upd in range(updates):
        O, A, ADV, ep_returns = [], [], [], []
        for _ in range(batch_episodes):
            obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
            obs_l, act_l, rew_l = [], [], []
            done = False
            while not done:
                a = policy.act(obs, rng)
                nobs, r, term, trunc, _ = env.step(a)
                obs_l.append(np.asarray(obs, float))
                act_l.append(np.asarray(a, float))
                rew_l.append(float(r))
                obs = nobs
                done = term or trunc
            G = _returns_to_go(rew_l, gamma)
            O.extend(obs_l); A.extend(act_l); ADV.extend(G)
            ep_returns.append(float(np.sum(rew_l)))

        O = np.array(O); A = np.array(A); ADV = np.array(ADV)
        ADV = ADV - ADV.mean()
        if standardize_adv and ADV.std() > 1e-8:
            ADV = ADV / ADV.std()

        _, gW, gb, gls = policy.logp_and_grads(O, A, ADV)
        grads = [*gW, *gb, gls]

        t_adam += 1
        for i, (p, g) in enumerate(zip(policy.params(), grads)):
            mvec[i] = b1 * mvec[i] + (1 - b1) * g
            vvec[i] = b2 * vvec[i] + (1 - b2) * g * g
            mh = mvec[i] / (1 - b1**t_adam)
            vh = vvec[i] / (1 - b2**t_adam)
            p -= lr * mh / (np.sqrt(vh) + eps)          # in-place: W/b/log_std views

        hist.append(float(np.mean(ep_returns)))
        if verbose and (upd % max(1, updates // 10) == 0 or upd == updates - 1):
            print(f"  update {upd:4d}  mean return {hist[-1]:.3f}  "
                  f"log_std {policy.log_std.round(2)}")

    res = ReinforceResult(returns=np.array(hist))
    return res


def evaluate_policy(env, policy: GaussianPolicy, *, episodes: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets, all_obs = [], []
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done, ep_r, obs_hist = False, 0.0, [np.asarray(obs, float)]
        while not done:
            obs, r, term, trunc, _ = env.step(policy.greedy(obs))
            ep_r += float(r)
            obs_hist.append(np.asarray(obs, float))
            done = term or trunc
        rets.append(ep_r)
        all_obs.append(np.array(obs_hist))
    return {"mean_return": float(np.mean(rets)), "returns": np.array(rets),
            "obs": all_obs}
