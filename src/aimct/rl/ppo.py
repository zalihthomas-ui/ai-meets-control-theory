r"""Proximal Policy Optimization — from scratch on NumPy.

Actor-critic. The actor is the diagonal-Gaussian :class:`GaussianPolicy`; the
critic is a plain :class:`aimct.ml.MLP` value head. Advantages use GAE(:math:`\lambda`)

.. math::
    \delta_t = r_t + \gamma\,V(s_{t+1})(1-d_t) - V(s_t),\qquad
    \hat A_t = \sum_{l\ge 0}(\gamma\lambda)^l \delta_{t+l},

and the policy takes several epochs of minibatch steps on the clipped surrogate

.. math::
    L^{\text{CLIP}} = \mathbb E\Big[\min\big(\rho\hat A,\;
        \operatorname{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat A\big)\Big],\quad
    \rho = e^{\log\pi_\theta - \log\pi_{\theta_{\text{old}}}},

plus an entropy bonus. The surrogate gradient reuses
:meth:`GaussianPolicy.logp_and_grads` with a per-sample weight
:math:`w = \hat A\,\rho\,\mathbb 1[\rho\hat A \le \operatorname{clip}(\rho)\hat A]`.

No torch. A Stable-Baselines3 PPO is an external cross-check in the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ml import MLP
from .policy_gradient import GaussianPolicy

__all__ = ["PPO", "ppo", "PPOResult"]


@dataclass
class PPOResult:
    returns: np.ndarray                       # mean episode return per iteration
    kl: np.ndarray = None
    entropy: np.ndarray = None


class _Adam:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2, e = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + e)


class PPO:
    def __init__(self, env, *, hidden=(64, 64), seed=0, gamma=0.99, lam=0.95,
                 clip=0.2, lr_pi=3e-4, lr_v=1e-3, epochs=10, minibatch=64,
                 rollout_steps=2048, ent_coef=0.0, target_kl=0.03,
                 normalize_adv=True):
        self.env = env
        obs_dim = int(np.asarray(env.observation_space.shape).prod())
        act_dim = int(np.asarray(env.action_space.shape).prod())
        lo = np.asarray(env.action_space.low, float).ravel()
        hi = np.asarray(env.action_space.high, float).ravel()
        # start with std ~ 30% of the half-range so exploration actually covers
        # the action box (a fixed -0.5 log-std is invisible on a +/-20 N range)
        log_std0 = float(np.log(0.3 * np.mean(hi - lo) / 2.0))
        self.pi = GaussianPolicy(obs_dim, act_dim, hidden=hidden,
                                 act_low=lo, act_high=hi,
                                 log_std_init=log_std0, seed=seed)
        self.vf = MLP([obs_dim, *hidden, 1], activation="tanh", seed=seed + 1)

        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch, self.rollout_steps = epochs, minibatch, rollout_steps
        self.ent_coef, self.target_kl, self.norm_adv = ent_coef, target_kl, normalize_adv
        self.opt_pi = _Adam(self.pi.params(), lr_pi)
        self.opt_vf = _Adam([*self.vf.W, *self.vf.b], lr_v)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- rollout

    def _collect(self):
        O, A, LOGP, R, D, V = [], [], [], [], [], []
        ep_returns, ep_r = [], 0.0
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))
        for _ in range(self.rollout_steps):
            a = self.pi.act(obs, self.rng)
            lp = float(self.pi.log_prob(obs, a)[0])
            v = float(self.vf(np.atleast_2d(obs))[0, 0])
            nobs, r, term, trunc, _ = self.env.step(a)
            O.append(np.asarray(obs, float)); A.append(np.asarray(a, float))
            LOGP.append(lp); R.append(float(r)); D.append(term); V.append(v)
            ep_r += r
            obs = nobs
            if term or trunc:
                ep_returns.append(ep_r); ep_r = 0.0
                obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))
        last_v = float(self.vf(np.atleast_2d(obs))[0, 0])
        if not ep_returns:
            ep_returns.append(ep_r)
        return (np.array(O), np.array(A), np.array(LOGP), np.array(R),
                np.array(D, float), np.array(V), last_v, float(np.mean(ep_returns)))

    def _gae(self, R, D, V, last_v):
        T = len(R)
        adv = np.zeros(T)
        gae = 0.0
        for t in range(T - 1, -1, -1):
            v_next = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + self.gamma * v_next * (1 - D[t]) - V[t]
            gae = delta + self.gamma * self.lam * (1 - D[t]) * gae
            adv[t] = gae
        return adv, adv + V

    # ------------------------------------------------------------- update

    def _update(self, O, A, logp_old, adv, ret):
        if self.norm_adv and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / adv.std()
        n = len(O)
        idx = np.arange(n)
        approx_kl = 0.0
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                logp_new = self.pi.log_prob(O[mb], A[mb])
                ratio = np.exp(logp_new - logp_old[mb])
                a_mb = adv[mb]
                surr1 = ratio * a_mb
                surr2 = np.clip(ratio, 1 - self.clip, 1 + self.clip) * a_mb
                w = np.where(surr1 <= surr2, a_mb * ratio, 0.0)

                _, gW, gb, gls = self.pi.logp_and_grads(O[mb], A[mb], w)
                gls = gls - self.ent_coef            # -d/d log_std of (+ent*H)
                self.opt_pi.step([*gW, *gb, gls])

                # value fit
                v_pred = self.vf(O[mb])[:, 0]
                vresid = (v_pred - ret[mb])
                delta = (2.0 / len(mb)) * vresid[:, None]
                _, vacts = self.vf.forward(O[mb], cache=True)
                gVW, gVb = self.vf.backprop(vacts, delta)
                self.opt_vf.step([*gVW, *gVb])

            approx_kl = float(np.mean(logp_old - self.pi.log_prob(O, A)))
            if abs(approx_kl) > 1.5 * self.target_kl:
                break
        return approx_kl

    # ------------------------------------------------------------- driver

    def train(self, iterations=50, verbose=False) -> PPOResult:
        hist, kls, ents = [], [], []
        for it in range(iterations):
            O, A, logp_old, R, D, V, last_v, mean_ret = self._collect()
            adv, ret = self._gae(R, D, V, last_v)
            kl = self._update(O, A, logp_old, adv, ret)
            hist.append(mean_ret); kls.append(kl); ents.append(self.pi.entropy())
            if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
                print(f"  iter {it:3d}  return {mean_ret:8.1f}  kl {kl:+.3f}  "
                      f"log_std {self.pi.log_std.round(2)}")
        return PPOResult(returns=np.array(hist), kl=np.array(kls),
                         entropy=np.array(ents))

    def evaluate(self, episodes=10, seed=123):
        rng = np.random.default_rng(seed)
        rets, obs_hist = [], []
        for _ in range(episodes):
            obs, _ = self.env.reset(seed=int(rng.integers(1 << 31)))
            done, ep_r, hist = False, 0.0, [np.asarray(obs, float)]
            while not done:
                obs, r, term, trunc, _ = self.env.step(self.pi.greedy(obs))
                ep_r += r
                hist.append(np.asarray(obs, float))
                done = term or trunc
            rets.append(ep_r); obs_hist.append(np.array(hist))
        return {"mean_return": float(np.mean(rets)), "returns": np.array(rets),
                "obs": obs_hist}


def ppo(env, **kwargs) -> PPO:
    return PPO(env, **kwargs)
