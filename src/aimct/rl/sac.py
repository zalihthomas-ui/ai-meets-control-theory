r"""Soft Actor-Critic -- off-policy maximum-entropy actor-critic, from scratch.

SAC (Haarnoja et al., 2018) maximises reward *plus* policy entropy:

.. math::
    J(\pi) = \mathbb E\Big[\sum_t \gamma^t\big(r_t + \alpha\,
             \mathcal H(\pi(\cdot|s_t))\big)\Big].

The pieces, all NumPy + :class:`aimct.ml.MLP`:

* a **squashed-Gaussian actor** -- the net emits ``(mu, log_std)``, a sample is
  ``a = tanh(mu + sigma\,\varepsilon)`` mapped to the action box, with the
  ``tanh`` change-of-variables term in the log-probability;
* **twin critics** ``Q_1, Q_2`` (``(s,a) -> scalar``) and Polyak-averaged
  targets -- ``min(Q_1, Q_2)`` in the bootstrap fights value overestimation;
* an **auto-tuned temperature** ``alpha`` driven to a target entropy of
  ``-act_dim`` (or a fixed value).

The actor gradient is the reparameterised one: it flows through the ``tanh``
squash and through the critic *with respect to its action input*
(:meth:`aimct.ml.MLP.grad_input`).

Same env contract as :class:`aimct.rl.PPO` / :class:`aimct.rl.DQN`
(``reset(seed=) -> (obs, info)``, ``step(a) -> (obs, r, term, trunc, info)``,
a ``Box`` action space).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ml import MLP

__all__ = ["SAC", "sac", "SACResult"]

_LOG_STD_MIN, _LOG_STD_MAX = -20.0, 2.0
_EPS = 1e-6


@dataclass
class SACResult:
    steps: np.ndarray          # env steps at each eval
    returns: np.ndarray        # mean greedy return at each eval
    alpha: np.ndarray = None   # temperature over training


class _Adam:
    """In-place Adam over a flat list of parameter arrays."""

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


class _Critic:
    """``Q(s, a) -> scalar`` MLP with an Adam optimiser and Polyak targets."""

    def __init__(self, obs_dim, act_dim, hidden, seed):
        self.net = MLP([obs_dim + act_dim, *hidden, 1], activation="relu", seed=seed)
        self.opt = _Adam([*self.net.W, *self.net.b], lr=3e-4)

    def q(self, S, A):
        return self.net(np.hstack([S, A]))[:, 0]

    def q_cached(self, S, A):
        out, acts = self.net.forward(np.hstack([S, A]), cache=True)
        return out[:, 0], acts

    def update(self, S, A, target):
        pred, acts = self.q_cached(S, A)
        resid = (pred - target)[:, None]
        n = S.shape[0]
        gW, gb = self.net.backprop(acts, (2.0 / n) * resid)
        self.opt.step([*gW, *gb])
        return float(np.mean(resid ** 2))

    def dq_da(self, S, A, act_dim):
        """``dQ/da`` at ``(S, A)`` -- (B, act_dim)."""
        _, acts = self.q_cached(S, A)
        g = self.net.grad_input(acts, np.ones((S.shape[0], 1)))
        return g[:, -act_dim:]

    def clone_to(self, other):
        other.net.W = [w.copy() for w in self.net.W]
        other.net.b = [b.copy() for b in self.net.b]

    def polyak_to(self, other, tau):
        for i in range(len(self.net.W)):
            other.net.W[i] += tau * (self.net.W[i] - other.net.W[i])
            other.net.b[i] += tau * (self.net.b[i] - other.net.b[i])


class SquashedGaussianActor:
    """MLP -> ``(mu, log_std)``; ``a = tanh(u) * scale + bias``, ``u ~ N(mu, sigma)``."""

    def __init__(self, obs_dim, act_dim, hidden, act_low, act_high, seed):
        self.net = MLP([obs_dim, *hidden, 2 * act_dim], activation="relu", seed=seed)
        self.opt = _Adam([*self.net.W, *self.net.b], lr=3e-4)
        self.act_dim = act_dim
        lo = np.asarray(act_low, float).ravel()
        hi = np.asarray(act_high, float).ravel()
        self.scale = 0.5 * (hi - lo)
        self.bias = 0.5 * (hi + lo)

    def _mu_logstd(self, S, cache=False):
        out = self.net.forward(S, cache=cache)
        raw, acts = (out if cache else (out, None))
        mu = raw[:, :self.act_dim]
        log_std = np.clip(raw[:, self.act_dim:], _LOG_STD_MIN, _LOG_STD_MAX)
        return mu, log_std, acts

    def sample(self, S, rng, *, deterministic=False, cache=False):
        """Return ``(action, logp, extras)`` -- ``extras`` carries the tensors
        the actor gradient needs when ``cache=True``."""
        S = np.atleast_2d(np.asarray(S, float))
        mu, log_std, acts = self._mu_logstd(S, cache=cache)
        std = np.exp(log_std)
        eps = np.zeros_like(mu) if deterministic else rng.standard_normal(mu.shape)
        u = mu + std * eps
        tanh_u = np.tanh(u)
        a = tanh_u * self.scale + self.bias
        # log N(u; mu, std) - sum log( (1 - tanh(u)^2) * scale )
        logp_gauss = (-0.5 * eps ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1)
        logp = logp_gauss - np.log(self.scale * (1 - tanh_u ** 2) + _EPS).sum(1)
        extras = dict(S=S, mu=mu, log_std=log_std, u=u, tanh_u=tanh_u,
                      eps=eps, acts=acts) if cache else None
        return a, logp, extras

    def greedy(self, obs):
        a, _, _ = self.sample(obs, np.random.default_rng(0), deterministic=True)
        return a[0]

    def grad_step(self, extras, dJ_da, alpha):
        """One Adam step on the reparameterised actor loss
        ``mean(alpha * logp - Q(s, a))``. ``dJ_da`` is ``-dQ/da`` (B, act_dim)."""
        mu, log_std, u, tanh_u = (extras[k] for k in ("mu", "log_std", "u", "tanh_u"))
        B = mu.shape[0]
        sech2 = 1.0 - tanh_u ** 2
        da_du = sech2 * self.scale                        # d a / d u
        umm = u - mu                                      # = sigma * eps

        # d(logp)/d(mu), d(logp)/d(log_std)  (total, through the reparam sample)
        dlogp_dmu = 2.0 * tanh_u
        dlogp_dlogstd = -1.0 + 2.0 * tanh_u * umm

        # dJ/d(mu,log_std) = alpha * dlogp - dQ/da * da/d(mu,log_std)
        g_mu = alpha * dlogp_dmu + dJ_da * da_du
        g_logstd = alpha * dlogp_dlogstd + dJ_da * da_du * umm
        delta = np.hstack([g_mu, g_logstd]) / B

        _, acts = self.net.forward(extras["S"], cache=True)
        gW, gb = self.net.backprop(acts, delta)
        self.opt.step([*gW, *gb])


class _ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, seed):
        self.cap = int(capacity)
        self.S = np.zeros((self.cap, obs_dim))
        self.A = np.zeros((self.cap, act_dim))
        self.R = np.zeros(self.cap)
        self.S2 = np.zeros((self.cap, obs_dim))
        self.D = np.zeros(self.cap)
        self.i, self.full = 0, False
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        k = self.i
        self.S[k], self.A[k], self.R[k], self.S2[k], self.D[k] = s, a, r, s2, done
        self.i = (self.i + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n):
        idx = self.rng.integers(0, len(self), size=n)
        return (self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx])


class SAC:
    def __init__(
        self,
        env,
        *,
        hidden=(256, 256),
        seed=0,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch_size=256,
        buffer_size=100_000,
        warmup=1000,
        train_every=1,
        gradient_steps=1,
        alpha="auto",
        target_entropy=None,
    ):
        self.env = env
        self.obs_dim = int(np.asarray(env.observation_space.shape).prod())
        self.act_dim = int(np.asarray(env.action_space.shape).prod())
        lo = np.asarray(env.action_space.low, float).ravel()
        hi = np.asarray(env.action_space.high, float).ravel()

        self.actor = SquashedGaussianActor(self.obs_dim, self.act_dim, hidden,
                                           lo, hi, seed)
        self.q1 = _Critic(self.obs_dim, self.act_dim, hidden, seed + 1)
        self.q2 = _Critic(self.obs_dim, self.act_dim, hidden, seed + 2)
        self.q1t = _Critic(self.obs_dim, self.act_dim, hidden, seed + 1)
        self.q2t = _Critic(self.obs_dim, self.act_dim, hidden, seed + 2)
        self.q1.clone_to(self.q1t)
        self.q2.clone_to(self.q2t)
        for c in (self.actor, self.q1, self.q2):
            c.opt.lr = lr

        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.warmup, self.train_every, self.gradient_steps = warmup, train_every, gradient_steps
        self.buf = _ReplayBuffer(buffer_size, self.obs_dim, self.act_dim, seed)
        self.rng = np.random.default_rng(seed)
        self._lo, self._hi = lo, hi

        self.auto_alpha = alpha == "auto"
        self.log_alpha = np.array(0.0)
        self.alpha = 1.0 if self.auto_alpha else float(alpha)
        self.target_entropy = (-float(self.act_dim) if target_entropy is None
                               else float(target_entropy))
        self._alpha_opt = _Adam([self.log_alpha], lr=lr) if self.auto_alpha else None

    # -- one gradient update ------------------------------------------------

    def _update(self):
        S, A, R, S2, D = self.buf.sample(self.batch_size)

        # --- critic targets
        a2, logp2, _ = self.actor.sample(S2, self.rng)
        q_next = np.minimum(self.q1t.q(S2, a2), self.q2t.q(S2, a2))
        y = R + self.gamma * (1.0 - D) * (q_next - self.alpha * logp2)
        q1_loss = self.q1.update(S, A, y)
        q2_loss = self.q2.update(S, A, y)

        # --- actor (reparameterised)
        a, logp, extras = self.actor.sample(S, self.rng, cache=True)
        q1a, q2a = self.q1.q(S, a), self.q2.q(S, a)
        use1 = (q1a <= q2a)[:, None]
        dq_da = np.where(use1, self.q1.dq_da(S, a, self.act_dim),
                         self.q2.dq_da(S, a, self.act_dim))
        self.actor.grad_step(extras, -dq_da, self.alpha)

        # --- temperature
        if self.auto_alpha:
            g = np.array(float(np.mean(-(logp + self.target_entropy)))
                         * np.exp(self.log_alpha))
            self._alpha_opt.step([g])
            self.alpha = float(np.exp(self.log_alpha))

        # --- targets
        self.q1.polyak_to(self.q1t, self.tau)
        self.q2.polyak_to(self.q2t, self.tau)
        return q1_loss, q2_loss

    # -- training loop ---------------------------------------------------

    def train(self, total_steps=20_000, *, eval_every=2000, eval_episodes=5,
              verbose=False) -> SACResult:
        steps_hist, ret_hist, alpha_hist = [], [], []
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))
        for t in range(1, total_steps + 1):
            if t <= self.warmup:
                a = self.rng.uniform(self._lo, self._hi)
            else:
                a, _, _ = self.actor.sample(obs, self.rng)
                a = a[0]
            nobs, r, term, trunc, _ = self.env.step(a)
            self.buf.add(obs, a, r, nobs, float(term))
            obs = nobs
            if term or trunc:
                obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))

            if t > self.warmup and t % self.train_every == 0:
                for _ in range(self.gradient_steps):
                    self._update()

            if t % eval_every == 0:
                ret = self.evaluate(eval_episodes)
                steps_hist.append(t); ret_hist.append(ret); alpha_hist.append(self.alpha)
                if verbose:
                    print(f"  step {t:6d}  eval_return {ret:8.2f}  alpha {self.alpha:.3f}")

        return SACResult(np.array(steps_hist), np.array(ret_hist), np.array(alpha_hist))

    def evaluate(self, episodes=10, seed=123):
        rng = np.random.default_rng(seed)
        total = 0.0
        for _ in range(episodes):
            obs, _ = self.env.reset(seed=int(rng.integers(1 << 31)))
            done = False
            while not done:
                obs, r, term, trunc, _ = self.env.step(self.actor.greedy(obs))
                total += r
                done = term or trunc
        return total / episodes


def sac(env, *, total_steps=20_000, **kwargs) -> tuple[SAC, SACResult]:
    """Build a :class:`SAC` and train it. Returns ``(agent, result)``."""
    train_kw = {k: kwargs.pop(k) for k in
                ("eval_every", "eval_episodes", "verbose") if k in kwargs}
    agent = SAC(env, **kwargs)
    return agent, agent.train(total_steps=total_steps, **train_kw)
