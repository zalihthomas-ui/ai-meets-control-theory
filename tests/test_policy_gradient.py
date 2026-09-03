"""Tests for the from-scratch REINFORCE policy gradient."""

import numpy as np

from aimct.rl import make
from aimct.rl.policy_gradient import (
    GaussianPolicy,
    evaluate_policy,
    reinforce,
)


# ------------------------------------------------------------------ policy math

def test_gaussian_policy_shapes_and_clipping():
    pol = GaussianPolicy(3, 1, act_low=-2.0, act_high=2.0, seed=0)
    rng = np.random.default_rng(0)
    a = pol.act(np.zeros(3), rng)
    assert a.shape == (1,)
    pol.log_std[:] = 5.0                         # huge noise -> clipping must bite
    for _ in range(50):
        a = pol.act(rng.standard_normal(3), rng)
        assert -2.0 - 1e-9 <= a[0] <= 2.0 + 1e-9
    assert pol.greedy(np.zeros(3)).shape == (1,)


def test_logp_mean_gradient_matches_finite_difference():
    pol = GaussianPolicy(2, 1, seed=1)
    rng = np.random.default_rng(0)
    O = rng.standard_normal((4, 2))
    A = rng.standard_normal((4, 1))
    adv = np.array([1.0, 1.0, 1.0, 1.0])         # so grad of loss == -grad of mean logp

    _, gW, gb, _ = pol.logp_and_grads(O, A, adv)

    eps = 1e-6
    W0 = pol.net.W[0]
    for idx in [(0, 0), (W0.shape[0] - 1, W0.shape[1] - 1)]:
        W0[idx] += eps
        lp_p, _, _, _ = pol.logp_and_grads(O, A, adv)
        W0[idx] -= 2 * eps
        lp_m, _, _, _ = pol.logp_and_grads(O, A, adv)
        W0[idx] += eps
        # loss L = -mean(logp * adv) = -mean(logp); dL/dW = -(dlp_mean/dW)
        num = -(lp_p - lp_m) / (2 * eps)
        assert abs(num - gW[0][idx]) < 1e-5


def test_logp_logstd_gradient_matches_finite_difference():
    pol = GaussianPolicy(2, 1, seed=2)
    rng = np.random.default_rng(1)
    O = rng.standard_normal((6, 2))
    A = rng.standard_normal((6, 1))
    adv = rng.standard_normal(6)
    _, _, _, gls = pol.logp_and_grads(O, A, adv)

    eps = 1e-6

    def adv_loss():
        mu = pol.net.forward(np.atleast_2d(O))
        z = (A - mu) / np.exp(pol.log_std)
        lp = (-0.5 * np.sum(z**2, 1) - np.sum(pol.log_std)
              - 0.5 * pol.act_dim * np.log(2 * np.pi))
        return float(-np.mean(lp * adv))

    pol.log_std[0] += eps
    l_p = adv_loss()
    pol.log_std[0] -= 2 * eps
    l_m = adv_loss()
    pol.log_std[0] += eps
    num = (l_p - l_m) / (2 * eps)
    assert abs(num - gls[0]) < 1e-5


# ------------------------------------------------------------------ learning

def test_reinforce_improves_on_a_one_step_target_env():
    class ReachEnv:
        """1-step env: reward = -(action - target)^2. Trivial policy-gradient test."""
        def __init__(self, target=1.3):
            self.target = target
            import gymnasium as gym
            self.observation_space = gym.spaces.Box(-1, 1, (1,), np.float64)
            self.action_space = gym.spaces.Box(-3, 3, (1,), np.float64)

        def reset(self, *, seed=None, options=None):
            return np.zeros(1), {}

        def step(self, a):
            a = float(np.clip(a, -3, 3)[0])
            return np.zeros(1), -(a - self.target) ** 2, True, False, {}

    env = ReachEnv(target=1.3)
    pol = GaussianPolicy(1, 1, act_low=-3.0, act_high=3.0, hidden=(16,),
                         log_std_init=0.0, seed=0)
    res = reinforce(env, pol, updates=60, batch_episodes=16, gamma=1.0,
                    lr=5e-2, seed=0)
    assert res.returns[-5:].mean() > res.returns[:5].mean()
    assert abs(pol.greedy(np.zeros(1))[0] - 1.3) < 0.25


def test_reinforce_learns_to_balance_cartpole():
    env = make("cartpole-balance", max_steps=200)
    obs_dim = env.observation_space.shape[0]
    pol = GaussianPolicy(obs_dim, 1, act_low=float(env.action_space.low[0]),
                         act_high=float(env.action_space.high[0]),
                         hidden=(64, 64), log_std_init=-0.5, seed=0)
    res = reinforce(env, pol, updates=80, batch_episodes=10, gamma=0.99,
                    lr=1e-2, seed=0)
    ev = evaluate_policy(env, pol, episodes=10, seed=123)
    assert res.returns[-10:].mean() > res.returns[:10].mean() + 5.0
    assert ev["mean_return"] > res.returns[:10].mean()
