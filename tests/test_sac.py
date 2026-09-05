"""From-scratch SAC: squashed-Gaussian actor, twin critics, auto-temperature."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.rl import make
from aimct.rl.sac import SAC, SquashedGaussianActor, _Critic


def test_squashed_actor_respects_the_action_box_and_logp_shape():
    rng = np.random.default_rng(0)
    actor = SquashedGaussianActor(3, 2, (16, 16), act_low=[-2.0, -1.0],
                                  act_high=[2.0, 1.0], seed=0)
    S = rng.standard_normal((8, 3))
    a, logp, extras = actor.sample(S, rng, cache=True)
    assert a.shape == (8, 2) and logp.shape == (8,)
    assert np.all(a[:, 0] >= -2.0 - 1e-9) and np.all(a[:, 0] <= 2.0 + 1e-9)
    assert np.all(a[:, 1] >= -1.0 - 1e-9) and np.all(a[:, 1] <= 1.0 + 1e-9)
    assert set(extras) >= {"S", "mu", "log_std", "u", "tanh_u", "eps"}
    # deterministic sample sits at tanh(mu) mapped into the box
    ad, _, _ = actor.sample(S, rng, deterministic=True)
    assert np.all(np.isfinite(ad))


def test_squashed_actor_logp_matches_a_finite_difference_density():
    # scalar action, no box scaling: logp should equal
    # log N(u; mu, sigma) - log(1 - tanh(u)^2)
    rng = np.random.default_rng(1)
    actor = SquashedGaussianActor(2, 1, (8,), act_low=[-1.0], act_high=[1.0], seed=2)
    S = rng.standard_normal((5, 2))
    mu, log_std, _ = actor._mu_logstd(S)
    std = np.exp(log_std)
    a, logp, _ = actor.sample(S, np.random.default_rng(3))
    u = np.arctanh(np.clip(a, -0.999999, 0.999999))
    manual = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1)
    manual -= np.log(1 - np.tanh(u) ** 2 + 1e-6).sum(1)
    assert np.allclose(logp, manual, atol=1e-5)


def test_critic_dq_da_matches_finite_difference():
    rng = np.random.default_rng(0)
    c = _Critic(3, 2, (16, 16), seed=0)
    S = rng.standard_normal((6, 3))
    A = rng.uniform(-1, 1, size=(6, 2))
    g = c.dq_da(S, A, act_dim=2)
    eps = 1e-6
    g_fd = np.zeros_like(A)
    for j in range(2):
        Ap, Am = A.copy(), A.copy()
        Ap[:, j] += eps
        Am[:, j] -= eps
        g_fd[:, j] = (c.q(S, Ap) - c.q(S, Am)) / (2 * eps)
    assert np.allclose(g, g_fd, atol=1e-6)


def test_sac_builds_and_one_update_runs_without_nans():
    env = make("pendulum-swingup", max_steps=50)
    ag = SAC(env, hidden=(32, 32), seed=0, warmup=40, batch_size=16, buffer_size=2000)
    # prime the buffer
    obs, _ = env.reset(seed=0)
    for _ in range(60):
        a = ag.rng.uniform(ag._lo, ag._hi)
        nobs, r, term, trunc, _ = env.step(a)
        ag.buf.add(obs, a, r, nobs, float(term))
        obs = nobs if not (term or trunc) else env.reset(seed=1)[0]
    q1_loss, q2_loss = ag._update()
    assert np.isfinite(q1_loss) and np.isfinite(q2_loss)
    assert np.isfinite(ag.alpha) and ag.alpha > 0
    for W in ag.actor.net.W:
        assert np.all(np.isfinite(W))


def test_auto_alpha_decreases_as_the_policy_sharpens():
    env = make("pendulum-swingup", max_steps=100)
    ag = SAC(env, hidden=(64, 64), seed=0, warmup=200, batch_size=64, buffer_size=5000)
    res = ag.train(total_steps=1500, eval_every=750, eval_episodes=2)
    assert res.alpha[-1] < res.alpha[0]          # temperature was tuned down
    assert np.all(np.isfinite(res.returns))


@pytest.mark.slow
def test_sac_learns_to_swing_up_the_pendulum():
    env = make("pendulum-swingup", max_steps=200)
    ag = SAC(env, hidden=(128, 128), seed=0, warmup=1000, batch_size=128,
             buffer_size=30_000)
    res = ag.train(total_steps=12_000, eval_every=3000, eval_episodes=5)
    # sample-efficient: a big jump within ~12k env steps
    assert res.returns[-1] > res.returns[0] + 500.0
    assert res.returns[-1] > -900.0
