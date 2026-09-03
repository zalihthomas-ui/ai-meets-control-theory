"""From-scratch DQN: network step, replay buffer, and learning on cart-pole."""

import numpy as np
import pytest

from aimct.rl import DQN, QNetwork, ReplayBuffer, make


def test_qnetwork_train_step_reduces_loss_on_a_fixed_target():
    qn = QNetwork(obs_dim=3, n_actions=4, hidden=(16,), seed=0)
    rng = np.random.default_rng(0)
    S = rng.standard_normal((32, 3))
    A = rng.integers(0, 4, size=32)
    target = rng.standard_normal(32)
    l0 = qn.train_step(S, A, target, lr=1e-2)
    for _ in range(200):
        l = qn.train_step(S, A, target, lr=1e-2)
    assert l < 0.1 * l0
    # only the taken-action heads were trained toward the targets
    q = qn.q(S)[np.arange(32), A]
    assert np.mean((q - target) ** 2) < 0.05


def test_replay_buffer_capacity_and_sampling():
    rb = ReplayBuffer(capacity=100, seed=0)
    for i in range(250):
        rb.add(np.zeros(2) + i, i % 3, float(i), np.ones(2), i % 7 == 0)
    assert len(rb) == 100
    S, A, R, S2, D = rb.sample(16)
    assert S.shape == (16, 2) and A.shape == (16,) and D.dtype == float
    assert R.min() >= 150.0                       # only the most recent 100 kept


def test_target_network_soft_update_moves_toward_online():
    a = QNetwork(2, 2, hidden=(8,), seed=1)
    b = QNetwork(2, 2, hidden=(8,), seed=2)
    before = np.linalg.norm(a.net.W[0] - b.net.W[0])
    b.soft_update_from(a, tau=0.5)
    after = np.linalg.norm(a.net.W[0] - b.net.W[0])
    assert after < before


@pytest.mark.slow
def test_dqn_learns_to_balance_cartpole():
    env = make("cartpole-balance", max_steps=200)
    ag = DQN(env, n_actions=5, hidden=(64, 64), seed=0, lr=1e-3, batch_size=64,
             warmup=500, target_tau=0.02, eps_decay_steps=6000)
    res = ag.train(episodes=120)
    ev = ag.evaluate(episodes=10, seed=7)

    assert res.returns[-20:].mean() > res.returns[:20].mean() + 15.0
    # a do-nothing / random policy tips over fast and scores far worse
    assert ev["mean_return"] > res.returns[:20].mean()
    # greedy rollouts keep the pole up for most of the horizon
    assert np.mean([len(o) for o in ev["obs"]]) > 150
