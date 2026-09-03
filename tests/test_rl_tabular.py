"""Tests for :mod:`aimct.rl.tabular` - discretiser + tabular Q-learning."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from aimct.rl.env import make, wrap_to_pi
from aimct.rl.tabular import Discretizer, QLearning, evaluate, train


# --------------------------------------------------------------- discretiser

def test_discretizer_indices_are_in_range_and_cover_corners():
    d = Discretizer([-1.0, -2.0], [1.0, 2.0], [5, 7], -3.0, 3.0, 4)
    assert d.n_states == 35 and d.n_actions == 4
    for obs in ([-1.0, -2.0], [1.0, 2.0], [0.0, 0.0], [-5.0, 9.0]):  # last is clipped
        s = d.encode(obs)
        assert 0 <= s < d.n_states
    assert d.encode([-1.0, -2.0]) != d.encode([1.0, 2.0])
    assert np.allclose(d.action(0), [-3.0]) and np.allclose(d.action(3), [3.0])


def test_discretizer_state_index_round_trips_bin():
    d = Discretizer([0.0, 0.0], [1.0, 1.0], [4, 4], 0.0, 1.0, 2)
    s = d.encode([0.9, 0.1])
    assert tuple(d.state_index(s)) == (3, 0)


# ------------------------------------------------------------------ q-learning

def test_update_moves_q_toward_the_td_target():
    ag = QLearning(2, 2, alpha=0.5, gamma=0.9)
    ag.Q[1] = np.array([1.0, 3.0])            # max Q(s'=1) = 3
    td = ag.update(0, 0, r=1.0, s_next=1, terminal=False)
    # target = 1 + 0.9*3 = 3.7 ; Q(0,0): 0 -> 0.5*3.7
    assert ag.Q[0, 0] == pytest.approx(1.85)
    assert td == pytest.approx(3.7)
    # terminal transition ignores the bootstrap
    ag.update(0, 1, r=2.0, s_next=1, terminal=True)
    assert ag.Q[0, 1] == pytest.approx(1.0)   # 0 -> 0.5*(2.0 - 0)


def test_epsilon_greedy_and_decay():
    ag = QLearning(1, 3, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5, seed=0)
    ag.Q[0] = np.array([0.0, 5.0, 0.0])
    assert ag.act(0, greedy=True) == 1
    picks = [ag.act(0) for _ in range(400)]
    assert set(picks) == {0, 1, 2}                       # eps=1 -> explores all
    for _ in range(10):
        ag.decay()
    assert ag.epsilon == pytest.approx(0.1)              # clamped at the floor


def test_q_learning_converges_on_a_two_state_chain():
    # s0 --a-> s1 (r=0), s1 --a-> s1 terminal (r=1). optimal Q(s0)=gamma, Q(s1)=1.
    ag = QLearning(2, 1, alpha=0.5, gamma=0.9, epsilon=0.0)
    for _ in range(200):
        ag.update(0, 0, 0.0, 1, False)
        ag.update(1, 0, 1.0, 1, True)
    assert ag.Q[1, 0] == pytest.approx(1.0, abs=1e-6)
    assert ag.Q[0, 0] == pytest.approx(0.9, abs=1e-3)


def test_train_is_reproducible_and_reports_shapes():
    env = make("pendulum-swingup", max_steps=40)
    disc = Discretizer([-1, -1, -10], [1, 1, 10], [7, 7, 9], -4.0, 4.0, 5)

    def run():
        ag = QLearning(disc.n_states, disc.n_actions, seed=0)
        return train(env, ag, disc, episodes=30, seed=0)

    a, b = run(), run()
    assert a.returns.shape == (30,) and a.td_errors.shape == (30,)
    assert np.array_equal(a.returns, b.returns)


# ------------------------------------------------------- learns the swing-up

def test_q_learning_swings_the_pendulum_up():
    """~30 s: trains a tabular agent from a hanging pendulum until its greedy
    policy swings the pole up near vertical. Seeded, so deterministic."""
    env = make("pendulum-swingup")
    disc = Discretizer([-1, -1, -10], [1, 1, 10], [15, 15, 25], -4.0, 4.0, 11)
    agent = QLearning(disc.n_states, disc.n_actions, alpha=0.25, gamma=0.99,
                      epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9975, seed=0)
    res = train(env, agent, disc, episodes=800, seed=0)
    ev = evaluate(env, agent, disc, episodes=10)

    baseline = res.returns[:20].mean()
    assert ev["mean_return"] > baseline + 700              # learned a lot
    err = np.abs(wrap_to_pi(ev["states"][:, 0] - np.pi))
    assert err.min() < 0.55                                # reaches ~upright
    assert agent.greedy_policy().shape == (disc.n_states,)  # inspectable policy
