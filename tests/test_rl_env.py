"""Gymnasium-API conformance and task-behaviour tests for :mod:`aimct.rl.env`."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from aimct.rl.env import ControlEnv, make, wrap_to_pi
from aimct.systems import MassSpringDamper


@pytest.mark.parametrize("task", ["cartpole-balance", "pendulum-swingup"])
def test_reset_and_step_signature_and_spaces(task):
    env = make(task)
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.observation_space.shape == obs.shape

    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "state" in info


@pytest.mark.parametrize("task", ["cartpole-balance", "pendulum-swingup"])
def test_seeded_rollouts_are_identical(task):
    a, b = make(task), make(task)
    o_a, _ = a.reset(seed=123)
    o_b, _ = b.reset(seed=123)
    assert np.array_equal(o_a, o_b)
    rng = np.random.default_rng(0)
    for _ in range(50):
        act = rng.uniform(a.action_space.low, a.action_space.high).astype(np.float32)
        ra = a.step(act)
        rb = b.step(act)
        assert np.allclose(ra[0], rb[0]) and ra[1] == rb[1]
        assert ra[2] == rb[2] and ra[3] == rb[3]


def test_action_is_clipped_to_bounds():
    env = make("pendulum-swingup")
    env.reset(seed=0)
    hi = float(env.action_space.high[0])
    _, _, _, _, info1 = env.step(np.array([1e6]))
    env.reset(seed=0)
    _, _, _, _, info2 = env.step(np.array([hi]))
    assert np.allclose(info1["state"], info2["state"])   # 1e6 clipped to hi


def test_cartpole_terminates_when_pole_falls():
    env = make("cartpole-balance")
    env.reset(seed=0)
    terminated = False
    for _ in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(np.array([20.0]))  # slam one way
        if terminated or truncated:
            break
    assert terminated and not truncated


def test_pendulum_truncates_and_never_terminates():
    env = make("pendulum-swingup")
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert not terminated
        if truncated:
            break
    assert steps == env.max_steps


def test_default_quadratic_reward_matches_formula():
    msd = MassSpringDamper()
    Q, R = np.diag([2.0, 0.5]), np.array([[0.1]])
    env = ControlEnv(msd, dt=0.02, max_steps=10, action_bounds=(-5.0, 5.0), Q=Q, R=R)
    env.reset(seed=0)
    x = env.state
    u = np.array([1.3])
    _, reward, *_ = env.step(u)
    assert reward == pytest.approx(-(x @ Q @ x + u @ R @ u))


def test_make_unknown_task_raises_and_overrides_apply():
    with pytest.raises(KeyError):
        make("no-such-task")
    env = make("pendulum-swingup", max_steps=17)
    assert env.max_steps == 17


def test_gymnasium_env_checker_passes():
    checker = pytest.importorskip("gymnasium.utils.env_checker")
    checker.check_env(make("pendulum-swingup"), skip_render_check=True)
