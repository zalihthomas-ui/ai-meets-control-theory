"""Gymnasium-API conformance and task-behaviour tests for :mod:`aimct.rl.env`."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from aimct.controllers import LQR
from aimct.rl.env import (ControlEnv, figure8_obs, figure8_reference, make,
                          wrap_to_pi)
from aimct.systems import MassSpringDamper, PlanarQuadrotor


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


# ----------------------------------------------- time-aware obs / reward / term

def test_time_is_threaded_to_obs_reward_and_terminated_callbacks():
    seen = {}

    def obs_fn(x, t):
        seen["obs_t"] = t
        return np.asarray(x, float)

    def reward_fn(x, u, x_next, t):
        seen["rew_t"] = t
        return -t                                   # reward == -(time of x_next)

    def terminated_fn(x_next, t):
        return t >= 0.05

    env = ControlEnv(MassSpringDamper(), dt=0.02, max_steps=100,
                     action_bounds=(-1.0, 1.0), obs_fn=obs_fn,
                     reward_fn=reward_fn, terminated_fn=terminated_fn)
    env.reset(seed=0)
    assert seen["obs_t"] == 0.0                     # obs after reset is at t = 0
    _, r1, term1, _, _ = env.step([0.0])
    assert r1 == pytest.approx(-0.02) and seen["rew_t"] == pytest.approx(0.02)
    assert seen["obs_t"] == pytest.approx(0.02) and not term1
    env.step([0.0])                                 # t -> 0.04
    _, _, term3, _, _ = env.step([0.0])             # t -> 0.06 >= 0.05
    assert term3


# ------------------------------------------------------- quad figure-8 tracking

def test_figure8_reference_and_obs_shapes_and_phase_clock():
    q = PlanarQuadrotor()
    x_ref, u_ref = figure8_reference(0.0, q)
    assert x_ref.shape == (6,) and u_ref.shape == (2,)
    assert np.all(u_ref >= 0.0) and np.all(u_ref <= q.thrust_max)
    # at t = 0 the lemniscate is at the origin, hover height, moving
    assert x_ref[0] == pytest.approx(0.0) and x_ref[1] == pytest.approx(1.0)

    o0 = figure8_obs(x_ref, 0.0, q)                 # zero error -> only the clock
    assert o0.shape == (8,)
    assert np.allclose(o0[:6], 0.0)
    assert o0[6] == pytest.approx(0.0) and o0[7] == pytest.approx(1.0)  # sin,cos
    # a quarter period later the phase clock has advanced
    o1 = figure8_obs(figure8_reference(1.5, q)[0], 1.5, q)
    assert o1[6] == pytest.approx(1.0, abs=1e-6)


def test_quad_figure8_track_gym_api_and_is_solvable_by_lqr_ff():
    env = make("quad-figure8-track")
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,) and obs.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert float(env.action_space.low[0]) == 0.0
    assert env.max_steps == 600

    q = PlanarQuadrotor()
    A, B = q.linearize()
    K = LQR(A, B,
            np.diag(1.0 / np.array([.1, .1, .2, .5, .5, 3.]) ** 2),
            np.diag(1.0 / np.array([.15, .15]) ** 2)).K

    t, term, trunc, sq = 0.0, False, False, 0.0
    while not (term or trunc):
        xr, ur = figure8_reference(t, q)
        u = np.clip(ur - K @ (env.state - xr), 0.0, q.thrust_max)
        _, r, term, trunc, _ = env.step(u)
        t += env.dt
        sq += float(np.sum((env.state[:2] - figure8_reference(t, q)[0][:2]) ** 2))
    assert trunc and not term                        # LQR+FF flies the whole lap
    assert np.sqrt(sq / 600) < 0.2                   # and tracks to < 20 cm RMS


def test_quad_figure8_track_terminates_when_it_falls_behind():
    env = make("quad-figure8-track")
    env.reset(seed=0)
    term = False
    for _ in range(env.max_steps):
        _, _, term, trunc, _ = env.step(np.zeros(2))     # no thrust -> drop
        if term or trunc:
            break
    assert term and not trunc


def test_quad_figure8_track_is_seed_reproducible():
    a, b = make("quad-figure8-track"), make("quad-figure8-track")
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert np.array_equal(oa, ob)
    q = PlanarQuadrotor()
    uh = np.asarray(q.u_hover, float)
    for _ in range(30):
        ra, rb = a.step(uh), b.step(uh)
        assert np.allclose(ra[0], rb[0]) and ra[1] == pytest.approx(rb[1])
