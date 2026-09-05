"""Tests for :mod:`aimct.rl.imitation` - behaviour cloning and DAgger."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.rl.imitation import BehaviorCloning, aggregate, dagger


# A trivial 1-D "plant": x_{k+1} = x_k + u_k, expert drives x -> 0 with u = -0.5 x.
# obs is just the state. The learner sees the expert only near its own visited
# states; DAgger should expand that coverage.
def _expert(x):
    return np.array([-0.5 * float(np.ravel(x)[0])])


def _observe(x):
    return np.asarray(x, dtype=float).reshape(1)


def _rollout(act_fn, x0=3.0, n=40):
    x = np.array([float(x0)])
    xs = [x.copy()]
    for _ in range(n):
        u = np.clip(np.asarray(act_fn(x), dtype=float), -1.0, 1.0)
        x = x + u
        xs.append(x.copy())
    return np.array(xs)


# --------------------------------------------------------------------- BC

def test_bc_fits_expert_actions_on_the_training_states():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(400, 1))
    U = np.array([_expert(x) for x in X])
    bc = BehaviorCloning(1, 1, act_low=-3.0, act_high=3.0, seed=0)
    bc.fit(X, U, epochs=300)
    assert bc.loss_history[-1] < 1e-2
    assert bc.loss_history[-1] < bc.loss_history[0]           # it learned something
    # matches the expert closely on held-out states in the trained range
    Xt = rng.uniform(-3, 3, size=(50, 1))
    pred = np.array([bc.act(x) for x in Xt])
    assert np.allclose(pred[:, 0], -0.5 * Xt[:, 0], atol=0.08)


def test_bc_action_is_clipped_to_the_box():
    bc = BehaviorCloning(1, 1, act_low=-0.2, act_high=0.2, seed=1)
    bc.fit(np.array([[5.0], [-5.0]]), np.array([[-2.5], [2.5]]), epochs=50)
    for x in (10.0, -10.0, 0.3):
        u = bc.act(np.array([x]))
        assert -0.2 - 1e-9 <= u[0] <= 0.2 + 1e-9


def test_bc_fit_rejects_mismatched_counts():
    bc = BehaviorCloning(1, 1)
    with pytest.raises(ValueError):
        bc.fit(np.zeros((5, 1)), np.zeros((4, 1)))


def test_bc_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, size=(200, 1))
    U = np.array([_expert(x) for x in X])
    bc = BehaviorCloning(1, 1, act_low=-1.0, act_high=1.0, seed=0)
    bc.fit(X, U, epochs=80)
    p = bc.save(tmp_path / "bc.npz")
    bc2 = BehaviorCloning.load(p)
    for x in (-1.7, 0.0, 2.3):
        assert np.allclose(bc.act(np.array([x])), bc2.act(np.array([x])))


def test_aggregate_stacks_and_skips_empty():
    o1, a1 = np.ones((3, 2)), np.ones((3, 1))
    o2, a2 = 2 * np.ones((4, 2)), 2 * np.ones((4, 1))
    O, A = aggregate((o1, a1), (np.empty((0, 2)), np.empty((0, 1))), (o2, a2))
    assert O.shape == (7, 2) and A.shape == (7, 1)
    assert O[:3].mean() == 1.0 and O[3:].mean() == 2.0


# ------------------------------------------------------------------ DAgger

def test_dagger_aggregates_a_growing_expert_labelled_dataset():
    rng = np.random.default_rng(0)
    X0 = rng.uniform(-0.3, 0.3, size=(100, 1))
    bc = BehaviorCloning(1, 1, act_low=-1.0, act_high=1.0, seed=0)
    bc.fit(X0, np.array([_expert(x) for x in X0]), epochs=80)

    visited = []

    def roll(act_fn):
        X = _rollout(act_fn, x0=3.0, n=25)
        visited.append(X)
        return X

    dagger(bc, rollout_states=roll, expert=_expert, observe=_observe,
           iterations=3, fit_kwargs=dict(epochs=60))

    # one dataset per iteration, each labelled with the expert at the states the
    # rollout actually visited
    assert len(bc.dagger_datasets) == 3
    sizes = [len(o) for o, _ in bc.dagger_datasets]
    assert all(s == len(visited[i]) for i, s in enumerate(sizes))
    for k, (obs_i, act_i) in enumerate(bc.dagger_datasets):
        want = np.array([_expert(x) for x in visited[k]])
        assert np.allclose(act_i, want)
        assert np.allclose(obs_i, visited[k])            # observe() is identity here


def test_dagger_improves_expert_match_on_the_states_the_learner_visits():
    # A *curved* expert BC cannot extrapolate: u = -0.8 * tanh(1.5 x) (near-linear
    # at the origin, flat far out). Train plain BC on |x| < 0.4 only; then run
    # DAgger, whose rollouts wander out to x ~ 3. DAgger's relabelling should
    # cut the expert-imitation error on those far, previously-unseen states.
    def expert(x):
        return np.array([-0.8 * np.tanh(1.5 * float(np.ravel(x)[0]))])

    def roll(act_fn):
        return _rollout(act_fn, x0=3.0, n=30)

    rng = np.random.default_rng(0)
    Xn = rng.uniform(-0.4, 0.4, size=(200, 1))
    Un = np.array([expert(x) for x in Xn])
    far = np.linspace(1.0, 3.0, 40).reshape(-1, 1)          # never in the BC set
    far_expert = np.array([expert(x) for x in far])[:, 0]

    bc = BehaviorCloning(1, 1, act_low=-1.0, act_high=1.0, seed=0)
    bc.fit(Xn, Un, epochs=250)
    bc_err = np.mean(np.abs(np.array([bc.act(x) for x in far])[:, 0] - far_expert))

    dbc = BehaviorCloning(1, 1, act_low=-1.0, act_high=1.0, seed=0)
    dbc.fit(Xn, Un, epochs=250)
    dagger(dbc, rollout_states=roll, expert=expert, observe=_observe,
           iterations=5, fit_kwargs=dict(epochs=150))
    dbc_err = np.mean(np.abs(np.array([dbc.act(x) for x in far])[:, 0] - far_expert))

    assert dbc_err < bc_err * 0.5      # DAgger roughly halves the off-distribution gap


def test_dagger_beta_schedule_controls_which_source_the_rollout_uses():
    calls = {"expert": 0, "learner": 0}

    def expert(x):
        calls["expert"] += 1
        return np.array([9.0])

    bc = BehaviorCloning(1, 1, act_low=-20.0, act_high=20.0, seed=0)
    real_act = bc.act

    def counting_act(o):
        calls["learner"] += 1
        return real_act(o)

    bc.act = counting_act
    dagger(bc, rollout_states=lambda f: np.array([[f(np.array([1.0]))[0]]]),
           expert=expert, observe=_observe, iterations=3,
           beta_schedule=lambda i: 1.0 if i == 0 else 0.0,
           fit_kwargs=dict(epochs=5))
    # iter 0 rolled the expert (no learner call during that rollout);
    # iters 1-2 rolled the learner
    assert calls["learner"] >= 2
