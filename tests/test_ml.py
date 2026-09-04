"""Tests for the from-scratch ML stack: MLP, LearnedDynamics, SamplingMPC."""

import numpy as np
import pytest

from aimct.controllers import SamplingMPC
from aimct.ml import MLP, LearnedDynamics, system_step
from aimct.simulate import rk4_step, simulate
from aimct.systems import (CartPole, DifferentialDriveRobot, MassSpringDamper,
                          Pendulum, PlanarQuadrotor)


# ------------------------------------------------------------------ MLP

def test_mlp_backprop_matches_finite_differences():
    net = MLP([3, 8, 2], activation="tanh", seed=1)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3))
    Y = rng.standard_normal((5, 2))
    loss0, gW, gb = net._loss_grad(X, Y)

    eps = 1e-6
    for li in range(len(net.W)):
        Wref = net.W[li]
        for idx in [(0, 0), (Wref.shape[0] - 1, Wref.shape[1] - 1)]:
            Wref[idx] += eps
            lp, _, _ = net._loss_grad(X, Y)
            Wref[idx] -= 2 * eps
            lm, _, _ = net._loss_grad(X, Y)
            Wref[idx] += eps
            num = (lp - lm) / (2 * eps)
            assert abs(num - gW[li][idx]) < 1e-5


def test_mlp_fits_a_nonlinear_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(600, 1))
    Y = np.sin(X) + 0.3 * X
    net = MLP([1, 32, 32, 1], seed=0)
    hist = net.fit(X, Y, epochs=300, lr=5e-3, batch_size=64)
    assert hist[-1] < 0.01
    assert hist[-1] < hist[0]


# ------------------------------------------------------------------ LearnedDynamics

def _rollout(system, dt, n_steps, seed, amp=1.0, hold=5):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-amp, amp, size=(n_steps // hold + 1, system.n_inputs))
    useq = np.repeat(raw, hold, axis=0)[:n_steps]
    k = {"i": 0}

    def ctrl(y, _dt):
        i = min(k["i"], n_steps - 1)
        k["i"] += 1
        return useq[i]

    tr = simulate(system, ctrl, x0=rng.normal(scale=0.1, size=system.n_states),
                  dt=dt, t_final=n_steps * dt)
    return tr.x, tr.u[:-1]


def test_learned_dynamics_predicts_mass_spring_damper():
    dt = 0.02
    sys = MassSpringDamper(m=1.0, c=0.4, k=1.0)
    Xtr, Utr = _rollout(sys, dt, 1200, seed=0, amp=1.5)
    Xte, Ute = _rollout(sys, dt, 400, seed=99, amp=1.5)

    model = LearnedDynamics(2, 1, hidden=(48, 48), seed=0)
    model.fit(Xtr, Utr, epochs=400, lr=5e-3)

    assert model.prediction_error(Xte, Ute, horizon=1) < 5e-3
    assert model.prediction_error(Xte, Ute, horizon=20) < 5e-2


@pytest.mark.slow
def test_learned_residual_over_a_wrong_physics_model_beats_both():
    """Grey-box: a learned correction on an approximate physics model should
    beat the physics alone AND a pure black-box residual model."""
    from aimct.ml import system_step

    true = MassSpringDamper(m=1.0, c=0.4, k=1.2)
    wrong = MassSpringDamper(m=1.0, c=0.4, k=1.0)      # 17% stiffness error
    dt = 0.02
    base = system_step(wrong, dt)

    Xtr, Utr = _rollout(true, dt, 1500, seed=0, amp=1.5)
    Xte, Ute = _rollout(true, dt, 400, seed=7, amp=1.5)

    plain = LearnedDynamics(2, 1, hidden=(32, 32), seed=0)
    plain.fit(Xtr, Utr, epochs=300, lr=4e-3)
    grey = LearnedDynamics(2, 1, hidden=(32, 32), base_step=base, seed=0)
    grey.fit(Xtr, Utr, epochs=300, lr=4e-3)

    # wrong physics alone, 30-step open loop
    wrong_err = []
    for k in range(len(Xte) - 31):
        x = Xte[k].copy()
        for j in range(30):
            x = np.atleast_2d(base(x, Ute[k + j]))[0]
        wrong_err.append(x - Xte[k + 30])
    wrong_rms = float(np.sqrt(np.mean(np.asarray(wrong_err) ** 2)))

    grey_rms = grey.prediction_error(Xte, Ute, horizon=30)
    plain_rms = plain.prediction_error(Xte, Ute, horizon=30)
    assert grey_rms < plain_rms < wrong_rms


def test_learned_dynamics_batched_step_shapes():
    model = LearnedDynamics(4, 1, seed=0)
    X, U = _rollout(CartPole(), 0.01, 400, seed=1, amp=3.0)
    model.fit(X, U, epochs=50)
    xb = np.zeros((7, 4))
    ub = np.zeros((7, 1))
    out = model.step(xb, ub)
    assert out.shape == (7, 4)
    assert model.step(np.zeros(4), np.zeros(1)).shape == (4,)


# ------------------------------------------------------------------ batched fields

@pytest.mark.parametrize("sys", [MassSpringDamper(), Pendulum(), CartPole(),
                                 PlanarQuadrotor(), DifferentialDriveRobot()])
def test_batched_field_matches_scalar_dynamics(sys):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, sys.n_states))
    U = rng.standard_normal((6, sys.n_inputs))
    step = system_step(sys, 0.01)
    got = step(X, U)
    want = np.array([rk4_step(sys.dynamics, 0.0, x, u, 0.01) for x, u in zip(X, U)])
    assert np.allclose(got, want, atol=1e-10)


# ------------------------------------------------------------------ SamplingMPC

@pytest.mark.slow
def test_sampling_mpc_balances_pendulum_near_upright_with_true_model():
    sys = Pendulum(m=1.0, L=1.0, b=0.1)
    dt = 0.02
    step = system_step(sys, dt)

    def running_cost(X, U):               # hold theta at pi (upright)
        ang = np.arctan2(np.sin(X[:, 0] - np.pi), np.cos(X[:, 0] - np.pi))
        return ang**2 + 0.02 * X[:, 1] ** 2 + 0.001 * U[:, 0] ** 2

    mpc = SamplingMPC(step, running_cost, horizon=50, n_samples=400, n_elite=40,
                      n_iter=4, u_dim=1, u_bounds=(-6.0, 6.0), seed=0)

    traj = simulate(sys, mpc, x0=[np.pi - 0.3, 0.0], dt=dt, t_final=4.0,
                    u_bounds=(-6.0, 6.0))
    final_ang = abs(np.arctan2(np.sin(traj.x[-1, 0] - np.pi),
                               np.cos(traj.x[-1, 0] - np.pi)))
    assert final_ang < 0.2             # caught and held within ~11 deg of upright
    assert not traj.diverged


@pytest.mark.slow
def test_sampling_mpc_respects_action_box_and_regulates():
    sys = MassSpringDamper()
    dt = 0.05
    step = system_step(sys, dt)

    def running_cost(X, U):
        return X[:, 0] ** 2 + X[:, 1] ** 2 + 0.01 * U[:, 0] ** 2

    mpc = SamplingMPC(step, running_cost, horizon=20, n_samples=256, n_elite=25,
                      n_iter=4, u_dim=1, u_bounds=(-2.0, 2.0), seed=0)
    traj = simulate(sys, mpc, x0=[3.0, 0.0], dt=dt, t_final=8.0)
    assert np.all(np.abs(traj.u) <= 2.0 + 1e-9)
    assert np.linalg.norm(traj.x[-1]) < 0.35       # regulated toward the origin
