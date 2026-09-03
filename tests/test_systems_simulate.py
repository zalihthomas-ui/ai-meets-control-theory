"""Smoke + correctness tests for the systems interface and the simulator."""

import numpy as np
import pytest

from aimct.simulate import rk4_step, simulate
from aimct.systems import CartPole, LinearSystem, MassSpringDamper, Pendulum


def test_rk4_matches_exponential_decay():
    # xdot = -x  ->  x(t) = x0 e^{-t}
    f = lambda t, x, u: -x
    x = np.array([1.0])
    dt = 1e-3
    for _ in range(1000):
        x = rk4_step(f, 0.0, x, np.zeros(1), dt)
    assert x[0] == pytest.approx(np.exp(-1.0), rel=1e-6)


def test_mass_spring_damper_decays_to_origin():
    sys = MassSpringDamper(m=1.0, c=0.5, k=1.0)
    traj = simulate(sys, lambda y, dt: np.zeros(1), x0=[1.0, 0.0], dt=0.01, t_final=40.0)
    assert np.linalg.norm(traj.x[-1]) < 1e-2
    assert traj.t[-1] == pytest.approx(40.0)
    assert traj.x.shape == (len(traj.t), 2)


def test_linearize_numeric_matches_analytic_msd():
    sys = MassSpringDamper(m=2.0, c=0.7, k=3.0)
    A_num, B_num = super(MassSpringDamper, sys).linearize(np.zeros(2), np.zeros(1))
    A_an, B_an = sys.linearize()
    assert np.allclose(A_num, A_an, atol=1e-5)
    assert np.allclose(B_num, B_an, atol=1e-5)


def test_pendulum_upright_linearization_is_unstable():
    sys = Pendulum()
    A, _ = sys.linearize()  # about theta = pi (upright)
    assert np.max(np.linalg.eigvals(A).real) > 0


def test_cartpole_linearize_numeric_matches_analytic():
    sys = CartPole()
    x0, u0 = np.zeros(4), np.zeros(1)
    A_num, B_num = super(CartPole, sys).linearize(x0, u0)
    A_an, B_an = sys.linearize()
    assert np.allclose(A_num, A_an, atol=1e-4)
    assert np.allclose(B_num, B_an, atol=1e-4)


def test_divergent_run_stops_early_and_flags_diverged():
    # unstable scalar plant xdot = +x with no control -> blows up
    sys = LinearSystem(A=[[1.0]], B=[[1.0]])
    traj = simulate(sys, lambda y, dt: np.zeros(1), x0=[1.0], dt=0.1, t_final=1e6)
    assert traj.diverged is True
    assert len(traj) < int(round(1e6 / 0.1)) + 1     # stopped early
    assert not np.all(np.isfinite(traj.x))           # keeps the blow-up sample


def test_finite_run_is_not_flagged_diverged():
    sys = MassSpringDamper()
    traj = simulate(sys, lambda y, dt: np.zeros(1), x0=[1.0, 0.0], dt=0.01, t_final=5.0)
    assert traj.diverged is False


def test_input_bounds_are_enforced():
    sys = LinearSystem(A=[[0.0]], B=[[1.0]])
    traj = simulate(
        sys, lambda y, dt: np.array([100.0]), x0=[0.0], dt=0.1, t_final=1.0,
        u_bounds=(-1.0, 1.0),
    )
    assert np.all(traj.u <= 1.0 + 1e-12)
