"""
Unit tests for the Coupled Two-Tank dynamical system.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import LQR, StateFeedback
from aimct.simulate import simulate
from aimct.systems.linear import LinearSystem
from aimct.systems.two_tank import TwoTank


def test_two_tank_dimensions_and_defaults() -> None:
    tank = TwoTank()
    assert tank.n_states == 2
    assert tank.n_inputs == 1
    assert tank.n_outputs == 2
    assert tank.h_max == 0.30
    assert tank.v_max == 12.0
    assert tank.A1 == 1.555e-3
    assert tank.A2 == 1.555e-3


def test_two_tank_operating_point_equilibrium() -> None:
    tank = TwoTank()
    # At target level h2 = 0.10 m (10 cm)
    x_eq, u_eq = tank.steady_state_operating_point(h2_target=0.10)
    assert x_eq[1] == pytest.approx(0.10)
    assert x_eq[0] == pytest.approx(0.20)  # h1_0 = 2 * h2_0 when a12 == a2
    assert 0.0 < u_eq[0] < tank.v_max

    # Dynamics at equilibrium must evaluate to zero
    xdot = tank.dynamics(0.0, x_eq, u_eq)
    np.testing.assert_allclose(xdot, 0.0, atol=1e-10)


def test_two_tank_analytical_linearization_matches_numerical() -> None:
    tank = TwoTank()
    x_eq, u_eq = tank.steady_state_operating_point(0.10)

    A_anal, B_anal = tank.linearize(x_eq=x_eq, u_eq=u_eq)

    # Numerical finite-difference linearisation
    A_num, B_num = super(TwoTank, tank).linearize(x_eq=x_eq, u_eq=u_eq, eps=1e-6)

    np.testing.assert_allclose(A_anal, A_num, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(B_anal, B_num, rtol=1e-3, atol=1e-3)


def test_two_tank_controllability_and_stability() -> None:
    tank = TwoTank()
    A, B = tank.linearize()

    # Controllability matrix rank
    Ctrb = np.column_stack([B, A @ B])
    assert np.linalg.matrix_rank(Ctrb) == 2

    # Open-loop poles must be real and strictly negative (self-regulating stable process)
    poles = np.linalg.eigvals(A)
    assert np.all(np.real(poles) < 0.0)
    assert np.all(np.imag(poles) == 0.0)


def test_two_tank_lqr_linear_regulation() -> None:
    tank = TwoTank()
    x_eq, u_eq = tank.steady_state_operating_point(0.10)
    A, B = tank.linearize(x_eq=x_eq, u_eq=u_eq)

    Q = np.diag([1.0, 50.0])
    R = np.array([[1.0]])
    lqr = LQR(A, B, Q, R)
    ctrl = StateFeedback(lqr.K, x_ref=np.zeros(2))

    # Test linearised system regulation around operating point perturbation
    lin_sys = LinearSystem(A, B)
    delta_x0 = np.array([0.02, -0.01])
    traj_lin = simulate(lin_sys, ctrl, x0=delta_x0, dt=0.05, t_final=150.0)

    # State perturbation must converge to zero
    np.testing.assert_allclose(traj_lin.x[-1], 0.0, atol=1e-3)


def test_two_tank_physical_bounding() -> None:
    tank = TwoTank()
    # Zero voltage -> tanks should drain toward zero without going negative
    x0 = np.array([0.05, 0.05])
    traj_drain = simulate(tank, lambda t, x: np.array([0.0]), x0=x0, dt=0.1, t_final=50.0)
    assert np.all(traj_drain.x >= -1e-6)
    assert traj_drain.x[-1, 1] < 1e-3
    assert traj_drain.x[-1, 0] < 1e-3
