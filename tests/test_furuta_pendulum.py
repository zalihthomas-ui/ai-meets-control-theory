"""
Unit tests for the Furuta Pendulum (Rotary Inverted Pendulum) dynamical system.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import LQR, StateFeedback
from aimct.simulate import simulate
from aimct.systems.furuta_pendulum import FurutaPendulum


def test_furuta_dimensions_and_defaults() -> None:
    furuta = FurutaPendulum()
    assert furuta.n_states == 4
    assert furuta.n_inputs == 1
    assert furuta.n_outputs == 4
    assert furuta.mr == 0.095
    assert furuta.mp == 0.024
    assert furuta.g == 9.81


def test_furuta_equilibrium_dynamics() -> None:
    furuta = FurutaPendulum()
    # At upright equilibrium [0, 0, 0, 0] with u=0, dynamics must evaluate to 0
    x_eq = np.zeros(4)
    u_eq = np.zeros(1)
    xdot = furuta.dynamics(0.0, x_eq, u_eq)
    np.testing.assert_allclose(xdot, 0.0, atol=1e-10)

    # At downward hanging equilibrium [0, pi, 0, 0] with u=0, dynamics must also evaluate to 0
    x_down = np.array([0.0, np.pi, 0.0, 0.0])
    xdot_down = furuta.dynamics(0.0, x_down, u_eq)
    np.testing.assert_allclose(xdot_down, 0.0, atol=1e-10)


def test_furuta_analytical_linearization_matches_numerical() -> None:
    furuta = FurutaPendulum()
    A_anal, B_anal = furuta.linearize(x_eq=np.zeros(4), u_eq=np.zeros(1))

    # Numerical finite difference linearisation
    A_num, B_num = super(FurutaPendulum, furuta).linearize(
        x_eq=np.zeros(4), u_eq=np.zeros(1), eps=1e-6
    )

    np.testing.assert_allclose(A_anal, A_num, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(B_anal, B_num, rtol=1e-3, atol=1e-3)


def test_furuta_controllability_and_poles() -> None:
    furuta = FurutaPendulum()
    A, B = furuta.linearize()

    # Controllability matrix rank
    Ctrb = np.column_stack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    rank = np.linalg.matrix_rank(Ctrb)
    assert rank == 4

    # Verify open-loop unstable pole exists
    poles = np.linalg.eigvals(A)
    assert np.any(np.real(poles) > 5.0)  # ~ +12.7 rad/s


def test_furuta_energy_conservation_undamped() -> None:
    # Zero damping, zero input -> total mechanical energy must be conserved
    furuta_undamped = FurutaPendulum(Dr=0.0, Dp=0.0)
    x0 = np.array([0.0, 0.5, 0.0, 0.0])  # Displaced pendulum released from rest

    # Step simulation with small dt RK4
    x = x0.copy()
    dt = 0.0005
    E_init = furuta_undamped.total_energy(x)

    for _ in range(2000):  # 1.0 second
        k1 = furuta_undamped.dynamics(0.0, x, [0.0])
        k2 = furuta_undamped.dynamics(0.0, x + 0.5 * dt * k1, [0.0])
        k3 = furuta_undamped.dynamics(0.0, x + 0.5 * dt * k2, [0.0])
        k4 = furuta_undamped.dynamics(0.0, x + dt * k3, [0.0])
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    E_final = furuta_undamped.total_energy(x)
    assert abs(E_final - E_init) < 1e-4


def test_furuta_pendulum_relative_energy() -> None:
    furuta = FurutaPendulum()
    # Upright rest -> E = 0
    assert furuta.pendulum_energy([0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-10)

    # Hanging rest -> E = -2 * mp * g * lp
    E_hanging_expected = -2.0 * furuta.mp * furuta.g * furuta.lp
    assert furuta.pendulum_energy([0.0, np.pi, 0.0, 0.0]) == pytest.approx(
        E_hanging_expected, rel=1e-5
    )


def test_furuta_lqr_stabilization() -> None:
    furuta = FurutaPendulum()
    A, B = furuta.linearize()

    Q = np.diag([2.0, 10.0, 0.2, 0.5])
    R = np.array([[20.0]])
    lqr = LQR(A, B, Q, R)
    ctrl = StateFeedback(lqr.K, x_ref=np.zeros(4))

    # Initial perturbation alpha = 0.05 rad (~2.9 deg)
    x0 = np.array([0.0, 0.05, 0.0, 0.0])
    traj = simulate(furuta, ctrl, x0=x0, dt=0.001, t_final=3.0, u_bounds=(-0.15, 0.15))

    # Assert convergence to upright equilibrium within 1e-3
    np.testing.assert_allclose(traj.x[-1], 0.0, atol=1e-3)
    assert not traj.diverged
