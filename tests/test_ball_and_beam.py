"""
Unit tests for the Ball and Beam dynamical system.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import LQR, StateFeedback
from aimct.simulate import simulate
from aimct.systems.ball_and_beam import BallAndBeam


def test_ball_and_beam_dimensions_and_defaults() -> None:
    bb = BallAndBeam()
    assert bb.n_states == 4
    assert bb.n_inputs == 1
    assert bb.n_outputs == 4
    assert bb.m == 0.064
    assert bb.M == 0.20
    assert bb.L == 0.425
    assert bb.r_max == 0.20
    assert bb.theta_max == 0.45


def test_ball_and_beam_equilibrium() -> None:
    bb = BallAndBeam()
    # At center horizontal equilibrium [r=0, r_dot=0, theta=0, theta_dot=0] with tau=0
    x_eq = np.zeros(4)
    u_eq = np.zeros(1)
    xdot = bb.dynamics(0.0, x_eq, u_eq)
    np.testing.assert_allclose(xdot, 0.0, atol=1e-10)


def test_ball_and_beam_analytical_linearization_matches_numerical() -> None:
    bb = BallAndBeam()
    A_anal, B_anal = bb.linearize(x_eq=np.zeros(4), u_eq=np.zeros(1))

    # Numerical finite-difference linearisation
    A_num, B_num = super(BallAndBeam, bb).linearize(
        x_eq=np.zeros(4), u_eq=np.zeros(1), eps=1e-6
    )

    np.testing.assert_allclose(A_anal, A_num, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(B_anal, B_num, rtol=1e-3, atol=1e-3)


def test_ball_and_beam_controllability_and_poles() -> None:
    bb = BallAndBeam()
    A, B = bb.linearize()

    # Controllability matrix rank
    Ctrb = np.column_stack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    assert np.linalg.matrix_rank(Ctrb) == 4

    # Must contain one positive unstable eigenvalue (gravity rolling)
    poles = np.linalg.eigvals(A)
    assert np.any(np.real(poles) > 2.0)  # ~ +4.12 rad/s


def test_ball_and_beam_lqr_stabilization() -> None:
    bb = BallAndBeam()
    A, B = bb.linearize()

    Q = np.diag([20.0, 2.0, 5.0, 0.5])
    R = np.array([[1.0]])
    lqr = LQR(A, B, Q, R)
    ctrl = StateFeedback(lqr.K, x_ref=np.zeros(4))

    # Initial perturbation: ball displaced by 5 cm (r0 = 0.05 m)
    x0 = np.array([0.05, 0.0, 0.0, 0.0])
    traj = simulate(bb, ctrl, x0=x0, dt=0.001, t_final=4.0, u_bounds=(-1.5, 1.5))

    # Ball position and beam angle must stabilize back to zero
    np.testing.assert_allclose(traj.x[-1], 0.0, atol=1e-3)
    assert not traj.diverged


def test_ball_and_beam_energy() -> None:
    bb = BallAndBeam()
    # At rest at origin -> E = 0
    assert bb.total_energy([0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
