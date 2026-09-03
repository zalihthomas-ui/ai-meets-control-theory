"""Tests for :mod:`aimct.controllers.lqr` — from-scratch continuous-time LQR.

The CARE solver is checked against (a) closed-form solutions, (b) the CARE
residual itself, and (c) ``scipy.linalg.solve_continuous_are`` where importable
(``scipy.linalg`` works on this box even though ``scipy.signal`` / python-control
do not).  A nonlinear cart-pole stabilisation run and the classic LQR gain-margin
property are exercised too.
"""

from __future__ import annotations

import numpy as np
import pytest

from scipy.linalg import solve_continuous_are as _scipy_care

from aimct.controllers import LQR, solve_care
from aimct.controllers.state_feedback import StateFeedback
from aimct.systems import CartPole, MassSpringDamper
from aimct.simulate import simulate


A_DI = np.array([[0.0, 1.0], [0.0, 0.0]])
B_DI = np.array([[0.0], [1.0]])


def _care_residual(A, B, Q, R, P):
    return A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q


# --------------------------------------------------------------- known answers

def test_scalar_care_closed_form():
    # 2aP - P^2 b^2/r + q = 0 ; a=0,b=1,q=1,r=1  ->  P=1, K=1
    P = solve_care([[0.0]], [[1.0]], [[1.0]], [[1.0]])
    assert P.item() == pytest.approx(1.0)
    lqr = LQR([[0.0]], [[1.0]], [[1.0]], [[1.0]], check_controllable=False)
    assert lqr.K.item() == pytest.approx(1.0)


def test_double_integrator_bryson_solution():
    # Q = I, R = 1  ->  P = [[sqrt3, 1], [1, sqrt3]], K = [1, sqrt3]
    lqr = LQR(A_DI, B_DI, np.eye(2), [[1.0]])
    s3 = np.sqrt(3.0)
    assert lqr.P == pytest.approx(np.array([[s3, 1.0], [1.0, s3]]))
    assert lqr.K == pytest.approx(np.array([[1.0, s3]]))


# --------------------------------------------------------------- CARE residual

@pytest.mark.parametrize(
    "A, B, Q, R",
    [
        (A_DI, B_DI, np.eye(2), np.array([[1.0]])),
        (A_DI, B_DI, np.diag([10.0, 1.0]), np.array([[0.05]])),
        (MassSpringDamper(k=4.0, c=0.2).linearize()[0],
         MassSpringDamper(k=4.0, c=0.2).linearize()[1],
         np.eye(2), np.array([[0.5]])),
        (CartPole().linearize()[0], CartPole().linearize()[1],
         np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.1]])),
    ],
)
def test_care_residual_is_negligible(A, B, Q, R):
    A, B, Q, R = map(np.asarray, (A, B, Q, R))
    P = solve_care(A, B, Q, R)
    assert np.max(np.abs(_care_residual(A, B, Q, R, P))) < 1e-8
    assert np.allclose(P, P.T)                          # symmetric
    assert np.all(np.linalg.eigvalsh(P) > -1e-9)        # positive semidefinite


@pytest.mark.parametrize(
    "A, B, Q, R",
    [
        (A_DI, B_DI, np.eye(2), np.array([[1.0]])),
        (CartPole().linearize()[0], CartPole().linearize()[1],
         np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.1]])),
    ],
)
def test_matches_scipy_solve_continuous_are(A, B, Q, R):
    A, B, Q, R = map(np.asarray, (A, B, Q, R))
    assert solve_care(A, B, Q, R) == pytest.approx(_scipy_care(A, B, Q, R), abs=1e-8)


# ------------------------------------------------------------------ properties

def test_closed_loop_is_hurwitz():
    lqr = LQR(A_DI, B_DI, np.eye(2), [[1.0]])
    assert np.all(np.linalg.eigvals(A_DI - B_DI @ lqr.K).real < 0)


def test_lqr_is_a_state_feedback_controller():
    lqr = LQR(A_DI, B_DI, np.eye(2), [[1.0]])
    assert isinstance(lqr, StateFeedback)
    assert lqr.update(np.array([2.0, 0.0])) == pytest.approx(-(lqr.K @ np.array([2.0, 0.0]))[0])


def test_cost_to_go_is_x_P_x():
    lqr = LQR(A_DI, B_DI, np.eye(2), [[1.0]])
    x = np.array([1.5, -0.7])
    assert lqr.cost_to_go(x) == pytest.approx(x @ lqr.P @ x)


def test_guaranteed_gain_margin():
    """SISO LQR tolerates loop-gain scaling in [1/2, inf) without losing
    stability — the classic Kalman gain-margin guarantee."""
    lqr = LQR(A_DI, B_DI, np.eye(2), [[1.0]])
    for scale in (0.5001, 1.0, 2.0, 5.0, 50.0):
        cl = A_DI - scale * B_DI @ lqr.K
        assert np.all(np.linalg.eigvals(cl).real < 1e-9)


# -------------------------------------------------------------------- guards

def test_R_not_positive_definite_raises():
    with pytest.raises(ValueError, match="positive definite"):
        solve_care(A_DI, B_DI, np.eye(2), [[-1.0]])


def test_asymmetric_Q_is_symmetrised():
    Q_asym = np.array([[1.0, 2.0], [0.0, 1.0]])
    Q_sym = 0.5 * (Q_asym + Q_asym.T)
    k_asym = LQR(A_DI, B_DI, Q_asym, [[1.0]]).K
    k_sym = LQR(A_DI, B_DI, Q_sym, [[1.0]]).K
    assert k_asym == pytest.approx(k_sym)


def test_uncontrollable_system_rejected():
    A = np.diag([1.0, -2.0])          # unstable mode 1 gets no actuation
    B = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        LQR(A, B, np.eye(2), [[1.0]])


# ------------------------------------------------------------ nonlinear cart-pole

def test_lqr_stabilises_nonlinear_cartpole():
    cp = CartPole()
    A, B = cp.linearize()
    Q = np.diag([1.0, 1.0, 10.0, 1.0])
    R = np.array([[0.1]])
    lqr = LQR(A, B, Q, R)

    traj = simulate(cp, lqr, x0=np.array([0.0, 0.0, 0.2, 0.0]), dt=0.01, t_final=8.0)
    # pole upright, cart near origin, everything quiet at the end
    assert np.abs(traj.x[-1, 2]) < 1e-2          # theta
    assert np.linalg.norm(traj.x[-1]) < 5e-2
    assert np.max(np.abs(traj.x[:, 2])) < 0.35   # never flails past ~20 deg


def test_heavier_angle_weight_reduces_peak_pole_angle():
    cp = CartPole()
    A, B = cp.linearize()
    R = np.array([[0.1]])
    x0 = np.array([0.0, 0.0, 0.25, 0.0])

    soft = simulate(cp, LQR(A, B, np.diag([1.0, 1.0, 1.0, 1.0]), R),
                    x0=x0, dt=0.01, t_final=6.0)
    stiff = simulate(cp, LQR(A, B, np.diag([1.0, 1.0, 200.0, 1.0]), R),
                     x0=x0, dt=0.01, t_final=6.0)
    assert np.max(np.abs(stiff.x[:, 2])) <= np.max(np.abs(soft.x[:, 2])) + 1e-9
