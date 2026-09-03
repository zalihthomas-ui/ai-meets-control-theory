"""Estimation tests — observability + Luenberger + Kalman, against the golden
values in docs/references/observers-kalman-reference.md.
"""

import numpy as np
import pytest

from aimct.estimation import (
    DiscreteKalmanFilter,
    KalmanFilter,
    LuenbergerObserver,
    is_observable,
    observability_rank,
    place_observer,
    solve_fare,
)
from aimct.systems import CartPole

C_FULL = np.eye(4)
C_CART = np.array([[1.0, 0.0, 0.0, 0.0]])
C_ANGLE = np.array([[0.0, 0.0, 1.0, 0.0]])
C_DUAL = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def cartpole_lin():
    return CartPole().linearize()  # (A, B) about upright


# ---------------------------------------------------------------- observability

@pytest.mark.parametrize(
    "C, rank, observable",
    [(C_FULL, 4, True), (C_CART, 4, True), (C_ANGLE, 2, False), (C_DUAL, 4, True)],
)
def test_cartpole_observability_matches_reference(C, rank, observable):
    A, _ = cartpole_lin()
    assert observability_rank(A, C) == rank
    assert is_observable(A, C) is observable


# ---------------------------------------------------------------- Luenberger

def test_place_observer_puts_error_poles_where_asked():
    A, B = cartpole_lin()
    poles = np.array([-6.0, -7.0, -8.0, -9.0])
    L = place_observer(A, C_DUAL, poles)
    got = np.sort(np.linalg.eigvals(A - L @ C_DUAL).real)
    assert np.allclose(got, np.sort(poles), atol=1e-6)


def test_luenberger_error_decays_to_zero():
    # stable plant (mass-spring-damper) so the truth state stays bounded while
    # the observer, started at the wrong estimate, catches up.
    from aimct.systems import MassSpringDamper

    sys = MassSpringDamper(m=1.0, c=0.4, k=1.0)
    A, B = sys.linearize()
    C = np.array([[1.0, 0.0]])                    # measure position only
    obs = LuenbergerObserver(A, B, C, poles=[-8.0, -10.0], x_hat0=np.zeros(2))

    x = np.array([1.0, 0.5])                      # true state, observer thinks [0, 0]
    dt = 1e-3
    for _ in range(6000):
        u = np.array([0.2])
        obs.update(C @ x, u, dt)
        x = x + dt * (A @ x + B @ u)
    # residual ~2e-4 is the Euler-truth vs RK4-observer discretisation mismatch,
    # not a design error: the initial estimate gap was ~1.1.
    assert np.linalg.norm(x - obs.x_hat) < 1e-3


# ---------------------------------------------------------------- Kalman (LQE)

def test_continuous_kalman_matches_golden_fixture():
    A, B = cartpole_lin()
    W = np.diag([1e-4, 1e-2, 1e-4, 1e-2])
    V = np.diag([1e-4, 1e-4])

    kf = KalmanFilter(A, B, C_DUAL, W, V)

    # NOTE: observers-kalman-reference.md sec 6 lists Sigma[1,1] and Sigma[3,3]
    # about 10x too large (a transcription error — its L and error-pole values
    # are correct and are what matters). These are the values our solver and
    # scipy.linalg.solve_continuous_are agree on to ~1e-16.
    Sigma_ref = np.array([
        [4.58519455e-04,  1.00189989e-03, -1.18273921e-05, -4.46351545e-05],
        [1.00189989e-03,  4.59825005e-03, -1.08509852e-04, -4.25671375e-04],
        [-1.18273921e-05, -1.08509852e-04,  8.36313808e-04,  3.44780336e-03],
        [-4.46351545e-05, -4.25671375e-04,  3.44780336e-03,  1.56329119e-02],
    ])
    L_ref = np.array([
        [4.5852, -0.1183],
        [10.0190, -1.0851],
        [-0.1183, 8.3631],
        [-0.4464, 34.4780],
    ])
    poles_ref = np.array([-2.2912 + 2.1805j, -2.2912 - 2.1805j,
                          -4.1830 + 1.0957j, -4.1830 - 1.0957j])

    assert np.allclose(kf.Sigma, Sigma_ref, rtol=1e-5, atol=1e-9)
    assert np.allclose(kf.L, L_ref, rtol=0, atol=5e-3)
    assert np.allclose(kf.fare_residual(), 0.0, atol=1e-6)

    got = np.sort_complex(kf.error_poles)
    assert np.allclose(got, np.sort_complex(poles_ref), atol=2e-3)
    assert np.all(kf.error_poles.real < 0)


def test_solve_fare_residual_is_zero():
    A, B = cartpole_lin()
    W = np.diag([1e-4, 1e-2, 1e-4, 1e-2])
    V = np.diag([1e-4, 1e-4])
    Sigma = solve_fare(A, C_DUAL, W, V)
    res = Sigma @ A.T + A @ Sigma - Sigma @ C_DUAL.T @ np.linalg.inv(V) @ C_DUAL @ Sigma + W
    assert np.allclose(res, 0.0, atol=1e-6)
    assert np.allclose(Sigma, Sigma.T, atol=1e-10)


# ---------------------------------------------------------------- discrete KF

def test_discrete_kalman_converges_on_noiseless_run():
    A, B = cartpole_lin()
    W = np.diag([1e-6, 1e-6, 1e-6, 1e-6])
    V = np.diag([1e-6, 1e-6])
    dt = 2e-3
    dkf = DiscreteKalmanFilter(A, B, C_DUAL, W, V, dt, x_hat0=np.zeros(4))

    x = np.array([0.05, 0.0, 0.08, 0.0])
    F = np.eye(4) + A * dt + 0.5 * (A @ A) * dt**2
    for _ in range(1500):
        x = F @ x                                 # noiseless truth (same discretisation)
        dkf.step(C_DUAL @ x, u=np.zeros(1))
    assert np.linalg.norm(x - dkf.x_hat) < 1e-2
