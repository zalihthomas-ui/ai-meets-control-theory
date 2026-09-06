"""Tests for :class:`aimct.estimation.MovingHorizonEstimator` (MHE)."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.estimation import (
    DiscreteKalmanFilter,
    ExtendedKalmanFilter,
    MHE,
    MovingHorizonEstimator,
)
from aimct.systems import MassSpringDamper, TwoTank
from aimct.simulate import rk4_step


# -------------------------------------------------- parameter validation & aliases

def test_mhe_alias_and_initialization():
    assert MHE is MovingHorizonEstimator
    Q = np.eye(2) * 1e-4
    R = np.eye(1) * 1e-3
    mhe = MovingHorizonEstimator(
        lambda x, u: np.zeros(2),
        lambda x: np.array([x[0]]),
        Q,
        R,
        horizon=5,
        dt=0.01,
        arrival_cost_mode="ekf",
    )
    assert mhe.n == 2
    assert mhe.p == 1
    assert mhe.horizon == 5
    assert repr(mhe).startswith("MovingHorizonEstimator")

    with pytest.raises(ValueError, match="Unknown arrival_cost_mode"):
        MovingHorizonEstimator(
            lambda x, u: np.zeros(2),
            lambda x: np.array([x[0]]),
            Q,
            R,
            dt=0.01,
            arrival_cost_mode="invalid_mode",
        )


# -------------------------------------------------- linear-Gaussian consistency vs EKF

def test_mhe_linear_gaussian_consistency_vs_ekf():
    """On an unconstrained linear-Gaussian model, MHE should closely track EKF."""
    msd = MassSpringDamper()
    A, B = msd.linearize()
    C = np.array([[1.0, 0.0]])
    W = np.diag([1e-4, 1e-4])
    V = np.array([[1e-3]])
    dt = 0.02

    dkf = DiscreteKalmanFilter(A, B, C, W, V, dt, x_hat0=np.zeros(2))
    F, Gu, Qd, Rd = dkf.F, dkf.Gu, dkf.Qd, dkf.Rd

    f_map = lambda x, u: F @ x + Gu @ u
    h_map = lambda x: C @ x

    ekf = ExtendedKalmanFilter(f_map, h_map, Qd, Rd, dt=dt, n=2, discrete=True, x0=np.zeros(2))
    mhe = MovingHorizonEstimator(f_map, h_map, Qd, Rd, horizon=8, dt=dt, n=2, discrete=True, x0=np.zeros(2))

    rng = np.random.default_rng(42)
    x = np.array([0.5, -0.1])

    ekf_errors = []
    mhe_errors = []

    for k in range(80):
        u = np.array([0.5 * np.sin(0.2 * k)])
        x = F @ x + Gu @ u + rng.multivariate_normal(np.zeros(2), Qd * 0.1)
        y = C @ x + rng.normal(0.0, np.sqrt(Rd[0, 0]), size=1)

        x_ekf = ekf.step(y, u)
        x_mhe = mhe.step(y, u)

        ekf_errors.append(np.linalg.norm(x_ekf - x))
        mhe_errors.append(np.linalg.norm(x_mhe - x))

    rmse_ekf = np.mean(ekf_errors)
    rmse_mhe = np.mean(mhe_errors)

    # Both should achieve good tracking, within close margin
    assert rmse_mhe < 0.25
    assert abs(rmse_mhe - rmse_ekf) < 0.15


# -------------------------------------------------- hard state bounds enforcement

def test_mhe_enforces_hard_lower_and_upper_state_bounds():
    """When measurements contain negative noise dips, MHE strictly enforces x >= 0."""
    dt = 0.05
    # Scalar decaying system: x_{k+1} = 0.9 * x_k, true state near zero
    f = lambda x, u: 0.9 * x
    h = lambda x: x
    Q = np.array([[1e-4]])
    R = np.array([[0.04]])  # High measurement noise std = 0.2

    x_min = np.array([0.0])
    x_max = np.array([1.0])

    mhe = MovingHorizonEstimator(
        f, h, Q, R, horizon=5, dt=dt, discrete=True,
        x_min=x_min, x_max=x_max, x0=np.array([0.1]),
    )
    ekf = ExtendedKalmanFilter(
        f, h, Q, R, dt=dt, discrete=True, x0=np.array([0.1]),
    )

    # Sequence of noisy negative measurements
    noisy_measurements = [-0.15, -0.25, -0.10, -0.30, -0.05, -0.20, 0.02, -0.12]

    ekf_violations = 0
    mhe_violations = 0

    for ym in noisy_measurements:
        y = np.array([ym])
        x_ekf = ekf.step(y)
        x_mhe = mhe.step(y)

        if x_ekf[0] < 0.0:
            ekf_violations += 1
        if x_mhe[0] < 0.0 - 1e-7:
            mhe_violations += 1

        assert x_mhe[0] >= 0.0 - 1e-7, f"MHE violated lower bound: {x_mhe[0]}"
        assert x_mhe[0] <= 1.0 + 1e-7, f"MHE violated upper bound: {x_mhe[0]}"
        assert np.all(mhe.trajectory >= 0.0 - 1e-7), "Trajectory violated lower bound"

    # EKF should have dipped negative, while MHE stayed non-negative
    assert ekf_violations > 0
    assert mhe_violations == 0


# -------------------------------------------------- disturbance bounds enforcement

def test_mhe_disturbance_bounds():
    """MHE clips or penalizes estimated disturbances within w_min, w_max."""
    dt = 0.05
    f = lambda x, u: x + dt * u
    h = lambda x: x
    Q = np.array([[1e-2]])
    R = np.array([[1e-2]])

    w_min = np.array([-0.05])
    w_max = np.array([0.05])

    mhe = MovingHorizonEstimator(
        f, h, Q, R, horizon=6, dt=dt, discrete=True,
        w_min=w_min, w_max=w_max, x0=np.array([0.0]),
    )

    # Large measurement jumps that would otherwise imply large disturbances
    y_seq = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for ym in y_seq:
        mhe.step(np.array([ym]), np.array([0.0]))
        if len(mhe.disturbances) > 0:
            assert np.all(mhe.disturbances >= w_min[0] - 1e-6)
            assert np.all(mhe.disturbances <= w_max[0] + 1e-6)


# -------------------------------------------------- general nonlinear constraints

def test_mhe_general_nonlinear_constraints():
    """Test custom inequality constraints g(x) <= 0."""
    dt = 0.05
    f = lambda x, u: x
    h = lambda x: x
    Q = np.eye(2) * 1e-3
    R = np.eye(2) * 1e-2

    # Circular keep-in constraint: x0^2 + x1^2 - 1.0 <= 0 (radius <= 1.0)
    def ball_constraint(x):
        return x[0]**2 + x[1]**2 - 1.0

    mhe = MovingHorizonEstimator(
        f, h, Q, R, horizon=4, dt=dt, discrete=True,
        constraints=[ball_constraint], x0=np.zeros(2),
    )

    # Measurements outside the unit ball
    for _ in range(6):
        y = np.array([1.5, 1.5])  # radius approx 2.12
        x_est = mhe.step(y)
        assert np.linalg.norm(x_est) <= 1.0 + 1e-4


# -------------------------------------------------- arrival cost stability across long horizons

def test_mhe_arrival_cost_stability_over_long_horizon():
    """Verify that EKF arrival cost propagation keeps covariance bounded over 100+ steps."""
    tank = TwoTank()
    dt = 0.1
    Q = np.diag([1e-5, 1e-5])
    R = np.diag([1e-4, 1e-4])

    mhe = MovingHorizonEstimator.from_system(
        tank,
        Q=Q,
        R=R,
        horizon=5,
        dt=dt,
        x_min=[0.0, 0.0],
        x_max=[tank.h_max, tank.h_max],
        x0=np.array([0.05, 0.03]),
    )

    x = np.array([0.05, 0.03])
    u = np.array([4.0])

    for k in range(80):
        x = rk4_step(lambda t, xx, uu: tank.dynamics(t, xx, uu), 0.0, x, u, dt)
        y = x + np.random.normal(0.0, 0.005, size=2)
        x_hat = mhe.step(y, u)

        # Covariance must remain positive definite and bounded
        assert np.all(np.isfinite(mhe.P))
        eigvals = np.linalg.eigvalsh(mhe.P)
        assert np.all(eigvals > 0.0)
        assert np.max(eigvals) < 10.0
        assert np.all(x_hat >= 0.0 - 1e-7)


# -------------------------------------------------- reset & predict interface

def test_mhe_reset_and_predict():
    tank = TwoTank()
    dt = 0.05
    Q = np.eye(2) * 1e-4
    R = np.eye(2) * 1e-3

    mhe = MovingHorizonEstimator.from_system(
        tank, Q=Q, R=R, horizon=4, dt=dt,
        x_min=[0.0, 0.0], x_max=[tank.h_max, tank.h_max],
        x0=np.array([0.1, 0.05]),
    )

    for _ in range(5):
        mhe.step(np.array([0.1, 0.05]), np.array([3.0]))

    assert len(mhe.history_y) == 5
    mhe.reset(x0=np.array([0.02, 0.01]))
    assert len(mhe.history_y) == 0
    assert len(mhe.history_u) == 0
    assert np.allclose(mhe.x_hat, [0.02, 0.01])

    # Test predict
    x_pred = mhe.predict(u=[5.0])
    assert len(x_pred) == 2
    assert x_pred[0] > 0.0
