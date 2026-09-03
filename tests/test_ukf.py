"""Tests for :class:`aimct.estimation.UnscentedKalmanFilter`."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import LQR, ObserverFeedback, wrap_angle
from aimct.estimation import (
    DiscreteKalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
)
from aimct.systems import MassSpringDamper, Pendulum
from aimct.simulate import rk4_step, simulate


def _pend():
    p = Pendulum()
    return p, (lambda x, u: p.dynamics(0.0, x, u))


# ------------------------------------------------------------------ weights

def test_sigma_weights_and_points():
    ukf = UnscentedKalmanFilter(lambda x, u: x, lambda x: x[:1],
                                np.eye(3), np.array([[1.0]]),
                                dt=0.1, n=3, discrete=True, alpha=0.7, beta=2.0)
    assert ukf.Wm.shape == (7,) and ukf.Wc.shape == (7,)
    assert ukf.Wm.sum() == pytest.approx(1.0)
    assert ukf.Wc.sum() == pytest.approx(1.0 + (1.0 - 0.7**2 + 2.0))
    pts = ukf._sigma_points()
    assert pts.shape == (7, 3)
    assert pts[0] == pytest.approx(ukf.x_hat)
    # symmetric about the mean
    assert (pts[1:4] + pts[4:7]) / 2 == pytest.approx(np.tile(ukf.x_hat, (3, 1)))


# -------------------------------------------------- exact on a linear model

def test_matches_the_linear_kalman_filter_on_a_linear_model():
    msd = MassSpringDamper()
    A, B = msd.linearize()
    C = np.array([[1.0, 0.0]])
    W, V, dt = np.diag([1e-4, 1e-3]), np.array([[1e-3]]), 0.02

    dkf = DiscreteKalmanFilter(A, B, C, W, V, dt, x_hat0=np.zeros(2))
    F, Gu, Qd, Rd = dkf.F, dkf.Gu, dkf.Qd, dkf.Rd
    ukf = UnscentedKalmanFilter(lambda x, u: F @ x + Gu @ u, lambda x: C @ x,
                                Qd, Rd, dt=dt, n=2, discrete=True, x0=np.zeros(2))

    rng = np.random.default_rng(0)
    x = np.array([0.6, -0.2])
    for k in range(120):
        u = np.array([np.sin(0.1 * k)])
        x = F @ x + Gu @ u
        y = C @ x + rng.normal(0.0, np.sqrt(Rd[0, 0]), size=1)
        assert ukf.step(y, u) == pytest.approx(dkf.step(y, u), abs=1e-7)
    assert ukf.P == pytest.approx(dkf.P, abs=1e-7)


def test_agrees_with_the_ekf_on_a_mild_nonlinearity():
    p, f = _pend()
    h = lambda x: np.array([np.sin(x[0]), x[1]])                 # noqa: E731
    Q, R = np.diag([1e-6, 1e-4]), np.diag([1e-3, 1e-3])
    kw = dict(dt=0.02, n=2, x0=np.array([2.7, 0.0]), P0=0.2 * np.eye(2))
    ekf = ExtendedKalmanFilter(f, h, Q, R, **kw)
    ukf = UnscentedKalmanFilter(f, h, Q, R, alpha=1e-2, **kw)

    rng = np.random.default_rng(1)
    xt = np.array([3.0, 0.4])
    e, u_ = [], []
    for k in range(300):
        step_u = np.array([0.3 * np.cos(0.05 * k)])
        xt = rk4_step(lambda t, a, b: p.dynamics(0, a, b), 0.0, xt, step_u, 0.02)
        y = h(xt) + rng.normal(0.0, np.sqrt(np.diag(R)))
        e.append(np.linalg.norm(ekf.step(y, step_u) - xt))
        u_.append(np.linalg.norm(ukf.step(y, step_u) - xt))
    # a weak nonlinearity over a small covariance -> the two are within a few %
    assert np.mean(u_[-100:]) == pytest.approx(np.mean(e[-100:]), rel=0.1)


# --------------------------------------------------- nonlinear convergence

def test_converges_and_keeps_a_valid_covariance_on_the_pendulum():
    p, f = _pend()
    h = lambda x: np.array([np.sin(x[0]), x[1]])                 # noqa: E731
    Q, R = np.diag([1e-6, 1e-4]), np.diag([2e-3, 2e-3])
    ukf = UnscentedKalmanFilter(f, h, Q, R, dt=0.02, n=2, alpha=0.5,
                                x0=np.array([2.4, 0.0]), P0=0.5 * np.eye(2))
    rng = np.random.default_rng(2)
    xt = np.array([3.0, 0.6])
    err0 = np.linalg.norm(ukf.x_hat - xt)
    errs = []
    for k in range(500):
        u = np.array([0.4 * np.sin(0.04 * k)])
        xt = rk4_step(lambda t, a, b: p.dynamics(0, a, b), 0.0, xt, u, 0.02)
        y = h(xt) + rng.normal(0.0, np.sqrt(np.diag(R)))
        ukf.step(y, u)
        errs.append(np.linalg.norm(ukf.x_hat - xt))
        assert np.allclose(ukf.P, ukf.P.T)
        assert np.all(np.linalg.eigvalsh(ukf.P) > -1e-9)
    assert err0 > 0.4 and np.mean(errs[-50:]) < 0.05


def test_ukf_recovers_a_bad_initial_angle_where_the_ekf_stays_stuck():
    """From a hopeless initial guess and a sparse, curved measurement, the EKF's
    single linearisation keeps it in the wrong basin; the UKF's spread sigma
    points see the true measurement map and pull it in."""
    p, f = _pend()
    h = lambda x: np.array([np.sin(x[0]), np.cos(x[0])])        # noqa: E731
    Q, R = np.diag([1e-5, 1e-3]), np.diag([1e-2, 1e-2])
    kw = dict(dt=0.02, n=2, x0=np.array([0.0, 0.0]), P0=np.diag([15.0, 15.0]))
    ekf = ExtendedKalmanFilter(f, h, Q, R, **kw)
    ukf = UnscentedKalmanFilter(f, h, Q, R, alpha=1.0, beta=2.0, kappa=0.0, **kw)

    rng = np.random.default_rng(4)
    xt = np.array([3.1, 2.0])
    e_ekf = e_ukf = []
    ekf_e, ukf_e = [], []
    for k in range(400):
        u = np.array([0.4 * np.sin(0.05 * k)])
        xt = rk4_step(lambda t, a, b: p.dynamics(0, a, b), 0.0, xt, u, 0.02)
        if k % 5 == 0:
            y = h(xt) + rng.normal(0.0, 0.1, 2)
            ekf.step(y, u); ukf.step(y, u)
        else:
            ekf.predict(u); ukf.predict(u)
        ekf_e.append(np.linalg.norm(ekf.x_hat - xt))
        ukf_e.append(np.linalg.norm(ukf.x_hat - xt))

    assert np.mean(ekf_e[-100:]) > 2.0                          # EKF never recovers
    assert np.mean(ukf_e[-100:]) < 0.4 * np.mean(ekf_e[-100:])  # UKF does


# ------------------------------------------------------------- mechanics

def test_residual_hook_controls_the_innovation():
    p, f = _pend()
    h = lambda x: np.array([np.sin(x[0]), x[1]])                 # noqa: E731
    Q, R = np.diag([1e-5, 1e-4]), np.diag([1e-3, 1e-3])
    ukf0 = UnscentedKalmanFilter(f, h, Q, R, dt=0.02, n=2,
                                 residual=lambda y, yp: np.zeros(2),
                                 x0=np.array([1.0, 0.2]))
    before = ukf0.x_hat.copy()
    ukf0.update(np.array([9.0, 9.0]))
    assert ukf0.x_hat == pytest.approx(before)                  # zero innovation -> no move


def test_step_equals_predict_then_update_and_reset_restores():
    p, f = _pend()
    h = lambda x: np.array([np.sin(x[0])])                       # noqa: E731
    Q, R = np.diag([1e-5, 1e-4]), np.array([[1e-3]])
    x0, P0 = np.array([1.0, 0.2]), 0.3 * np.eye(2)
    a = UnscentedKalmanFilter(f, h, Q, R, dt=0.02, n=2, x0=x0, P0=P0)
    b = UnscentedKalmanFilter(f, h, Q, R, dt=0.02, n=2, x0=x0, P0=P0)
    u, y = np.array([0.5]), np.array([0.1])
    a.step(y, u)
    b.predict(u)
    b.update(y)
    assert a.x_hat == pytest.approx(b.x_hat) and a.P == pytest.approx(b.P)
    a.reset()
    assert a.x_hat == pytest.approx(x0) and a.P == pytest.approx(P0)


def test_ukf_drops_into_observer_feedback():
    p, f = _pend()
    A, B = p.linearize()                                        # about upright
    K = LQR(A, B, np.diag([10.0, 1.0]), [[0.5]]).K
    h = lambda x: np.array([np.sin(wrap_angle(x[0] - np.pi)), x[1]])  # noqa: E731
    ukf = UnscentedKalmanFilter(f, h, np.diag([1e-6, 1e-4]), np.diag([1e-4, 1e-4]),
                                dt=0.01, n=2, alpha=0.5,
                                x0=np.array([np.pi + 0.25, 0.0]))
    ofb = ObserverFeedback(ukf, K, x_ref=np.array([np.pi, 0.0]))

    def measure(t, x, u):
        return np.array([np.sin(wrap_angle(x[0] - np.pi)), x[1]])

    traj = simulate(p, ofb, x0=np.array([np.pi + 0.25, 0.0]), dt=0.01, t_final=6.0,
                    measurement_fn=measure)
    assert abs(wrap_angle(traj.x[-1, 0] - np.pi)) < 0.05
