"""Tests for :class:`aimct.estimation.ExtendedKalmanFilter`."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.estimation import (
    DiscreteKalmanFilter,
    ExtendedKalmanFilter,
    finite_diff_jacobian,
)
from aimct.controllers import LQR, ObserverFeedback, wrap_angle
from aimct.systems import MassSpringDamper, Pendulum
from aimct.simulate import rk4_step, simulate


# --------------------------------------------------------------- fd Jacobian

def test_finite_diff_jacobian_matches_analytic():
    def fn(x):
        return np.array([x[0] ** 2 + x[1], np.sin(x[0]) * x[2], x[1] - x[2]])
    x = np.array([0.7, -1.3, 0.4])
    J_analytic = np.array([
        [2 * x[0], 1.0, 0.0],
        [np.cos(x[0]) * x[2], 0.0, np.sin(x[0])],
        [0.0, 1.0, -1.0],
    ])
    assert finite_diff_jacobian(fn, x) == pytest.approx(J_analytic, abs=1e-7)


# -------------------------------------------------- reduces to the linear KF

def test_matches_discrete_kalman_filter_on_a_linear_model():
    msd = MassSpringDamper()
    A, B = msd.linearize()
    C = np.array([[1.0, 0.0]])
    W, V, dt = np.diag([1e-4, 1e-3]), np.array([[1e-3]]), 0.02

    dkf = DiscreteKalmanFilter(A, B, C, W, V, dt, x_hat0=np.zeros(2))
    F, Gu, Qd, Rd = dkf.F, dkf.Gu, dkf.Qd, dkf.Rd
    # feed the EKF the *exact* same discrete transition -> they must agree
    ekf = ExtendedKalmanFilter(
        lambda x, u: F @ x + Gu @ u, lambda x: C @ x, Qd, Rd,
        dt=dt, n=2, discrete=True, x0=np.zeros(2),
    )

    rng = np.random.default_rng(0)
    x = np.array([0.6, -0.2])
    for k in range(120):
        u = np.array([np.sin(0.1 * k)])
        x = F @ x + Gu @ u
        y = C @ x + rng.normal(0.0, np.sqrt(Rd[0, 0]), size=1)
        xd = dkf.step(y, u)
        xe = ekf.step(y, u)
        assert xe == pytest.approx(xd, abs=1e-7)
    assert ekf.P == pytest.approx(dkf.P, abs=1e-7)


# --------------------------------------------- analytic vs finite-diff agree

def test_analytic_and_finite_diff_jacobians_give_the_same_estimate():
    p = Pendulum()
    dt = 0.02

    def f(x, u):
        return p.dynamics(0.0, x, u)

    def h(x):
        return np.array([np.sin(x[0]), x[1]])

    def H_jac(x):
        return np.array([[np.cos(x[0]), 0.0], [0.0, 1.0]])

    def F_jac(x, u):                       # Jacobian of one RK4 step
        return finite_diff_jacobian(
            lambda xx: rk4_step(lambda t, a, b: p.dynamics(0, a, b), 0.0, xx, u, dt),
            x, 1e-7,
        )

    Q, R = np.diag([1e-5, 1e-4]), np.diag([1e-3, 1e-3])
    common = dict(dt=dt, n=2, x0=np.array([2.6, 0.0]), P0=np.eye(2))
    ekf_fd = ExtendedKalmanFilter(f, h, Q, R, **common)
    ekf_an = ExtendedKalmanFilter(f, h, Q, R, F_jac=F_jac, H_jac=H_jac, **common)

    rng = np.random.default_rng(1)
    xt = np.array([3.0, 0.4])
    for k in range(200):
        u = np.array([0.3 * np.cos(0.05 * k)])
        xt = rk4_step(lambda t, a, b: p.dynamics(0, a, b), 0.0, xt, u, dt)
        y = h(xt) + rng.normal(0.0, np.sqrt(np.diag(R)))
        a = ekf_fd.step(y, u)
        b = ekf_an.step(y, u)
        assert a == pytest.approx(b, abs=1e-6)


# ---------------------------------------------- converges on a nonlinear model

def test_ekf_converges_and_keeps_a_valid_covariance_on_the_pendulum():
    p = Pendulum()
    dt = 0.02

    def f(x, u):
        return p.dynamics(0.0, x, u)

    def h(x):
        return np.array([np.sin(x[0]), x[1]])       # nonlinear angle measurement

    Q, R = np.diag([1e-6, 1e-4]), np.diag([2e-3, 2e-3])
    ekf = ExtendedKalmanFilter(f, h, Q, R, dt=dt, n=2,
                               x0=np.array([2.4, 0.0]), P0=0.5 * np.eye(2))

    rng = np.random.default_rng(2)
    xt = np.array([3.0, 0.6])
    err0 = np.linalg.norm(ekf.x_hat - xt)             # before any measurement
    errs = []
    for k in range(500):
        u = np.array([0.4 * np.sin(0.04 * k)])
        xt = rk4_step(lambda t, a, b: p.dynamics(0, a, b), 0.0, xt, u, dt)
        y = h(xt) + rng.normal(0.0, np.sqrt(np.diag(R)))
        xh = ekf.step(y, u)
        errs.append(np.linalg.norm(xh - xt))
        assert np.allclose(ekf.P, ekf.P.T)
        assert np.all(np.linalg.eigvalsh(ekf.P) > -1e-9)

    assert err0 > 0.4                                 # started off
    assert np.mean(errs[-50:]) < 0.05                 # tracked the true state


def test_residual_hook_controls_the_innovation():
    p = Pendulum()
    f = lambda x, u: p.dynamics(0.0, x, u)            # noqa: E731
    h = lambda x: np.array([np.sin(x[0]), x[1]])      # noqa: E731
    Q, R = np.diag([1e-5, 1e-4]), np.diag([1e-3, 1e-3])

    seen = []

    def resid(y, y_pred):
        seen.append((np.array(y, float), np.array(y_pred, float)))
        return np.asarray(y, float) - np.asarray(y_pred, float)

    ekf = ExtendedKalmanFilter(f, h, Q, R, dt=0.02, n=2, residual=resid,
                               x0=np.array([1.0, 0.2]))
    ekf.update(np.array([0.5, 0.1]))
    assert len(seen) == 1
    y, y_pred = seen[0]
    assert y == pytest.approx([0.5, 0.1])
    assert y_pred == pytest.approx(h(np.array([1.0, 0.2])))    # prediction at the prior mean

    # a residual that reports zero innovation must leave the state untouched
    ekf0 = ExtendedKalmanFilter(f, h, Q, R, dt=0.02, n=2,
                                residual=lambda y, yp: np.zeros(2),
                                x0=np.array([1.0, 0.2]))
    before = ekf0.x_hat.copy()
    ekf0.update(np.array([9.0, 9.0]))
    assert ekf0.x_hat == pytest.approx(before)


def test_angle_wrapping_residual_beats_the_naive_one_across_the_branch_cut():
    p = Pendulum()
    dt = 0.02
    f = lambda x, u: p.dynamics(0.0, x, u)            # noqa: E731
    h = lambda x: np.array([wrap_angle(x[0])])        # a wrapped-angle sensor  # noqa: E731

    def wrapped_residual(y, y_pred):
        return np.array([wrap_angle(y[0] - y_pred[0])])

    # prior mean sits just below +pi; the true angle is just above it, so the
    # sensor reads ~ -pi and a naive (y - h) innovation is ~ -2pi.
    Q = np.diag([1e-6, 1e-4])
    R = np.array([[4e-4]])
    kw = dict(dt=dt, n=2, x0=np.array([3.10, 0.0]), P0=np.diag([1e-3, 1e-3]))
    plain = ExtendedKalmanFilter(f, h, Q, R, **kw)
    wrapped = ExtendedKalmanFilter(f, h, Q, R, residual=wrapped_residual, **kw)

    xt = np.array([3.16, 0.0])                        # just past +pi
    for _ in range(60):
        xt = rk4_step(lambda t, a, b: p.dynamics(0, a, b), 0.0, xt, np.array([0.0]), dt)
        y = np.array([wrap_angle(xt[0])])
        plain.step(y, np.array([0.0]))
        wrapped.step(y, np.array([0.0]))
    e_wrap = abs(wrap_angle(wrapped.x_hat[0] - xt[0]))
    e_plain = abs(wrap_angle(plain.x_hat[0] - xt[0]))
    assert e_wrap < 0.05 and e_wrap < e_plain


# ------------------------------------------------- step / reset / drop-in

def test_step_equals_predict_then_update_and_reset_restores():
    p = Pendulum()
    f = lambda x, u: p.dynamics(0.0, x, u)            # noqa: E731
    h = lambda x: np.array([np.sin(x[0])])            # noqa: E731
    Q, R = np.diag([1e-5, 1e-4]), np.array([[1e-3]])
    x0, P0 = np.array([1.0, 0.2]), 0.3 * np.eye(2)

    a = ExtendedKalmanFilter(f, h, Q, R, dt=0.02, n=2, x0=x0, P0=P0)
    b = ExtendedKalmanFilter(f, h, Q, R, dt=0.02, n=2, x0=x0, P0=P0)
    u, y = np.array([0.5]), np.array([0.1])
    a.step(y, u)
    b.predict(u)
    b.update(y)
    assert a.x_hat == pytest.approx(b.x_hat) and a.P == pytest.approx(b.P)

    a.reset()
    assert a.x_hat == pytest.approx(x0) and a.P == pytest.approx(P0)


def test_ekf_drops_into_observer_feedback():
    p = Pendulum()
    A, B = p.linearize()                              # about upright
    K = LQR(A, B, np.diag([10.0, 1.0]), [[0.5]]).K

    def f(x, u):
        return p.dynamics(0.0, x, u)

    def h(x):
        return np.array([np.sin(wrap_angle(x[0] - np.pi)), x[1]])  # ~angle-from-upright + rate

    ekf = ExtendedKalmanFilter(
        f, h, np.diag([1e-6, 1e-4]), np.diag([1e-4, 1e-4]),
        dt=0.01, n=2, x0=np.array([np.pi + 0.25, 0.0]),
    )
    ofb = ObserverFeedback(ekf, K, x_ref=np.array([np.pi, 0.0]))

    def measure(t, x, u):
        return np.array([np.sin(wrap_angle(x[0] - np.pi)), x[1]])

    traj = simulate(p, ofb, x0=np.array([np.pi + 0.25, 0.0]), dt=0.01, t_final=6.0,
                    measurement_fn=measure)
    assert abs(wrap_angle(traj.x[-1, 0] - np.pi)) < 0.05     # balanced via EKF output feedback
