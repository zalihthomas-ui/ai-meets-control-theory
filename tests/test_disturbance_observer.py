"""Unit tests for Disturbance Observer (DOB) and QFilter."""

import numpy as np
import pytest

from aimct.controllers.disturbance_observer import DisturbanceObserver, QFilter
from aimct.controllers.lqr import LQR
from aimct.systems.mass_spring_damper import MassSpringDamper
from aimct.systems.quadrotor import PlanarQuadrotor


def test_qfilter_dc_gain_and_step_response():
    """Verify Q-filters of orders 1, 2, 3 have unity DC gain (Q(0) = 1.0)."""
    dt = 0.005
    n_steps = 1000  # 5.0 seconds
    cutoff = 10.0

    for order in (1, 2, 3):
        q = QFilter(cutoff_freq=cutoff, damping=1.0, order=order, dt=dt, n_channels=2)
        u_step = np.array([2.5, -1.8])

        y_out = None
        for _ in range(n_steps):
            y_out = q.filter(u_step, dt=dt)

        # Steady-state output must match step input within 1e-4
        assert y_out is not None
        np.testing.assert_allclose(y_out, u_step, rtol=1e-3, atol=1e-4)


def test_qfilter_high_frequency_attenuation():
    """Verify Q-filter attenuates high-frequency noise (omega >> omega_Q)."""
    dt = 0.001
    cutoff = 10.0  # rad/s
    q = QFilter(cutoff_freq=cutoff, damping=1.0, order=2, dt=dt, n_channels=1)

    # Fast sinusoidal input at 200 rad/s (~31.8 Hz)
    freq = 200.0
    t = np.arange(2000) * dt
    signal = np.sin(freq * t)

    filtered = np.array([q.filter([s], dt=dt)[0] for s in signal])

    # In steady state (last 500 samples), amplitude should be attenuated by > 20 dB (< 0.10)
    # Theoretical magnitude |Q(j*200)| = 100 / sqrt((100 - 40000)^2 + (2*10*200)^2) ~= 100 / 40100 ~= 0.0025
    amplitude_out = np.max(np.abs(filtered[1500:]))
    assert amplitude_out < 0.05, f"Expected heavy attenuation, got {amplitude_out}"


def test_qfilter_invalid_parameters():
    """Verify argument validation in QFilter."""
    with pytest.raises(ValueError, match="cutoff_freq must be positive"):
        QFilter(cutoff_freq=-5.0)
    with pytest.raises(ValueError, match="damping must be positive"):
        QFilter(damping=0.0)
    with pytest.raises(ValueError, match="order must be 1, 2, or 3"):
        QFilter(order=4)


def test_linear_dob_matched_disturbance_rejection():
    """Test DOB on linear MassSpringDamper with unknown constant load force."""
    msd = MassSpringDamper(m=1.0, k=4.0, c=0.5)
    A, B = msd.linearize()

    # Base LQR regulator
    Q = np.diag([20.0, 2.0])
    R = np.eye(1)
    lqr = LQR(A, B, Q, R)

    # Wrap in DOB
    dob = DisturbanceObserver(
        base_controller=lqr,
        plant=msd,
        cutoff_freq=15.0,
        filter_order=2,
    )

    # Simulation with constant disturbance d = 2.0 N
    dt = 0.005
    x = np.array([0.0, 0.0])  # starts at origin
    d_true = 2.0  # N

    for _ in range(800):  # 4.0 seconds
        u = dob.update(x, dt)
        # Plant dynamics: xdot = A*x + B*(u + d_true)
        xdot = A @ x + B.flatten() * (u[0] + d_true)
        x = x + xdot * dt

    # Without DOB, static error would be e_ss = d / (k + K_0) ~= 2.0 / (4 + 4.47) ~= 0.23 m
    # With DOB, steady-state position error is < 0.01 m
    assert abs(x[0]) < 0.01
    assert abs(dob.d_hat[0] - d_true) < 0.05


def test_quadrotor_dob_matched_vertical_wind():
    """Verify DOB cancels vertical wind on PlanarQuadrotor."""
    quad = PlanarQuadrotor()
    A, B = quad.linearize()

    Q = np.diag([10.0, 20.0, 10.0, 1.0, 2.0, 0.5])
    R = np.diag([1.0, 1.0])
    lqr = LQR(A, B, Q, R, u_ref=quad.u_hover)

    dob = DisturbanceObserver(
        base_controller=lqr,
        plant=quad,
        cutoff_freq=12.0,
    )

    dt = 0.002
    x = np.zeros(6)
    f_wind_z = -0.04  # -0.04 N downward gust

    for _ in range(1500):  # 3.0 seconds
        u = dob.update(x, dt)
        xdot = quad.dynamics(0.0, x, u)
        # Inject vertical wind
        xdot[4] += f_wind_z / quad.m
        x = x + xdot * dt

    # DOB must estimate vertical acceleration discrepancy within 5%
    # expected d_z_hat = f_wind_z / m = -0.04 / 0.028 ~= -1.428 m/s^2
    expected_dz = f_wind_z / quad.m
    np.testing.assert_allclose(dob.d_hat[1], expected_dz, rtol=0.08)
    # Altitude drift should be < 1.0 cm
    assert abs(x[1]) < 0.01


def test_quadrotor_dob_unmatched_horizontal_wind():
    """Verify DOB eliminates horizontal drift under unmatched horizontal wind."""
    quad = PlanarQuadrotor()
    A, B = quad.linearize()

    Q = np.diag([15.0, 20.0, 10.0, 2.0, 2.0, 0.5])
    R = np.diag([1.0, 1.0])
    lqr = LQR(A, B, Q, R, u_ref=quad.u_hover)

    dob = DisturbanceObserver(
        base_controller=lqr,
        plant=quad,
        cutoff_freq=10.0,
    )

    dt = 0.002
    x = np.zeros(6)
    f_wind_x = 0.03  # 0.03 N horizontal wind (+x direction)

    for _ in range(2500):  # 5.0 seconds
        u = dob.update(x, dt)
        xdot = quad.dynamics(0.0, x, u)
        # Inject horizontal wind
        xdot[3] += f_wind_x / quad.m
        x = x + xdot * dt

    # DOB must estimate horizontal acceleration discrepancy within 8%
    # expected d_x_hat = f_wind_x / m = 0.03 / 0.028 ~= 1.071 m/s^2
    expected_dx = f_wind_x / quad.m
    np.testing.assert_allclose(dob.d_hat[0], expected_dx, rtol=0.08)

    # Drone must tilt nose into wind (theta > 0 in our sign convention where xdd = -thrust*sin(th)/m)
    # sin(theta_trim) = f_wind_x / (m*g) ~= 0.03 / (0.028*9.81) ~= 0.109 rad (~6.25 deg)
    assert x[2] > 0.05
    # Steady state position error must be < 2.0 cm
    assert abs(x[0]) < 0.02
