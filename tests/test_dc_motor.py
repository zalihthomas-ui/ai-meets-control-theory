"""DCMotor (armature-controlled) model checks."""

import numpy as np
import pytest

from aimct.controllers import LQR
from aimct.controllers.state_feedback import is_controllable
from aimct.simulate import simulate
from aimct.systems import DCMotor


def test_analytic_linearization_matches_numeric():
    m = DCMotor()
    A_num, B_num = super(DCMotor, m).linearize(np.zeros(3), np.zeros(1))
    A_an, B_an = m.linearize()
    assert np.allclose(A_num, A_an, rtol=1e-4, atol=1e-6)
    assert np.allclose(B_num, B_an, rtol=1e-4, atol=1e-6)


def test_step_voltage_spins_up_to_a_finite_speed():
    m = DCMotor(tau_load=0.0)
    traj = simulate(m, lambda y, dt: np.array([6.0]), x0=[0, 0, 0],
                    dt=1e-4, t_final=1.0)
    omega_ss = traj.x[-1, 1]
    # no-load speed ~ V / Ke (minus a little friction)
    assert 0.7 * 6.0 / m.Ke < omega_ss < 6.0 / m.Ke
    assert traj.x[-1, 2] > 0                      # small holding current vs friction


def test_open_loop_has_an_integrator_and_is_controllable():
    A, B = DCMotor().linearize()
    assert is_controllable(A, B)
    assert np.min(np.abs(np.linalg.eigvals(A))) < 1e-9      # free integrator on angle


def test_lqr_position_control_holds_near_target_under_load():
    m = DCMotor(tau_load=0.02)
    A, B = m.linearize()
    K = LQR(A, B, np.diag([200.0, 0.1, 0.01]), np.array([[1.0]])).K

    class PosCtl:
        name = "lqr-pos"

        def reset(self):
            pass

        def update(self, x, dt):
            xr = np.array([1.0, 0.0, 0.0])          # 1 rad target
            return -K @ (np.asarray(x) - xr)

    traj = simulate(m, PosCtl(), x0=[0, 0, 0], dt=1e-4, t_final=2.0,
                    u_bounds=(-m.v_max, m.v_max))
    # LQR has no integral action -> a small residual offset under constant load
    assert 0.0 < abs(traj.x[-1, 0] - 1.0) < 0.12
    assert not traj.diverged


def test_reduced_two_state_matches_full_model_at_low_frequency():
    full = DCMotor()
    red = full.reduced()
    # same steady-state speed for a step voltage
    tf = simulate(full, lambda y, dt: np.array([4.0]), x0=[0, 0, 0], dt=1e-4, t_final=1.5)
    tr = simulate(red, lambda y, dt: np.array([4.0]), x0=[0, 0], dt=1e-4, t_final=1.5)
    assert abs(tf.x[-1, 1] - tr.x[-1, 1]) / tf.x[-1, 1] < 0.02
