"""DifferentialDriveRobot (unicycle + first-order actuator lag) model checks."""

import numpy as np
import pytest

from aimct.controllers import LQR
from aimct.controllers.state_feedback import is_controllable
from aimct.simulate import simulate
from aimct.systems import DifferentialDriveRobot


def test_shapes_and_analytic_linearization_matches_numeric():
    r = DifferentialDriveRobot()
    xd = r.dynamics(0.0, np.zeros(5), np.array([0.1, 0.2]))
    assert xd.shape == (5,)

    x_eq = np.array([0.0, 0.0, 0.3, r.v_ref, 0.0])
    u_eq = np.array([r.v_ref, 0.0])
    A_an, B_an = r.linearize(x_eq, u_eq)
    A_num, B_num = super(DifferentialDriveRobot, r).linearize(x_eq, u_eq)
    assert A_an.shape == (5, 5) and B_an.shape == (5, 2)
    assert np.allclose(A_an, A_num, rtol=1e-4, atol=1e-6)
    assert np.allclose(B_an, B_num, rtol=1e-4, atol=1e-6)


def test_straight_line_command_drives_forward_along_x():
    r = DifferentialDriveRobot()
    v0 = 0.15
    traj = simulate(r, lambda y, dt: np.array([v0, 0.0]),
                    x0=np.zeros(5), dt=1e-3, t_final=4.0)
    x, y, theta, v, omega = traj.x[-1]
    assert v == pytest.approx(v0, abs=1e-3)          # speed loop settled
    assert abs(theta) < 1e-6 and abs(omega) < 1e-6
    assert abs(y) < 1e-6
    # travelled ~ v0 * (t - a bit of lag transient)
    assert v0 * 4.0 - 0.05 < x < v0 * 4.0


def test_constant_turn_traces_a_circle_of_the_expected_radius():
    r = DifferentialDriveRobot()
    v0, w0 = 0.15, 0.6
    traj = simulate(r, lambda y, dt: np.array([v0, w0]),
                    x0=np.zeros(5), dt=1e-3, t_final=8.0)
    assert traj.x[-1, 3] == pytest.approx(v0, abs=1e-3)   # speed loop settled
    assert traj.x[-1, 4] == pytest.approx(w0, abs=1e-3)   # yaw-rate loop settled

    # algebraic circle fit on the post-transient arc: x^2+y^2 = a*x + b*y + c
    xy = traj.x[len(traj.x) // 2:, :2]
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    a, b, c = np.linalg.lstsq(A, xy[:, 0] ** 2 + xy[:, 1] ** 2, rcond=None)[0]
    radius = np.sqrt(c + (a / 2) ** 2 + (b / 2) ** 2)
    assert radius == pytest.approx(v0 / w0, rel=0.05)     # R = v / omega


def test_actuator_lag_has_the_declared_time_constant():
    r = DifferentialDriveRobot(tau_v=0.05)
    v_cmd = 0.2
    traj = simulate(r, lambda y, dt: np.array([v_cmd, 0.0]),
                    x0=np.zeros(5), dt=1e-4, t_final=r.tau_v)
    # one time constant -> ~63% of the step
    assert 0.55 * v_cmd < traj.x[-1, 3] < 0.70 * v_cmd


def test_command_limits_are_clipped_inside_dynamics():
    r = DifferentialDriveRobot()
    xd_big = r.dynamics(0.0, np.zeros(5), np.array([100.0, 100.0]))
    xd_lim = r.dynamics(0.0, np.zeros(5), np.array([r.v_max, r.omega_max]))
    assert np.allclose(xd_big, xd_lim)


def test_wheel_kinematics_round_trip():
    r = DifferentialDriveRobot()
    for v, w in [(0.1, 0.0), (0.0, 1.0), (0.18, -0.7)]:
        wl, wr = r.wheel_speeds(v, w)
        v2, w2 = r.body_twist(wl, wr)
        assert v2 == pytest.approx(v) and w2 == pytest.approx(w)


def test_lateral_path_error_is_controllable_and_lqr_regulates_it():
    r = DifferentialDriveRobot()
    A, B = r.linearize()                              # straight path at v_ref
    # drop the x integrator (uncontrollable along-track pose) -> [y, theta, v, omega]
    idx = np.array([1, 2, 3, 4])
    Ar, Br = A[np.ix_(idx, idx)], B[idx, :]
    assert is_controllable(Ar, Br)

    K = LQR(Ar, Br, np.diag([20.0, 5.0, 1.0, 1.0]), np.diag([1.0, 1.0])).K

    class PathLQR:
        def reset(self): pass

        def update(self, x, dt):
            e = np.array([x[1], x[2], x[3] - r.v_ref, x[4]])   # lateral error state
            u = np.array([r.v_ref, 0.0]) - K @ e
            return u

    traj = simulate(r, PathLQR(), x0=np.array([0.0, 0.20, 0.0, r.v_ref, 0.0]),
                    dt=1e-3, t_final=12.0)
    assert abs(traj.x[-1, 1]) < 0.02                  # pulled back onto y = 0
    assert abs(traj.x[-1, 1]) < abs(traj.x[0, 1]) / 5 # and it is a big reduction
    assert not traj.diverged
