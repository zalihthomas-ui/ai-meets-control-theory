"""Quadrotor3D (full 6-DOF, Crazyflie 2.0) model checks."""

import numpy as np

from aimct.controllers import LQR
from aimct.controllers.state_feedback import is_controllable
from aimct.simulate import simulate
from aimct.systems import Quadrotor3D, rotation_matrix


def test_rotation_matrix_is_orthonormal():
    R = rotation_matrix(0.3, -0.2, 1.1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_hover_is_an_equilibrium():
    q = Quadrotor3D()
    assert np.allclose(q.dynamics(0.0, np.zeros(12), q.u_hover), 0.0, atol=1e-12)


def test_thrust_and_torque_signs():
    q = Quadrotor3D()
    # extra collective thrust at level attitude -> climbs (+z accel)
    assert q.dynamics(0.0, np.zeros(12), q.u_hover + [0.05, 0, 0, 0])[5] > 0
    # positive body torque about x -> positive roll-rate derivative
    assert q.dynamics(0.0, np.zeros(12), q.u_hover + [0, 1e-3, 0, 0])[9] > 0
    # zero thrust -> free fall
    assert q.dynamics(0.0, np.zeros(12), np.zeros(4))[5] == -q.g


def test_hover_linearization_shape_and_controllability():
    q = Quadrotor3D()
    A, B = q.linearize()
    assert A.shape == (12, 12) and B.shape == (12, 4)
    assert is_controllable(A, B)
    assert np.max(np.linalg.eigvals(A).real) < 1e-6           # no unstable mode


def test_lqr_recovers_hover_from_a_6dof_perturbation():
    q = Quadrotor3D()
    A, B = q.linearize()
    Q = np.diag([4, 4, 4, 1, 1, 1, 4, 4, 1, 0.1, 0.1, 0.1])
    R = np.diag([1.0, 5e3, 5e3, 5e3])
    K = LQR(A, B, Q, R).K

    class Hold:
        name = "lqr3d"

        def reset(self):
            pass

        def update(self, x, dt):
            return q.u_hover - K @ np.asarray(x)

    x0 = np.array([0.5, -0.3, 0.4, 0, 0, 0, 0.2, -0.15, 0.3, 0, 0, 0])
    tr = simulate(q, Hold(), x0=x0, dt=0.002, t_final=6.0)
    assert np.linalg.norm(tr.x[-1, :3]) < 5e-3               # back to the origin
    assert np.linalg.norm(tr.x[-1, 6:9]) < 5e-3              # level attitude
    assert not tr.diverged
