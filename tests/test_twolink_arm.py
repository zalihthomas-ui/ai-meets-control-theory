"""TwoLinkArm (planar Euler-Lagrange manipulator) model checks."""

import numpy as np
import pytest

from aimct.controllers import LQR
from aimct.simulate import simulate
from aimct.systems import TwoLinkArm


def test_shapes_and_inertia_matrix_is_symmetric_positive_definite():
    arm = TwoLinkArm()
    assert arm.dynamics(0.0, np.zeros(4), np.zeros(2)).shape == (4,)
    for q in ([0.0, 0.0], [0.5, -1.2], [np.pi / 2, 2.0]):
        M = arm.M(q)
        assert M.shape == (2, 2)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvals(M) > 0.0)


def test_gravity_torque_holds_the_arm_static():
    arm = TwoLinkArm()
    q0 = np.array([0.0, 0.0])                         # stretched along +x, worst gravity
    x0 = np.array([q0[0], q0[1], 0.0, 0.0])
    held = simulate(arm, lambda y, dt: arm.G(y[:2]), x0=x0, dt=1e-3, t_final=1.0)
    assert np.max(np.abs(held.x[:, :2] - q0)) < 1e-3          # stays put
    dropped = simulate(arm, lambda y, dt: np.zeros(2), x0=x0, dt=1e-3, t_final=0.5)
    assert dropped.x[-1, 0] < q0[0] - 0.2                     # sags under gravity


def test_energy_is_conserved_with_no_gravity_no_friction():
    arm = TwoLinkArm(g=0.0, b=0.0)
    x0 = np.array([0.3, -0.5, 1.2, -0.8])
    traj = simulate(arm, lambda y, dt: np.zeros(2), x0=x0, dt=2e-4, t_final=2.0)

    def kinetic(x):
        dq = x[2:]
        return 0.5 * dq @ arm.M(x[:2]) @ dq

    e = np.array([kinetic(x) for x in traj.x])
    assert np.ptp(e) / e[0] < 5e-3                    # KE ~ constant, no drift


def test_payload_increases_inertia_and_gravity_load():
    base = TwoLinkArm(payload=0.0)
    loaded = TwoLinkArm(payload=0.5)
    q = np.array([0.4, 0.7])
    assert np.all(np.diag(loaded.M(q)) > np.diag(base.M(q)))
    assert abs(loaded.G(q)[1]) > abs(base.G(q)[1])
    # setting the attribute back to zero recovers the base model exactly
    loaded.payload = 0.0
    assert np.allclose(loaded.M(q), base.M(q))
    assert np.allclose(loaded.G(q), base.G(q))


def test_forward_kinematics_and_jacobian_agree_with_finite_difference():
    arm = TwoLinkArm()
    q = np.array([0.6, -0.9])
    J = arm.jacobian(q)
    eps = 1e-6
    J_fd = np.zeros((2, 2))
    for j in range(2):
        dq = np.zeros(2)
        dq[j] = eps
        J_fd[:, j] = (arm.forward_kinematics(q + dq)
                      - arm.forward_kinematics(q - dq)) / (2 * eps)
    assert np.allclose(J, J_fd, atol=1e-6)
    # reach check: fully extended tip is at l1 + l2 on the x axis
    assert np.allclose(arm.forward_kinematics([0.0, 0.0]), [arm.l1 + arm.l2, 0.0])


def test_computed_torque_tracks_a_joint_setpoint():
    arm = TwoLinkArm()
    q_des = np.array([np.pi / 2, -0.6])
    Kp, Kd = 120.0, 22.0

    class ComputedTorque:
        def reset(self): pass

        def update(self, x, dt):
            q, dq = np.asarray(x[:2]), np.asarray(x[2:])
            a = Kp * (q_des - q) - Kd * dq                # desired ddq
            return arm.M(q) @ a + arm.C(q, dq) @ dq + arm.G(q) + arm.b * dq

    traj = simulate(arm, ComputedTorque(), x0=np.array([0.2, 0.2, 0.0, 0.0]),
                    dt=1e-3, t_final=3.0)
    assert np.allclose(traj.x[-1, :2], q_des, atol=2e-2)
    assert not traj.diverged


def test_numeric_linearization_default_equilibrium_is_the_gravity_comp_hold():
    arm = TwoLinkArm()
    A, B = arm.linearize()                            # about [pi/2, 0, 0, 0]
    assert A.shape == (4, 4) and B.shape == (4, 2)
    # top-right block is identity (dq feeds q); LQR about the pose stabilises it
    assert np.allclose(A[:2, 2:], np.eye(2))
    K = LQR(A, B, np.diag([10.0, 10.0, 1.0, 1.0]), np.eye(2)).K
    q_eq = np.array([np.pi / 2, 0.0])

    class PoseLQR:
        def reset(self): pass

        def update(self, x, dt):
            e = np.asarray(x) - np.array([q_eq[0], q_eq[1], 0.0, 0.0])
            return arm.G(x[:2]) - K @ e

    traj = simulate(arm, PoseLQR(), x0=np.array([np.pi / 2 + 0.3, -0.3, 0.0, 0.0]),
                    dt=1e-3, t_final=4.0)
    assert np.allclose(traj.x[-1, :2], q_eq, atol=3e-2)
    assert not traj.diverged
