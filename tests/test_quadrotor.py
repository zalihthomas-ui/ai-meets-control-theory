"""PlanarQuadrotor (Crazyflie 2.0) model checks."""

import numpy as np
import pytest

from aimct.controllers import LQR
from aimct.controllers.state_feedback import is_controllable
from aimct.simulate import simulate
from aimct.systems import PlanarQuadrotor


def test_hover_is_an_equilibrium():
    q = PlanarQuadrotor()
    xdot = q.dynamics(0.0, np.zeros(6), q.u_hover)
    assert np.allclose(xdot, 0.0, atol=1e-12)


def test_free_fall_and_torque_signs():
    q = PlanarQuadrotor()
    # no thrust -> falls at -g in z
    assert q.dynamics(0.0, np.zeros(6), np.zeros(2))[4] == pytest.approx(-q.g)
    # right rotor harder than left -> positive pitch accel
    u = q.u_hover + np.array([1e-3, -1e-3])
    assert q.dynamics(0.0, np.zeros(6), u)[5] > 0.0
    # level, extra collective thrust -> climbs
    assert q.dynamics(0.0, np.zeros(6), q.u_hover * 1.1)[4] > 0.0


def test_analytic_linearization_matches_numeric():
    q = PlanarQuadrotor()
    A_num, B_num = super(PlanarQuadrotor, q).linearize(np.zeros(6), q.u_hover)
    A_an, B_an = q.linearize()
    assert np.allclose(A_num, A_an, atol=1e-4)
    assert np.allclose(B_num, B_an, atol=1e-4)


def test_hover_linearization_is_controllable_but_open_loop_unstable():
    q = PlanarQuadrotor()
    A, B = q.linearize()
    assert is_controllable(A, B)
    # rigid-body integrators + tilt coupling -> not asymptotically stable
    assert np.max(np.linalg.eigvals(A).real) >= -1e-9


def test_lqr_hover_hold_from_a_perturbation():
    q = PlanarQuadrotor()
    A, B = q.linearize()
    # Bryson-scaled cost: B is very badly scaled (pitch-torque gain ~3300 vs
    # thrust ~36), so an unscaled R yields a nonsensical LQR that saturates and
    # limit-cycles. Q_ii = 1/max(x_i)^2, R_jj = 1/max(du_j)^2.
    Q = np.diag(1.0 / np.array([0.1, 0.1, 0.2, 0.5, 0.5, 3.0]) ** 2)
    R = np.diag(1.0 / np.array([0.15, 0.15]) ** 2)
    K = LQR(A, B, Q, R).K

    class Hold:
        name = "lqr-hover"

        def reset(self):
            pass

        def update(self, x, dt):
            return q.u_hover - K @ np.asarray(x)

    traj = simulate(q, Hold(), x0=[0.3, -0.2, 0.15, 0.0, 0.0, 0.0],
                    dt=0.002, t_final=6.0,
                    u_bounds=(0.0, q.thrust_max))
    assert np.linalg.norm(traj.x[-1, :3]) < 5e-3          # back to hover pose
    assert not traj.diverged
