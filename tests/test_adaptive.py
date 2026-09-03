"""Adaptive control: Lyapunov solve, gain scheduling, MRAC."""

import numpy as np
import pytest

from aimct.controllers import MRAC, GainScheduledLQR, LQR, solve_lyapunov
from aimct.simulate import simulate
from aimct.systems import MassSpringDamper, Pendulum


# ----------------------------------------------------------------- Lyapunov

def test_solve_lyapunov_residual_is_zero():
    A = np.array([[0.0, 1.0], [-4.0, -1.5]])          # Hurwitz
    Q = np.diag([2.0, 1.0])
    P = solve_lyapunov(A, Q)
    assert np.allclose(A.T @ P + P @ A + Q, 0.0, atol=1e-10)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvals(P) > 0)


# ----------------------------------------------------------------- gain scheduling

def _pend_lin(theta):
    return Pendulum().linearize(np.array([theta, 0.0]))


def test_gain_scheduled_matches_pointwise_lqr_on_the_grid():
    grid = np.array([np.pi - 0.4, np.pi, np.pi + 0.4])
    Q, R = np.diag([10.0, 1.0]), np.array([[1.0]])
    gs = GainScheduledLQR(_pend_lin, grid, lambda x: x[0], Q, R)
    for s in grid:
        A, B = _pend_lin(float(s))
        assert np.allclose(gs.gain([s, 0.0]), LQR(A, B, Q, R).K, atol=1e-9)


def test_gain_scheduled_lqr_stabilises_pendulum_over_a_wide_angle_band():
    grid = np.linspace(np.pi - 0.9, np.pi + 0.9, 7)
    Q, R = np.diag([10.0, 1.0]), np.array([[0.5]])
    sys = Pendulum()
    gs = GainScheduledLQR(_pend_lin, grid, lambda x: x[0], Q, R,
                          x_ref=[np.pi, 0.0])
    for th0 in (np.pi - 0.7, np.pi + 0.6):
        traj = simulate(sys, gs, x0=[th0, 0.0], dt=0.005, t_final=6.0,
                        u_bounds=(-20.0, 20.0))
        assert abs(traj.x[-1, 0] - np.pi) < 0.02
        assert not traj.diverged


# ----------------------------------------------------------------- MRAC

def _mrac_for_msd(nominal_k=1.0, gamma=40.0):
    sys = MassSpringDamper(m=1.0, c=0.4, k=nominal_k)
    A_nom, B = sys.linearize()
    # reference model: same structure, a bit faster and well damped
    A_m = np.array([[0.0, 1.0], [-4.0, -2.8]])
    B_m = np.array([[0.0], [4.0]])
    ctrl = MRAC(A_m, B_m, B, A_nom=A_nom, gamma=gamma,
                Q=np.diag([5.0, 1.0]), u_bounds=(-50.0, 50.0))
    return ctrl


def test_mrac_tracks_the_reference_model_despite_unmodelled_stiffness():
    # true plant is 3x stiffer than the nominal the baseline was built for
    true = MassSpringDamper(m=1.0, c=0.4, k=3.0)
    ctrl = _mrac_for_msd(nominal_k=1.0, gamma=60.0)
    ctrl.r = np.array([1.0])                         # step command

    traj = simulate(true, ctrl, x0=[0.0, 0.0], dt=0.002, t_final=25.0,
                    u_bounds=(-50.0, 50.0))
    # reference model steady state for r=1: -A_m^{-1} B_m r, position component
    xm_ss = (-np.linalg.solve(ctrl.A_m, ctrl.B_m @ ctrl.r))[0]
    assert abs(traj.x[-1, 0] - xm_ss) < 0.05
    assert np.linalg.norm(traj.x[-1] - [xm_ss, 0.0]) < 0.05


def test_mrac_beats_a_fixed_baseline_when_the_plant_changes_mid_run():
    class DriftingMSD(MassSpringDamper):
        """stiffness k jumps from 1 -> 4 at t = 12 s."""
        def dynamics(self, t, x, u):
            self.k = 4.0 if t >= 12.0 else 1.0
            return super().dynamics(t, x, u)

    r = np.array([1.0])

    ctrl = _mrac_for_msd(nominal_k=1.0, gamma=80.0)
    ctrl.r = r
    tr_mrac = simulate(DriftingMSD(), ctrl, x0=[0, 0], dt=0.002, t_final=24.0,
                       u_bounds=(-50.0, 50.0))

    # fixed baseline = MRAC's baseline law with adaptation frozen off
    base = _mrac_for_msd(nominal_k=1.0)
    base.r = r
    base.Gamma = np.zeros_like(base.Gamma)
    tr_base = simulate(DriftingMSD(), base, x0=[0, 0], dt=0.002, t_final=24.0,
                       u_bounds=(-50.0, 50.0))

    xm_ss = (-np.linalg.solve(ctrl.A_m, ctrl.B_m @ r))[0]
    err_mrac = abs(tr_mrac.x[-1, 0] - xm_ss)
    err_base = abs(tr_base.x[-1, 0] - xm_ss)
    assert err_mrac < 0.03
    assert err_mrac < 0.5 * err_base


def test_mrac_theta_estimate_stays_bounded():
    ctrl = _mrac_for_msd(gamma=100.0)
    ctrl.r = np.array([0.5])
    traj = simulate(MassSpringDamper(k=2.5), ctrl, x0=[0.2, 0.0], dt=0.002,
                    t_final=20.0, u_bounds=(-50.0, 50.0))
    assert np.all(np.isfinite(ctrl.theta))
    assert np.linalg.norm(ctrl.theta) < 50.0
    assert not traj.diverged
