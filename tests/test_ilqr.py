"""iLQR trajectory optimiser + receding-horizon (real-time-iteration) NMPC."""

import numpy as np
import pytest

from aimct.controllers import ILQR, iLQR
from aimct.controllers.mpc import _discretize, dare
from aimct.simulate import simulate
from aimct.systems import CartPole, MassSpringDamper, PlanarQuadrotor


def test_ilqr_first_move_matches_discrete_lqr_on_a_linear_system():
    # With a long horizon and Qf = DARE solution, the first iLQR move must be
    # the exact infinite-horizon discrete-LQR move (iLQR == LQR on a linear
    # system with a quadratic cost).
    sys = MassSpringDamper()
    dt = 0.05
    A, B = sys.linearize(np.zeros(2), np.zeros(1))
    Ad, Bd = _discretize(A, B, dt)
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.5]])
    P = dare(Ad, Bd, Q, R)
    Kd = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    x0 = np.array([1.0, -0.3])
    opt = iLQR.from_system(sys, dt, horizon=120, Q=Q, R=R, Qf=P,
                           max_iter=50, tol=1e-12)
    res = opt.solve(x0)

    assert res.converged
    assert np.allclose(res.U[0], -Kd @ x0, atol=1e-5)


def test_ilqr_backward_pass_is_a_descent_direction():
    sys = MassSpringDamper()
    opt = iLQR.from_system(sys, 0.05, horizon=60, Q=np.diag([10.0, 1.0]),
                           R=np.array([[0.1]]))
    res = opt.solve([1.0, 0.0])
    # cost is monotone non-increasing across accepted iterations
    h = np.array(res.cost_history)
    assert np.all(np.diff(h) <= 1e-6)
    assert h[-1] < h[0]
    assert res.converged


def test_ilqr_swings_up_the_cart_pole():
    cp = CartPole()
    dt = 0.02
    opt = iLQR.from_system(
        cp, dt, horizon=150,
        Q=np.diag([1.0, 1.0, 5.0, 0.5]),
        Qf=np.diag([50.0, 50.0, 200.0, 50.0]),
        R=np.array([[0.05]]),
        x_ref=np.zeros(4), u_bounds=(-20.0, 20.0), max_iter=200,
    )
    res = opt.solve(np.array([0.0, 0.0, np.pi, 0.0]))
    # reached the upright: |theta| and rates small at the horizon end
    assert abs(res.X[-1, 2]) < 0.15
    assert abs(res.X[-1, 3]) < 0.5
    assert res.cost < 0.5 * res.cost_history[0]
    assert np.all(np.abs(res.U) <= 20.0 + 1e-9)   # box respected by clamping


def test_ilqr_respects_the_input_box():
    cp = CartPole()
    opt = iLQR.from_system(cp, 0.02, horizon=80,
                           Q=np.eye(4), R=np.array([[1e-4]]),
                           u_bounds=(-3.0, 3.0), max_iter=60)
    res = opt.solve(np.array([1.5, 0.0, 0.4, 0.0]))
    assert res.U.min() >= -3.0 - 1e-9
    assert res.U.max() <= 3.0 + 1e-9


def test_receding_horizon_ilqr_regulates_through_simulate():
    cp = CartPole()
    dt = 0.02
    ctrl = ILQR.from_system(cp, dt, horizon=40,
                            Q=np.diag([1.0, 1.0, 10.0, 1.0]),
                            R=np.array([[0.1]]), u_bounds=(-15.0, 15.0),
                            warm_iters=100, rti_iters=2)
    tr = simulate(cp, ctrl, x0=np.array([0.3, 0.0, 0.25, 0.0]),
                  dt=dt, t_final=3.0)
    assert not tr.diverged
    assert abs(tr.x[-1, 2]) < 0.05           # pole held upright
    assert abs(tr.x[-1, 3]) < 0.3
    ctrl.reset()
    assert ctrl.U is None and ctrl._t == 0.0


def test_ilqr_controller_tracks_a_time_varying_reference():
    # moving set-point given as a callable of absolute time
    sys = MassSpringDamper()
    dt = 0.02

    def x_ref(t):
        return np.array([0.5 * np.sin(0.8 * t), 0.4 * np.cos(0.8 * t)])

    ctrl = ILQR.from_system(sys, dt, horizon=30, Q=np.diag([50.0, 1.0]),
                            R=np.array([[0.01]]), x_ref=x_ref,
                            warm_iters=60, rti_iters=1)
    tr = simulate(sys, ctrl, x0=np.array([0.0, 0.0]), dt=dt, t_final=8.0)
    assert not tr.diverged
    ref = np.array([x_ref(t)[0] for t in tr.t])
    err = np.sqrt(np.mean((tr.x[:, 0] - ref) ** 2))
    assert err < 0.05                         # tracks the sine to < 5 cm RMS


def test_ilqr_finite_difference_jacobians_are_step_size_independent():
    # _linearize does one central difference on step() with eps=1e-6; a second,
    # independent central difference at a coarser eps must land on the same
    # Jacobian (rules out a broken perturbation / assembly).
    q = PlanarQuadrotor()
    dt = 0.02
    opt = iLQR.from_system(q, dt, horizon=3, Q=np.eye(6), R=np.eye(2))
    x = np.array([0.1, 1.0, 0.05, -0.2, 0.1, 0.3])
    u = np.array([q.m * q.g / 2 + 0.01, q.m * q.g / 2 - 0.02])
    step = opt.step

    def cdiff(nx, nu, e):
        Fx = np.zeros((6, nx.size))
        for j in range(nx.size):
            d = np.zeros(nx.size); d[j] = e
            Fx[:, j] = (step(nx + d, nu) - step(nx - d, nu)) / (2 * e)
        Fu = np.zeros((6, nu.size))
        for j in range(nu.size):
            d = np.zeros(nu.size); d[j] = e
            Fu[:, j] = (step(nx, nu + d) - step(nx, nu - d)) / (2 * e)
        return Fx, Fu

    fx, fu = opt._linearize(np.tile(x, (4, 1)), np.tile(u, (3, 1)))
    fx_ref, fu_ref = cdiff(x, u, 1e-4)
    assert np.allclose(fx[0], fx_ref, rtol=1e-4, atol=1e-7)
    assert np.allclose(fu[0], fu_ref, rtol=1e-4, atol=1e-7)
    assert fx.shape == (3, 6, 6) and fu.shape == (3, 6, 2)
    # leading term of the RK4 state Jacobian is I + dt * A_c
    A, _ = q.linearize(x, u)
    assert np.allclose(np.diag(fx[0]), np.diag(np.eye(6) + dt * A), atol=1e-4)
