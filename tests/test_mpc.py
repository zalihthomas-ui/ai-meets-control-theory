"""Tests for :class:`aimct.controllers.LinearMPC` - condensed receding-horizon
linear MPC with the from-scratch QP.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from aimct.benchmarks import compare
from aimct.controllers import LQR, LinearMPC, dare
from aimct.systems import LinearSystem, MassSpringDamper
from aimct.simulate import simulate

A_DI = np.array([[0.0, 1.0], [0.0, 0.0]])
B_DI = np.array([[0.0], [1.0]])


@pytest.fixture
def msd():
    m = MassSpringDamper(m=1.0, c=0.4, k=1.0)
    A, B = m.linearize()
    return m, A, B


# ------------------------------------------------------------------------ DARE

@pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
def test_dare_matches_scipy(dt):
    from aimct.controllers.mpc import _discretize

    Ad, Bd = _discretize(A_DI, B_DI, dt)
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    P = dare(Ad, Bd, Q, R)
    assert P == pytest.approx(solve_discrete_are(Ad, Bd, Q, R), abs=1e-6)


# --------------------------------------------------------- reduces to LQR

@pytest.mark.parametrize("N", [1, 5, 20, 40])
def test_unconstrained_first_move_is_the_discrete_lqr_move(msd, N):
    _, A, B = msd
    dt = 0.02
    mpc = LinearMPC(A, B, Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]), N=N)
    Kd = mpc.discrete_lqr_gain(dt)
    for x in (np.array([0.7, -0.3]), np.array([-1.2, 0.5]), np.array([0.0, 2.0])):
        mpc.reset()
        assert mpc.update(x, dt) == pytest.approx(float(-(Kd @ x)[0]), abs=1e-6)


def test_unconstrained_closed_loop_matches_discrete_lqr(msd):
    plant, A, B = msd
    dt = 0.02
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    mpc = LinearMPC(A, B, Q=Q, R=R, N=30)
    Kd = mpc.discrete_lqr_gain(dt)

    class _DLQR:
        def update(self, x, dt):
            return float(-(Kd @ np.atleast_1d(x))[0])
        def reset(self):
            pass

    t_mpc = simulate(plant, mpc, x0=np.array([1.0, 0.0]), dt=dt, t_final=6.0)
    t_lqr = simulate(plant, _DLQR(), x0=np.array([1.0, 0.0]), dt=dt, t_final=6.0)
    assert np.max(np.abs(t_mpc.x - t_lqr.x)) < 1e-7


def test_qf_defaults_to_the_dare_solution(msd):
    _, A, B = msd
    from aimct.controllers.mpc import _discretize
    dt = 0.02
    mpc = LinearMPC(A, B, Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]), N=10)
    built = mpc._build(dt)
    Ad, Bd = _discretize(A, B, dt)
    assert built["Qf"] == pytest.approx(dare(Ad, Bd, mpc.Q, mpc.R), abs=1e-9)


# -------------------------------------------------------------- constraints

def test_respects_the_input_box(msd):
    plant, A, B = msd
    mpc = LinearMPC(A, B, Q=np.diag([50.0, 1.0]), R=np.array([[1e-3]]), N=20,
                    u_bounds=(-2.0, 2.0))
    traj = simulate(plant, mpc, x0=np.array([1.5, 0.0]), dt=0.02, t_final=6.0)
    assert np.max(np.abs(traj.u)) <= 2.0 + 1e-6
    assert np.linalg.norm(traj.x[-1]) < 1e-2               # still regulates


def test_state_box_holds_where_the_unconstrained_plan_would_violate():
    sys = LinearSystem(A_DI, B_DI)
    common = dict(Q=np.diag([1.0, 1.0]), R=np.array([[0.5]]), N=25, u_bounds=(-5.0, 5.0))
    x0 = np.array([0.0, 2.0])                               # moving fast toward +x1

    free = simulate(sys, LinearMPC(A_DI, B_DI, **common), x0=x0, dt=0.04, t_final=10.0)
    capped = simulate(sys, LinearMPC(A_DI, B_DI, **common,
                                     x_bounds=([None, None], [0.6, None])),
                      x0=x0, dt=0.04, t_final=10.0)

    assert free.x[:, 0].max() > 0.62                        # unconstrained overshoots
    assert capped.x[:, 0].max() <= 0.6 + 5e-3               # softened box essentially held
    assert np.linalg.norm(capped.x[-1]) < 5e-3             # and still reaches the origin


def test_recovers_input_constrained_double_integrator_to_origin():
    sys = LinearSystem(A_DI, B_DI)
    mpc = LinearMPC(A_DI, B_DI, Q=np.diag([10.0, 1.0]), R=np.array([[0.1]]),
                    N=30, u_bounds=(-1.0, 1.0))
    traj = simulate(sys, mpc, x0=np.array([2.0, 0.0]), dt=0.05, t_final=20.0)
    assert np.max(np.abs(traj.u)) <= 1.0 + 1e-6
    assert np.linalg.norm(traj.x[-1]) < 1e-3


# -------------------------------------------------------------- mechanics

def test_reference_tracking_unconstrained(msd):
    plant, A, B = msd
    mpc = LinearMPC(A, B, Q=np.diag([100.0, 1.0]), R=np.array([[0.01]]), N=25,
                    x_ref=np.array([1.0, 0.0]), u_ref=np.array([1.0]))  # k*r feed-forward
    traj = simulate(plant, mpc, x0=np.zeros(2), dt=0.02, t_final=10.0)
    assert traj.x[-1] == pytest.approx([1.0, 0.0], abs=2e-3)


def test_horizon_plan_and_predicted_states_exposed(msd):
    _, A, B = msd
    mpc = LinearMPC(A, B, Q=np.eye(2), R=np.array([[0.1]]), N=12)
    mpc.update(np.array([0.5, 0.0]), 0.05)
    assert mpc.horizon_plan.shape == (12, 1)
    assert mpc.predicted_states.shape == (12, 2)
    # first predicted state is one ZOH step from x0 under u0
    Ad, Bd = mpc._build(0.05)["Ad"], mpc._build(0.05)["Bd"]
    x1 = Ad @ np.array([0.5, 0.0]) + Bd @ mpc.horizon_plan[0]
    assert mpc.predicted_states[0] == pytest.approx(x1, abs=1e-9)


def test_discretization_is_cached_per_dt(msd):
    _, A, B = msd
    mpc = LinearMPC(A, B, Q=np.eye(2), R=np.array([[0.1]]), N=8)
    mpc.update(np.array([1.0, 0.0]), 0.02)
    mpc.update(np.array([1.0, 0.0]), 0.02)
    assert set(mpc._cache) == {round(0.02, 12)}
    mpc.update(np.array([1.0, 0.0]), 0.05)
    assert set(mpc._cache) == {round(0.02, 12), round(0.05, 12)}


def test_invalid_horizon_raises(msd):
    _, A, B = msd
    with pytest.raises(ValueError, match="N must be"):
        LinearMPC(A, B, Q=np.eye(2), R=np.array([[0.1]]), N=0)


def test_mpc_and_lqr_agree_through_the_harness(msd):
    plant, A, B = msd
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    res = compare(
        plant,
        {"LQR": LQR(A, B, Q, R), "MPC (N=25)": LinearMPC(A, B, Q=Q, R=R, N=25)},
        x0=np.array([1.0, 0.0]), dt=0.02, t_final=8.0, reference=0.0, output_index=0,
    )
    assert res.status["LQR"] == "Stable"
    assert res.status["MPC (N=25)"] == "Stable"
    # unconstrained MPC ~ discrete LQR ~ continuous LQR at this dt
    assert res.metrics["MPC (N=25)"]["rmse"] == pytest.approx(
        res.metrics["LQR"]["rmse"], rel=0.05)
