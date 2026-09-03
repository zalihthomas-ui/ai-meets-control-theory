"""Tests for :class:`aimct.controllers.ObserverFeedback` - a state estimator in
the loop with a static gain (Luenberger compensator / LQG).
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.benchmarks import compare
from aimct.controllers import LQR, ObserverFeedback, StateFeedback
from aimct.estimation import DiscreteKalmanFilter, KalmanFilter, LuenbergerObserver
from aimct.systems import CartPole, LinearSystem, MassSpringDamper
from aimct.simulate import simulate


@pytest.fixture
def msd():
    m = MassSpringDamper(m=1.0, c=0.4, k=1.0)
    A, B = m.linearize()
    return m, A, B


# ----------------------------------------------------------- separation principle

@pytest.mark.parametrize("obs_poles", [[-8.0, -9.0], [-5.0 + 2j, -5.0 - 2j], [-12.0, -3.0]])
def test_separation_principle_augmented_poles(msd, obs_poles):
    _, A, B = msd
    C = np.array([[1.0, 0.0]])
    K = LQR(A, B, np.diag([10.0, 1.0]), [[0.1]]).K
    ofb = ObserverFeedback.luenberger(A, B, C, K, observer_poles=obs_poles)
    L = ofb.observer.L

    aug = np.block([[A - B @ K, B @ K], [np.zeros((2, 2)), A - L @ C]])
    got = np.sort_complex(np.linalg.eigvals(aug))
    want = np.sort_complex(np.concatenate([
        np.linalg.eigvals(A - B @ K), np.linalg.eigvals(A - L @ C)
    ]))
    assert got == pytest.approx(want, abs=1e-8)


# ------------------------------------------------- reduces to plain state feedback

def test_perfect_estimate_matches_state_feedback(msd):
    plant, A, B = msd
    C = np.eye(2)                                  # full state measured, no noise
    K = LQR(A, B, np.diag([10.0, 1.0]), [[0.1]]).K
    x0 = np.array([0.4, -0.2])

    obs = LuenbergerObserver(A, B, C, poles=[-20.0, -25.0], x_hat0=x0)
    ofb = ObserverFeedback(obs, K)
    sf = StateFeedback(K)

    t_of = simulate(plant, ofb, x0=x0, dt=1e-3, t_final=6.0)
    t_sf = simulate(plant, sf, x0=x0, dt=1e-3, t_final=6.0)
    # estimate starts exact and the model is exact; the two laws track closely,
    # differing only by the observer's O(dt) one-step input-delay coupling.
    assert np.max(np.abs(t_of.x - t_sf.x)) < 5e-3
    assert np.max(np.abs(t_of.u - t_sf.u)) < 5e-2


# ----------------------------------------------------------------- LQG regulation

def test_lqg_regulates_cartpole_from_noisy_dual_encoder():
    cp = CartPole()
    A, B = cp.linearize()
    C = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])   # measure cart x and pole angle
    Q, R = np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.1]])
    W = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
    V = np.diag([1e-5, 1e-5])                         # encoder noise covariance

    lqg = ObserverFeedback.lqg(A, B, C, Q, R, W, V)
    assert np.all(lqg.observer.error_poles.real < 0)

    rng = np.random.default_rng(0)
    noise = np.sqrt(np.diag(V))

    def measure(t, x, u):
        return C @ x + rng.normal(scale=noise)

    x0 = np.array([0.0, 0.0, 0.15, 0.0])
    traj = simulate(cp, lqg, x0=x0, dt=2e-3, t_final=8.0, measurement_fn=measure)

    assert np.abs(traj.x[-1, 2]) < 2e-2              # pole upright
    assert np.linalg.norm(traj.x[-1]) < 0.1
    assert np.linalg.norm(lqg.estimation_error(traj.x[-1])) < 5e-2


# --------------------------------------------------------------------- mechanics

def test_first_step_uses_zero_previous_input(msd):
    _, A, B = msd
    C = np.array([[1.0, 0.0]])
    obs = LuenbergerObserver(A, B, C, poles=[-10.0, -11.0])
    spy = {}
    real_update = obs.update

    def traced(y, u, dt):
        spy.setdefault("first_u", np.array(u, dtype=float))
        return real_update(y, u, dt)

    obs.update = traced
    ofb = ObserverFeedback(obs, np.array([[1.0, 2.0]]))
    ofb.update(np.array([0.5]), 0.01)
    assert spy["first_u"] == pytest.approx([0.0])


def test_reset_restores_observer_and_uprev(msd):
    _, A, B = msd
    C = np.array([[1.0, 0.0]])
    ofb = ObserverFeedback(
        LuenbergerObserver(A, B, C, poles=[-10.0, -11.0]),
        LQR(A, B, np.eye(2), [[1.0]]).K,
    )
    for _ in range(20):
        ofb.update(np.array([0.7]), 0.01)
    assert np.linalg.norm(ofb.x_hat) > 0
    assert np.linalg.norm(ofb._u_prev) > 0

    ofb.reset()
    assert ofb.x_hat == pytest.approx(np.zeros(2))
    assert ofb._u_prev == pytest.approx(np.zeros(1))


def test_accepts_lqr_instance_as_gain(msd):
    _, A, B = msd
    C = np.array([[1.0, 0.0]])
    lqr = LQR(A, B, np.eye(2), [[1.0]])
    ofb = ObserverFeedback(LuenbergerObserver(A, B, C, poles=[-9.0, -10.0]), lqr)
    assert ofb.K == pytest.approx(lqr.K)


def test_reference_tracking_offset(msd):
    plant, A, B = msd
    C = np.eye(2)
    K = LQR(A, B, np.diag([50.0, 1.0]), [[0.05]]).K
    obs = LuenbergerObserver(A, B, C, poles=[-15.0, -18.0])
    ofb = ObserverFeedback(obs, K, x_ref=np.array([1.0, 0.0]),
                           u_ref=np.array([1.0]))          # feed-forward k*r = 1
    traj = simulate(plant, ofb, x0=np.zeros(2), dt=1e-3, t_final=10.0)
    assert traj.x[-1] == pytest.approx([1.0, 0.0], abs=2e-3)


def test_state_dim_mismatch_raises(msd):
    _, A, B = msd
    C = np.array([[1.0, 0.0]])
    obs = LuenbergerObserver(A, B, C, poles=[-10.0, -11.0])
    with pytest.raises(ValueError, match="state dim"):
        ObserverFeedback(obs, np.array([[1.0, 2.0, 3.0]]))    # 3 columns, obs.n = 2


# --------------------------------------------------------- discrete-KF observer

def test_wraps_discrete_kalman_filter(msd):
    plant, A, B = msd
    C = np.array([[1.0, 0.0]])
    dt = 2e-3
    dkf = DiscreteKalmanFilter(A, B, C, np.diag([1e-4, 1e-4]), [[1e-5]], dt)
    ofb = ObserverFeedback(dkf, LQR(A, B, np.diag([10.0, 1.0]), [[0.1]]).K)

    traj = simulate(plant, ofb, x0=np.array([0.3, 0.0]), dt=dt, t_final=8.0,
                    measurement_fn=lambda t, x, u: C @ x)
    assert np.linalg.norm(traj.x[-1]) < 1e-2
    assert np.linalg.norm(ofb.estimation_error(traj.x[-1])) < 1e-2


# ------------------------------------------------------------ through the harness

def test_lqg_vs_full_state_lqr_through_compare():
    plant = MassSpringDamper(m=1.0, c=0.4, k=1.0)
    A, B = plant.linearize()
    C = np.array([[1.0, 0.0]])
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])

    lqr = LQR(A, B, Q, R)
    lqg = ObserverFeedback.lqg(A, B, C, Q, R, np.diag([1e-4, 1e-4]), [[1e-5]])

    res = compare(
        plant, {"LQR (full state)": lqr, "LQG (output)": lqg},
        x0=np.array([0.5, 0.0]), dt=1e-3, t_final=8.0, reference=0.0, output_index=0,
        measurement_fns={"LQG (output)": lambda t, x, u: C @ x},
    )
    assert res.status["LQR (full state)"] == "Stable"
    assert res.status["LQG (output)"] == "Stable"
    # the observer's transient can only cost tracking accuracy, never improve it
    assert res.metrics["LQG (output)"]["rmse"] >= res.metrics["LQR (full state)"]["rmse"] - 1e-9
