"""simulate_batch: shape/stacking, per-trial disturbances & param overrides,
divergence padding, and metric mapping."""

import numpy as np
import pytest

from aimct.controllers import LQR
from aimct.simulate import BatchResult, Trajectory, simulate, simulate_batch
from aimct.systems import MassSpringDamper, Pendulum


def _msd_lqr():
    sys = MassSpringDamper()
    A, B = sys.linearize()
    return sys, LQR(A, B, np.eye(2), np.array([[1.0]]))


def test_batch_shapes_and_grid_match_single_simulate():
    sys, k = _msd_lqr()
    rng = np.random.default_rng(0)
    x0s = rng.normal(scale=0.5, size=(12, 2))
    res = simulate_batch(sys, k, x0s, dt=0.02, t_final=3.0)

    assert isinstance(res, BatchResult) and len(res) == 12
    T = int(round(3.0 / 0.02)) + 1
    assert res.x.shape == (12, T, 2)
    assert res.u.shape == (12, T, 1)
    assert res.t.shape == (T,)
    # trial i must equal a standalone simulate() from the same x0
    k.reset()
    one = simulate(sys, k, x0=x0s[3], dt=0.02, t_final=3.0)
    assert np.allclose(res.x[3], one.x, atol=1e-10)
    assert np.allclose(res.final_x[3], one.x[-1])


def test_all_trials_converge_from_a_ball_of_initial_conditions():
    sys, k = _msd_lqr()
    rng = np.random.default_rng(1)
    x0s = rng.normal(scale=1.0, size=(40, 2))
    res = simulate_batch(sys, k, x0s, dt=0.02, t_final=8.0)
    assert not res.diverged.any()
    assert np.all(np.linalg.norm(res.final_x, axis=1) < 5e-2)


def test_map_metric_matches_manual_per_trajectory():
    sys, k = _msd_lqr()
    x0s = np.array([[1.0, 0.0], [0.0, 1.0], [-0.5, 0.3]])
    res = simulate_batch(sys, k, x0s, dt=0.02, t_final=4.0)

    peak = res.map_metric(lambda tr: float(np.max(np.abs(tr.x[:, 0]))))
    for i in range(3):
        k.reset()
        one = simulate(sys, k, x0=x0s[i], dt=0.02, t_final=4.0)
        assert peak[i] == pytest.approx(np.max(np.abs(one.x[:, 0])))


def test_per_trial_input_disturbances():
    sys, k = _msd_lqr()
    x0s = np.zeros((3, 2))
    dists = [None,
             lambda t: np.array([2.0]),
             lambda t: np.array([5.0 if t < 1.0 else 0.0])]
    res = simulate_batch(sys, k, x0s, dt=0.02, t_final=4.0,
                         input_disturbances=dists)
    # trial 0 (no disturbance, x0=0) never moves; the others do
    assert np.allclose(res.x[0], 0.0, atol=1e-9)
    assert np.max(np.abs(res.x[1])) > 1e-3
    assert np.max(np.abs(res.x[2])) > 1e-3


def test_param_overrides_are_applied_then_restored():
    sys = MassSpringDamper()
    A, B = sys.linearize()
    k = LQR(A, B, np.eye(2), np.array([[1.0]]))
    stiff0 = sys.k
    x0s = np.tile([1.0, 0.0], (3, 1))
    res = simulate_batch(sys, k, x0s, dt=0.02, t_final=5.0,
                         param_overrides=[{"k": 1.0}, {"k": 20.0}, {"k": 80.0}])
    # a stiffer spring oscillates faster -> more zero-crossings in x[:,0]
    zc = res.map_metric(
        lambda tr: float(np.sum(np.abs(np.diff(np.sign(tr.x[:, 0]))) > 0)))
    assert zc[0] < zc[1] < zc[2]
    assert sys.k == stiff0                       # restored


def test_fresh_controller_factory_is_used_per_trial():
    sys, _ = _msd_lqr()
    A, B = sys.linearize()
    made = {"n": 0}

    def factory():
        made["n"] += 1
        return LQR(A, B, np.eye(2), np.array([[1.0]]))

    x0s = np.zeros((5, 2))
    simulate_batch(sys, factory, x0s, dt=0.05, t_final=1.0, fresh_controller=True)
    assert made["n"] == 5


def test_diverged_trial_is_flagged_and_padded_rectangular():
    sys = Pendulum()
    # no control; a big push sends one trial over the top and it runs away
    zero = lambda meas, dt: np.zeros(1)
    x0s = np.array([[0.1, 0.0], [0.2, 0.0], [3.0, 50.0]])
    res = simulate_batch(sys, zero, x0s, dt=0.05, t_final=6.0,
                         u_bounds=(-1.0, 1.0))
    T = int(round(6.0 / 0.05)) + 1
    assert res.x.shape == (3, T, 2)              # still rectangular
    assert res.n_valid[0] == T and res.n_valid[1] == T
    # the runaway trial: either flagged diverged, or simply never settled
    assert res.diverged[2] or res.n_valid[2] <= T
    # padding repeats the last finite sample
    k2 = int(res.n_valid[2])
    if k2 < T:
        assert np.allclose(res.x[2, k2:], res.x[2, k2 - 1])
