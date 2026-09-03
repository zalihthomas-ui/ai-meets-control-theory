"""System-identification tests: exact recovery, DMDc truncation, ZOH inversion."""

import numpy as np
import pytest

from aimct.simulate import simulate
from aimct.sysid import (
    dmdc,
    least_squares_id,
    model_mismatch,
    prediction_error,
    to_continuous,
)
from aimct.systems import LinearSystem, MassSpringDamper


def _prbs(n_steps, rng, hold=5, amp=1.0):
    """Piecewise-constant random input, one value held for `hold` steps."""
    raw = rng.uniform(-amp, amp, size=(n_steps // hold + 1, 1))
    return np.repeat(raw, hold, axis=0)[:n_steps]


def _rollout(system, dt, n_steps, rng):
    u_seq = _prbs(n_steps, rng)
    k = {"i": 0}

    def ctrl(y, _dt):
        i = min(k["i"], n_steps - 1)
        k["i"] += 1
        return u_seq[i]

    x0 = rng.normal(scale=0.1, size=system.n_states)
    traj = simulate(system, ctrl, x0=x0, dt=dt, t_final=n_steps * dt)
    return traj.x, traj.u[:-1]


def test_exact_recovery_of_known_discrete_system():
    rng = np.random.default_rng(0)
    Ad = np.array([[0.95, 0.05], [-0.1, 0.9]])
    Bd = np.array([[0.0], [0.1]])
    n_steps = 200
    U = rng.uniform(-1, 1, size=(n_steps, 1))
    X = np.zeros((n_steps + 1, 2))
    X[0] = [0.3, -0.2]
    for k in range(n_steps):
        X[k + 1] = Ad @ X[k] + Bd @ U[k]

    A_id, B_id = least_squares_id(X, U)
    assert np.allclose(A_id, Ad, atol=1e-10)
    assert np.allclose(B_id, Bd, atol=1e-10)
    assert prediction_error(A_id, B_id, X, U) < 1e-10


def test_identify_mass_spring_damper_then_back_to_continuous():
    dt = 0.02
    sys = MassSpringDamper(m=1.0, c=0.4, k=1.0)
    X, U = _rollout(sys, dt, 400, np.random.default_rng(1))

    A_d, B_d = least_squares_id(X, U)
    A_c, B_c = to_continuous(A_d, B_d, dt)
    A_true, B_true = sys.linearize()

    mm = model_mismatch(A_true, B_true, A_c, B_c)
    assert mm["A_rel_fro"] < 1e-3
    assert mm["B_rel_fro"] < 1e-3
    assert mm["eig_max_abs_diff"] < 1e-3


def test_least_squares_id_is_robust_to_small_measurement_noise():
    dt = 0.02
    sys = MassSpringDamper()
    X, U = _rollout(sys, dt, 800, np.random.default_rng(2))
    Xn = X + np.random.default_rng(3).normal(scale=1e-3, size=X.shape)

    A_d, B_d = least_squares_id(Xn, U)
    A_c, B_c = to_continuous(A_d, B_d, dt)
    A_true, B_true = sys.linearize()
    # noisy, but the identified continuous model still predicts well
    assert model_mismatch(A_true, B_true, A_c, B_c)["A_rel_fro"] < 0.1
    assert prediction_error(A_d, B_d, X, U, horizon=10) < 5e-2


def test_dmdc_full_rank_matches_plain_least_squares():
    rng = np.random.default_rng(4)
    sys = MassSpringDamper()
    X, U = _rollout(sys, 0.02, 300, rng)
    A1, B1 = least_squares_id(X, U)
    A2, B2 = dmdc(X, U, rank=None)
    assert np.allclose(A1, A2, atol=1e-9)
    assert np.allclose(B1, B2, atol=1e-9)


def test_dmdc_rank_truncation_runs_and_stays_sane():
    rng = np.random.default_rng(5)
    A = np.diag([0.99, 0.97, 0.5, 0.4]) + 0.01 * rng.standard_normal((4, 4))
    B = np.array([[0.1], [0.0], [0.0], [0.0]])
    n_steps = 400
    U = rng.uniform(-1, 1, size=(n_steps, 1))
    X = np.zeros((n_steps + 1, 4))
    for k in range(n_steps):
        X[k + 1] = A @ X[k] + B @ U[k]

    # full rank (snapshot matrix is n+m = 5 wide) -> exact recovery
    A_full, B_full = dmdc(X, U, rank=5)
    assert prediction_error(A_full, B_full, X, U) < 1e-9

    # truncating below the true rank loses accuracy but stays finite & bounded,
    # and monotonically improves as rank grows
    errs = [prediction_error(*dmdc(X, U, rank=r), X, U) for r in (2, 3, 4, 5)]
    assert all(np.isfinite(errs))
    assert errs[0] >= errs[1] >= errs[2] >= errs[3]
    assert errs[-1] < 1e-9


def test_stack_rejects_bad_shapes():
    with pytest.raises(ValueError):
        least_squares_id(np.zeros((1, 2)), np.zeros((0, 1)))
