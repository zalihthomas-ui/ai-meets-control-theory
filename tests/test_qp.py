"""Tests for :mod:`aimct.controllers._qp` - the from-scratch active-set QP,
cross-checked against ``scipy.optimize.minimize`` (SLSQP).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from aimct.controllers._qp import solve_qp


def _slsqp(H, g, lb, ub, C=None, d=None):
    n = g.size
    cons = []
    if C is not None:
        cons.append({"type": "ineq", "fun": lambda z, C=C, d=d: d - C @ z})
    r = minimize(
        lambda z: 0.5 * z @ H @ z + g @ z, np.zeros(n),
        jac=lambda z: H @ z + g,
        bounds=list(zip(lb, ub)), constraints=cons, method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    return r.x


def _rand_qp(rng, n, n_ineq=0):
    M = rng.standard_normal((n, n))
    H = M @ M.T + 0.1 * np.eye(n)
    g = rng.standard_normal(n)
    lb = -rng.uniform(0.5, 2.0, n)
    ub = rng.uniform(0.5, 2.0, n)
    C = d = None
    if n_ineq:
        C = rng.standard_normal((n_ineq, n))
        d = rng.uniform(0.2, 1.5, n_ineq)
    return H, g, lb, ub, C, d


def test_unconstrained_reaches_stationary_point():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((5, 5))
    H = M @ M.T + np.eye(5)
    g = rng.standard_normal(5)
    x = solve_qp(H, g, lb=np.full(5, -1e3), ub=np.full(5, 1e3)).x
    assert H @ x + g == pytest.approx(np.zeros(5), abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_box_only_matches_slsqp(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    H, g, lb, ub, *_ = _rand_qp(rng, n)
    x = solve_qp(H, g, lb=lb, ub=ub).x
    xs = _slsqp(H, g, lb, ub)
    assert 0.5 * x @ H @ x + g @ x == pytest.approx(0.5 * xs @ H @ xs + g @ xs, abs=1e-8)
    assert np.all(x >= lb - 1e-7) and np.all(x <= ub + 1e-7)


@pytest.mark.parametrize("seed", range(6))
def test_linear_inequalities_match_slsqp(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(3, 8))
    H, g, lb, ub, C, d = _rand_qp(rng, n, n_ineq=int(rng.integers(1, 4)))
    x = solve_qp(H, g, C=C, d=d, lb=lb, ub=ub).x
    xs = _slsqp(H, g, lb, ub, C, d)
    assert 0.5 * x @ H @ x + g @ x == pytest.approx(0.5 * xs @ H @ xs + g @ xs, abs=1e-7)
    assert np.all(C @ x - d <= 1e-6)
    assert np.all(x >= lb - 1e-7) and np.all(x <= ub + 1e-7)


def test_active_inequality_is_tight_and_multiplier_nonnegative():
    # min (x1-2)^2 + (x2-2)^2  s.t. x1 + x2 <= 1  -> solution on the line
    H = 2 * np.eye(2)
    g = np.array([-4.0, -4.0])
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    res = solve_qp(H, g, C=C, d=d, lb=[-10, -10], ub=[10, 10])
    assert C @ res.x == pytest.approx(d, abs=1e-8)
    assert res.x == pytest.approx([0.5, 0.5], abs=1e-7)


def test_infeasible_start_is_projected_into_the_feasible_set():
    H = np.eye(3)
    g = np.zeros(3)
    C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    d = np.array([-1.0, -1.0])                       # requires x1 <= -1, x2 <= -1
    res = solve_qp(H, g, C=C, d=d, lb=np.full(3, -5.0), ub=np.full(3, 5.0),
                   z0=np.array([4.0, 4.0, 0.0]))
    assert np.all(C @ res.x - d <= 1e-6)
    assert res.feasible


def test_result_carries_iteration_and_active_set_info():
    H = 2 * np.eye(2)
    g = np.array([-4.0, -4.0])
    res = solve_qp(H, g, lb=[-1.0, -1.0], ub=[1.0, 1.0])   # optimum at the +1/+1 corner
    assert res.x == pytest.approx([1.0, 1.0], abs=1e-8)
    assert res.iterations >= 1
    assert len(res.active_set) == 2
