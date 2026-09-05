"""Tests for :mod:`aimct.planning` - Hermite-Simpson direct collocation."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.planning import CollocationResult, DirectCollocation
from aimct.systems import CartPole


# A double integrator:  x = [pos, vel],  xdot = [vel, u].
def _double_integrator(x, u):
    return np.array([x[1], u[0]])


def _rollout_foh(f, x0, U, t, sub=50):
    """Fine RK4 with a first-order-hold input - the model collocation assumes."""
    x = np.array(x0, float)
    out = [x.copy()]
    for k in range(len(t) - 1):
        h = (t[k + 1] - t[k]) / sub
        for j in range(sub):
            frac = (j + 0.5) / sub
            u = (1 - frac) * U[k] + frac * U[k + 1]
            k1 = f(x, u)
            k2 = f(x + 0.5 * h * k1, u)
            k3 = f(x + 0.5 * h * k2, u)
            k4 = f(x + h * k3, u)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x.copy())
    return np.array(out)


def test_returns_a_result_with_consistent_shapes():
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=15, t_final=1.0,
        x0=[0.0, 0.0], x_goal=[1.0, 0.0], Q=0.0, R=1.0,
    )
    res = dc.solve()
    assert isinstance(res, CollocationResult)
    assert res.X.shape == (15, 2)
    assert res.U.shape == (15, 1)
    assert res.t.shape == (15,)
    assert res.t[0] == 0.0 and res.t[-1] == pytest.approx(1.0)


def test_defects_vanish_at_the_solution():
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=25, t_final=2.0,
        x0=[0.0, 0.0], x_goal=[1.0, 0.0], Q=0.0, R=1.0, tol=1e-10,
    )
    res = dc.solve()
    assert res.success
    assert res.defect_norm < 1e-7


def test_boundary_conditions_are_enforced_exactly():
    x0 = np.array([-0.4, 0.2])
    xg = np.array([1.3, 0.0])
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=20, t_final=1.5,
        x0=x0, x_goal=xg, Q=0.0, R=1.0,
    )
    res = dc.solve()
    assert np.allclose(res.X[0], x0, atol=1e-9)
    assert np.allclose(res.X[-1], xg, atol=1e-7)


def test_minimum_effort_double_integrator_matches_the_analytic_optimum():
    # rest-to-rest, distance d over [0, T]:  u*(t) = (6 d / T^3) (T - 2 t),
    # optimal cost  J* = integral u^2 = 12 d^2 / T^3.
    d, T, N = 1.0, 2.0, 61
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=N, t_final=T,
        x0=[0.0, 0.0], x_goal=[d, 0.0], Q=0.0, R=1.0, Qf=0.0, tol=1e-11,
    )
    res = dc.solve()
    j_star = 12 * d**2 / T**3
    assert res.cost == pytest.approx(j_star, rel=2e-3)
    u_an = (6 * d / T**3) * (T - 2 * res.t)
    assert np.max(np.abs(res.U[:, 0] - u_an)) < 5e-2      # boundary ringing aside


def test_the_plan_flies_when_reintegrated_through_the_true_dynamics():
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=41, t_final=2.0,
        x0=[0.0, 0.0], x_goal=[1.0, 0.0], Q=0.0, R=1.0,
    )
    res = dc.solve()
    sim = _rollout_foh(_double_integrator, res.X[0], res.U, res.t)
    # linear dynamics + FOH: the collocation trajectory is exact up to solver tol
    assert np.linalg.norm(sim[-1] - res.X[-1]) < 1e-6
    assert np.max(np.linalg.norm(sim - res.X, axis=1)) < 1e-6


def test_input_box_is_respected():
    # rest-to-rest unit move in T=1: the unconstrained min-effort peak is |u|=6;
    # |u| <= 5 is feasible (bang-bang needs 4) and clips the peaks -> active.
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=31, t_final=1.0,
        x0=[0.0, 0.0], x_goal=[1.0, 0.0], Q=0.0, R=1.0, u_bounds=(-5.0, 5.0),
    )
    res = dc.solve()
    assert res.success
    assert np.all(res.U <= 5.0 + 1e-7) and np.all(res.U >= -5.0 - 1e-7)
    assert np.max(np.abs(res.U)) == pytest.approx(5.0, abs=1e-4)   # bound active


def test_state_box_is_respected():
    # unconstrained peak velocity for this move is 1.5; a 1.35 cap is feasible
    # (mean speed is 1.0) and bites.
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=31, t_final=1.0,
        x0=[0.0, 0.0], x_goal=[1.0, 0.0], Q=0.0, R=1.0,
        x_bounds=([-1.0, -2.0], [2.0, 1.35]),
    )
    res = dc.solve()
    assert res.success
    assert np.all(res.X[:, 1] <= 1.35 + 1e-7)              # velocity capped
    assert res.X[:, 1].max() == pytest.approx(1.35, abs=1e-3)   # ... and it bites


def test_path_inequality_is_enforced():
    # same velocity cap as the state-box test, but expressed as a general
    # path inequality  g(X, U) = v - 1.35 <= 0.
    def path_con(X, U):
        return X[:, 1] - 1.35

    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=31, t_final=1.0,
        x0=[0.0, 0.0], x_goal=[1.0, 0.0], Q=0.0, R=1.0, path_con=path_con,
    )
    res = dc.solve()
    assert res.success
    assert res.X[:, 1].max() <= 1.35 + 1e-6
    assert res.X[:, 1].max() == pytest.approx(1.35, abs=1e-3)   # constraint bites


def test_custom_running_and_terminal_cost_callables():
    # minimum time-ish: penalise distance from the goal each knot, light effort
    goal = np.array([1.0, 0.0])

    def running(x, u):
        return float((x - goal) @ (x - goal) + 1e-2 * u[0] ** 2)

    def terminal(x):
        return float(50.0 * (x - goal) @ (x - goal))

    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=21, t_final=2.0,
        x0=[0.0, 0.0], running_cost=running, terminal_cost=terminal,
        u_bounds=(-10.0, 10.0),
    )
    res = dc.solve()
    assert res.success
    assert res.defect_norm < 1e-6
    assert np.linalg.norm(res.X[-1] - goal) < 0.1          # driven to the goal


def test_deterministic():
    kw = dict(n_x=2, n_u=1, N=21, t_final=1.5, x0=[0.0, 0.0],
              x_goal=[1.0, 0.0], Q=0.0, R=1.0)
    a = DirectCollocation(_double_integrator, **kw).solve()
    b = DirectCollocation(_double_integrator, **kw).solve()
    assert np.array_equal(a.X, b.X) and np.array_equal(a.U, b.U)


@pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
def test_trust_constr_backend_agrees_with_slsqp():
    kw = dict(n_x=2, n_u=1, N=25, t_final=2.0, x0=[0.0, 0.0],
              x_goal=[1.0, 0.0], Q=0.0, R=1.0)
    a = DirectCollocation(_double_integrator, method="SLSQP", **kw).solve()
    b = DirectCollocation(_double_integrator, method="trust-constr",
                          max_iter=300, **kw).solve()
    assert b.defect_norm < 1e-5
    assert b.cost == pytest.approx(a.cost, rel=1e-3)


def test_analytic_defect_jacobian_matches_finite_difference():
    dc = DirectCollocation(
        _double_integrator, n_x=2, n_u=1, N=8, t_final=1.0,
        x0=[0.0, 0.0], x_goal=[1.0, 0.0], Q=0.0, R=1.0,
    )
    rng = np.random.default_rng(0)
    z = rng.standard_normal(dc._nz)
    J = dc._defects_jac(z)
    eps = 1e-6
    J_fd = np.zeros_like(J)
    for i in range(z.size):
        d = np.zeros_like(z)
        d[i] = eps
        J_fd[:, i] = (dc._defects(z + d) - dc._defects(z - d)) / (2 * eps)
    assert np.allclose(J, J_fd, atol=1e-6)


def test_from_system_builds_a_cartpole_swingup_that_hits_the_target():
    cp = CartPole()
    dc = DirectCollocation.from_system(
        cp, t_final=2.0, N=21, x0=[0.0, 0.0, np.pi, 0.0],
        x_goal=[0.0, 0.0, 0.0, 0.0], Q=0.0, R=1.0, u_bounds=(-20.0, 20.0),
        max_iter=400,
    )
    res = dc.solve()
    assert res.success
    assert res.defect_norm < 1e-6
    assert np.allclose(res.X[-1], [0.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert np.max(np.abs(res.U)) <= 20.0 + 1e-6


@pytest.mark.slow
def test_cartpole_swingup_plan_reintegrates_close_on_a_fine_mesh():
    cp = CartPole()
    f = lambda x, u: np.asarray(cp.dynamics(0.0, x, u), float)
    dc = DirectCollocation.from_system(
        cp, t_final=2.0, N=61, x0=[0.0, 0.0, np.pi, 0.0],
        x_goal=[0.0, 0.0, 0.0, 0.0], Q=0.0, R=1.0, u_bounds=(-20.0, 20.0),
        max_iter=600,
    )
    res = dc.solve()
    sim = _rollout_foh(f, res.X[0], res.U, res.t)
    # inter-knot quadrature error only; shrinks with the mesh
    assert np.linalg.norm(sim[-1] - res.X[-1]) < 0.05
