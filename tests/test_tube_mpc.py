"""Tube MPC: mRPI box, constraint tightening, robust constraint satisfaction."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import LinearMPC, MRPISet, TubeMPC, dare, mrpi_box
from aimct.controllers.mpc import _discretize


# ======================================================================
# mrpi_box
# ======================================================================
def test_mrpi_box_contains_the_disturbance_set_and_is_invariant_enough():
    A_K = np.array([[0.6, 0.1], [0.0, 0.5]])
    w = np.array([0.1, 0.05])
    Z = mrpi_box(A_K, w)
    assert isinstance(Z, MRPISet)
    assert np.all(Z.half_widths >= w - 1e-12)                 # W subset Z
    # for this nonnegative A_K the box IS RPI:  |A_K| Z + w <= Z
    assert Z.rpi_residual <= 1e-8
    # matches the closed form (I - A_K)^-1 w  (|A_K| = A_K here)
    assert np.allclose(Z.half_widths, np.linalg.solve(np.eye(2) - A_K, w),
                       rtol=1e-4)


def test_mrpi_box_conservatism_scales_linearly_with_W():
    A_K = np.array([[0.9, 0.3], [-0.2, 0.7]])                  # rho(|A_K|) > 1
    base = mrpi_box(A_K, np.array([0.1, 0.1])).half_widths
    for s in (0.5, 0.25, 0.1):
        z = mrpi_box(A_K, s * np.array([0.1, 0.1])).half_widths
        assert np.allclose(z, s * base, rtol=1e-9)


def test_mrpi_box_handles_a_disturbance_on_only_one_channel():
    A_K = np.array([[0.9, 0.1], [-0.15, 0.8]])
    Z = mrpi_box(A_K, np.array([0.0, 0.05]))                  # degenerate W
    assert Z.half_widths[0] > 0.0                              # coupling -> non-zero
    assert Z.half_widths[1] >= 0.05 - 1e-12


def test_mrpi_box_rejects_a_non_schur_A_K():
    with pytest.raises(ValueError):
        mrpi_box(np.array([[1.1, 0.0], [0.0, 0.5]]), np.array([0.1, 0.1]))


# ======================================================================
# TubeMPC
# ======================================================================
@pytest.fixture
def di_problem():
    # double integrator, position box, disturbance on the velocity channel
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    dt = 0.1
    Ad, Bd = _discretize(A, B, dt)
    w = np.array([0.0, 0.05])
    common = dict(Q=np.diag([20.0, 1.0]), R=np.array([[0.2]]), N=25,
                  x_bounds=(np.array([-1.0, -np.inf]), np.array([1.0, np.inf])),
                  u_bounds=(-5.0, 5.0), x_ref=np.array([0.95, 0.0]))
    return A, B, Ad, Bd, dt, w, common


def _rollout(ctrl, Ad, Bd, dt, w_hw, T=140, sign=+1.0):
    ctrl.reset()
    x = np.zeros(2)
    xs = [x.copy()]
    for _ in range(T):
        u = np.atleast_1d(ctrl.update(x, dt))
        x = Ad @ x + Bd @ u + np.array([0.0, sign * w_hw[1]])
        xs.append(x.copy())
    return np.array(xs)


def test_tube_mpc_keeps_the_true_state_in_the_box_where_nominal_mpc_does_not(di_problem):
    A, B, Ad, Bd, dt, w, common = di_problem
    nom = LinearMPC(A, B, **common)
    tube = TubeMPC(A, B, W=w, **common)

    xn = _rollout(nom, Ad, Bd, dt, w)
    xt = _rollout(tube, Ad, Bd, dt, w)

    assert xn[:, 0].max() > 1.0 + 1e-4          # nominal MPC violates the box
    assert xt[:, 0].max() <= 1.0 + 1e-3         # tube MPC does not


def test_tube_mpc_tightens_the_state_box_by_the_mrpi_width(di_problem):
    A, B, Ad, Bd, dt, w, common = di_problem
    tube = TubeMPC(A, B, W=w, **common)
    tube.update(np.zeros(2), dt)                 # triggers the lazy tube build
    z = tube.mrpi.half_widths
    x_lo_t, x_hi_t = tube.tightened_x_bounds
    assert np.isclose(x_hi_t[0], 1.0 - z[0])
    assert np.isclose(x_lo_t[0], -1.0 + z[0])
    u_lo_t, u_hi_t = tube.tightened_u_bounds
    kz = np.abs(tube._tube_build(dt)["K"]) @ z
    assert np.isclose(u_hi_t[0], 5.0 - kz[0])


def test_tube_conservatism_shrinks_as_the_disturbance_shrinks(di_problem):
    A, B, Ad, Bd, dt, w, common = di_problem
    widths = []
    for scale in (1.0, 0.5, 0.25):
        tube = TubeMPC(A, B, W=scale * w, **common)
        tube.update(np.zeros(2), dt)
        widths.append(tube.mrpi.half_widths[0])
    assert widths[1] == pytest.approx(0.5 * widths[0], rel=1e-6)
    assert widths[2] == pytest.approx(0.25 * widths[0], rel=1e-6)


def test_tube_mpc_raises_if_the_disturbance_eats_the_whole_constraint_set(di_problem):
    A, B, Ad, Bd, dt, w, common = di_problem
    huge = TubeMPC(A, B, W=np.array([0.0, 3.0]), **common)    # |x1| box is only +/-1
    with pytest.raises(ValueError, match="empty"):
        huge.update(np.zeros(2), dt)


def test_unconstrained_tube_mpc_reduces_to_the_ancillary_plus_nominal_lqr(di_problem):
    # with no state box and a zero disturbance, u should track the LQR move
    A, B, Ad, Bd, dt, _w, common = di_problem
    ck = dict(common)
    ck.pop("x_bounds")
    ck["x_ref"] = np.zeros(2)
    tube = TubeMPC(A, B, W=np.zeros(2), **ck)
    lqr_gain = LinearMPC(A, B, **ck).discrete_lqr_gain(dt)
    x = np.array([0.3, -0.1])
    u = np.atleast_1d(tube.update(x, dt))
    assert np.allclose(u, -lqr_gain @ x, atol=1e-6)


def test_tube_mpc_from_system_builds_from_a_model():
    from aimct.systems import MassSpringDamper

    msd = MassSpringDamper()
    tube = TubeMPC.from_system(
        msd, W=np.array([0.0, 0.02]), Q=np.diag([10.0, 1.0]),
        R=np.array([[0.5]]), N=15,
        x_bounds=(np.array([-0.5, -np.inf]), np.array([0.5, np.inf])),
        u_bounds=(-3.0, 3.0),
    )
    tube.update(np.array([0.2, 0.0]), 0.05)
    assert tube.mrpi is not None and np.all(tube.mrpi.half_widths >= 0)
