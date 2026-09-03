"""Tests for :mod:`aimct.controllers.state_feedback` — Ackermann pole placement
and the static ``u = u_ref - K (x - x_ref)`` law.

Correctness is checked two ways: against gains that have a closed-form answer,
and by confirming ``eig(A - B K)`` equals the requested pole set (a check that
needs no control library — ``scipy.signal.place_poles`` is unavailable on this
box anyway).
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import (
    StateFeedback,
    controllability_matrix,
    is_controllable,
    place_poles,
)
from aimct.systems import LinearSystem
from aimct.simulate import simulate


# double integrator: xddot = u
A_DI = np.array([[0.0, 1.0], [0.0, 0.0]])
B_DI = np.array([[0.0], [1.0]])


# --------------------------------------------------------------- controllability

def test_controllability_matrix_double_integrator():
    ctrb = controllability_matrix(A_DI, B_DI)
    assert ctrb == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_is_controllable_true_and_false():
    assert is_controllable(A_DI, B_DI)
    # mode at eigenvalue 2 gets no input
    A = np.diag([1.0, 2.0])
    B = np.array([[1.0], [0.0]])
    assert not is_controllable(A, B)


# --------------------------------------------------------------- Ackermann gain

def test_ackermann_matches_closed_form_gain():
    # poles -2, -3  ->  phi(s) = s^2 + 5s + 6  ->  A - BK = [[0,1],[-6,-5]]
    K = place_poles(A_DI, B_DI, [-2.0, -3.0])
    assert K == pytest.approx(np.array([[6.0, 5.0]]))


@pytest.mark.parametrize(
    "poles",
    [
        [-1.0, -2.0],
        [-3.0, -3.0],                 # repeated real
        [-2.0 + 3.0j, -2.0 - 3.0j],   # complex conjugate pair
        [-0.5, -10.0],                # widely separated
    ],
)
def test_placed_poles_are_closed_loop_eigenvalues(poles):
    K = place_poles(A_DI, B_DI, poles)
    got = np.sort_complex(np.linalg.eigvals(A_DI - B_DI @ K))
    want = np.sort_complex(np.asarray(poles, dtype=complex))
    # a repeated (defective) pole is only accurate to ~sqrt(eps)
    assert got == pytest.approx(want, abs=1e-6)


def test_placement_on_a_third_order_system():
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-2.0, -4.0, -6.0]])
    B = np.array([[0.0], [0.0], [1.0]])
    poles = [-1.0, -2.0 + 1.0j, -2.0 - 1.0j]
    K = place_poles(A, B, poles)
    got = np.sort_complex(np.linalg.eigvals(A - B @ K))
    assert got == pytest.approx(np.sort_complex(np.asarray(poles, dtype=complex)), abs=1e-8)


# ----------------------------------------------------------------------- guards

def test_unpaired_complex_poles_raise():
    with pytest.raises(ValueError, match="conjugate"):
        place_poles(A_DI, B_DI, [-1.0 + 1.0j, -1.0 + 2.0j])


def test_wrong_pole_count_raises():
    with pytest.raises(ValueError, match="exactly n"):
        place_poles(A_DI, B_DI, [-1.0, -2.0, -3.0])


def test_uncontrollable_pair_raises():
    A = np.diag([1.0, 2.0])
    B = np.array([[1.0], [0.0]])
    with pytest.raises(ValueError, match="controllable"):
        place_poles(A, B, [-1.0, -2.0])


def test_multi_input_placement_not_implemented():
    B2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(NotImplementedError):
        place_poles(A_DI, B2, [-1.0, -2.0])


# ------------------------------------------------------------------ control law

def test_regulator_law_is_minus_k_x():
    sf = StateFeedback(np.array([[2.0, 0.5]]))
    assert sf.update(np.array([3.0, -1.0])) == pytest.approx(-(2.0 * 3.0 + 0.5 * -1.0))


def test_reference_tracking_law():
    K = np.array([[4.0, 1.0]])
    sf = StateFeedback(K, x_ref=np.array([1.0, 0.0]), u_ref=np.array([2.5]))
    x = np.array([1.2, -0.3])
    expected = 2.5 - (K @ (x - np.array([1.0, 0.0])))[0]
    assert sf.update(x) == pytest.approx(expected)


def test_update_rejects_wrong_length_state():
    sf = StateFeedback(np.array([[1.0, 1.0]]))
    with pytest.raises(ValueError, match="length-2"):
        sf.update(np.array([1.0, 2.0, 3.0]))


def test_multi_input_gain_returns_vector():
    K = np.array([[1.0, 0.0], [0.0, 2.0]])
    sf = StateFeedback(K)
    u = sf.update(np.array([1.0, 1.0]))
    assert np.asarray(u) == pytest.approx([-1.0, -2.0])


def test_stateless_reset_and_call_alias():
    sf = StateFeedback(np.array([[1.0, 2.0]]))
    a = sf.update(np.array([1.0, 1.0]))
    sf.reset()
    b = sf(np.array([1.0, 1.0]), 0.1)
    assert a == pytest.approx(b)


# ------------------------------------------------------------------ constructors

def test_from_poles_stores_ab_and_reports_closed_loop_poles():
    sf = StateFeedback.from_poles(A_DI, B_DI, [-2.0, -4.0])
    got = np.sort_complex(sf.closed_loop_poles())
    assert got == pytest.approx(np.array([-4.0 + 0j, -2.0 + 0j]), abs=1e-9)


def test_closed_loop_poles_requires_ab_when_not_stored():
    sf = StateFeedback(np.array([[6.0, 5.0]]))
    with pytest.raises(ValueError, match="A and B"):
        sf.closed_loop_poles()


# ------------------------------------------------------------------- simulation

def test_stabilizes_an_open_loop_unstable_system():
    # A has eigenvalues +-1 (one unstable); place both well into the LHP.
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.max(np.linalg.eigvals(A).real) > 0
    sf = StateFeedback.from_poles(A, B_DI, [-2.0, -3.0])
    sys = LinearSystem(A, B_DI)
    traj = simulate(sys, sf, x0=np.array([1.0, -0.5]), dt=0.01, t_final=12.0)
    assert np.linalg.norm(traj.x[-1]) < 1e-3
    assert np.all(np.abs(traj.u) < 1e3)  # no blow-up
