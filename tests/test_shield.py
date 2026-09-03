"""Tests for :mod:`aimct.hybrid.shield` - the safety shield."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import LQR, wrap_angle
from aimct.hybrid import ShieldedController, barrier_predicate, box_predicate
from aimct.hybrid.shield import _step  # noqa: for a bare-callable check
from aimct.simulate import simulate
from aimct.systems import CartPole, Pendulum

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _const(value):
    def law(x, dt):
        return float(value)
    law.reset = lambda: None
    return law


# ------------------------------------------------------------- switching logic

def test_switch_picks_base_or_fallback_by_predicate():
    flag = {"safe": True}
    sh = ShieldedController(_const(1.0), _const(-9.0),
                            is_safe=lambda x: flag["safe"])
    assert sh.update(np.zeros(2), 0.1) == 1.0 and sh.mode == "base"
    flag["safe"] = False
    assert sh.update(np.zeros(2), 0.1) == -9.0 and sh.mode == "fallback"
    assert sh.intervention_log == [False, True]
    assert sh.intervention_rate == pytest.approx(0.5)


def test_switching_is_deterministic():
    def make():
        return ShieldedController(
            _const(2.0), _const(-2.0),
            is_safe=box_predicate([-1.0, -1.0], [1.0, 1.0]),
        )
    a, b = make(), make()
    xs = [np.array([0.5, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 5.0]),
          np.array([-0.2, 0.1])]
    for x in xs:
        assert a.update(x, 0.1) == b.update(x, 0.1)
    assert a.intervention_log == b.intervention_log == [False, True, True, False]


def test_reset_clears_log_and_resets_children():
    child_reset = {"n": 0}

    class C:
        def update(self, x, dt):
            return 0.0
        def reset(self):
            child_reset["n"] += 1

    sh = ShieldedController(C(), C(), is_safe=lambda x: True)
    sh.update(np.zeros(1), 0.1)
    assert sh.intervention_log
    sh.reset()
    assert sh.intervention_log == [] and sh.mode == "base"
    assert child_reset["n"] == 4                      # 2 at construction, 2 at reset


def test_invalid_blend_and_missing_predict_raise():
    with pytest.raises(ValueError, match="blend"):
        ShieldedController(_const(0.0), _const(0.0), is_safe=lambda x: True,
                           blend="bogus")
    with pytest.raises(ValueError, match="predict"):
        ShieldedController(_const(0.0), _const(0.0), is_safe=lambda x: True,
                           blend="filter")


# ---------------------------------------------------------------- predicates

def test_box_predicate_handles_infinite_bounds():
    safe = box_predicate([-1.0, -np.inf], [1.0, np.inf])
    assert safe([0.9, 1e6]) and safe([-1.0, -1e9])
    assert not safe([1.1, 0.0])


def test_barrier_predicate_uses_margin():
    safe = barrier_predicate(lambda x: 0.5 - abs(x[0]), margin=0.0)
    assert safe([0.4, 0.0]) and not safe([0.6, 0.0])
    sh = ShieldedController(_const(1.0), _const(-1.0), is_safe=safe)
    assert sh.update(np.array([0.3, 0.0]), 0.1) == 1.0
    assert sh.update(np.array([0.9, 0.0]), 0.1) == -1.0


# ------------------------------------------------------- containment guarantees

def test_bad_base_policy_is_contained_on_the_cartpole():
    cp = CartPole()
    A, B = cp.linearize()
    lqr = LQR(A, B, np.diag([1.0, 1.0, 10.0, 1.0]), [[0.1]])
    rng = np.random.default_rng(0)
    bad = lambda x, dt: float(rng.uniform(-20.0, 20.0))          # noqa: E731
    bad.reset = lambda: None
    # trigger bounds angle AND rate, with margin inside the guarantee box
    trigger = box_predicate([-0.8, -1.5, -0.20, -1.0], [0.8, 1.5, 0.20, 1.0])
    sh = ShieldedController(bad, lqr, is_safe=trigger)

    traj = simulate(cp, sh, x0=np.array([0.0, 0.0, 0.1, 0.0]),
                    dt=0.01, t_final=8.0, u_bounds=(-20.0, 20.0))
    assert not traj.diverged
    assert np.max(np.abs(traj.x[:, 2])) < 0.5          # guarantee: pole angle
    assert np.max(np.abs(traj.x[:, 0])) < 2.4          # guarantee: on the rail
    assert 0.0 < sh.intervention_rate < 1.0


def test_bad_base_policy_is_contained_on_the_pendulum():
    p = Pendulum()
    A, B = p.linearize()                               # about upright
    K = LQR(A, B, np.diag([10.0, 1.0]), [[0.5]]).K

    def lqr_up(x, dt):
        err = wrap_angle(x[0] - np.pi)
        return float(np.clip(-(K @ np.array([err, x[1]]))[0], -4.0, 4.0))
    lqr_up.reset = lambda: None
    rng = np.random.default_rng(1)
    bad = lambda x, dt: float(rng.uniform(-4.0, 4.0))            # noqa: E731
    bad.reset = lambda: None

    def wrapped_err(x):
        return abs(wrap_angle(x[0] - np.pi))

    # tight trigger (angle AND rate) so the weak +/-4 N.m fallback keeps its grip
    trigger = lambda x: wrapped_err(x) < 0.10 and abs(x[1]) < 0.4   # noqa: E731
    sh = ShieldedController(bad, lqr_up, is_safe=trigger)

    traj = simulate(p, sh, x0=np.array([np.pi, 0.0]), dt=0.01, t_final=8.0,
                    u_bounds=(-4.0, 4.0))
    assert not traj.diverged
    assert max(wrapped_err(x) for x in traj.x) < 0.35  # never falls far from upright
    assert 0.0 < sh.intervention_rate < 1.0


def test_good_base_never_triggers_the_fallback():
    cp = CartPole()
    A, B = cp.linearize()
    good = LQR(A, B, np.diag([1.0, 1.0, 10.0, 1.0]), [[0.1]])
    fallback = LQR(A, B, np.diag([1.0, 1.0, 100.0, 10.0]), [[0.01]])
    trigger = box_predicate([-1.0, -3.0, -0.4, -3.0], [1.0, 3.0, 0.4, 3.0])
    sh = ShieldedController(good, fallback, is_safe=trigger)

    traj = simulate(cp, sh, x0=np.array([0.0, 0.0, 0.15, 0.0]),
                    dt=0.02, t_final=6.0, u_bounds=(-20.0, 20.0))
    assert sh.intervention_rate == 0.0
    assert np.abs(traj.x[-1, 2]) < 1e-2
    assert len(sh.intervention_log) == len(traj) - 1


# ------------------------------------------------------------------- filter mode

def test_filter_mode_projects_unsafe_actions_and_passes_safe_ones():
    # scalar integrator x' = u ; safe box |x_next| <= 1
    def predict(x, u, dt):
        return np.asarray(x, float) + dt * np.asarray(u, float)
    safe = box_predicate([-1.0], [1.0])
    sh = ShieldedController(_const(50.0), _const(0.0), is_safe=safe,
                            blend="filter", predict=predict)

    # from x=0, u=50, dt=0.1 -> x_next=5 unsafe -> projected toward fallback (0)
    u = sh.update(np.array([0.0]), 0.1)
    assert abs(0.0 + 0.1 * u) <= 1.0 + 1e-6            # prediction is safe
    assert sh.intervention_log == [True]

    # a base action that stays safe passes straight through
    sh2 = ShieldedController(_const(5.0), _const(0.0), is_safe=safe,
                             blend="filter", predict=predict)
    assert sh2.update(np.array([0.0]), 0.1) == pytest.approx(5.0)
    assert sh2.intervention_log == [False]


def test_bare_callables_work_as_base_and_fallback():
    sh = ShieldedController(lambda x, dt: 3.0, lambda x, dt: -3.0,
                            is_safe=lambda x: x[0] < 1.0)
    assert sh.update(np.array([0.0]), 0.1) == 3.0
    assert sh.update(np.array([2.0]), 0.1) == -3.0
