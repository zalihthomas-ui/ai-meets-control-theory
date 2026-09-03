"""
Unit tests for Intelligent Control Challenge (ICC) scoring engine and wrappers.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.benchmarks.challenge_scoring import (
    NORMALISERS,
    WEIGHTS,
    BaselineCosts,
    SafetyEnvelope,
    ScoreWeights,
    evaluate_safety,
    robust_degradation,
    score_run,
)
from aimct.benchmarks.challenge_wrappers import (
    ActuatorLag,
    BlackBoxPlant,
    ImpulseDisturbance,
    ImpulseInjector,
    ParamPerturbed,
    perturbed_system,
)
from aimct.systems.cartpole import CartPole
from aimct.systems.linear import LinearSystem
from aimct.systems.mass_spring_damper import MassSpringDamper


# ============================================================================
# Scoring Tests
# ============================================================================

def test_challenge_score_run_nominal_and_degradation() -> None:
    baseline = {"itae": 10.0, "control_energy": 100.0, "slew_rate": 50.0}
    metrics_match = {"itae": 10.0, "control_energy": 100.0, "slew_rate": 50.0}

    # At baseline: norm_cost = 0.50*(1) + 0.30*(1) + 0.20*(1) = 1.0
    # composite = 100 * exp(-1.0) = 36.7879...
    res = score_run(metrics_match, baseline, robust=1.0, safety_ok=True)
    assert res["status"] == "PASS"
    assert res["composite"] == pytest.approx(100.0 * np.exp(-1.0), rel=1e-3)
    assert res["terms"]["norm_cost"] == pytest.approx(1.0, rel=1e-3)

    # Test with robustness degradation
    res_rob = score_run(metrics_match, baseline, robust=0.80, safety_ok=True)
    assert res_rob["composite"] == pytest.approx(100.0 * np.exp(-1.0) * 0.80, rel=1e-3)


def test_challenge_score_run_disqualification() -> None:
    baseline = {"itae": 10.0, "control_energy": 100.0, "slew_rate": 50.0}
    metrics = {"itae": 5.0, "control_energy": 50.0, "slew_rate": 25.0}

    # Safety violation must result in 0.0 composite score
    res_unsafe = score_run(metrics, baseline, safety_ok=False)
    assert res_unsafe["status"] == "DQ_SAFETY"
    assert res_unsafe["composite"] == 0.0

    # Explicit DQ reason
    res_dq = score_run(metrics, baseline, safety_ok=True, dq_reasons=["Timeout exceeded"])
    assert res_dq["status"] == "FAILED"
    assert res_dq["composite"] == 0.0
    assert "Timeout exceeded" in res_dq["dq_reasons"]


def test_robust_degradation_calculation() -> None:
    # 0% degradation
    assert robust_degradation(10.0, 10.0) == 1.0

    # 50% degradation -> 0.50
    assert robust_degradation(10.0, 15.0) == pytest.approx(0.50, rel=1e-3)

    # 100% or worse degradation floored at 0.20
    assert robust_degradation(10.0, 20.0) == 0.20
    assert robust_degradation(10.0, 50.0) == 0.20

    # Better performance under perturbation clamped to 1.0
    assert robust_degradation(10.0, 8.0) == 1.0


def test_score_run_ratio_capping() -> None:
    baseline = {"itae": 1.0, "control_energy": 1.0, "slew_rate": 1.0}
    # Extreme noise on slew rate: ratio = 10000.0 -> capped at 10.0
    metrics_noisy = {"itae": 1.0, "control_energy": 1.0, "slew_rate": 10000.0}
    res = score_run(metrics_noisy, baseline, max_ratio=10.0)
    assert res["terms"]["slew_ratio"] == 10.0
    # norm_cost = 0.5*1 + 0.3*1 + 0.2*10 = 2.8 -> composite = 100*exp(-2.8) > 0
    assert res["composite"] > 5.0


# ============================================================================
# Wrapper Tests
# ============================================================================

def test_param_perturbed_wrapper() -> None:
    sys = MassSpringDamper(m=1.0, c=0.5, k=5.0)
    pert = ParamPerturbed(sys, scale=0.30, seed=42)

    # Check that perturbed mass, damping, stiffness are within +-30%
    assert 0.70 <= pert.base_system.m <= 1.30
    assert 0.35 <= pert.base_system.c <= 0.65
    assert 3.50 <= pert.base_system.k <= 6.50

    # Check dynamics execution
    x = np.array([1.0, 0.0])
    u = np.array([0.0])
    xdot = pert.dynamics(0.0, x, u)
    assert len(xdot) == 2
    assert np.all(np.isfinite(xdot))

    # Test functional constructor
    pert_func = perturbed_system(lambda: MassSpringDamper(m=2.0), frac=0.20, rng=10)
    assert isinstance(pert_func, ParamPerturbed)


def test_actuator_lag_wrapper() -> None:
    sys = MassSpringDamper(m=1.0, c=0.5, k=5.0)
    lag_sys = ActuatorLag(sys, tau_a=0.05)

    assert lag_sys.n_states == sys.n_states + sys.n_inputs  # 2 + 1 = 3
    assert lag_sys.n_inputs == 1

    # State x_aug = [pos, vel, u_act]
    x_aug = np.array([0.0, 0.0, 0.0])
    u_cmd = np.array([2.0])

    # At t=0, u_act = 0 -> u_dot_act = (2.0 - 0.0) / 0.05 = 40.0
    xdot = lag_sys.dynamics(0.0, x_aug, u_cmd)
    assert xdot[2] == pytest.approx(40.0, rel=1e-3)


def test_impulse_injector() -> None:
    inj = ImpulseInjector(rng=42, b_scale=5.0, rate_hz=2.0, duration=0.05, t_final=5.0)
    # Ensure callable works and returns 1D array
    out = inj(1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1,)


def test_blackbox_plant_environment() -> None:
    sys = CartPole()
    env = BlackBoxPlant(
        system=sys,
        dt=0.001,
        max_step_budget=10,
        action_limit=20.0,
        state_safety_bounds=(np.array([-2.4, -10.0, -1.0, -10.0]), np.array([2.4, 10.0, 1.0, 10.0])),
    )

    assert env.n_states == 4
    assert env.n_inputs == 1
    assert env.action_bounds == (-20.0, 20.0)

    obs = env.reset([0.0, 0.0, 0.05, 0.0])
    assert len(obs) == 4

    # Execute 5 valid steps
    for _ in range(5):
        obs, cost, done, info = env.step([1.0])
        assert not done
        assert not env.is_disqualified

    # Test safety envelope breach (rail boundary violation)
    env.reset([2.5, 0.0, 0.0, 0.0])  # Starts outside [-2.4, +2.4]
    obs, cost, done, info = env.step([0.0])
    assert done
    assert env.is_disqualified
    assert len(env.dq_reasons) > 0
