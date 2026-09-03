"""
Unit tests for the Module 09 Capstone evaluation rubric and scoring engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.benchmarks.capstone_scoring import (
    CAPSTONE_BASELINES,
    CAPSTONE_WEIGHTS,
    capstone_leaderboard_table,
    score_capstone,
    score_capstone_entry,
)


def test_score_capstone_entry_nominal() -> None:
    # Baseline matching costs
    metrics_match = {
        "rmse": 0.050,
        "energy": 150.0,
        "slew": 2500.0,
        "violation_penalty": 0.0,
    }

    # norm_cost = 0.40*(1) + 0.20*(1) + 0.10*(1) + 0.15*(0) = 0.70
    # composite = 100 * exp(-0.70) = 49.6585...
    res = score_capstone_entry(metrics_match, robust_factor=1.0, safety_ok=True)
    assert res["status"] == "PASS"
    assert res["composite"] == pytest.approx(100.0 * np.exp(-0.70), rel=1e-3)
    assert res["terms"]["norm_cost"] == pytest.approx(0.70, rel=1e-3)


def test_score_capstone_entry_disqualification() -> None:
    metrics = {"rmse": 0.02, "energy": 100.0, "slew": 1000.0, "violation_penalty": 5.0}

    # Hard safety violation
    res_dq = score_capstone_entry(metrics, safety_ok=False, dq_reasons=["Obstacle collision"])
    assert res_dq["status"] == "DISQUALIFIED"
    assert res_dq["composite"] == 0.0
    assert "Obstacle collision" in res_dq["dq_reasons"]


def test_score_capstone_five_way_bakeoff() -> None:
    bakeoff_metrics = {
        "LQR+Flatness": {
            "rmse": 0.040,
            "energy": 120.0,
            "slew": 2000.0,
            "violation_penalty": 0.0,
            "s_robust": 0.85,
            "mean_latency_ms": 0.02,
            "hard_fail": 0.0,
        },
        "Linear MPC Preview": {
            "rmse": 0.025,
            "energy": 135.0,
            "slew": 2200.0,
            "violation_penalty": 0.0,
            "s_robust": 0.90,
            "mean_latency_ms": 0.35,
            "hard_fail": 0.0,
        },
        "Learned Sampling MPC": {
            "rmse": 0.080,
            "energy": 180.0,
            "slew": 3200.0,
            "violation_penalty": 0.0,
            "s_robust": 0.95,
            "mean_latency_ms": 1.80,
            "hard_fail": 0.0,
        },
        "Deep RL (PPO)": {
            "rmse": 0.120,
            "energy": 220.0,
            "slew": 6500.0,
            "violation_penalty": 0.0,
            "s_robust": 0.40,
            "mean_latency_ms": 0.08,
            "hard_fail": 0.0,
        },
        "Shielded Hybrid RL": {
            "rmse": 0.060,
            "energy": 150.0,
            "slew": 2800.0,
            "violation_penalty": 0.0,
            "s_robust": 0.88,
            "mean_latency_ms": 0.25,
            "hard_fail": 0.0,
        },
    }

    eval_result = score_capstone(bakeoff_metrics)
    assert len(eval_result["ranked"]) == 5
    # Linear MPC should lead due to highest tracking precision & strong robustness
    assert eval_result["winner"] == "Linear MPC Preview"

    table_md = eval_result["leaderboard_md"]
    assert "| Rank | Controller Entry |" in table_md
    assert "Linear MPC Preview" in table_md
    assert "Deep RL (PPO)" in table_md
