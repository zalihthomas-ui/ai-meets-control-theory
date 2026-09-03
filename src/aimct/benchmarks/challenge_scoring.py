"""
Intelligent Control Challenge (ICC) — Scoring Rubric and Evaluation Engine.

Conforms to docs/references/challenge-spec.md §3 & §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..simulate import Trajectory
from .metrics import compute_all_metrics

# Standard Challenge Composite Weights (spec §4)
WEIGHTS: dict[str, float] = {
    "itae": 0.50,
    "energy": 0.30,
    "slew": 0.20,
}

# Display / metric normalization ranges for 0-1 per-dimension radar/breakdown
NORMALISERS: dict[str, tuple[float, float]] = {
    "itae": (0.1, 10.0),
    "energy": (1.0, 500.0),
    "slew": (0.1, 100.0),
    "latency_us": (10.0, 1000.0),
    "rmse": (0.01, 1.0),
}


def robust_degradation(j_nominal: float, j_perturbed_mean: float, floor: float = 0.20) -> float:
    """
    Computes robustness degradation factor in [floor, 1.0] (spec §3.4):
      S_robust = max(floor, 1.0 - (J_pert_mean - J_nom) / J_nom)
    """
    if j_nominal <= 0.0:
        return 1.0
    degradation = (j_perturbed_mean - j_nominal) / j_nominal
    raw_factor = 1.0 - max(0.0, degradation)
    return float(np.clip(max(floor, raw_factor), floor, 1.0))


def score_run(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    robust: float = 1.0,
    safety_ok: bool = True,
    weights: Mapping[str, float] | None = None,
    dq_reasons: Sequence[str] | None = None,
    max_ratio: float = 10.0,
) -> dict[str, object]:
    """
    Scores a single evaluation rollout against baseline costs (spec §4).

    Formula:
      S = 100 * exp( - [ w_itae * (J_itae / J_itae_base) +
                         w_energy * (J_energy / J_energy_base) +
                         w_slew * (J_slew / J_slew_base) ] )
          * S_robust * I(safety_ok)

    Individual cost ratios are capped at max_ratio (default 10.0) to prevent
    high-frequency numerical noise from causing exponential underflow to 0.0.

    Returns:
      {
        "composite": float (0-100),
        "status": "PASS" | "DQ_SAFETY" | "FAILED",
        "terms": {"itae_ratio": float, "energy_ratio": float, "slew_ratio": float, "norm_cost": float},
        "breakdown": {"performance": float, "effort": float, "smoothness": float, "robustness": float, "safety": float},
        "metrics": dict,
        "dq_reasons": list[str],
      }
    """
    w = dict(weights or WEIGHTS)
    reasons = list(dq_reasons or [])

    # Extract required keys (fallback aliases supported)
    m_itae = float(metrics.get("itae", metrics.get("j_itae", 1.0)))
    m_energy = float(metrics.get("control_energy", metrics.get("energy", metrics.get("j_energy", 1.0))))
    m_slew = float(metrics.get("slew_rate", metrics.get("slew", metrics.get("j_slew", 0.0))))

    b_itae = float(baseline.get("itae", baseline.get("j_itae", 1.0)))
    b_energy = float(baseline.get("control_energy", baseline.get("energy", baseline.get("j_energy", 1.0))))
    b_slew = float(baseline.get("slew_rate", baseline.get("slew", baseline.get("j_slew", 1.0))) or 1.0)

    # Ratio capping at max_ratio (default 10.0)
    r_itae = min(max_ratio, m_itae / max(1e-6, b_itae))
    r_energy = min(max_ratio, m_energy / max(1e-6, b_energy))
    r_slew = min(max_ratio, m_slew / max(1e-6, b_slew))

    norm_cost = (
        w.get("itae", 0.50) * r_itae
        + w.get("energy", 0.30) * r_energy
        + w.get("slew", 0.20) * r_slew
    )

    perf_base = float(100.0 * np.exp(-norm_cost))
    r_factor = float(np.clip(robust, 0.0, 1.0))

    if not safety_ok or len(reasons) > 0:
        status = "DQ_SAFETY" if not safety_ok else "FAILED"
        composite = 0.0
    else:
        status = "PASS"
        composite = float(perf_base * r_factor)

    terms = {
        "itae_ratio": r_itae,
        "energy_ratio": r_energy,
        "slew_ratio": r_slew,
        "norm_cost": norm_cost,
    }

    breakdown = {
        "performance": float(np.exp(-r_itae)),
        "effort": float(np.exp(-r_energy)),
        "smoothness": float(np.exp(-r_slew)),
        "robustness": r_factor,
        "safety": 1.0 if safety_ok else 0.0,
    }

    return {
        "composite": composite,
        "status": status,
        "terms": terms,
        "breakdown": breakdown,
        "metrics": dict(metrics),
        "dq_reasons": reasons,
    }


# ============================================================================
# Dataclasses & Helper APIs
# ============================================================================

@dataclass(frozen=True)
class ScoreWeights:
    w_itae: float = 0.50
    w_energy: float = 0.30
    w_slew: float = 0.20


@dataclass(frozen=True)
class BaselineCosts:
    j_itae: float
    j_energy: float
    j_slew: float


@dataclass
class ChallengeScoreResult:
    """Outcome of scoring a controller submission against the ICC rubric."""

    controller_name: str
    composite_score: float
    performance_score: float
    robustness_factor: float
    safety_preserved: bool
    disqualified: bool
    dq_reasons: list[str] = field(default_factory=list)
    raw_metrics: dict[str, float] = field(default_factory=dict)
    summary_text: str = ""


@dataclass(frozen=True)
class SafetyEnvelope:
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    action_limit: float | None = None
    max_step_latency_sec: float = 0.001


def evaluate_safety(
    trajectory: Trajectory,
    safety_envelope: SafetyEnvelope | None = None,
    step_latencies: Sequence[float] | np.ndarray | None = None,
) -> tuple[bool, list[str], float]:
    dq_reasons: list[str] = []
    barrier_penalty: float = 0.0

    if trajectory.diverged or not np.all(np.isfinite(trajectory.x)) or not np.all(np.isfinite(trajectory.u)):
        dq_reasons.append("State or action diverged / non-finite values.")
        return False, dq_reasons, float("inf")

    if safety_envelope is None:
        return True, dq_reasons, 0.0

    x = trajectory.x
    if safety_envelope.state_min is not None:
        s_min = np.asarray(safety_envelope.state_min)
        under = s_min - x
        violations = np.maximum(0.0, under)
        if np.any(violations > 1e-4):
            dq_reasons.append(f"State lower bound violated: observed {np.min(x, axis=0)} < {s_min}")
            barrier_penalty += float(np.sum(violations**2))

    if safety_envelope.state_max is not None:
        s_max = np.asarray(safety_envelope.state_max)
        over = x - s_max
        violations = np.maximum(0.0, over)
        if np.any(violations > 1e-4):
            dq_reasons.append(f"State upper bound violated: observed {np.max(x, axis=0)} > {s_max}")
            barrier_penalty += float(np.sum(violations**2))

    if safety_envelope.action_limit is not None:
        u_max = np.max(np.abs(trajectory.u))
        if u_max > safety_envelope.action_limit * 1.02:
            dq_reasons.append(f"Actuator limit breached: peak control {u_max:.3f} > {safety_envelope.action_limit:.3f}")

    if step_latencies is not None and len(step_latencies) > 0:
        lat_arr = np.asarray(step_latencies, dtype=float)
        max_lat = float(np.max(lat_arr))
        if max_lat > safety_envelope.max_step_latency_sec * 2.0:
            dq_reasons.append(
                f"Step latency timeout: max {max_lat*1e3:.2f} ms > {safety_envelope.max_step_latency_sec*1e3:.2f} ms"
            )

    return len(dq_reasons) == 0, dq_reasons, barrier_penalty
