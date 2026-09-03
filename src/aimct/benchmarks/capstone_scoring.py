"""
Capstone Evaluation Rubric and Multi-Controller Scoring Engine.

Conforms to docs/references/capstone-rubric.md for the Module 09 five-way quadrotor bake-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# Standard Capstone Composite Weights (spec §2 & §3)
CAPSTONE_WEIGHTS: dict[str, float] = {
    "rmse": 0.40,
    "energy": 0.20,
    "slew": 0.10,
    "safety": 0.15,
    "robustness": 0.15,
}

# Canonical Quadrotor Figure-8 Baselines
CAPSTONE_BASELINES: dict[str, float] = {
    "rmse": 0.050,      # 5 cm position tracking error
    "energy": 150.0,    # 150 N^2*s total thrust effort
    "slew": 2500.0,     # Actuator slew rate penalty
    "safety": 0.10,     # Boundary penalty threshold
}


def score_capstone_entry(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
    *,
    robust_factor: float = 1.0,
    safety_ok: bool = True,
    dq_reasons: Sequence[str] | None = None,
    max_ratio: float = 10.0,
) -> dict[str, object]:
    """
    Scores a single capstone controller evaluation against baseline costs.

    Formula (spec §3):
      S = 100 * exp( - [ 0.40 * r_pos + 0.20 * r_energy + 0.10 * r_slew + 0.15 * r_safety ] )
          * S_robust * I(No Hard Disqualifications)
    """
    w = dict(weights or CAPSTONE_WEIGHTS)
    base = dict(baseline or CAPSTONE_BASELINES)
    reasons = list(dq_reasons or [])

    m_rmse = float(metrics.get("rmse", metrics.get("pos_rmse", metrics.get("itae", 1.0))))
    m_energy = float(metrics.get("control_energy", metrics.get("energy", 100.0)))
    m_slew = float(metrics.get("slew_rate", metrics.get("slew", 1000.0)))
    m_safety = float(metrics.get("violation_penalty", metrics.get("safety", 0.0)))

    b_rmse = float(base.get("rmse", 0.05))
    b_energy = float(base.get("energy", 150.0))
    b_slew = float(base.get("slew", 2500.0))
    b_safety = float(base.get("safety", 0.10))

    # Capped cost ratios
    r_rmse = min(max_ratio, m_rmse / max(1e-6, b_rmse))
    r_energy = min(max_ratio, m_energy / max(1e-6, b_energy))
    r_slew = min(max_ratio, m_slew / max(1e-6, b_slew))
    r_safety = min(max_ratio, (m_safety + 1e-4) / max(1e-6, b_safety)) if m_safety > 0 else 0.0

    norm_cost = (
        w.get("rmse", 0.40) * r_rmse
        + w.get("energy", 0.20) * r_energy
        + w.get("slew", 0.10) * r_slew
        + w.get("safety", 0.15) * r_safety
    )

    perf_base = float(100.0 * np.exp(-norm_cost))
    r_factor = float(np.clip(robust_factor, 0.20, 1.0))

    if not safety_ok or len(reasons) > 0:
        status = "DISQUALIFIED"
        composite = 0.0
    else:
        status = "PASS"
        composite = float(perf_base * r_factor)

    terms = {
        "rmse_ratio": r_rmse,
        "energy_ratio": r_energy,
        "slew_ratio": r_slew,
        "safety_ratio": r_safety,
        "norm_cost": norm_cost,
    }

    breakdown = {
        "precision": float(np.exp(-r_rmse)),
        "effort": float(np.exp(-r_energy)),
        "smoothness": float(np.exp(-r_slew)),
        "safety": 1.0 if safety_ok else 0.0,
        "robustness": r_factor,
    }

    return {
        "composite": composite,
        "status": status,
        "terms": terms,
        "breakdown": breakdown,
        "metrics": dict(metrics),
        "dq_reasons": reasons,
    }


def score_capstone(
    metrics_per_controller: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """
    Evaluates and ranks all controllers in the five-way bake-off.
    """
    scores: dict[str, dict] = {}
    for name, m in metrics_per_controller.items():
        is_safe = m.get("hard_fail", 0.0) == 0.0
        robust = float(m.get("s_robust", m.get("robustness", 1.0)))
        dq_reasons = []
        if not is_safe:
            dq_reasons.append("Safety envelope or obstacle penetration breached.")
        if float(m.get("mean_latency_ms", 0.0)) > 2.0:
            dq_reasons.append(f"Step latency {m.get('mean_latency_ms'):.2f} ms exceeded 2.0 ms deadline.")

        scores[name] = score_capstone_entry(
            m,
            baseline=baseline,
            weights=weights,
            robust_factor=robust,
            safety_ok=is_safe and len(dq_reasons) == 0,
            dq_reasons=dq_reasons,
        )

    # Rank controllers by composite score descending
    ranked = sorted(scores.items(), key=lambda item: item[1]["composite"], reverse=True)
    winner = ranked[0][0] if ranked and ranked[0][1]["composite"] > 0 else "None"

    return {
        "scores": scores,
        "ranked": ranked,
        "winner": winner,
        "leaderboard_md": capstone_leaderboard_table(scores),
    }


def capstone_leaderboard_table(scores: Mapping[str, dict]) -> str:
    """
    Generates a Markdown leaderboard table for the Capstone report.
    """
    lines = [
        "| Rank | Controller Entry | Score / 100 | Status | Tracking RMSE [m] | Energy $E_u$ | Slew Rate | $S_{\\text{robust}}$ | Latency [ms] |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    ranked = sorted(scores.items(), key=lambda item: item[1]["composite"], reverse=True)
    for rank, (name, s) in enumerate(ranked, 1):
        m = s["metrics"]
        rmse_val = m.get("rmse", m.get("pos_rmse", float("nan")))
        energy_val = m.get("energy", m.get("control_energy", float("nan")))
        slew_val = m.get("slew", m.get("slew_rate", float("nan")))
        rob_val = m.get("s_robust", s["breakdown"]["robustness"])
        lat_val = m.get("mean_latency_ms", float("nan"))

        lines.append(
            f"| **{rank}** | **{name}** | **{s['composite']:.1f}** | `{s['status']}` | "
            f"{rmse_val:.4g} | {energy_val:.4g} | {slew_val:.4g} | {rob_val:.2f} | {lat_val:.2f} |"
        )

    return "\n".join(lines) + "\n"
