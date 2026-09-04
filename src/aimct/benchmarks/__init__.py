"""
Benchmark suites, metrics, and comparison harnesses.
"""

from aimct.benchmarks.capstone_scoring import (
    CAPSTONE_BASELINES,
    CAPSTONE_WEIGHTS,
    capstone_leaderboard_table,
    score_capstone,
    score_capstone_entry,
)
from aimct.benchmarks.challenge_scoring import (
    NORMALISERS,
    WEIGHTS,
    BaselineCosts,
    ChallengeScoreResult,
    SafetyEnvelope,
    ScoreWeights,
    evaluate_safety,
    robust_degradation,
    score_run,
)
from aimct.benchmarks.challenge_wrappers import (
    ActuatorLag,
    BlackBoxEnvironment,
    BlackBoxPlant,
    ImpulseDisturbance,
    ImpulseInjector,
    ParamPerturbed,
    perturbed_system,
)
from aimct.benchmarks.harness import ComparisonResult, compare
from aimct.benchmarks.metrics import (
    compute_all_metrics,
    control_energy,
    iae,
    ise,
    itae,
    peak_control,
    peak_overshoot,
    peak_time,
    rise_time,
    rmse,
    saturation_duty_cycle,
    settling_time,
    slew_rate,
    steady_state_error,
)
from aimct.benchmarks.sweep import SweepResult, sweep
from aimct.benchmarks.tracking import TrackingResult, track_trajectory

__all__ = [
    "rise_time",
    "settling_time",
    "peak_overshoot",
    "peak_time",
    "steady_state_error",
    "rmse",
    "iae",
    "itae",
    "ise",
    "control_energy",
    "peak_control",
    "slew_rate",
    "saturation_duty_cycle",
    "compute_all_metrics",
    "compare",
    "ComparisonResult",
    "sweep",
    "SweepResult",
    "track_trajectory",
    "TrackingResult",
    # Challenge scoring & wrappers
    "WEIGHTS",
    "NORMALISERS",
    "ScoreWeights",
    "BaselineCosts",
    "SafetyEnvelope",
    "ChallengeScoreResult",
    "robust_degradation",
    "evaluate_safety",
    "score_run",
    "ParamPerturbed",
    "perturbed_system",
    "ActuatorLag",
    "ImpulseDisturbance",
    "ImpulseInjector",
    "BlackBoxPlant",
    "BlackBoxEnvironment",
    # Capstone scoring & bake-off
    "CAPSTONE_WEIGHTS",
    "CAPSTONE_BASELINES",
    "score_capstone_entry",
    "score_capstone",
    "capstone_leaderboard_table",
]
