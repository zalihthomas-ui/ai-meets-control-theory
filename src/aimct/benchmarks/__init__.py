"""
Benchmark suites, metrics, and comparison harnesses.
"""

from aimct.benchmarks.harness import ComparisonResult, compare
from aimct.benchmarks.sweep import SweepResult, sweep
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
]
