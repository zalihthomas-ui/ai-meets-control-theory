"""
Benchmark evaluation metrics for dynamical systems and control algorithms.

Conforms to docs/comparison-report-spec.md.
"""

from __future__ import annotations

import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Numerical integration compatible across all NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # type: ignore[attr-defined]


def rise_time(
    t: np.ndarray,
    y: np.ndarray,
    target: float = 1.0,
    threshold_low: float = 0.1,
    threshold_high: float = 0.9,
) -> float:
    """
    Computes rise time t_r = t_90% - t_10% for a step response.
    
    Args:
        t: 1D time array.
        y: 1D response array.
        target: Target step setpoint value.
        threshold_low: Lower fraction (default 0.1 for 10%).
        threshold_high: Upper fraction (default 0.9 for 90%).
        
    Returns:
        Rise time in seconds. Returns 0.0 if thresholds are not crossed.
    """
    if target == 0.0:
        return 0.0
        
    y_norm = y / target
    idx_low = np.where(y_norm >= threshold_low)[0]
    idx_high = np.where(y_norm >= threshold_high)[0]
    
    if len(idx_low) == 0 or len(idx_high) == 0:
        return 0.0
        
    t_low = float(t[idx_low[0]])
    t_high = float(t[idx_high[0]])
    return max(0.0, t_high - t_low)


def settling_time(
    t: np.ndarray,
    y: np.ndarray,
    target: float = 1.0,
    band: float = 0.02,
) -> float:
    """
    Computes settling time t_s: earliest time after which |y(t) - target| <= band * |target|.
    
    Args:
        t: 1D time array.
        y: 1D response array.
        target: Target setpoint value.
        band: Error tolerance fraction (default 0.02 for 2% band).
        
    Returns:
        Settling time in seconds. Returns total duration if not settled.
    """
    error = np.abs(y - target)
    threshold = band * abs(target) if target != 0.0 else band
    
    out_of_band = np.where(error > threshold)[0]
    if len(out_of_band) == 0:
        return float(t[0])
        
    last_out_idx = out_of_band[-1]
    if last_out_idx + 1 < len(t):
        return float(t[last_out_idx + 1])
    return float(t[-1])


def peak_overshoot(
    t: np.ndarray,
    y: np.ndarray,
    target: float = 1.0,
) -> float:
    """
    Computes peak percentage overshoot M_p [%].
    
    Args:
        t: 1D time array.
        y: 1D response array.
        target: Target setpoint value.
        
    Returns:
        Peak overshoot in percent (e.g. 16.3 for 16.3%). Returns 0.0 if no overshoot.
    """
    if target == 0.0:
        return 0.0
    max_val = np.max(y)
    overshoot = (max_val - target) / abs(target) * 100.0
    return float(max(0.0, overshoot))


def peak_time(
    t: np.ndarray,
    y: np.ndarray,
    target: float = 1.0,
) -> float:
    """
    Computes peak time t_p: time at which response reaches maximum absolute value or overshoot.
    """
    idx_max = int(np.argmax(y if target >= 0 else -y))
    return float(t[idx_max])


def steady_state_error(
    t: np.ndarray,
    y: np.ndarray,
    target: float = 1.0,
    tail_fraction: float = 0.05,
) -> float:
    """
    Computes residual steady-state error e_ss averaged over final tail_fraction of trajectory.
    """
    n_tail = max(1, int(tail_fraction * len(y)))
    return float(np.mean(np.abs(target - y[-n_tail:])))


def rmse(
    t: np.ndarray,
    y: np.ndarray,
    target: float | np.ndarray = 1.0,
) -> float:
    """
    Computes Root Mean Square Tracking Error across the trajectory.
    """
    error = target - y
    return float(np.sqrt(np.mean(error**2)))


def iae(
    t: np.ndarray,
    y: np.ndarray,
    target: float | np.ndarray = 1.0,
) -> float:
    """
    Computes Integral of Absolute Error: integral |r(t) - y(t)| dt.
    """
    error = np.abs(target - y)
    return _trapz(error, t)


def itae(
    t: np.ndarray,
    y: np.ndarray,
    target: float | np.ndarray = 1.0,
) -> float:
    """
    Computes Integral of Time-weighted Absolute Error: integral t * |r(t) - y(t)| dt.
    """
    error = np.abs(target - y)
    return _trapz(t * error, t)


def ise(
    t: np.ndarray,
    y: np.ndarray,
    target: float | np.ndarray = 1.0,
) -> float:
    """
    Computes Integral of Squared Error: integral (r(t) - y(t))^2 dt.
    """
    error = target - y
    return _trapz(error**2, t)


def control_energy(
    t: np.ndarray,
    u: np.ndarray,
) -> float:
    """
    Computes control effort / energy: integral ||u(t)||^2 dt.
    """
    u_arr = np.asarray(u)
    if u_arr.ndim == 1:
        u_squared = u_arr**2
    else:
        u_squared = np.sum(u_arr**2, axis=1)
    return _trapz(u_squared, t)


def peak_control(
    t: np.ndarray,
    u: np.ndarray,
) -> float:
    """
    Computes maximum instantaneous control effort: max ||u(t)||_inf.
    """
    return float(np.max(np.abs(u)))


def slew_rate(
    t: np.ndarray,
    u: np.ndarray,
) -> float:
    """
    Computes actuator slew rate / jerk penalty: integral ||du/dt||^2 dt.
    """
    u_arr = np.asarray(u)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    u_diff = np.gradient(u_arr, dt, axis=0)
    if u_diff.ndim == 1:
        slew_sq = u_diff**2
    else:
        slew_sq = np.sum(u_diff**2, axis=1)
    return _trapz(slew_sq, t)


def saturation_duty_cycle(
    t: np.ndarray,
    u: np.ndarray,
    u_limit: float,
    threshold: float = 0.99,
) -> float:
    """
    Computes saturation percentage: percentage of simulation time |u(t)| >= threshold * u_limit.
    """
    if u_limit <= 0.0:
        return 0.0
    u_abs = np.abs(u)
    saturated = (u_abs >= threshold * u_limit).astype(float)
    total_time = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    sat_time = _trapz(saturated, t)
    return float((sat_time / total_time) * 100.0)


def compute_all_metrics(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    target: float = 1.0,
    u_limit: float | None = None,
) -> dict[str, float]:
    """
    Evaluates all standard performance metrics and returns a summary dict conforming to
    docs/comparison-report-spec.md.
    """
    metrics = {
        "rise_time": rise_time(t, y, target),
        "settling_time": settling_time(t, y, target),
        "peak_overshoot_pct": peak_overshoot(t, y, target),
        "peak_time": peak_time(t, y, target),
        "steady_state_error": steady_state_error(t, y, target),
        "rmse": rmse(t, y, target),
        "iae": iae(t, y, target),
        "itae": itae(t, y, target),
        "ise": ise(t, y, target),
        "control_energy": control_energy(t, u),
        "peak_control": peak_control(t, u),
        "slew_rate": slew_rate(t, u),
    }
    if u_limit is not None:
        metrics["saturation_pct"] = saturation_duty_cycle(t, u, u_limit)
    return metrics
