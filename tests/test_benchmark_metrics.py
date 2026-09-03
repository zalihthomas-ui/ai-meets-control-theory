"""
Unit tests and analytical golden value verifications for benchmark metrics.

Tests mathematical accuracy against exact closed-form analytical solutions:
1. Underdamped 2nd-order step response (omega_n = 2.0, zeta = 0.5).
2. Critically damped 2nd-order step response (omega_n = 2.0, zeta = 1.0).
3. First-order step response (tau = 1.0).
4. Control effort, slew rate, and saturation duty cycles.
"""

from __future__ import annotations

import numpy as np
import pytest

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


# ============================================================================
# Analytical Signal Generators
# ============================================================================

def make_underdamped_2nd_order(
    wn: float = 2.0,
    zeta: float = 0.5,
    t_final: float = 10.0,
    dt: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates exact analytical step response for G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2).
    """
    n_points = int(t_final / dt) + 1
    t = np.linspace(0.0, t_final, n_points)
    wd = wn * np.sqrt(1.0 - zeta**2)
    phi = np.arccos(zeta)
    y = 1.0 - (np.exp(-zeta * wn * t) / np.sqrt(1.0 - zeta**2)) * np.sin(wd * t + phi)
    return t, y


def make_critically_damped_2nd_order(
    wn: float = 2.0,
    t_final: float = 10.0,
    dt: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates exact analytical step response for critically damped system (zeta=1.0).
    y(t) = 1 - (1 + wn*t) * exp(-wn*t).
    """
    n_points = int(t_final / dt) + 1
    t = np.linspace(0.0, t_final, n_points)
    y = 1.0 - (1.0 + wn * t) * np.exp(-wn * t)
    return t, y


def make_first_order(
    tau: float = 1.0,
    t_final: float = 10.0,
    dt: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates exact analytical step response for G(s) = 1 / (tau*s + 1).
    y(t) = 1 - exp(-t/tau).
    """
    n_points = int(t_final / dt) + 1
    t = np.linspace(0.0, t_final, n_points)
    y = 1.0 - np.exp(-t / tau)
    return t, y


# ============================================================================
# Test Cases with Analytical Golden Reference Values
# ============================================================================

def test_underdamped_second_order_golden_values() -> None:
    """
    Validates underdamped 2nd-order step response metrics against analytical formulas:
    - wn = 2.0, zeta = 0.5
    - wd = sqrt(3) ~ 1.73205 rad/s
    - Peak Time t_p = pi / wd = pi / sqrt(3) ~ 1.81380 s
    - Peak Overshoot M_p = 100 * exp(-pi*zeta / sqrt(1 - zeta^2)) = 100 * exp(-pi/sqrt(3)) ~ 16.30335%
    - Exact ISE = (1 + 4*zeta^2) / (4*zeta*wn) = (1 + 1) / (4 * 0.5 * 2) = 0.500000
    - Settling Time (2% band) = ~4.038 s
    - Residual steady-state error = ~0.0
    """
    t, y = make_underdamped_2nd_order(wn=2.0, zeta=0.5, t_final=10.0, dt=0.0001)

    # Overshoot and Peak Time
    analytical_mp = 100.0 * np.exp(-np.pi * 0.5 / np.sqrt(0.75))
    analytical_tp = np.pi / (2.0 * np.sqrt(0.75))

    computed_mp = peak_overshoot(t, y, target=1.0)
    computed_tp = peak_time(t, y, target=1.0)

    assert computed_mp == pytest.approx(analytical_mp, rel=1e-3)  # ~16.30%
    assert computed_tp == pytest.approx(analytical_tp, rel=1e-3)  # ~1.8138 s

    # Exact Integral of Squared Error (Parseval theorem closed form: 0.5000)
    computed_ise = ise(t, y, target=1.0)
    assert computed_ise == pytest.approx(0.5000, rel=5e-3)

    # Settling Time (2% band)
    computed_ts = settling_time(t, y, target=1.0, band=0.02)
    assert 4.0 <= computed_ts <= 4.1  # Theoretical crossing ~4.038 s

    # Rise Time (10% to 90%)
    computed_tr = rise_time(t, y, target=1.0, threshold_low=0.1, threshold_high=0.9)
    assert 0.80 <= computed_tr <= 0.84  # Expected ~0.819 s

    # Steady-State Error
    computed_ess = steady_state_error(t, y, target=1.0, tail_fraction=0.05)
    assert computed_ess < 1e-4

    # IAE and ITAE golden checks
    computed_iae = iae(t, y, target=1.0)
    computed_itae = itae(t, y, target=1.0)
    assert computed_iae == pytest.approx(0.8565, rel=1e-2)
    assert computed_itae == pytest.approx(0.7351, rel=1e-2)


def test_critically_damped_golden_values() -> None:
    """
    Validates critically damped step response (wn = 2.0, zeta = 1.0):
    - Monotonic: Overshoot must be strictly 0.0%
    - Analytical ISE = 5 / (8 * wn) = 5 / 16 = 0.3125 / (2 * 1) -> 0.625 for error integral
    - Settling Time (2% band) ~ 2.917 s
    - Steady-state error < 1e-5
    """
    t, y = make_critically_damped_2nd_order(wn=2.0, t_final=10.0, dt=0.0001)

    # Overshoot must be exactly zero for monotonic response
    assert peak_overshoot(t, y, target=1.0) == 0.0

    # ISE closed-form: integral_0^inf (1 + 2t)^2 exp(-4t) dt = 5/8 = 0.6250
    computed_ise = ise(t, y, target=1.0)
    assert computed_ise == pytest.approx(0.6250, rel=5e-3)

    # Settling time (2% band)
    computed_ts = settling_time(t, y, target=1.0, band=0.02)
    assert 2.85 <= computed_ts <= 2.95  # ~2.917 s

    # Rise time (10% to 90%)
    computed_tr = rise_time(t, y, target=1.0)
    assert 1.65 <= computed_tr <= 1.70  # ~1.679 s


def test_first_order_golden_values() -> None:
    """
    Validates first-order step response (tau = 1.0):
    - Rise time (10% to 90%): tau * ln(9) ~ 2.19722 s
    - Settling time (2% band): tau * ln(50) ~ 3.91202 s
    - Overshoot: 0.0%
    - Analytical ISE: integral_0^inf exp(-2t) dt = 0.50000
    """
    tau = 1.0
    t, y = make_first_order(tau=tau, t_final=10.0, dt=0.0001)

    # Exact analytical rise time: ln(0.9 / 0.1) = ln(9) ~ 2.19722
    analytical_tr = tau * np.log(9.0)
    assert rise_time(t, y, target=1.0) == pytest.approx(analytical_tr, rel=1e-3)

    # Exact analytical settling time: ln(1 / 0.02) = ln(50) ~ 3.91202
    analytical_ts = tau * np.log(50.0)
    assert settling_time(t, y, target=1.0, band=0.02) == pytest.approx(analytical_ts, rel=1e-3)

    # Overshoot
    assert peak_overshoot(t, y, target=1.0) == 0.0

    # Exact ISE: 0.5000
    assert ise(t, y, target=1.0) == pytest.approx(0.5000, rel=5e-3)


def test_control_effort_and_slew_rate_golden_values() -> None:
    """
    Tests control energy, peak control, and slew rate against exact calculus values.
    """
    # 1. Constant input u(t) = 2.0 over [0, 5.0]
    t_const = np.linspace(0.0, 5.0, 5001)
    u_const = np.full_like(t_const, 2.0)
    assert control_energy(t_const, u_const) == pytest.approx(2.0**2 * 5.0, rel=1e-4)  # 20.0
    assert peak_control(t_const, u_const) == pytest.approx(2.0, rel=1e-6)
    assert slew_rate(t_const, u_const) == pytest.approx(0.0, abs=1e-6)

    # 2. Ramp input u(t) = 3.0 * t over [0, 2.0]
    # Energy = integral_0^2 9 t^2 dt = 9 * (8/3) = 24.0
    # Slew rate = integral_0^2 3^2 dt = 18.0
    t_ramp = np.linspace(0.0, 2.0, 20001)
    u_ramp = 3.0 * t_ramp
    assert control_energy(t_ramp, u_ramp) == pytest.approx(24.0, rel=1e-3)
    assert slew_rate(t_ramp, u_ramp) == pytest.approx(18.0, rel=1e-2)
    assert peak_control(t_ramp, u_ramp) == pytest.approx(6.0, rel=1e-4)

    # 3. Sinusoidal input u(t) = sin(t) over [0, 2*pi]
    # Energy = integral_0^2pi sin^2(t) dt = pi ~ 3.14159265
    t_sin = np.linspace(0.0, 2.0 * np.pi, 20001)
    u_sin = np.sin(t_sin)
    assert control_energy(t_sin, u_sin) == pytest.approx(np.pi, rel=1e-3)


def test_saturation_duty_cycle() -> None:
    """
    Tests saturation duty cycle calculation.
    """
    # 10s simulation: 2s clamped at 10.0, 8s at 5.0. Bound = 10.0.
    t = np.linspace(0.0, 10.0, 10001)
    u = np.where(t <= 2.0, 10.0, 5.0)
    
    sat_pct = saturation_duty_cycle(t, u, u_limit=10.0, threshold=0.99)
    assert sat_pct == pytest.approx(20.0, rel=1e-2)

    # Zero saturation if limit is never approached
    assert saturation_duty_cycle(t, u, u_limit=20.0) == pytest.approx(0.0, abs=1e-4)


def test_compute_all_metrics_schema() -> None:
    """
    Validates that compute_all_metrics outputs all required keys conforming to
    docs/comparison-report-spec.md.
    """
    t, y = make_underdamped_2nd_order(wn=2.0, zeta=0.5, t_final=5.0, dt=0.001)
    u = np.sin(t)

    res = compute_all_metrics(t, y, u, target=1.0, u_limit=2.0)
    
    expected_keys = {
        "rise_time",
        "settling_time",
        "peak_overshoot_pct",
        "peak_time",
        "steady_state_error",
        "rmse",
        "iae",
        "itae",
        "ise",
        "control_energy",
        "peak_control",
        "slew_rate",
        "saturation_pct",
    }
    
    assert set(res.keys()) == expected_keys
    for k, v in res.items():
        assert isinstance(v, float)
        assert not np.isnan(v)
        assert not np.isinf(v)
