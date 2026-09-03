"""Numerical-correctness tests for :class:`aimct.controllers.PID`.

The suite checks each term in isolation, the discretisation choices (derivative
on measurement vs error, derivative filtering), the anti-windup behaviour, and a
couple of closed-loop properties that have closed-form answers so no control
library is needed.  A python-control cross-check is included but skipped when
``control`` is not importable (e.g. environments that block SciPy's compiled
extensions).
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import PID


# --------------------------------------------------------------------------- P

def test_proportional_only_is_gain_times_error():
    pid = PID(kp=2.0, setpoint=1.0, dt=0.1)
    # error = 1.0 - 0.25 = 0.75  ->  u = 2.0 * 0.75
    assert pid.update(0.25) == pytest.approx(1.5)
    # ki = kd = 0  ->  those terms contribute nothing and the integrator, whose
    # channel gain is zero, does not accumulate.
    assert pid.integral == pytest.approx(0.0)
    assert pid.terms[1] == pytest.approx(0.0)
    assert pid.terms[2] == pytest.approx(0.0)


def test_proportional_sign_follows_error():
    pid = PID(kp=1.0, setpoint=0.0, dt=0.1)
    assert pid.update(-3.0) == pytest.approx(3.0)   # y below setpoint -> push up
    assert pid.update(+3.0) == pytest.approx(-3.0)  # y above setpoint -> push down


# --------------------------------------------------------------------------- I

def test_integral_accumulates_linearly_for_constant_error():
    dt = 0.1
    pid = PID(ki=2.0, setpoint=1.0, dt=dt)
    err = 1.0
    for k in range(1, 6):
        u = pid.update(0.0)
        assert pid.integral == pytest.approx(err * dt * k)
        assert u == pytest.approx(2.0 * err * dt * k)


def test_integral_removes_steady_state_offset_p_would_leave():
    # P-only on a first-order plant y' = -a y + b u has a known static error:
    #   y_ss / r = (kp b) / (a + kp b)
    # Adding integral action must drive that offset to zero.
    a, b, r = 1.0, 1.0, 1.0
    dt = 1e-3
    pid = PID(kp=2.0, ki=5.0, setpoint=r, dt=dt)
    y = 0.0
    for _ in range(int(40 / dt)):
        u = pid.update(y)
        y += dt * (-a * y + b * u)
    assert y == pytest.approx(r, abs=1e-3)


# --------------------------------------------------------------------------- D

def test_derivative_on_error_responds_to_measurement_rate():
    dt = 0.5
    pid = PID(kd=2.0, setpoint=0.0, dt=dt, derivative_on="error")
    assert pid.update(0.0) == pytest.approx(0.0)   # first sample: no derivative
    # error goes 0 -> -1 over dt=0.5  =>  de/dt = -2  =>  u = kd * (-2) = -4
    assert pid.update(1.0) == pytest.approx(-4.0)
    # error constant now  =>  derivative back to zero
    assert pid.update(1.0) == pytest.approx(0.0)


def test_derivative_on_measurement_has_no_setpoint_kick():
    dt = 0.1
    pid = PID(kd=10.0, setpoint=0.0, dt=dt, derivative_on="measurement")
    pid.update(0.0)
    pid.update(0.0)
    # Jump the set-point while the measurement is unchanged: derivative-on-error
    # would spike, derivative-on-measurement must not.
    u = pid.update(0.0, setpoint=5.0)
    assert pid.terms[2] == pytest.approx(0.0)
    assert u == pytest.approx(0.0)


def test_derivative_first_call_is_zero_both_modes():
    for mode in ("measurement", "error"):
        pid = PID(kd=1.0, setpoint=1.0, dt=0.01, derivative_on=mode)
        pid.update(0.0)
        assert pid.terms[2] == pytest.approx(0.0)


def test_derivative_lowpass_attenuates_and_relaxes():
    dt, tau = 0.01, 0.1
    unfiltered = PID(kd=1.0, dt=dt, derivative_on="error")
    filtered = PID(kd=1.0, dt=dt, tau_d=tau, derivative_on="error")
    unfiltered.update(0.0)
    filtered.update(0.0)

    # Same ramp into both; filtered derivative must lag the raw one on the
    # first step and then approach it.
    # Long enough (>> tau/dt steps) that the filtered term fully catches up.
    ramp = [0.1 * k for k in range(1, 200)]
    d_unf = d_fil = None
    for i, y in enumerate(ramp):
        unfiltered.update(y)
        filtered.update(y)
        if i == 0:
            first_ratio = abs(filtered.terms[2]) / abs(unfiltered.terms[2])
        d_unf, d_fil = unfiltered.terms[2], filtered.terms[2]

    assert first_ratio == pytest.approx(dt / (tau + dt), rel=1e-6)
    assert d_fil == pytest.approx(d_unf, rel=1e-2)  # converged after many steps


# ------------------------------------------------------------------ saturation

def test_output_limits_saturate_command():
    pid = PID(kp=100.0, setpoint=1.0, dt=0.1, output_limits=(-5.0, 5.0))
    assert pid.update(0.0) == pytest.approx(5.0)
    assert pid.update(2.0) == pytest.approx(-5.0)


def test_one_sided_output_limit():
    pid = PID(kp=1.0, setpoint=0.0, dt=0.1, output_limits=(0.0, None))
    assert pid.update(10.0) == pytest.approx(0.0)   # would be -10, clamped at 0
    assert pid.update(-10.0) == pytest.approx(10.0)


def test_integral_limits_clamp_integral_term():
    dt = 0.1
    pid = PID(ki=1.0, setpoint=1.0, dt=dt, integral_limits=(-0.3, 0.3))
    for _ in range(50):
        pid.update(0.0)
    assert pid.terms[1] == pytest.approx(0.3)


# ------------------------------------------------------------------ anti-windup

def test_conditional_integration_freezes_integral_while_saturated():
    pid = PID(kp=1.0, ki=1.0, setpoint=10.0, dt=0.1, output_limits=(-1.0, 1.0))
    for _ in range(100):
        pid.update(0.0)          # huge positive error, command pinned at +1
    assert pid.integral == pytest.approx(0.0)   # never allowed to wind up

    # Reference: identical loop without limits winds up substantially.
    naive = PID(kp=1.0, ki=1.0, setpoint=10.0, dt=0.1)
    for _ in range(100):
        naive.update(0.0)
    assert naive.integral == pytest.approx(10.0 * 0.1 * 100)


def test_anti_windup_gives_faster_recovery_after_saturation():
    """Anti-windup must recover quicker once the set-point is reachable again."""
    a, b, dt = 1.0, 1.0, 0.01
    u_max = 2.0

    def settle_time(anti_windup: bool) -> float:
        pid = PID(kp=2.0, ki=6.0, dt=dt,
                  output_limits=(-u_max, u_max) if anti_windup else (None, None))
        y = 0.0
        t = 0.0
        t_settled = float("inf")
        while t < 60.0:
            pid.setpoint = 8.0 if t < 4.0 else 1.0   # 8.0 is unreachable (u_max=2)
            u = pid.update(y)
            if not anti_windup:                       # emulate a naive clamped actuator
                u = float(np.clip(u, -u_max, u_max))
            y += dt * (-a * y + b * u)
            t += dt
            if t >= 4.0:
                if abs(y - 1.0) < 0.02:
                    t_settled = min(t_settled, t - 4.0)
                else:
                    t_settled = float("inf")         # left the band again; keep waiting
        return t_settled

    t_aw = settle_time(anti_windup=True)
    t_naive = settle_time(anti_windup=False)
    assert np.isfinite(t_aw), "anti-windup loop never settled"
    assert t_aw < t_naive          # windup delays recovery (often to 'never' within 60 s)


# ----------------------------------------------------------------------- state

def test_reset_clears_all_internal_state():
    pid = PID(kp=1.0, ki=1.0, kd=1.0, setpoint=1.0, dt=0.1, derivative_on="error")
    for y in (0.0, 0.2, 0.5):
        pid.update(y)
    assert pid.integral != pytest.approx(0.0)

    pid.reset()
    assert pid.integral == pytest.approx(0.0)
    assert pid.terms == (0.0, 0.0, 0.0)
    # First post-reset call behaves like a fresh controller: no derivative yet.
    pid.update(0.0)
    assert pid.terms[2] == pytest.approx(0.0)
    assert pid.integral == pytest.approx(0.1)


def test_call_is_alias_for_update():
    a = PID(kp=1.5, ki=0.4, kd=0.2, setpoint=1.0, dt=0.05, derivative_on="error")
    b = PID(kp=1.5, ki=0.4, kd=0.2, setpoint=1.0, dt=0.05, derivative_on="error")
    for y in (0.0, 0.1, 0.35, 0.6, 0.9):
        assert a(y, 0.05) == pytest.approx(b.update(y, 0.05))


def test_setpoint_override_persists_on_instance():
    pid = PID(kp=1.0, dt=0.1)
    pid.update(0.0, setpoint=3.0)
    assert pid.setpoint == pytest.approx(3.0)
    assert pid.update(0.0) == pytest.approx(3.0)   # uses stored set-point


# --------------------------------------------------------------------- guards

def test_update_requires_dt_when_not_configured():
    pid = PID(kp=1.0)
    with pytest.raises(ValueError, match="dt"):
        pid.update(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"derivative_on": "bogus"},
        {"dt": -0.1},
        {"dt": 0.0},
        {"tau_d": -1.0},
    ],
)
def test_constructor_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        PID(kp=1.0, **kwargs)


def test_update_rejects_non_positive_dt():
    pid = PID(kp=1.0)
    with pytest.raises(ValueError):
        pid.update(0.0, dt=-0.01)


# ---------------------------------------------------------------------- vector

def test_vector_pid_runs_independent_per_channel_loops():
    pid = PID(
        kp=np.array([1.0, 2.0]),
        ki=np.array([0.0, 1.0]),
        setpoint=np.array([1.0, 1.0]),
        dt=0.1,
    )
    u = pid.update(np.array([0.0, 0.0]))
    # channel 0: pure P = 1.0 (ki=0, no accumulation)
    # channel 1: P + ki*(e*dt) = 2.0 + 1.0*(1.0*0.1) = 2.1  (integrates from step 1)
    assert u == pytest.approx([1.0, 2.1])
    u = pid.update(np.array([0.0, 0.0]))
    # channel 1: 2.0 + 1.0*(0.1*2) = 2.2
    assert u == pytest.approx([1.0, 2.2])


def test_vector_and_scalar_agree_channelwise():
    scalar = PID(kp=1.0, ki=0.5, kd=0.3, setpoint=1.0, dt=0.1, derivative_on="error")
    vector = PID(kp=1.0, ki=0.5, kd=0.3, setpoint=np.ones(3), dt=0.1,
                 derivative_on="error")
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.normal()
        us = scalar.update(y)
        uv = vector.update(np.full(3, y))
        assert uv == pytest.approx(np.full(3, us))


# ---------------------------------------------------- cross-check (python-control)

def test_pi_step_response_matches_python_control():
    import control

    kp, ki = 2.0, 3.0
    # plant G(s) = 1 / (s + 1)
    G = control.tf([1.0], [1.0, 1.0])
    C = control.tf([kp, ki], [1.0, 0.0])          # kp + ki/s
    closed = control.feedback(C * G, 1)

    dt = 1e-3
    t = np.arange(0.0, 8.0 + dt, dt)
    _, y_ref = control.forced_response(closed, T=t, U=np.ones_like(t))

    pid = PID(kp=kp, ki=ki, setpoint=1.0, dt=dt)
    y = 0.0
    ys = [0.0]
    for _ in range(len(t) - 1):
        u = pid.update(y)
        y += dt * (-y + u)                        # same G(s), explicit Euler
        ys.append(y)
    ys = np.asarray(ys)

    # Loose tolerance: our loop is explicit-Euler, python-control uses a stiff
    # ODE solver; they must still agree closely at this step size.
    assert np.max(np.abs(ys - y_ref)) < 5e-3
