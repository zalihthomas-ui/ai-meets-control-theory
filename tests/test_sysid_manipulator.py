"""Manipulator identification: exact base-parameter recovery, arm round-trip,
derivative estimation, excitation diagnostics, and a noisy finite-difference log.
"""

import numpy as np
import pytest

from aimct.simulate import simulate
from aimct.sysid import (
    finite_difference_derivatives,
    identify_manipulator,
    manipulator_regressor,
)
from aimct.systems import TwoLinkArm

# --- true base parameters of the default TwoLinkArm (see manipulator.py) ------
TRUE = {
    "p1": 1.0 * 0.25**2 + 0.8 * (0.5**2 + 0.20**2) + 0.02083 + 0.01067,
    "p2": 0.8 * 0.5 * 0.20,
    "p3": 0.8 * 0.20**2 + 0.01067,
    "p4": 1.0 * 0.25 + 0.8 * 0.5,
    "p5": 0.8 * 0.20,
    "b1": 0.10,
    "b2": 0.10,
}


def _excitation_log(arm, *, dt=0.002, t_final=12.0, seed=0):
    """PD-track a joint-space multi-sine -> a rich (q, dq, ddq, tau) log with
    torque well inside the actuator limits and ddq taken exactly from the model."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.3, 1.6, size=(2, 4))
    phase = rng.uniform(0, 2 * np.pi, size=(2, 4))
    amp = np.array([0.5, 0.7])
    bias = np.array([np.pi / 2, -0.3])

    def ref(t):
        s = np.array([np.sum(np.sin(2 * np.pi * freqs[j] * t + phase[j])) for j in range(2)])
        return bias + amp * s / 4.0

    Kp, Kd = np.array([40.0, 25.0]), np.array([8.0, 5.0])

    def ctrl(y, _dt):
        t = ctrl.t
        ctrl.t += dt
        q, dq = y[:2], y[2:]
        e = ref(t) - q
        de = -dq  # reference rate ~ small; PD on position + damping
        return Kp * e + Kd * de + arm.G(q)  # gravity feed-forward keeps it gentle

    ctrl.t = 0.0
    x0 = np.array([*ref(0.0), 0.0, 0.0])
    traj = simulate(arm, ctrl, x0=x0, dt=dt, t_final=t_final)

    q = traj.x[:, :2]
    dq = traj.x[:, 2:]
    tau = traj.u.copy()
    # exact model acceleration for the recorded command (no discretisation noise)
    ddq = np.array([arm.dynamics(0.0, traj.x[k], tau[k])[2:] for k in range(len(traj))])
    assert np.all(np.abs(tau) < arm.tau_max - 1e-6), "command hit the torque limit"
    return q, dq, ddq, tau


def test_regressor_recovers_known_base_parameters():
    arm = TwoLinkArm()
    q, dq, ddq, tau = _excitation_log(arm)
    res = identify_manipulator(q, dq, ddq, tau, g=arm.g, friction="viscous")

    assert res.cond < 1e4, f"trajectory under-exciting: cond={res.cond:.3g}"
    for name, truth in TRUE.items():
        assert res.params[name] == pytest.approx(truth, rel=1e-4, abs=1e-6), name
    assert np.all(res.resid_rms_train < 1e-6)
    assert np.all(res.resid_rms_val < 1e-6)
    assert res.r2_val == pytest.approx(1.0, abs=1e-9)
    assert res.implied_l1 == pytest.approx(arm.l1, rel=1e-4)


def test_regressor_shape_and_friction_modes():
    q = np.zeros((5, 2))
    for fric, p in (("none", 5), ("viscous", 7), ("coulomb", 9)):
        Y = manipulator_regressor(q, q, q, friction=fric)
        assert Y.shape == (10, p)


def test_to_twolink_arm_round_trips_dynamics():
    arm = TwoLinkArm()
    q, dq, ddq, tau = _excitation_log(arm)
    res = identify_manipulator(q, dq, ddq, tau, g=arm.g)
    ident = res.to_twolink_arm(l1=arm.l1, l2=arm.l2)

    rng = np.random.default_rng(1)
    for _ in range(20):
        qi = rng.uniform(-np.pi, np.pi, size=2)
        dqi = rng.uniform(-2, 2, size=2)
        assert ident.M(qi) == pytest.approx(arm.M(qi), rel=1e-3, abs=1e-4)
        assert ident.G(qi) == pytest.approx(arm.G(qi), rel=1e-3, abs=1e-4)
        assert ident.C(qi, dqi) == pytest.approx(arm.C(qi, dqi), rel=1e-3, abs=1e-4)


def test_finite_difference_derivatives_matches_analytic():
    t = np.linspace(0, 4, 801)
    w = 2.3
    q = np.stack([np.sin(w * t), 0.5 * np.cos(0.7 * w * t)], axis=1)
    dq_true = np.stack([w * np.cos(w * t), -0.35 * w * np.sin(0.7 * w * t)], axis=1)
    ddq_true = np.stack([-(w**2) * np.sin(w * t), -0.245 * w**2 * np.cos(0.7 * w * t)], axis=1)

    dq, ddq = finite_difference_derivatives(t, q)
    # interior points only -- endpoints are one-sided
    assert dq[5:-5] == pytest.approx(dq_true[5:-5], abs=2e-3)
    assert ddq[5:-5] == pytest.approx(ddq_true[5:-5], abs=5e-2)


def test_poor_excitation_is_flagged_by_condition_number():
    arm = TwoLinkArm()
    # hold near a single configuration: the regressor columns become collinear
    rng = np.random.default_rng(0)
    q = np.full((400, 2), [np.pi / 2, -0.3]) + rng.normal(scale=1e-3, size=(400, 2))
    dq = rng.normal(scale=1e-3, size=(400, 2))
    ddq = rng.normal(scale=1e-3, size=(400, 2))
    tau = np.array([arm.dynamics(0.0, [*q[k], *dq[k]], np.zeros(2)) for k in range(400)])[:, 2:] * 0
    tau = np.array([arm.M(q[k]) @ ddq[k] + arm.C(q[k], dq[k]) @ dq[k] + arm.G(q[k]) + arm.b * dq[k]
                    for k in range(400)])
    res = identify_manipulator(q, dq, ddq, tau, g=arm.g)
    assert res.cond > 1e4
    assert "POORLY EXCITED" in res.summary()


def test_noisy_log_through_finite_differences():
    arm = TwoLinkArm()
    q, dq, ddq, tau = _excitation_log(arm, dt=0.001, t_final=20.0, seed=3)
    t = np.arange(len(q)) * 0.001
    rng = np.random.default_rng(7)
    q_noisy = q + rng.normal(scale=1e-4, size=q.shape)   # ~0.1 mrad encoder noise
    tau_noisy = tau + rng.normal(scale=2e-3, size=tau.shape)

    # Savitzky-Golay derivative estimate is the realistic tool for measured data
    dq_fd, ddq_fd = finite_difference_derivatives(t, q_noisy, smooth=81)
    res = identify_manipulator(q_noisy, dq_fd, ddq_fd, tau_noisy, g=arm.g, ridge=1e-6)

    # honest expectation: gravity params (best conditioned) land close, the
    # inertia params degrade but stay the right order, prediction still holds
    assert res.params["p4"] == pytest.approx(TRUE["p4"], rel=0.15)
    assert res.params["p5"] == pytest.approx(TRUE["p5"], rel=0.15)
    assert res.implied_l1 == pytest.approx(arm.l1, rel=0.20)
    assert res.r2_val > 0.9
