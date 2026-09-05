"""aimct.deploy: JSON round-trip, sim-path == deploy-path equivalence, codegen
cross-check, and the unsupported-controller guard.
"""

import json

import numpy as np
import pytest

from aimct.controllers import LQR, PID, LinearMPC, StateFeedback
from aimct.deploy import (
    ControllerSpec,
    UnsupportedControllerError,
    emit_c,
    emit_micropython,
    export_controller,
    load_controller,
)
from aimct.simulate import simulate
from aimct.systems import CartPole, MassSpringDamper


def _cartpole_lqr():
    cp = CartPole()
    A, B = cp.linearize()
    return cp, LQR(A, B, np.diag([10.0, 1.0, 10.0, 1.0]), np.array([[0.1]]))


def test_state_feedback_json_round_trip():
    _, k = _cartpole_lqr()
    spec = export_controller(k, dt=0.02, u_bounds=(-20.0, 20.0), meta={"system": "cartpole"})
    txt = spec.to_json()
    back = ControllerSpec.from_json(txt)
    assert back.kind == "state_feedback"
    assert np.allclose(np.asarray(back.K), np.asarray(spec.K))
    assert back.u_min == [-20.0] and back.u_max == [20.0]
    assert back.meta["system"] == "cartpole"
    assert json.loads(txt)["schema"] == "aimct.deploy/1"


def test_portable_state_feedback_matches_original_pointwise():
    _, k = _cartpole_lqr()
    port = load_controller(export_controller(k, dt=0.02))
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.normal(scale=0.5, size=4)
        assert port.update(x, 0.02) == pytest.approx(float(np.atleast_1d(k.update(x))[0]), rel=1e-9, abs=1e-9)


def test_portable_controller_runs_in_simulate_identically():
    cp, k = _cartpole_lqr()
    port = load_controller(export_controller(k, dt=0.02, u_bounds=(-20.0, 20.0)))
    x0 = np.array([0.0, 0.0, 0.25, 0.0])
    a = simulate(cp, k, x0=x0, dt=0.02, t_final=4.0, u_bounds=(-20.0, 20.0))
    b = simulate(cp, port, x0=x0, dt=0.02, t_final=4.0, u_bounds=(-20.0, 20.0))
    assert np.allclose(a.x, b.x, atol=1e-9)
    assert np.allclose(a.u, b.u, atol=1e-9)


def test_portable_pid_matches_original_in_closed_loop():
    sys = MassSpringDamper()
    pid = PID(kp=40.0, ki=12.0, kd=8.0, setpoint=1.0, dt=0.01,
              output_limits=(-8.0, 8.0), tau_d=0.02, derivative_on="measurement")
    spec = export_controller(pid)
    assert spec.kind == "pid" and spec.dt == 0.01 and spec.tau_d == 0.02
    port = load_controller(spec)

    meas = lambda t, x, u: x[[0]]      # position feedback
    x0 = np.zeros(2)
    a = simulate(sys, pid, x0=x0, dt=0.01, t_final=6.0, measurement_fn=meas)
    pid.reset()
    b = simulate(sys, port, x0=x0, dt=0.01, t_final=6.0, measurement_fn=meas)
    assert np.allclose(a.x, b.x, atol=1e-7)


def test_emit_micropython_executor_cross_checks_against_numpy():
    _, k = _cartpole_lqr()
    spec = export_controller(k, dt=0.02, u_bounds=(-20.0, 20.0))
    src = emit_micropython(spec)
    ns: dict = {}
    exec(compile(src, "<emitted>", "exec"), ns)          # MicroPython source is CPython-valid here
    port = load_controller(spec)

    rng = np.random.default_rng(1)
    for _ in range(100):
        x = list(rng.normal(scale=0.4, size=4))
        got = ns["controller_update"](x, 0.02)[0]
        assert got == pytest.approx(port.update(np.array(x), 0.02), rel=1e-9, abs=1e-9)


def test_emit_micropython_pid_cross_checks():
    pid = PID(kp=5.0, ki=2.0, kd=0.5, setpoint=0.3, dt=0.01, output_limits=(-2.0, 2.0))
    spec = export_controller(pid)
    ns: dict = {}
    exec(compile(emit_micropython(spec), "<emitted>", "exec"), ns)
    port = load_controller(spec)

    rng = np.random.default_rng(2)
    for _ in range(300):
        y = list(rng.normal(scale=0.5, size=1))
        got = ns["controller_update"](y, 0.01)[0]
        exp = port.update(np.array(y), 0.01)
        assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_emit_c_is_well_formed_and_bakes_in_gains():
    _, k = _cartpole_lqr()
    c_sf = emit_c(export_controller(k, dt=0.02, u_bounds=(-20.0, 20.0)))
    assert "void controller_update(const float y[AIMCT_N_X], float u[AIMCT_N_U])" in c_sf
    assert "AIMCT_K[i][j]" in c_sf and "AIMCT_U_MIN" in c_sf
    assert c_sf.count("{") == c_sf.count("}")

    pid = PID(kp=5.0, ki=2.0, kd=0.5, setpoint=0.3, dt=0.01, output_limits=(-2.0, 2.0))
    c_pid = emit_c(export_controller(pid))
    assert "controller_reset" in c_pid and "anti-windup" in c_pid
    assert c_pid.count("{") == c_pid.count("}")


def test_unsupported_controller_is_rejected_with_a_helpful_message():
    cp = CartPole()
    A, B = cp.linearize()
    mpc = LinearMPC(A, B, Q=np.eye(4), R=np.array([[0.1]]), N=10)
    with pytest.raises(UnsupportedControllerError, match="online solver"):
        export_controller(mpc)


def test_bare_state_feedback_without_bounds_round_trips():
    k = StateFeedback(np.array([[1.0, 2.0, 3.0, 4.0]]), x_ref=np.array([0.1, 0, 0, 0]))
    port = load_controller(export_controller(k))
    x = np.array([0.2, -0.1, 0.05, 0.0])
    assert port.update(x) == pytest.approx(float(np.atleast_1d(k.update(x))[0]), rel=1e-12)
