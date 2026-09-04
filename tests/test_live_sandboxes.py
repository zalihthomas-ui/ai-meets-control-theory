"""The interactive 2-D sandboxes (arm, diff-drive) — physics + controllers,
headless.  The GUI (`.run()`) is never exercised here."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

_EXP = Path(__file__).resolve().parents[1] / "experiments"


def _load(name):
    script = _EXP / name / "run.py"
    spec = importlib.util.spec_from_file_location(name.replace("_", ""), script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- live_arm
arm = _load("live_arm")


def test_live_arm_headless_smoke(capsys):
    assert arm.main.__module__  # module loaded
    rc = arm._headless()
    out = capsys.readouterr().out
    assert rc == 0 and "headless check OK" in out


def test_live_arm_no_payload_all_controllers_hold_the_target():
    for name in ("PD + gravity comp", "Computed torque (0 kg model)",
                 "Adaptive computed torque"):
        settled, _, _ = arm._run_scenario(name, payload=0.0, steps=700)
        assert settled < 2.0, (name, settled)    # < 2 mm with the true model


def test_live_arm_adaptive_beats_fixed_computed_torque_under_an_unknown_load():
    pd, _, _ = arm._run_scenario("PD + gravity comp", 0.4, 900)
    ct, _, _ = arm._run_scenario("Computed torque (0 kg model)", 0.4, 900)
    ada, _, mhat = arm._run_scenario("Adaptive computed torque", 0.4, 900)
    assert ada < ct                              # adaptive recovers, fixed CT does not
    assert ada < pd
    assert 0.30 <= mhat <= 0.50                  # identifies the payload (true 0.40)


def test_live_arm_reference_model_is_finite_under_a_target_jump():
    box, pth = arm.build()
    box.set_controller("Adaptive computed torque")
    pth.centre[:] = [0.30, 0.10]                 # a hard jump
    for _ in range(200):
        box.step()
    assert np.all(np.isfinite(box.x))


# ------------------------------------------------------------ live_diffdrive
dd = _load("live_diffdrive")


def test_live_diffdrive_headless_smoke(capsys):
    rc = dd._headless()
    out = capsys.readouterr().out
    assert rc == 0 and "headless check OK" in out


def test_live_diffdrive_all_followers_track_the_loop_undisturbed():
    for name in ("Pure pursuit", "Stanley", "Path LQR"):
        box, _ = dd.build()
        box.set_controller(name)
        errs = []
        for _ in range(600):
            box.step()
            errs.append(abs(dd.PATH.frame(box.x[:2])[2]))
        assert np.all(np.isfinite(box.x))
        assert np.mean(errs[-200:]) < 0.08, (name, np.mean(errs[-200:]))


def test_live_diffdrive_recovers_from_a_shove():
    box, _ = dd.build()
    box.set_controller("Stanley")
    for k in range(500):
        if k == 150:
            box.kick([0.25, -0.15, 0, 0, 0])
        box.step()
    assert abs(dd.PATH.frame(box.x[:2])[2]) < 0.05      # back on the line


def test_live_diffdrive_lookahead_has_no_teleport_across_the_crossing():
    # regression: a global-argmin nearest point flips branches at the
    # figure-8's self-intersection; progress hysteresis must prevent that.
    box, shared = dd.build()
    box.set_controller("Pure pursuit")
    prev = None
    for _ in range(1200):
        box.step()
        la = shared["lookahead"]
        if la is not None and prev is not None:
            assert np.linalg.norm(np.asarray(la) - prev) < 0.5, "look-ahead teleported"
        prev = None if la is None else np.asarray(la).copy()


# --------------------------------------------------------- live_arm_balance
ab = _load("live_arm_balance")


def test_live_arm_balance_headless_smoke(capsys):
    rc = ab._headless()
    out = capsys.readouterr().out
    assert rc == 0 and "headless check OK" in out


def test_live_arm_balance_upright_is_a_true_equilibrium():
    # zero torque at x_eq, and dynamics(x_eq, u_eq) == 0 exactly
    assert np.allclose(ab.UEQ, 0.0, atol=1e-8)
    xdot = ab.ARM.dynamics(0.0, ab.XEQ, ab.UEQ)
    assert np.allclose(xdot, 0.0, atol=1e-8)


def test_live_arm_balance_is_open_loop_unstable_there():
    A, _ = ab.ARM.linearize()          # about XEQ by default
    assert np.any(np.linalg.eigvals(A).real > 0)


def test_live_arm_balance_stiff_and_integral_recover_a_poke_soft_does_not():
    results = {}
    for name in ("LQR (stiff)", "LQR + integral (wind-adaptive)", "LQR (soft)"):
        box = ab.build()
        box.set_controller(name)
        box.kick([0, 0, 1.5, -1.0])
        for _ in range(2000):
            box.step()
        results[name] = np.degrees(ab._tilt(box.x))
    assert results["LQR (stiff)"] < 1.0
    assert results["LQR + integral (wind-adaptive)"] < 1.0
    assert results["LQR (soft)"] > 10.0        # settles into a wrong offset


def test_live_arm_balance_integral_nulls_a_steady_wind_stiff_does_not():
    tilts = {}
    for name in ("LQR (stiff)", "LQR + integral (wind-adaptive)"):
        box = ab.build()
        box.set_controller(name)
        box.knobs["wind q1 [N.m]"] = 0.6
        box.knobs["wind q2 [N.m]"] = 0.4
        for _ in range(2500):
            box.step()
        tilts[name] = np.degrees(ab._tilt(box.x))
    assert tilts["LQR + integral (wind-adaptive)"] < 0.5
    assert tilts["LQR (stiff)"] > tilts["LQR + integral (wind-adaptive)"] + 2.0


def test_live_arm_balance_autoreset_after_a_fall():
    box = ab.build()
    box.set_controller("LQR (soft)")
    before = ab._falls["count"]
    box.x[:] = ab.XEQ + np.array([1.35, 0.0, 0.0, 0.0])   # already past FALL_LIMIT
    box.step()
    assert ab._falls["count"] == before + 1
    assert abs(box.x[0] - ab.XEQ[0]) < 0.2                 # respawned near vertical
