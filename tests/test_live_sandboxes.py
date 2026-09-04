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
