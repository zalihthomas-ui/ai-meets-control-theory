"""The 3-D live drone sandbox - physics + controllers, headless."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "live_drone_3d" / "sim3d.py"
_spec = importlib.util.spec_from_file_location("sim3d", _SCRIPT)
sim3d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim3d)


def _roll(name, steps, wind):
    e = sim3d.Engine(name)
    for k in range(steps):
        e.steady_wind = np.asarray(wind(k), float)
        e.step_frame()
    return e


def test_all_controllers_hold_hover_with_no_wind():
    for name in sim3d.build_controllers(sim3d.Quadrotor3D()):
        e = _roll(name, 400, lambda k: (0, 0, 0))
        assert np.all(np.isfinite(e.x))
        assert np.linalg.norm(e.x[:3] - sim3d.HOVER) < 5e-3, name


def test_integral_controller_rejects_steady_3d_wind():
    ctrls = sim3d.build_controllers(sim3d.Quadrotor3D())
    err = {}
    for name in ctrls:
        e = _roll(name, 900, lambda k: (0.03, -0.02, 0.015))
        err[name] = float(np.linalg.norm(e.x[:3] - sim3d.HOVER))
    integ = next(n for n in ctrls if "integral" in n)
    plain = [v for n, v in err.items() if "integral" not in n]
    assert err[integ] < 0.02
    assert err[integ] < min(plain) / 3


def test_frame_contract():
    e = sim3d.Engine()
    fr = e.step_frame()
    assert fr.pos.shape == (3,)
    assert fr.R.shape == (3, 3)
    assert np.allclose(fr.R @ fr.R.T, np.eye(3), atol=1e-9)
    assert fr.rotors.shape == (4,) and np.all(fr.rotors >= 0)
    assert fr.wind.shape == (3,)
    assert isinstance(fr.hud, str) and "controller" in fr.hud


def test_headless_entrypoint_runs():
    r = subprocess.run([sys.executable, str(_SCRIPT), "--headless"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "headless check OK" in r.stdout


def test_pyvista_renderer_builds_its_meshes():
    import importlib
    if importlib.util.find_spec("pyvista") is None:
        import pytest
        pytest.skip("pyvista not installed")
    import pyvista as pv
    pv_spec = importlib.util.spec_from_file_location(
        "pv3d", _SCRIPT.parent / "pv3d.py")
    pv3d = importlib.util.module_from_spec(pv_spec)
    pv_spec.loader.exec_module(pv3d)
    body, arms, tips, props = pv3d._drone_meshes(pv, 0.046)
    assert len(arms) == 4 and len(props) == 4
    assert pv3d._pose_matrix(np.array([1.0, 2, 3]), np.eye(3)).shape == (4, 4)


def test_pyvista_renderer_main_builds_and_ticks_without_a_window(monkeypatch):
    # regression: main() used pl.add_callback(), which does not exist on every
    # PyVista version (it didn't here) - the interactive path raised
    # AttributeError the moment you actually launched it, silently, because
    # this test previously only exercised the two pure helper functions above.
    import importlib

    if importlib.util.find_spec("pyvista") is None:
        import pytest
        pytest.skip("pyvista not installed")
    import pyvista as pv

    captured = {}
    monkeypatch.setattr(pv.Plotter, "add_timer_event",
                        lambda self, max_steps, duration, callback:
                        captured.__setitem__("tick", callback), raising=False)
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *a, **k: None,
                        raising=False)

    pv_spec = importlib.util.spec_from_file_location(
        "pv3d_ticktest", _SCRIPT.parent / "pv3d.py")
    pv3d = importlib.util.module_from_spec(pv_spec)
    pv_spec.loader.exec_module(pv3d)

    assert pv3d.main() == 0
    tick = captured["tick"]
    for k in range(10):
        tick(k)                      # must not raise (real AttributeError site)
