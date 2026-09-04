"""The 3-D PyVista arm renderers (live_arm/pv_arm.py, live_arm_balance/pv_arm.py)
- structural checks with the real window/render loop mocked out, mirroring
tests/test_live_drone_3d.py's pattern. Skips cleanly if pyvista is absent."""

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("pyvista") is None:
    pytest.skip("pyvista not installed", allow_module_level=True)

import pyvista as pv

_EXP = Path(__file__).resolve().parents[1] / "experiments"


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), _EXP / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mocked_plotter(monkeypatch):
    """Capture the timer tick + key-event callbacks instead of starting a real
    render loop / interactor."""
    captured = {"keys": {}}
    monkeypatch.setattr(
        pv.Plotter, "add_timer_event",
        lambda self, max_steps, duration, callback: captured.__setitem__("tick", callback),
        raising=False)
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *a, **k: None, raising=False)
    monkeypatch.setattr(
        pv.Plotter, "add_key_event",
        lambda self, key, callback: captured["keys"].__setitem__(key, callback),
        raising=False)
    return captured


def test_embed_lays_the_2d_plane_into_x_z():
    from aimct.viz.pv_arm import embed

    assert np.allclose(embed([0.3, -0.5]), [0.3, 0.0, -0.5])


def test_live_arm_pv_scene_builds_and_ticks(mocked_plotter):
    from aimct.viz.pv_arm import run_pyvista_arm

    mod = _load("live_arm/pv_arm.py")
    box, _path = mod.run.build()
    assert run_pyvista_arm(box, title="test", show_payload=True) == 0
    tick = mocked_plotter["tick"]
    for k in range(30):
        if k == 10:
            mod.run.ARM.payload = 0.4
        tick(k)                              # must not raise
    assert np.all(np.isfinite(box.x))


def test_live_arm_balance_pv_scene_builds_and_ticks(mocked_plotter):
    from aimct.viz.pv_arm import run_pyvista_arm

    mod = _load("live_arm_balance/pv_arm.py")
    box = mod.run.build()
    assert run_pyvista_arm(box, title="test") == 0
    tick = mocked_plotter["tick"]
    for k in range(30):
        tick(k)                              # must not raise, no target actor needed
    assert np.all(np.isfinite(box.x))


def test_pv_arm_help_surprise_snapshot(mocked_plotter, tmp_path, monkeypatch):
    from aimct.viz.pv_arm import run_pyvista_arm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pv, "OFF_SCREEN", True)          # screenshot() needs a
                                                          # rendered window or off_screen
    mod = _load("live_arm/pv_arm.py")
    box, _path = mod.run.build()
    assert run_pyvista_arm(box, title="test", show_payload=True) == 0
    keys = mocked_plotter["keys"]
    assert {"h", "g", "c", "r", "1", "2", "3"} <= set(keys)

    keys["h"]()                                        # show help
    keys["h"]()                                         # hide it again (no raise)

    before = dict(box.knobs)
    keys["g"]()                                          # surprise me
    assert box.knobs != before

    keys["c"]()
    saved = list((tmp_path / "snapshots").glob("*.png"))
    assert len(saved) == 1


def test_pv_arm_headless_delegates_to_run_headless(monkeypatch, capsys):
    import sys

    mod = _load("live_arm/pv_arm.py")
    monkeypatch.setattr(sys, "argv", ["pv_arm.py", "--headless"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0 and "headless check OK" in out
