"""The live drone sandbox — physics + controllers, headless."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT = (Path(__file__).resolve().parents[1] / "experiments" / "live_drone"
           / "live.py")
_spec = importlib.util.spec_from_file_location("live_drone", _SCRIPT)
live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live)


def _roll(ctl, q, steps, wind_fn):
    x = np.array([live.HOVER[0], live.HOVER[1], 0, 0, 0, 0], dtype=float)
    integ = np.zeros(2)
    for k in range(steps):
        q.wind = np.asarray(wind_fn(k), dtype=float)
        u = np.clip(ctl(x, integ), 0.0, q.thrust_max)
        for _ in range(live.SUBSTEPS):
            x = _rk4(q, x, u, live.PHYS_DT)
            integ += (x[:2] - live.HOVER) * live.PHYS_DT
            integ[:] = np.clip(integ, -0.5, 0.5)
    return x


def _rk4(q, x, u, h):
    k1 = q.dynamics(0, x, u)
    k2 = q.dynamics(0, x + 0.5 * h * k1, u)
    k3 = q.dynamics(0, x + 0.5 * h * k2, u)
    k4 = q.dynamics(0, x + h * k3, u)
    return x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def test_all_controllers_hold_hover_with_no_wind():
    q = live.LiveQuad()
    for name, ctl in live.build_controllers(q).items():
        xf = _roll(ctl, q, 400, lambda k: (0.0, 0.0))
        assert np.all(np.isfinite(xf)), name
        assert np.hypot(xf[0] - live.HOVER[0], xf[1] - live.HOVER[1]) < 1e-3, name


def test_integral_controller_rejects_steady_wind_better_than_plain_lqr():
    q = live.LiveQuad()
    ctrls = live.build_controllers(q)
    wind = lambda k: (0.03, 0.0)
    err = {}
    for name, ctl in ctrls.items():
        xf = _roll(ctl, q, 900, wind)
        err[name] = np.hypot(xf[0] - live.HOVER[0], xf[1] - live.HOVER[1])
    integ_name = next(n for n in ctrls if "integral" in n)
    plain = [v for n, v in err.items() if "integral" not in n]
    assert err[integ_name] < 0.03            # ~cm, steady wind nulled
    assert err[integ_name] < min(plain) / 3  # and far better than any plain LQR


def test_headless_entrypoint_runs():
    r = subprocess.run([sys.executable, "-m", "aimct", "live", "--headless"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "headless check OK" in r.stdout
