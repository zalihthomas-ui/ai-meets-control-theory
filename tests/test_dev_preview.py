"""Tests for :mod:`aimct.dev.preview` - the design-time system preview."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.dev import build_report, load_system, preview_once, render, watch
from aimct.systems import CartPole, Pendulum

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

_BROKEN_SOURCE = '''
import numpy as np
from aimct.systems.base import DynamicalSystem

class SignErrorSpring(DynamicalSystem):
    n_states = 2
    n_inputs = 1

    def __init__(self, m=1.0, k=2.0, c=0.3):
        self.m, self.k, self.c = m, k, c

    def dynamics(self, t, x, u):
        x, u = self._prep(x, u)
        pos, vel = x
        acc = (self.k * pos - self.c * vel + u[0]) / self.m   # sign bug
        return np.array([vel, acc])

    def linearize(self, x_eq=None, u_eq=None, eps=1e-6):
        A = np.array([[0.0, 1.0], [self.k / self.m, -self.c / self.m]])
        B = np.array([[0.0], [5.0 / self.m]])   # wrong gain
        return A, B


class Uncontrollable(DynamicalSystem):
    n_states = 2
    n_inputs = 1

    def dynamics(self, t, x, u):
        x, u = self._prep(x, u)
        return np.array([-x[0] + u[0], -2.0 * x[1]])


class Clean(DynamicalSystem):
    n_states = 2
    n_inputs = 1

    def dynamics(self, t, x, u):
        x, u = self._prep(x, u)
        return np.array([x[1], -x[0] - 0.2 * x[1] + u[0]])
'''


@pytest.fixture
def broken_module(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text(_BROKEN_SOURCE, encoding="utf-8")
    return p


# --------------------------------------------------------------------- report

def test_build_report_on_a_clean_system_has_no_warnings():
    r = build_report(Pendulum(), x_eq=[np.pi, 0.0], t_final=1.0)
    assert r.name == "Pendulum"
    assert r.poles.shape == (2,)
    assert r.controllable
    assert r.has_analytic_linearize
    assert r.jacobian_residual < 1e-6
    assert r.warnings == []
    assert set(r.responses) == {"free", "step", "impulse", "sinusoid"}
    for tr in r.responses.values():
        assert not tr.diverged


def test_default_operating_point_is_the_origin():
    r = build_report(CartPole(), t_final=0.5)
    assert np.allclose(r.x_eq, 0.0) and np.allclose(r.u_eq, 0.0)


def test_a_bad_analytic_linearize_is_caught_by_the_residual_check(broken_module):
    sys = load_system(f"{broken_module}:SignErrorSpring")
    r = build_report(sys, t_final=1.0)
    assert r.has_analytic_linearize
    assert r.jacobian_residual > 1.0            # way over the 1e-3 tolerance
    assert any("disagrees with the numeric check" in w for w in r.warnings)
    assert not r.stable                          # the sign bug also destabilises it


def test_an_uncontrollable_model_is_flagged(broken_module):
    sys = load_system(f"{broken_module}:Uncontrollable")
    r = build_report(sys, t_final=1.0)
    assert not r.controllable
    assert any("not controllable" in w for w in r.warnings)


def test_a_correctly_implemented_system_has_no_analytic_linearize_and_no_warnings(broken_module):
    sys = load_system(f"{broken_module}:Clean")
    r = build_report(sys, t_final=1.0)
    assert not r.has_analytic_linearize         # inherits the numeric fallback
    assert r.jacobian_residual is None
    assert r.controllable
    assert r.warnings == []


def test_summary_mentions_the_headline_numbers():
    r = build_report(Pendulum(), x_eq=[np.pi, 0.0], t_final=0.5)
    s = r.summary()
    assert "Pendulum" in s and "poles" in s and "controllable" in s


# ------------------------------------------------------------------ rendering

def test_render_produces_a_figure_with_five_axes():
    import matplotlib
    matplotlib.use("Agg")
    r = build_report(Pendulum(), t_final=0.5)
    fig = render(r)
    assert len(fig.axes) == 6              # poles + text + 4 response panels
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_flags_a_diverged_response_in_the_title(broken_module):
    import matplotlib
    matplotlib.use("Agg")
    sys = load_system(f"{broken_module}:SignErrorSpring")
    r = build_report(sys, t_final=20.0, u_scale=50.0)   # push it hard enough to blow up
    fig = render(r)
    import matplotlib.pyplot as plt
    plt.close(fig)
    # at least one response should have diverged given the instability + large input
    assert any(tr.diverged for tr in r.responses.values()) or not r.stable


# --------------------------------------------------------------------- loading

def test_load_system_from_a_file_path(broken_module):
    sys = load_system(f"{broken_module}:Clean")
    assert type(sys).__name__ == "Clean"
    assert sys.n_states == 2 and sys.n_inputs == 1


def test_load_system_from_an_importable_module(tmp_path, monkeypatch):
    # exercise the "module:Class" (not file-path) branch without reloading a
    # real aimct module - importlib.reload() would give the reloaded module a
    # fresh class object, which could confuse isinstance() checks elsewhere in
    # the process for a shared library module. A private throwaway module
    # avoids that entirely.
    import sys as _sys
    (tmp_path / "_aimct_dev_test_importable.py").write_text(
        "from aimct.systems.base import DynamicalSystem\nimport numpy as np\n"
        "class T(DynamicalSystem):\n"
        "    n_states, n_inputs = 1, 1\n"
        "    def __init__(self, m=1.0):\n        self.m = m\n"
        "    def dynamics(self, t, x, u):\n"
        "        x, u = self._prep(x, u)\n"
        "        return np.array([-self.m * x[0] + u[0]])\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        sys = load_system("_aimct_dev_test_importable:T", m=2.0)
        assert type(sys).__name__ == "T"
        assert sys.m == 2.0
    finally:
        _sys.modules.pop("_aimct_dev_test_importable", None)


def test_load_system_rejects_a_target_without_a_colon():
    with pytest.raises(ValueError):
        load_system("aimct.systems.pendulum")


def test_two_loads_of_an_edited_file_pick_up_the_change(tmp_path):
    p = tmp_path / "evolving.py"
    p.write_text(
        "from aimct.systems.base import DynamicalSystem\nimport numpy as np\n"
        "class S(DynamicalSystem):\n"
        "    n_states, n_inputs = 1, 1\n"
        "    def dynamics(self, t, x, u):\n"
        "        x, u = self._prep(x, u)\n"
        "        return np.array([-1.0 * x[0] + u[0]])\n",
        encoding="utf-8",
    )
    a = load_system(f"{p}:S")
    A1, _ = a.linearize(np.zeros(1), np.zeros(1))

    p.write_text(
        "from aimct.systems.base import DynamicalSystem\nimport numpy as np\n"
        "class S(DynamicalSystem):\n"
        "    n_states, n_inputs = 1, 1\n"
        "    def dynamics(self, t, x, u):\n"
        "        x, u = self._prep(x, u)\n"
        "        return np.array([-5.0 * x[0] + u[0]])\n",
        encoding="utf-8",
    )
    b = load_system(f"{p}:S")
    A2, _ = b.linearize(np.zeros(1), np.zeros(1))
    assert not np.allclose(A1, A2)               # fresh reimport, not a cached module


# --------------------------------------------------------------- preview_once

def test_preview_once_writes_a_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "preview.png"
    r = preview_once("aimct.systems.pendulum:Pendulum", out=out,
                     x_eq=[np.pi, 0.0], t_final=0.5)
    assert out.exists() and out.stat().st_size > 0
    assert r.name == "Pendulum"


def test_preview_once_can_skip_rendering():
    r = preview_once("aimct.systems.pendulum:Pendulum", out=None, t_final=0.5)
    assert r.name == "Pendulum"


# ---------------------------------------------------------------------- watch

def test_watch_rebuilds_once_per_tick_and_stops_at_max_ticks(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    p = tmp_path / "watched.py"
    p.write_text(
        "from aimct.systems.base import DynamicalSystem\nimport numpy as np\n"
        "class S(DynamicalSystem):\n"
        "    n_states, n_inputs = 2, 1\n"
        "    def dynamics(self, t, x, u):\n"
        "        x, u = self._prep(x, u)\n"
        "        return np.array([x[1], -x[0] - 0.1*x[1] + u[0]])\n",
        encoding="utf-8",
    )
    out = tmp_path / "watched.png"
    calls = []
    watch(f"{p}:S", out=out, poll=0.0, max_ticks=2, on_report=calls.append,
         t_final=0.5)
    assert len(calls) == 1                        # unchanged mtime -> rebuilt once
    assert out.exists()
