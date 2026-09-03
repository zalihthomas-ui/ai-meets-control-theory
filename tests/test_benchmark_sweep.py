"""Tests for :mod:`aimct.benchmarks.sweep` - the parameter / robustness sweep
built on :func:`aimct.benchmarks.compare`.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.benchmarks.sweep import SweepResult, sweep
from aimct.controllers import LQR, PID
from aimct.systems import CartPole, MassSpringDamper

# Driving the cart-pole past its basin makes the state diverge to inf/nan on
# purpose; numpy's overflow / invalid-value warnings on those samples are
# expected and not a signal of a bug in the harness.
diverges = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ---------------------------------------------------------------- fixtures

def _msd_damping_case(c):
    """MSD step response, PID + LQR, plant damping = c (both rebuilt per point)."""
    msd = MassSpringDamper(m=1.0, c=c, k=1.0)
    A, B = msd.linearize()
    return dict(
        system=msd,
        controllers={
            "PID": PID(kp=20.0, ki=10.0, kd=8.0, tau_d=0.02, setpoint=1.0),
            "LQR": LQR(A, B, np.diag([10.0, 1.0]), [[0.1]], x_ref=np.array([1.0, 0.0])),
        },
        x0=np.zeros(2),
        dt=2e-3,
        t_final=6.0,
        reference=1.0,
        measurement_fns={"PID": lambda t, x, u: x[[0]]},
    )


def _cartpole_theta0_case(theta0):
    """LQR (linearised about upright) vs a growing initial pole angle."""
    cp = CartPole()
    A, B = cp.linearize()
    return dict(
        system=cp,
        controllers={"LQR": LQR(A, B, np.diag([1.0, 1.0, 10.0, 1.0]), [[0.1]])},
        x0=np.array([0.0, 0.0, theta0, 0.0]),
        dt=0.02,
        t_final=5.0,
        reference=0.0,
        output_index=2,
    )


# ---------------------------------------------------------------- contract

def test_one_compare_result_per_grid_point():
    res = sweep([0.2, 0.4, 0.8], _msd_damping_case, param_name="c")
    assert isinstance(res, SweepResult)
    assert res.param_name == "c"
    assert res.param_values == [0.2, 0.4, 0.8]
    assert len(res.results) == 3
    assert res.controllers == ["PID", "LQR"]


def test_numpy_scalar_grid_is_coerced_to_python_floats():
    res = sweep(np.linspace(0.2, 0.8, 4), _msd_damping_case, param_name="c")
    assert all(type(v) is float for v in res.param_values)


def test_empty_grid_raises():
    with pytest.raises(ValueError, match="empty"):
        sweep([], _msd_damping_case)


def test_make_case_missing_key_raises():
    def bad_case(v):
        d = _msd_damping_case(v)
        del d["reference"]
        return d

    with pytest.raises(ValueError, match="reference"):
        sweep([0.2, 0.4], bad_case)


def test_controller_set_must_be_stable_across_sweep():
    def shifting(v):
        d = _msd_damping_case(v)
        if v > 0.3:
            d["controllers"] = {"PID": d["controllers"]["PID"]}
        return d

    with pytest.raises(ValueError, match="controller set"):
        sweep([0.2, 0.4], shifting, param_name="c")


# ---------------------------------------------------------------- accessors

def test_series_status_and_mask_align_with_grid():
    res = sweep([0.2, 0.4, 0.8], _msd_damping_case, param_name="c")
    s = res.series("settling_time", "LQR")
    assert s.shape == (3,)
    assert np.all(np.isfinite(s))
    assert res.status("LQR") == ["Stable"] * 3 or "Marginal" in res.status("LQR")
    assert res.stable_mask("PID").dtype == bool
    assert len(res.stable_mask("PID")) == 3


@diverges
def test_basin_edge_helpers_on_cartpole():
    res = sweep([0.2, 0.8, 1.3], _cartpole_theta0_case, param_name="theta0")
    st = res.status("LQR")
    assert st[0] == "Stable" and st[1] == "Stable"
    assert st[2] == "Diverged"
    assert res.max_stable("LQR") == 0.8
    assert res.first_unstable("LQR") == 1.3
    assert res.stable_mask("LQR").tolist() == [True, True, False]


@diverges
def test_recovered_and_max_recoverable_treat_marginal_as_survived():
    # 0.8 rad settles slowly here -> "Marginal", not "Diverged"; still recovered.
    res = sweep([0.2, 0.8, 1.3, 1.4], _cartpole_theta0_case, param_name="theta0")
    assert res.first_diverged("LQR") == 1.3
    assert res.max_recoverable("LQR") == 0.8       # largest value below first divergence
    assert res.recovered_mask("LQR").tolist() == [True, True, False, False]


def test_max_recoverable_is_last_value_when_nothing_diverges():
    res = sweep([0.2, 0.4, 0.8], _msd_damping_case, param_name="c")
    assert res.first_diverged("LQR") is None
    assert res.max_recoverable("LQR") == 0.8


def test_result_at_finds_by_value():
    res = sweep([0.2, 0.4, 0.8], _msd_damping_case, param_name="c")
    assert res.result_at(0.4) is res.results[1]
    with pytest.raises(KeyError):
        res.result_at(0.5)


# ---------------------------------------------------------------- reporting

def test_pivot_table_shape():
    res = sweep([0.2, 0.4, 0.8], _msd_damping_case, param_name="c")
    lines = [ln for ln in res.table("settling_time").strip().splitlines() if ln.startswith("|")]
    assert len(lines) == 2 + 3                       # header + align + 3 rows
    assert lines[0].strip("| ").split(" | ") == ["c", "PID", "LQR"]


@diverges
def test_status_table_lists_every_point():
    res = sweep([0.2, 0.8, 1.3], _cartpole_theta0_case, param_name="theta0")
    body = [ln for ln in res.status_table().strip().splitlines() if ln.startswith("|")][2:]
    assert len(body) == 3
    assert "Diverged" in body[-1]


def test_csv_is_long_format_with_all_metrics():
    res = sweep([0.2, 0.4], _msd_damping_case, param_name="c")
    rows = res.to_csv().strip().splitlines()
    header = rows[0].split(",")
    assert header[:3] == ["c", "controller", "status"]
    assert "settling_time" in header and "control_energy" in header
    assert len(rows) == 1 + 2 * 2                    # header + (points x controllers)


# ---------------------------------------------------------------- plotting

@diverges
def test_plot_returns_twin_axes_and_shades_unstable():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = sweep([0.2, 0.8, 1.3, 1.4], _cartpole_theta0_case, param_name="theta0")
    fig, ax_l, ax_r = res.plot(left="settling_time", right="control_energy")
    assert ax_r is not None
    assert len(ax_l.patches) >= 1                    # >=1 axvspan for the divergent tail
    plt.close(fig)

    fig, ax_l, ax_r = res.plot(left="rmse", right=None)
    assert ax_r is None
    plt.close(fig)


def test_save_writes_csv_summary_and_figure(tmp_path):
    res = sweep([0.2, 0.4, 0.8], _msd_damping_case, param_name="c")
    written = res.save(tmp_path, metric="settling_time")
    for key in ("csv", "summary_md", "figure_png", "figure_svg"):
        assert written[key].exists() and written[key].stat().st_size > 0
    assert (tmp_path / "robustness_sweep.png").exists()


def test_progress_prints_per_point(capsys):
    sweep([0.2, 0.4], _msd_damping_case, param_name="c", progress=True)
    out = capsys.readouterr().out
    assert out.count("c=") == 2
