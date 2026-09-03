"""Tests for :mod:`aimct.benchmarks.harness` — the multi-controller ``compare``
runner and its :class:`ComparisonResult`.

Golden closed-form metric values are famo's ``test_benchmark_metrics``; here we
check the *harness* contract: identical conditions across controllers, correct
reference handling, status classification, actuator-bound enforcement, output
channel selection, and the emitted table / files.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.benchmarks.harness import SPEC_COLUMNS, ComparisonResult, compare
from aimct.controllers import LQR, PID, StateFeedback
from aimct.systems import CartPole, LinearSystem, MassSpringDamper


@pytest.fixture
def msd():
    return MassSpringDamper(m=1.0, c=0.4, k=1.0)


@pytest.fixture
def msd_controllers(msd):
    A, B = msd.linearize()
    return {
        "PID": PID(kp=20.0, ki=10.0, kd=8.0, tau_d=0.02, setpoint=1.0),
        "LQR": LQR(A, B, np.diag([10.0, 1.0]), [[0.1]], x_ref=np.array([1.0, 0.0])),
    }


def _run(msd, controllers, **kw):
    kw.setdefault("reference", 1.0)
    return compare(
        msd, controllers, x0=np.zeros(2), dt=2e-3, t_final=6.0,
        measurement_fns={"PID": lambda t, x, u: x[[0]]}, **kw,
    )


# --------------------------------------------------------------------- contract

def test_result_has_one_entry_per_controller_in_order(msd, msd_controllers):
    res = _run(msd, msd_controllers)
    assert isinstance(res, ComparisonResult)
    assert res.names == ["PID", "LQR"]
    for name in res.names:
        assert name in res.trajectories
        assert name in res.metrics
        assert name in res.status


def test_all_controllers_share_time_grid_and_initial_state(msd, msd_controllers):
    res = _run(msd, msd_controllers)
    trajs = list(res.trajectories.values())
    for tr in trajs:
        assert np.array_equal(tr.t, res.t)
        assert tr.x[0] == pytest.approx(np.zeros(2))
    assert res.t[0] == 0.0
    assert res.t[-1] == pytest.approx(6.0)


def test_empty_controllers_rejected(msd):
    with pytest.raises(ValueError, match="at least one"):
        compare(msd, {}, x0=np.zeros(2), dt=1e-2, t_final=1.0, reference=0.0)


# -------------------------------------------------------------------- reference

def test_scalar_array_and_callable_references_are_equivalent(msd, msd_controllers):
    n = int(round(6.0 / 2e-3)) + 1
    base = _run(msd, dict(msd_controllers)).metrics["LQR"]
    arr = _run(msd, dict(msd_controllers), reference=np.ones(n)).metrics["LQR"]
    fn = compare(
        msd, dict(msd_controllers), x0=np.zeros(2), dt=2e-3, t_final=6.0,
        reference=lambda t: 1.0, measurement_fns={"PID": lambda t, x, u: x[[0]]},
    ).metrics["LQR"]
    assert base["rmse"] == pytest.approx(arr["rmse"]) == pytest.approx(fn["rmse"])


def test_reference_array_length_mismatch_raises(msd, msd_controllers):
    with pytest.raises(ValueError, match="one per sample"):
        compare(msd, msd_controllers, x0=np.zeros(2), dt=2e-3, t_final=6.0,
                reference=np.ones(10))


# ------------------------------------------------------------ status classification

def test_stable_controller_is_labelled_stable():
    di = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    sys = LinearSystem(di, B)
    lqr = LQR(di, B, np.eye(2), [[1.0]])
    res = compare(sys, {"LQR": lqr}, x0=np.array([1.0, 0.0]), dt=1e-2,
                  t_final=20.0, reference=0.0)
    assert res.status["LQR"] == "Stable"
    assert res.metrics["LQR"]["steady_state_error"] < 1e-3


def test_divergent_run_is_labelled_diverged():
    # open-loop-unstable plant (poles at +-2) with a controller that does nothing
    A = np.array([[0.0, 1.0], [4.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    sys = LinearSystem(A, B, C=np.array([[1.0, 0.0]]))
    res = compare(sys, {"open loop": PID(kp=0.0)}, x0=np.array([0.1, 0.0]),
                  dt=1e-2, t_final=15.0, reference=0.0,
                  measurement_fns={"open loop": lambda t, x, u: x[[0]]})
    assert res.status["open loop"] == "Diverged"


# ------------------------------------------------------------- actuator handling

def test_u_bounds_enforced_and_reported(msd, msd_controllers):
    res = _run(msd, msd_controllers, u_bounds=(-3.0, 3.0))
    for name in res.names:
        u = res.trajectories[name].u
        assert np.all(u <= 3.0 + 1e-9) and np.all(u >= -3.0 - 1e-9)
        assert res.metrics[name]["peak_control"] <= 3.0 + 1e-9
        assert "saturation_pct" in res.metrics[name]


def test_no_u_bounds_means_no_saturation_metric(msd, msd_controllers):
    res = _run(msd, msd_controllers)
    assert "saturation_pct" not in res.metrics["LQR"]
    # spec table then prints a placeholder for that column
    assert "--" in "".join(res.rows()[0])


def test_disturbance_changes_the_trajectory(msd, msd_controllers):
    quiet = _run(msd, dict(msd_controllers))
    kicked = _run(msd, dict(msd_controllers),
                  disturbance=lambda t: np.array([0.5]) if t >= 3.0 else np.array([0.0]))
    assert not np.allclose(quiet.trajectories["LQR"].x, kicked.trajectories["LQR"].x)


# --------------------------------------------------------------- output channel

def test_output_index_scores_the_requested_channel():
    cp = CartPole()
    A, B = cp.linearize()
    lqr = LQR(A, B, np.diag([1.0, 1.0, 10.0, 1.0]), [[0.1]])
    res = compare(cp, {"LQR": lqr}, x0=np.array([0.0, 0.0, 0.2, 0.0]),
                  dt=1e-2, t_final=8.0, reference=0.0, output_index=2)
    # metrics are on theta (channel 2), which LQR drives to ~0
    assert res.status["LQR"] == "Stable"
    assert res.metrics["LQR"]["steady_state_error"] < 1e-2
    assert res.deriv_index == 3


def test_measurement_fn_required_for_output_feedback_on_full_state_system(msd):
    pid = PID(kp=10.0, ki=5.0, setpoint=1.0)
    # no measurement_fns: PID receives the length-2 state and returns length-2 -> error
    with pytest.raises(ValueError):
        compare(msd, {"PID": pid}, x0=np.zeros(2), dt=2e-2, t_final=1.0, reference=1.0)


# --------------------------------------------------------------------- reporting

def test_markdown_table_shape_matches_spec_columns(msd, msd_controllers):
    res = _run(msd, msd_controllers, u_bounds=(-50.0, 50.0))
    md = res.to_markdown()
    lines = [ln for ln in md.strip().splitlines() if ln.startswith("|")]
    assert len(lines) == 2 + len(res.names)                 # header + align + N rows
    n_cols = len(SPEC_COLUMNS) + 2                           # + Controller + Status
    assert lines[0].count("|") == n_cols + 1
    assert "Status" in lines[0]
    for name in res.names:
        assert any(f"**{name}**" in ln for ln in lines)


def test_csv_roundtrips_every_controller(msd, msd_controllers):
    res = _run(msd, msd_controllers, u_bounds=(-50.0, 50.0))
    csv_lines = res.to_csv().strip().splitlines()
    assert csv_lines[0].startswith("controller,")
    assert len(csv_lines) == 1 + len(res.names)


def test_full_metrics_csv_lists_all_metric_keys(msd, msd_controllers):
    res = _run(msd, msd_controllers)
    header = res.full_metrics_csv().splitlines()[0].split(",")
    for key in ("rise_time", "settling_time", "iae", "itae", "ise", "slew_rate"):
        assert key in header


def test_summary_names_precision_and_energy_winners(msd, msd_controllers):
    res = _run(msd, msd_controllers)
    s = res.summary()
    assert "Precision:" in s and "Energy:" in s


def test_save_writes_table_and_figure_files(tmp_path, msd, msd_controllers):
    res = _run(msd, msd_controllers, u_bounds=(-50.0, 50.0))
    written = res.save(tmp_path)
    for key in ("metrics_md", "metrics_csv", "metrics_full_csv", "figure_png", "figure_svg"):
        assert written[key].exists() and written[key].stat().st_size > 0
    assert (tmp_path / "comparison_benchmark.png").exists()


def test_figure_builds_four_panels(msd, msd_controllers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = _run(msd, msd_controllers)
    fig, axes = res.figure()
    assert np.asarray(axes).size == 4
    plt.close(fig)
