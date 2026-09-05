"""Reference trajectories + the trajectory-tracking benchmark."""

import numpy as np
import pytest

from aimct.benchmarks import track_trajectory
from aimct.controllers import LQR
from aimct.systems import PlanarQuadrotor
from aimct.trajectories import (
    Circle,
    Dubins,
    Lemniscate,
    Lissajous,
    MinimumJerk,
    Rose,
    Setpoint,
    Spiral,
    Spline,
)

_ALL = [
    Lemniscate(0.6, 0.35, 6.0),
    Lemniscate(0.6, 0.35, 6.0, z0=1.0, Cz=0.1),
    Circle(1.0, 6.0),
    MinimumJerk([0.0, 0.0], [1.0, 2.0], 2.0),
    Spline([[0, 0], [1, 1], [2, 0], [3, 2]]),
    Dubins(v=0.5),
    Lissajous(0.6, 0.4, 3, 2, period=8.0),
    Lissajous(0.5, 0.5, 5, 4, period=10.0),
    Rose(0.6, 3, period=10.0),
    Rose(0.6, 4, period=10.0),
    Spiral(0.15, 0.05, 1.2, 12.0),
]


@pytest.mark.parametrize("traj", _ALL, ids=lambda t: type(t).__name__)
def test_vel_and_acc_are_the_time_derivatives(traj):
    t, h = 0.7, 1e-5
    p0, v, a = traj(t)
    pp, vp, _ = traj(t + h)
    pm, vm, _ = traj(t - h)
    assert np.allclose((pp - pm) / (2 * h), v, atol=1e-6)
    assert np.allclose((vp - vm) / (2 * h), a, atol=1e-5)
    assert p0.size == traj.dim


def test_minimum_jerk_endpoints_are_at_rest():
    mj = MinimumJerk([1.0, -1.0], [2.0, 3.0], T=1.5)
    for t in (0.0, 1.5):
        _, v, a = mj(t)
        assert np.allclose(v, 0.0, atol=1e-9)
        assert np.allclose(a, 0.0, atol=1e-9)
    assert np.allclose(mj(1.5)[0], [2.0, 3.0])


def test_setpoint_is_constant():
    sp = Setpoint([1.0, 2.0, 3.0])
    p, v, a = sp(99.0)
    assert np.allclose(p, [1, 2, 3]) and np.allclose(v, 0) and np.allclose(a, 0)


def test_dubins_path_is_continuous_and_has_expected_length():
    d = Dubins(start=(0, 0), heading=0.0, straight1=1.0, radius=0.6,
               turn=np.pi / 2, straight2=1.0, v=0.5)
    pts = np.array([d(t)[0] for t in np.linspace(0, d.duration, 400)])
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() / steps.mean() < 1.05          # no jumps between segments
    assert abs(d.length - (1.0 + np.pi / 2 * 0.6 + 1.0)) < 0.05


def test_closest_point_and_completion():
    circ = Circle(1.0, 6.0)
    pt, s, frac = circ.closest([1.0, 0.0])            # start of the circle
    assert np.allclose(pt, [1.0, 0.0], atol=0.05)
    assert 0.0 <= frac <= 0.05


def test_track_trajectory_scores_a_quad_following_a_lemniscate(tmp_path):
    quad = PlanarQuadrotor()
    A, B = quad.linearize()
    K = LQR(A, B, np.diag(1 / np.array([.1, .1, .2, .5, .5, 3.]) ** 2),
            np.diag(1 / np.array([.15, .15]) ** 2)).K
    traj = Lemniscate(0.5, 0.3, 8.0)

    class Tracker:
        name = "lqr+ff"

        def __init__(self):
            self._t = 0.0

        def reset(self):
            self._t = 0.0

        def update(self, x, dt):
            p, v, a = traj(self._t)
            th = -a[0] / quad.g
            xr = np.array([p[0], p[1] + 1.0, th, v[0], v[1], 0.0])
            uff = np.array([0.5 * quad.m * (quad.g + a[1])] * 2)
            self._t += dt
            return np.clip(uff - K @ (np.asarray(x) - xr), 0.0, quad.thrust_max)

    x0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    res = track_trajectory(quad, {"lqr+ff": Tracker()}, traj, x0, dt=0.01,
                           t_final=16.0, pos_index=(0, 1), u_bounds=(0.0, quad.thrust_max))
    m = res.metrics["lqr+ff"]
    # the tracker here is deliberately crude (partial flatness feed-forward) -
    # the point is that track_trajectory runs and returns sane, finite metrics
    assert m["status"] == "OK"
    assert 0.0 < m["rms_err_mm"] < 2500.0
    assert m["rms_cross_track_mm"] <= m["rms_err_mm"] + 1e-6
    assert 30.0 < m["completion_pct"] <= 100.0
    assert set(res.trajectories) == {"lqr+ff"}
    assert "cross-track" in res.to_markdown() or "cross_track" in res.to_markdown()

    # -- the reporting surface: markdown / csv / figure / save -----------
    import csv
    import io

    csv_text = res.to_csv()
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0][0] == "controller" and rows[1][0] == "lqr+ff"
    assert len(rows) == 2 and "OK" in rows[1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = res.figure()
    assert len(ax) == 2                                # path panel + error panel
    plt.close(fig)

    res.save(tmp_path)
    for f in ("tracking.md", "tracking.csv", "tracking.png"):
        assert (tmp_path / f).exists() and (tmp_path / f).stat().st_size > 0


def test_track_trajectory_result_animates(tmp_path):
    quad = PlanarQuadrotor()
    traj = Circle(0.6, 6.0)

    class Hold:
        def reset(self): pass
        def update(self, x, dt): return quad.u_hover

    res = track_trajectory(quad, {"hold": Hold()}, traj,
                           np.array([0.6, 1.0, 0, 0, 0, 0]), dt=0.05, t_final=3.0,
                           pos_index=(0, 1))
    rep = res.animate(quad, controller="hold", fps=10)
    out = rep.save(tmp_path / "track.gif")
    assert out.exists() and out.stat().st_size > 0
