"""aimct.viz — replay animator, system artists, interactive sandbox (headless)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from aimct.benchmarks import track_trajectory
from aimct.controllers import LQR
from aimct.simulate import simulate
from aimct.systems import (
    CartPole,
    DCMotor,
    DifferentialDriveRobot,
    Pendulum,
    PlanarQuadrotor,
    TwoLinkArm,
)
from aimct.trajectories import Lemniscate
from aimct.viz import Disturbance, Sandbox, animate, get_artist, has_artist
from aimct.viz.artists import (
    CartPoleArtist,
    DiffDriveArtist,
    PendulumArtist,
    PlanarQuadrotorArtist,
    TwoLinkArmArtist,
)

_CASES = [
    (Pendulum(), np.array([0.4, -0.2]), np.array([0.3]), PendulumArtist),
    (CartPole(), np.array([0.2, 0.0, 0.15, 0.0]), np.array([1.0]), CartPoleArtist),
    (TwoLinkArm(), np.array([0.3, 0.5, 0.0, 0.0]), np.array([1.0, -0.5]), TwoLinkArmArtist),
    (DifferentialDriveRobot(), np.array([0.0, 0.0, 0.2, 0.1, 0.05]),
     np.array([0.15, 0.3]), DiffDriveArtist),
    (PlanarQuadrotor(), np.array([0.1, 1.0, 0.05, 0.0, 0.0, 0.0]),
     np.array([0.14, 0.13]), PlanarQuadrotorArtist),
]


@pytest.mark.parametrize("system, x, u, artist_cls", _CASES,
                         ids=lambda v: getattr(v, "__name__", type(v).__name__))
def test_every_registered_artist_builds_and_draws(system, x, u, artist_cls):
    import matplotlib.pyplot as plt

    assert has_artist(system)
    art = get_artist(system)
    assert isinstance(art, artist_cls)

    fig, ax = plt.subplots()
    (xlim, ylim) = art.bounds(np.atleast_2d(x))
    assert xlim[0] < xlim[1] and ylim[0] < ylim[1]
    handles = art.build(ax)
    assert handles                                   # created some artists
    aux = {"trail": np.array([art.position(x), art.position(x)])}
    art.draw(x, u, 0.0, aux)                          # must not raise
    lines = art.hud_lines(x, u, 1.23, aux)
    assert lines and all(isinstance(s, str) for s in lines)
    p = art.position(x)
    assert np.asarray(p, float).shape == (2,)
    plt.close(fig)


def test_get_artist_raises_for_an_unregistered_system():
    with pytest.raises(LookupError):
        get_artist(DCMotor())
    assert not has_artist(DCMotor())


def _pendulum_swing():
    p = Pendulum()
    A, B = p.linearize()
    K = LQR(A, B, np.diag([10.0, 1.0]), np.array([[0.5]])).K

    class Ctl:
        def reset(self):
            pass

        def update(self, x, dt):
            return float((-K @ np.array([x[0] - np.pi, x[1]]))[0])

    tr = simulate(p, Ctl(), x0=np.array([0.5, 0.0]), dt=0.02, t_final=3.0)
    return p, tr


def test_animate_returns_a_playable_replay_and_frames_render():
    p, tr = _pendulum_swing()
    rep = animate(tr, p, fps=15, title="swing-up")
    assert rep.fig is not None and rep.ax is not None
    # drive a handful of frames by hand — the frame fn must be pure/no-raise
    for i in (0, 1, 5, 10):
        rep.anim._func(i)
    assert "swing-up" in rep.ax.get_title()
    import matplotlib.pyplot as plt
    plt.close(rep.fig)


@pytest.mark.slow
def test_animate_saves_a_gif(tmp_path):
    p, tr = _pendulum_swing()
    out = animate(tr, p, fps=12).save(tmp_path / "swing.gif")
    assert out.exists() and out.stat().st_size > 0


def test_animate_accepts_a_tracking_result():
    quad = PlanarQuadrotor()
    A, B = quad.linearize()
    K = LQR(A, B, np.diag([6.0, 6.0, 2.0, 1.0, 1.0, 1.0]), np.eye(2)).K
    traj = Lemniscate(0.5, 0.3, 8.0)

    class T:
        def __init__(self):
            self._t = 0.0

        def reset(self):
            self._t = 0.0

        def update(self, x, dt):
            p = traj.pos(self._t)
            self._t += dt
            xr = np.array([p[0], p[1] + 1.0, 0, 0, 0, 0])
            return np.clip(quad.u_hover - K @ (np.asarray(x) - xr), 0, quad.thrust_max)

    res = track_trajectory(quad, {"lqr": T()}, traj, np.array([0.0, 1.0, 0, 0, 0, 0]),
                           dt=0.02, t_final=8.0, pos_index=(0, 1))
    rep = animate(res, quad, fps=12)                  # ref comes from the result
    rep.anim._func(3)
    import matplotlib.pyplot as plt
    plt.close(rep.fig)


def test_animate_needs_a_controller_name_when_ambiguous():
    quad = PlanarQuadrotor()
    traj = Lemniscate(0.5, 0.3, 8.0)

    class Z:
        def reset(self): pass
        def update(self, x, dt): return quad.u_hover

    res = track_trajectory(quad, {"a": Z(), "b": Z()}, traj,
                           np.array([0.0, 1.0, 0, 0, 0, 0]), dt=0.05, t_final=2.0)
    with pytest.raises(ValueError):
        animate(res, quad)


def test_sandbox_headless_runs_and_scores():
    arm = TwoLinkArm()
    target = np.array([0.45, 0.25])

    class PD:
        def reset(self):
            pass

        def update(self, x, dt):
            q, dq = np.asarray(x)[:2], np.asarray(x)[2:]
            return arm.G(q) + 50.0 * (_ik(arm, target) - q) - 10.0 * dq

    box = Sandbox(arm, {"pd": PD()}, x0=np.array([0.2, 0.4, 0.0, 0.0]),
                  target=target, dt=0.02, substeps=4)
    res = box.headless(steps=200, quiet=True)
    assert np.isfinite(res["mean_err_tail_mm"])
    assert res["final_state"].shape == (4,)


def test_sandbox_disturbance_hooks_fire():
    arm = TwoLinkArm()
    seen = {}

    class Hold:
        def reset(self): pass
        def update(self, x, dt): return arm.G(np.asarray(x)[:2])

    dist = Disturbance(
        sliders=[("payload [kg]", 0.0, 0.5, 0.2)],
        hotkeys=[("p", "poke", lambda s: s.kick([0, 0, 2.0, -2.0]))],
        on_slider=lambda s, name, v: seen.__setitem__(name, v),
    )
    box = Sandbox(arm, {"hold": Hold()}, x0=np.zeros(4), disturbance=dist)
    assert seen.get("payload [kg]") == 0.2             # on_slider ran at start-up
    box.knobs["payload [kg]"] = 0.5
    x_before = box.x.copy()
    dist.hotkeys[0][2](box)                            # simulate the key press
    assert not np.allclose(box.x, x_before)


def _press(fig, key):
    from matplotlib.backend_bases import KeyEvent

    fig.canvas.callbacks.process("key_press_event",
                                 KeyEvent("key_press_event", fig.canvas, key))


def test_sandbox_creative_additives_help_surprise_snapshot(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from aimct.systems import Pendulum

    monkeypatch.chdir(tmp_path)
    p = Pendulum()

    class Hold:
        def reset(self): pass
        def update(self, x, dt): return 0.0

    seen = []
    dist = Disturbance(
        sliders=[("bias", -1.0, 1.0, 0.0)],
        hotkeys=[("p", "poke", lambda s: seen.append(s.t))],
        on_slider=lambda s, name, v: None,
        help_text="test disturbance",
    )
    box = Sandbox(p, {"hold": Hold()}, x0=np.array([0.1, 0.0]),
                  target=np.array([0.0, -1.0]), disturbance=dist)
    orig_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        box.run()
    finally:
        plt.show = orig_show
    fig = box._anim._fig
    box._anim._func(0)                                    # populate best_mm
    assert box._best_mm and np.isfinite(next(iter(box._best_mm.values())))

    _press(fig, "h")                                       # help overlay toggles
    help_texts = [t for t in fig.axes[0].texts if "controllers" in t.get_text()]
    assert help_texts and help_texts[0].get_visible()
    _press(fig, "h")
    assert not help_texts[0].get_visible()

    before = box.knobs["bias"]
    _press(fig, "g")                                       # surprise me
    assert box.knobs["bias"] != before or len(seen) > 0     # slider moved or hotkey fired

    _press(fig, "c")                                        # snapshot
    saved = list((tmp_path / "snapshots").glob("*.png"))
    assert len(saved) == 1 and saved[0].stat().st_size > 0
    plt.close(fig)


def _ik(arm, target):
    """crude 2-link inverse kinematics (elbow-down)."""
    x, y = target
    r2 = x * x + y * y
    c2 = np.clip((r2 - arm.l1**2 - arm.l2**2) / (2 * arm.l1 * arm.l2), -1, 1)
    q2 = np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(arm.l2 * np.sin(q2),
                                      arm.l1 + arm.l2 * np.cos(q2))
    return np.array([q1, q2])
