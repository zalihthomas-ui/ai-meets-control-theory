r"""Live differential-drive sandbox — path followers vs. a shove.

The interactive counterpart of Experiment 22.  A
:class:`aimct.systems.DifferentialDriveRobot` drives a fixed figure-8 loop
under one of three path-followers while **you** shove it off course (arrow
keys) and steal wheel traction (slip slider).  Watch how each recovers:

* **Pure pursuit** — chases a look-ahead point; smooth, cuts corners, forgives
  a big lateral error but drifts back slowly.
* **Stanley** — nulls heading error plus an ``atan`` cross-track term at the
  front axle; snaps back to the line hard, can wag near high curvature.
* **Path LQR** — LQR on the path-error state (``y, θ, v, ω``) with a curvature
  feed-forward; the balanced option.

Run:  python experiments/live_diffdrive/run.py     (or:  python -m aimct live diffdrive)
      python experiments/live_diffdrive/run.py --headless
"""

from __future__ import annotations

import sys

import numpy as np

from aimct.controllers import LQR
from aimct.controllers.swingup import wrap_angle
from aimct.systems import DifferentialDriveRobot
from aimct.trajectories import Lemniscate
from aimct.viz import Disturbance, Sandbox

ROBOT = DifferentialDriveRobot()
V = 0.16                                  # cruise speed [m/s]
DT = 0.05
LOOP = Lemniscate(A=1.2, B=0.7, period=1.0)     # period rescaled below by speed


class Path:
    """A closed reference loop with the nearest-point / look-ahead / curvature
    queries the followers need, all off one memoised polyline.

    A figure-8 self-intersects, so a *global* nearest-point search is
    ambiguous right at the crossing: two arclength-distant samples sit at the
    same spot, and a per-frame argmin can flip between them - the look-ahead
    point (and the Stanley/LQR cross-track term) then jumps discontinuously
    ("the ball teleports"). :meth:`nearest` fixes this with **progress
    hysteresis**: after the first call it only searches a window of the
    polyline around where it was last found, so it tracks the branch the
    robot is actually on instead of re-resolving the ambiguity every frame.
    """

    def __init__(self, traj, n=1400, window_frac: float = 0.05):
        s = np.linspace(0.0, traj.duration, n)
        self.P = np.array([traj.pos(t) for t in s])
        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.S = np.concatenate([[0.0], np.cumsum(d)])
        self.length = float(self.S[-1])
        # curvature per sample from finite differences of the polyline
        dP = np.gradient(self.P, axis=0)
        ddP = np.gradient(dP, axis=0)
        sp = np.hypot(dP[:, 0], dP[:, 1]) + 1e-12
        self.kappa = (dP[:, 0] * ddP[:, 1] - dP[:, 1] * ddP[:, 0]) / sp ** 3
        self.tang = np.arctan2(dP[:, 1], dP[:, 0])
        self._n = len(self.P)
        self._window = max(30, int(window_frac * self._n))
        self._last: int | None = None

    def reset_progress(self) -> None:
        """Force the next :meth:`nearest` to do a full search (e.g. a hard
        teleport / re-spawn, as opposed to an ordinary shove)."""
        self._last = None

    def nearest(self, p):
        p = np.asarray(p, float)
        if self._last is None:
            i = int(np.argmin(np.hypot(self.P[:, 0] - p[0], self.P[:, 1] - p[1])))
        else:
            idx = (np.arange(self._last - self._window,
                             self._last + self._window + 1) % self._n)
            local = np.argmin(np.hypot(self.P[idx, 0] - p[0], self.P[idx, 1] - p[1]))
            i = int(idx[local])
        self._last = i
        return i

    def frame(self, p):
        """``(point, path_heading, signed_cross_track, curvature, arclen)``."""
        i = self.nearest(p)
        th = self.tang[i]
        d = np.asarray(p, float)[:2] - self.P[i]
        e_cross = -(np.sin(th) * d[0] - np.cos(th) * d[1])
        return self.P[i], th, float(e_cross), float(self.kappa[i]), float(self.S[i])

    def lookahead(self, p, ld):
        i = self.nearest(p)
        target = (self.S[i] + ld) % self.length
        k = int(np.searchsorted(self.S, target))
        return self.P[min(k, len(self.P) - 1)]


PATH = Path(Lemniscate(A=1.2, B=0.7, period=10.0))


# --------------------------------------------------------------- controllers
class PurePursuit:
    name = "Pure pursuit"
    ld = 0.5

    def reset(self):
        pass

    def update(self, x, dt):
        p, th = x[:2], x[2]
        tgt = PATH.lookahead(p, self.ld)
        alpha = wrap_angle(np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - th)
        return np.array([V, 2.0 * V * np.sin(alpha) / self.ld]), tgt


class Stanley:
    name = "Stanley"
    k_e, k_gain = 1.6, 2.4

    def reset(self):
        pass

    def update(self, x, dt):
        _, th_path, e_cross, _, _ = PATH.frame(x[:2])
        psi = wrap_angle(th_path - x[2])
        delta = psi - np.arctan2(self.k_e * e_cross, V)
        return np.array([V, self.k_gain * delta]), None


class PathLQR:
    name = "Path LQR"

    def __init__(self):
        A, B = ROBOT.linearize()                       # straight path at v_ref
        idx = np.array([1, 2, 3, 4])                   # [y, theta, v, omega] error
        self.K = LQR(A[np.ix_(idx, idx)], B[idx, :],
                     np.diag([25.0, 8.0, 1.0, 0.3]), np.diag([2.0, 1.0])).K

    def reset(self):
        pass

    def update(self, x, dt):
        _, th_path, e_cross, kappa, _ = PATH.frame(x[:2])
        e = np.array([e_cross, wrap_angle(x[2] - th_path), x[3] - V,
                      x[4] - V * kappa])
        u = np.array([V, V * kappa]) - self.K @ e
        return u, None


def _clip_cmd(u):
    return np.array([np.clip(u[0], -ROBOT.v_max, ROBOT.v_max),
                     np.clip(u[1], -ROBOT.omega_max, ROBOT.omega_max)])


class _Wrapped:
    """Adapt a follower's ``(u, lookahead)`` return to the Controller protocol
    and stash the look-ahead point for the artist."""

    def __init__(self, inner, box_ref):
        self.inner = inner
        self.name = inner.name
        self.box = box_ref

    def reset(self):
        self.inner.reset()

    def update(self, x, dt):
        u, la = self.inner.update(np.asarray(x, float), dt)
        self.box["lookahead"] = la
        return _clip_cmd(u)


# --------------------------------------------------------------- disturbances
def _slip(t, x, u, knobs):
    """Wheel-slip: a fraction of commanded speed is lost, as an extra drag on
    the actual body speed / yaw rate."""
    frac = knobs.get("wheel slip", 0.0)
    return np.array([0.0, 0.0, 0.0, -frac * x[3] / ROBOT.tau_v,
                     -frac * x[4] / ROBOT.tau_omega])


def build():
    shared = {"lookahead": None}
    followers = [PurePursuit(), Stanley(), PathLQR()]
    ctrls = {f.name: _Wrapped(f, shared) for f in followers}
    x0 = np.array([PATH.P[0, 0], PATH.P[0, 1] - 0.15, 0.0, 0.0, 0.0])
    box = Sandbox(
        ROBOT, ctrls, x0=x0, dt=DT, substeps=6,
        path=PATH.P,
        title="Live diff-drive - path followers vs. a shove",
        disturbance=Disturbance(
            sliders=[("wheel slip", 0.0, 0.6, 0.0)],
            hotkeys=[
                ("left", "shove -x", lambda s: s.kick([-0.12, 0, 0, 0, 0])),
                ("right", "shove +x", lambda s: s.kick([0.12, 0, 0, 0, 0])),
                ("up", "shove +y", lambda s: s.kick([0, 0.12, 0, 0, 0])),
                ("down", "shove -y", lambda s: s.kick([0, -0.12, 0, 0, 0])),
                ("t", "spin", lambda s: s.kick([0, 0, 0.8, 0, 0])),
            ],
            xdot_extra=_slip,
            help_text="arrows: shove   t: spin   slider: wheel slip",
        ),
        aux_extra=lambda s: {
            "cross_track_mm": abs(PATH.frame(s.x[:2])[2]) * 1e3,
            **({"lookahead": shared["lookahead"]}
               if shared["lookahead"] is not None else {}),
        },
    )
    return box, shared


def _headless() -> int:
    print("live_diffdrive headless check - follow the figure-8, then a shove "
          "+ 30% wheel slip\n")
    from aimct.viz import get_artist

    a = get_artist(ROBOT)
    for name in ("Pure pursuit", "Stanley", "Path LQR"):
        box, _ = build()
        box.set_controller(name)
        box.knobs["wheel slip"] = 0.0
        xt = []
        for k in range(700):
            if k == 200:
                box.kick([0.25, -0.15, 0, 0, 0])       # a hard shove
                box.knobs["wheel slip"] = 0.30
            box.step()
            _, _, e_cross, _, _ = PATH.frame(box.x[:2])
            xt.append(abs(e_cross))
        recov = np.array(xt)
        settle = next((k for k in range(210, len(recov)) if recov[k] < 0.03), None)
        print(f"  {name:<14}  peak x-track {recov[200:].max() * 1e3:5.0f} mm   "
              f"back < 30 mm after {'-' if settle is None else f'{(settle-200)*DT:.1f} s'}")
    print("\nheadless check OK")
    return 0


def main() -> int:
    if "--headless" in sys.argv:
        return _headless()
    box, _ = build()
    box.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
