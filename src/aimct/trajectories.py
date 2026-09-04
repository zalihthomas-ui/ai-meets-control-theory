r"""Reference trajectories - shared by the RL envs, the live sandboxes, and the
trajectory-tracking experiments.

Every trajectory is a callable ``traj(t) -> (pos, vel, acc)`` where each is an
``(n,)`` array (n = 2 or 3). ``.duration`` is the nominal time to traverse it;
path-like trajectories also expose ``.length`` (metres) and ``.closest(p)``
(nearest point + arc length, for cross-track error).

    from aimct.trajectories import Lemniscate, MinimumJerk, Dubins, Spline
    traj = Lemniscate(A=0.6, B=0.35, period=6.0)
    p, v, a = traj(1.3)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Trajectory", "Setpoint", "Circle", "Lemniscate", "MinimumJerk",
    "Spline", "Dubins",
]


class Trajectory:
    """Base: subclasses implement :meth:`__call__`. Provides finite-difference
    ``vel``/``acc`` fallbacks and a default polyline sampling for path metrics."""

    duration: float = 1.0
    dim: int = 2

    def __call__(self, t: float):                       # -> (pos, vel, acc)
        raise NotImplementedError

    def pos(self, t):
        return np.asarray(self(t)[0], dtype=float)

    # -- path geometry (sampled; good enough for cross-track / completion) ----

    def _polyline(self, n: int = 400):
        ts = np.linspace(0.0, self.duration, n)
        return ts, np.array([self.pos(t) for t in ts])

    @property
    def length(self) -> float:
        _, P = self._polyline()
        return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))

    def closest(self, p):
        """Nearest sampled point on the path -> ``(point, arclength, frac)``."""
        _, P = self._polyline()
        p = np.asarray(p, dtype=float)[: P.shape[1]]
        d = np.linalg.norm(P - p, axis=1)
        i = int(np.argmin(d))
        s = float(np.sum(np.linalg.norm(np.diff(P[: i + 1], axis=0), axis=1)))
        return P[i], s, (s / max(self.length, 1e-9))


class Setpoint(Trajectory):
    """A constant target (a step, once you start away from it)."""

    def __init__(self, point):
        self.p = np.asarray(point, dtype=float)
        self.dim = self.p.size
        self.duration = 1.0

    def __call__(self, t):
        z = np.zeros(self.dim)
        return self.p.copy(), z, z


class Circle(Trajectory):
    def __init__(self, radius=1.0, period=6.0, center=(0.0, 0.0), z=None):
        self.r, self.w = float(radius), 2 * np.pi / float(period)
        self.c = np.asarray(center, dtype=float)
        self.z = z
        self.dim = 3 if z is not None else 2
        self.duration = float(period)

    def __call__(self, t):
        s, co = np.sin(self.w * t), np.cos(self.w * t)
        p = np.array([self.c[0] + self.r * co, self.c[1] + self.r * s])
        v = np.array([-self.r * self.w * s, self.r * self.w * co])
        a = np.array([-self.r * self.w**2 * co, -self.r * self.w**2 * s])
        if self.z is not None:
            p = np.append(p, self.z); v = np.append(v, 0.0); a = np.append(a, 0.0)
        return p, v, a


class Lemniscate(Trajectory):
    """Figure-8 (lemniscate of Gerono): ``x = A sin(wt)``, ``y = B sin(2wt)``.
    Optionally a third coordinate held at ``z0`` or oscillating."""

    def __init__(self, A=0.6, B=0.35, period=6.0, z0=None, Cz=0.0):
        self.A, self.B = float(A), float(B)
        self.w = 2 * np.pi / float(period)
        self.z0, self.Cz = z0, float(Cz)
        self.dim = 3 if z0 is not None else 2
        self.duration = float(period)

    def __call__(self, t):
        w = self.w
        s1, c1 = np.sin(w * t), np.cos(w * t)
        s2, c2 = np.sin(2 * w * t), np.cos(2 * w * t)
        p = [self.A * s1, self.B * s2]
        v = [self.A * w * c1, 2 * self.B * w * c2]
        a = [-self.A * w**2 * s1, -4 * self.B * w**2 * s2]
        if self.z0 is not None:
            p.append(self.z0 + self.Cz * s2)
            v.append(2 * self.Cz * w * c2)
            a.append(-4 * self.Cz * w**2 * s2)
        return np.array(p), np.array(v), np.array(a)


class MinimumJerk(Trajectory):
    """Point-to-point minimum-jerk (5th-order) segment from ``p0`` to ``p1`` in
    ``T`` seconds - zero velocity and acceleration at both ends."""

    def __init__(self, p0, p1, T=2.0):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.T = float(T)
        self.dim = self.p0.size
        self.duration = self.T

    def __call__(self, t):
        tau = np.clip(t / self.T, 0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.T
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / self.T**2
        d = self.p1 - self.p0
        return self.p0 + s * d, sd * d, sdd * d


class Spline(Trajectory):
    """Natural cubic spline through ``waypoints`` at times ``knots`` (default:
    evenly spaced, unit-speed-ish)."""

    def __init__(self, waypoints, knots=None):
        from scipy.interpolate import CubicSpline

        P = np.asarray(waypoints, dtype=float)
        if knots is None:
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            knots = np.concatenate([[0.0], np.cumsum(np.maximum(seg, 1e-6))])
        self.cs = CubicSpline(knots, P, bc_type="natural")
        self.dim = P.shape[1]
        self.duration = float(knots[-1])

    def __call__(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        return (self.cs(t), self.cs(t, 1), self.cs(t, 2))


class Dubins(Trajectory):
    """A line -> arc -> line path in the plane (a lightweight Dubins segment):
    straight to a turn point, a constant-radius arc through ``turn`` rad, then
    straight again. Traversed at constant speed ``v``. Good for a car / robot."""

    def __init__(self, start=(0.0, 0.0), heading=0.0, straight1=1.0,
                 radius=0.6, turn=np.pi / 2, straight2=1.0, v=0.5):
        self.v = float(v)
        p = np.asarray(start, dtype=float)
        th = float(heading)
        segs = []                                        # (kind, params, length)
        d1 = np.array([np.cos(th), np.sin(th)])
        segs.append(("line", (p.copy(), d1), straight1))
        p = p + d1 * straight1
        sgn = np.sign(turn) or 1.0
        cx = p + radius * np.array([-np.sin(th) * sgn, np.cos(th) * sgn])
        segs.append(("arc", (cx, radius, th, sgn), abs(turn) * radius))
        th2 = th + turn
        p = cx + radius * np.array([np.sin(th2) * sgn, -np.cos(th2) * sgn])
        d2 = np.array([np.cos(th2), np.sin(th2)])
        segs.append(("line", (p.copy(), d2), straight2))
        self.segs = segs
        self.total_len = sum(s[2] for s in segs)
        self.duration = self.total_len / self.v
        self.dim = 2

    def __call__(self, t):
        s = np.clip(self.v * t, 0.0, self.total_len)
        for kind, prm, L in self.segs:
            if s <= L + 1e-9:
                if kind == "line":
                    p0, d = prm
                    return p0 + d * s, d * self.v, np.zeros(2)
                cx, r, th0, sgn = prm
                a = th0 + sgn * s / r                     # heading along the arc
                pos = cx + r * np.array([np.sin(a) * sgn, -np.cos(a) * sgn])
                tang = np.array([np.cos(a), np.sin(a)])
                acc = (self.v**2 / r) * (cx - pos) / np.linalg.norm(cx - pos)
                return pos, tang * self.v, acc
            s -= L
        p0, d = self.segs[-1][1]
        return p0 + d * self.segs[-1][2], d * self.v, np.zeros(2)
