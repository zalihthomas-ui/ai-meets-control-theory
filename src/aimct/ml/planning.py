r"""Vectorised one-step models for :class:`aimct.controllers.SamplingMPC`.

The CEM planner rolls hundreds of action sequences per control step, so it needs
a ``step(X, U) -> X_next`` that works on a whole batch at once. The analytic
dynamics of the Phase-0 reference systems are written here in batched form
(columns instead of tuple-unpacking) with a batched RK4; ``system_step`` picks
the right one, falling back to a slow per-row loop for anything else.

These batched equations mirror the corresponding ``aimct.systems`` classes
exactly -- the tests assert they agree with ``system.dynamics``.
"""

from __future__ import annotations

import numpy as np

from ..systems import CartPole, MassSpringDamper, Pendulum, PlanarQuadrotor

__all__ = ["batched_rk4", "system_step"]


def batched_rk4(fbatch, X, U, dt):
    """One RK4 step for a batched field ``fbatch(X, U) -> Xdot`` (all ``(B, n)``)."""
    k1 = fbatch(X, U)
    k2 = fbatch(X + 0.5 * dt * k1, U)
    k3 = fbatch(X + 0.5 * dt * k2, U)
    k4 = fbatch(X + dt * k3, U)
    return X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _msd_field(sys: MassSpringDamper):
    def f(X, U):
        pos, vel = X[:, 0], X[:, 1]
        acc = (U[:, 0] - sys.c * vel - sys.k * pos) / sys.m
        return np.stack([vel, acc], axis=1)
    return f


def _pendulum_field(sys: Pendulum):
    inertia = sys.m * sys.L**2

    def f(X, U):
        theta, omega = X[:, 0], X[:, 1]
        domega = (-(sys.g / sys.L) * np.sin(theta)
                  - (sys.b / inertia) * omega + U[:, 0] / inertia)
        return np.stack([omega, domega], axis=1)
    return f


def _cartpole_field(sys: CartPole):
    mc, mp, l, g = sys.mc, sys.mp, sys.l, sys.g
    total = mc + mp

    def f(X, U):
        xdot, th, thdot = X[:, 1], X[:, 2], X[:, 3]
        f_ = U[:, 0]
        s, c = np.sin(th), np.cos(th)
        temp = (f_ + mp * l * thdot**2 * s) / total
        thddot = (g * s - c * temp) / (l * (4.0 / 3.0 - mp * c**2 / total))
        xddot = temp - mp * l * thddot * c / total
        return np.stack([xdot, xddot, thdot, thddot], axis=1)
    return f


def _quadrotor_field(sys: PlanarQuadrotor):
    m, cd, g, l, Iyy = sys.m, sys.cd, sys.g, sys.l, sys.Iyy

    def f(X, U):
        th, xd, zd = X[:, 2], X[:, 3], X[:, 4]
        thd = X[:, 5]
        T = U[:, 0] + U[:, 1]
        s, c = np.sin(th), np.cos(th)
        xdd = -T * s / m - cd * xd / m
        zdd = T * c / m - g - cd * zd / m
        thdd = (U[:, 0] - U[:, 1]) * l / Iyy
        return np.stack([xd, zd, thd, xdd, zdd, thdd], axis=1)
    return f


_FIELDS = {
    MassSpringDamper: _msd_field,
    Pendulum: _pendulum_field,
    CartPole: _cartpole_field,
    PlanarQuadrotor: _quadrotor_field,
}


def system_step(system, dt: float):
    """Return a batched ``step(X, U) -> X_next`` (one RK4 step of ``dt``).

    Uses a vectorised analytic field for the known reference systems; otherwise
    falls back to a per-row loop over :func:`aimct.simulate.rk4_step`.
    """
    field_factory = _FIELDS.get(type(system))
    if field_factory is not None:
        f = field_factory(system)
        return lambda X, U: batched_rk4(f, np.atleast_2d(X), np.atleast_2d(U), dt)

    from ..simulate import rk4_step

    def slow(X, U):
        X, U = np.atleast_2d(X), np.atleast_2d(U)
        return np.array([rk4_step(system.dynamics, 0.0, x, u, dt) for x, u in zip(X, U)])

    return slow
