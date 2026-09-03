r"""Gymnasium adapter for :mod:`aimct.systems`.

``ControlEnv`` wraps any :class:`~aimct.systems.base.DynamicalSystem` as a
standard Gymnasium environment: a continuous ``Box`` action equal to the input
bounds, an observation that is the full state (or ``obs_fn(x)``), a fixed-step
RK4 transition (:func:`aimct.simulate.rk4_step`), a quadratic reward
``-(x^T Q x + u^T R u)`` by default, ``terminated`` on leaving a state box and
``truncated`` at ``max_steps``.

Two ready tasks are in :data:`TASKS`; build them with :func:`make`::

    env = make("pendulum-swingup")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - gymnasium is an ml extra
    raise ModuleNotFoundError(
        "aimct.rl needs the 'ml' extra: pip install -e '.[ml]'"
    ) from exc

from ..simulate import rk4_step
from ..systems.base import DynamicalSystem
from ..systems import CartPole, Pendulum, PlanarQuadrotor

__all__ = ["ControlEnv", "TASKS", "make", "wrap_to_pi",
           "figure8_reference", "figure8_obs", "FIGURE8_PERIOD"]

FIGURE8_PERIOD = 6.0                                  # seconds, shared by the ref + obs

# 1 / typical magnitude of each obs channel, so the policy net sees O(1) inputs
# (small pitch errors would otherwise be swamped by the O(1) phase clock).
_F8_OBS_SCALE = np.array([10.0, 10.0, 12.0, 2.0, 2.0, 0.6, 1.0, 1.0])


def wrap_to_pi(a):
    """Wrap angle(s) to ``(-pi, pi]``."""
    return np.pi - (np.pi - np.asarray(a, dtype=float)) % (2.0 * np.pi)


def figure8_reference(t, quad, *, A=0.55, B=0.30, period=FIGURE8_PERIOD, z0=1.0):
    """Lemniscate (figure-8) reference for :class:`PlanarQuadrotor`.

    ``x(t) = A sin(w t)``, ``z(t) = z0 + B sin(2 w t)``; pitch and the thrust
    feed-forward come from differential flatness (small-angle
    ``xdd ~ -g theta``).  Returns ``(x_ref[6], u_ref[2])`` with ``u_ref``
    clipped to ``[0, thrust_max]`` - the same trajectory flown in
    Experiments 14 and 20.
    """
    w = 2.0 * np.pi / period
    g, m, Iyy, l = quad.g, quad.m, quad.Iyy, quad.l
    s1, c1 = np.sin(w * t), np.cos(w * t)
    s2, c2 = np.sin(2 * w * t), np.cos(2 * w * t)
    x, xd, xdd = A * s1, A * w * c1, -A * w**2 * s1
    xddd, xdddd = -A * w**3 * c1, A * w**4 * s1
    z, zd, zdd = z0 + B * s2, 2 * B * w * c2, -4 * B * w**2 * s2
    th, thd, thdd = -xdd / g, -xddd / g, -xdddd / g
    u_ref = np.array([0.5 * m * (g + zdd) + 0.5 * Iyy * thdd / l,
                      0.5 * m * (g + zdd) - 0.5 * Iyy * thdd / l])
    x_ref = np.array([x, z, th, xd, zd, thd])
    return x_ref, np.clip(u_ref, 0.0, quad.thrust_max)


def figure8_obs(state, t, quad) -> np.ndarray:
    """Observation for the figure-8 tracking task / policy: the 6-D state error
    to the moving reference plus a ``(sin, cos)`` phase clock, each channel
    scaled to O(1).  The single source of truth shared by the ``quad-figure8-
    track`` env and any wrapper that reconstructs the obs from a raw state."""
    w = 2.0 * np.pi / FIGURE8_PERIOD
    xr, _ = figure8_reference(t, quad)
    raw = np.concatenate([np.asarray(state, float) - xr,
                          [np.sin(w * t), np.cos(w * t)]])
    return raw * _F8_OBS_SCALE


def _n_pos_args(fn) -> int:
    """Count the positional parameters a callable accepts (``*args`` -> big)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):       # builtins / C funcs: assume 1
        return 1
    n = 0
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            return 99
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


class ControlEnv(gym.Env):
    """A ``DynamicalSystem`` as a Gymnasium environment.

    Parameters
    ----------
    system:
        The plant.
    dt:
        Integration / control step (seconds).
    max_steps:
        Episode truncation length.
    action_bounds:
        ``(low, high)`` scalar or length-``n_inputs`` arrays -> the ``Box``
        action space.
    Q, R:
        Weights for the default reward ``-(x^T Q x + u^T R u)`` (identity if
        omitted).  Ignored when ``reward_fn`` is given.
    reward_fn:
        ``reward_fn(x, u, x_next) -> float`` overriding the default.
    obs_fn:
        ``obs_fn(x) -> obs`` (default: the full state).  ``obs_bounds`` sets the
        observation ``Box`` (``+/- inf`` otherwise).
    obs_bounds, state_bounds:
        ``(low, high)`` arrays.  Leaving ``state_bounds`` triggers
        ``terminated``; ``None`` disables it.
    x0, x0_fn, reset_noise:
        Initial state: fixed ``x0``, or ``x0_fn(np_random) -> x0``; plus optional
        uniform ``+/- reset_noise`` (scalar or per-state).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        system: DynamicalSystem,
        *,
        dt: float,
        max_steps: int,
        action_bounds: tuple,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        reward_fn: Callable | None = None,
        obs_fn: Callable | None = None,
        obs_bounds: tuple | None = None,
        state_bounds: tuple | None = None,
        terminated_fn: Callable | None = None,
        x0: np.ndarray | None = None,
        x0_fn: Callable | None = None,
        reset_noise: float | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.nx, self.nu = system.n_states, system.n_inputs

        lo, hi = action_bounds
        self._a_lo = np.broadcast_to(np.asarray(lo, np.float32), (self.nu,)).copy()
        self._a_hi = np.broadcast_to(np.asarray(hi, np.float32), (self.nu,)).copy()
        self.action_space = spaces.Box(self._a_lo, self._a_hi, dtype=np.float32)

        self._obs_fn = obs_fn or (lambda x: np.asarray(x, dtype=float))
        # obs_fn / reward_fn / terminated_fn may optionally take the episode time
        # as a trailing positional arg (time-varying references); detect it once.
        self._obs_t = _n_pos_args(self._obs_fn) >= 2
        self._reward_t = reward_fn is not None and _n_pos_args(reward_fn) >= 4
        self._term_t = terminated_fn is not None and _n_pos_args(terminated_fn) >= 2
        self._terminated_fn = terminated_fn
        obs_dim = int(np.asarray(self._call_obs(np.zeros(self.nx), 0.0)).size)
        if obs_bounds is None:
            o_lo = np.full(obs_dim, -np.inf, np.float32)
            o_hi = np.full(obs_dim, np.inf, np.float32)
        else:
            o_lo = np.broadcast_to(np.asarray(obs_bounds[0], np.float32), (obs_dim,)).copy()
            o_hi = np.broadcast_to(np.asarray(obs_bounds[1], np.float32), (obs_dim,)).copy()
        self.observation_space = spaces.Box(o_lo, o_hi, dtype=np.float32)

        self._Q = np.eye(self.nx) if Q is None else np.atleast_2d(np.asarray(Q, float))
        self._R = np.eye(self.nu) if R is None else np.atleast_2d(np.asarray(R, float))
        self._reward_fn = reward_fn

        if state_bounds is None:
            self._s_lo = self._s_hi = None
        else:
            self._s_lo = np.broadcast_to(np.asarray(state_bounds[0], float), (self.nx,)).copy()
            self._s_hi = np.broadcast_to(np.asarray(state_bounds[1], float), (self.nx,)).copy()

        self._x0 = None if x0 is None else np.asarray(x0, float).reshape(self.nx)
        self._x0_fn = x0_fn
        self._noise = (None if reset_noise is None
                       else np.broadcast_to(np.asarray(reset_noise, float), (self.nx,)).copy())

        self._x = np.zeros(self.nx)
        self._t = 0
        self._steps = 0

    # ------------------------------------------------------------------ api

    def _call_obs(self, x, t):
        return self._obs_fn(x, t) if self._obs_t else self._obs_fn(x)

    def _obs(self) -> np.ndarray:
        return np.asarray(self._call_obs(self._x, self._t), dtype=np.float32).ravel()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._x0_fn is not None:
            x = np.asarray(self._x0_fn(self.np_random), dtype=float).reshape(self.nx)
        elif self._x0 is not None:
            x = self._x0.copy()
        else:
            x = np.zeros(self.nx)
        if self._noise is not None:
            x = x + self.np_random.uniform(-self._noise, self._noise)
        self._x = x
        self._steps = 0
        self._t = 0.0
        return self._obs(), {}

    def _reward(self, x, u, x_next, t) -> float:
        if self._reward_fn is not None:
            return float(self._reward_fn(x, u, x_next, t) if self._reward_t
                         else self._reward_fn(x, u, x_next))
        return float(-(x @ self._Q @ x + u @ self._R @ u))

    def step(self, action):
        u = np.clip(np.atleast_1d(np.asarray(action, dtype=float)),
                    self._a_lo, self._a_hi).reshape(self.nu)
        x = self._x
        x_next = rk4_step(self.system.dynamics, self._t, x, u, self.dt)
        t_next = self._t + self.dt                 # time that x_next lives at

        reward = self._reward(x, u, x_next, t_next)
        terminated = bool(
            self._s_lo is not None
            and (np.any(x_next < self._s_lo) or np.any(x_next > self._s_hi))
        )
        if self._terminated_fn is not None:
            terminated = terminated or bool(
                self._terminated_fn(x_next, t_next) if self._term_t
                else self._terminated_fn(x_next)
            )
        self._x = x_next
        self._t = t_next
        self._steps += 1
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"state": x_next.copy()}

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()


# --------------------------------------------------------------------- tasks

@dataclass
class TaskSpec:
    system_factory: Callable[[], DynamicalSystem]
    kwargs: dict = field(default_factory=dict)


def _cartpole_balance() -> dict:
    Q = np.diag([1.0, 0.1, 10.0, 0.1])
    R = np.array([[1e-3]])
    return dict(
        dt=0.02, max_steps=500, action_bounds=(-20.0, 20.0), Q=Q, R=R,
        state_bounds=([-2.4, -np.inf, -0.8, -np.inf], [2.4, np.inf, 0.8, np.inf]),
        x0=np.zeros(4), reset_noise=np.array([0.05, 0.05, 0.05, 0.05]),
    )


def _pendulum_swingup() -> dict:
    def reward(x, u, x_next):
        err = wrap_to_pi(x_next[0] - np.pi)        # 0 at upright, +-pi hanging
        return float(-(err ** 2) - 1e-3 * u[0] ** 2)

    def obs_fn(x):
        # (cos, sin, omega): no wrap discontinuity for a value-function grid
        return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])

    return dict(
        dt=0.05, max_steps=250, action_bounds=(-4.0, 4.0),
        reward_fn=reward, obs_fn=obs_fn,
        obs_bounds=([-1.0, -1.0, -10.0], [1.0, 1.0, 10.0]),
        x0=np.array([0.0, 0.0]), reset_noise=np.array([0.05, 0.05]),
    )


def _quad_figure8_track() -> dict:
    """Planar quadrotor chasing the Exp-14/20 figure-8.  Obs is the 6-D state
    error to the moving reference plus a ``(sin, cos)`` phase clock; reward
    punishes position error, pitch and thrust effort; the episode ends early if
    the quad falls > 1.5 m behind the reference or pitches past 1 rad."""
    quad = PlanarQuadrotor()
    uh = quad.u_hover
    tmax = quad.thrust_max
    # attitude regulation is the hard part on this feather-light pitch axis, so
    # keep a small explicit pitch-rate term alongside the pitch penalty; position
    # tracking and effort are the softer terms.
    w_pos, w_pitch, w_rate, w_eff = 8.0, 2.0, 0.15, 0.2

    def obs_fn(x, t):
        return figure8_obs(x, t, quad)

    def reward(x, u, x_next, t):
        xr, _ = figure8_reference(t, quad)
        xn = np.asarray(x_next, float)
        pe = xn[:2] - xr[:2]
        eff = (np.asarray(u, float) - uh) / tmax
        return float(-(w_pos * pe @ pe
                       + w_pitch * xn[2] ** 2
                       + w_rate * xn[5] ** 2
                       + w_eff * eff @ eff))

    def terminated(x_next, t):
        xr, _ = figure8_reference(t, quad)
        pe = np.asarray(x_next, float)[:2] - xr[:2]
        return bool(np.hypot(*pe) > 1.5 or abs(float(x_next[2])) > 1.0)

    x0 = figure8_reference(0.0, quad)[0]
    return dict(
        dt=0.02, max_steps=600, action_bounds=(0.0, tmax),
        reward_fn=reward, obs_fn=obs_fn, terminated_fn=terminated,
        x0=x0, reset_noise=np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]),
    )


TASKS: dict[str, TaskSpec] = {
    "cartpole-balance": TaskSpec(CartPole, _cartpole_balance()),
    "pendulum-swingup": TaskSpec(Pendulum, _pendulum_swingup()),
    "quad-figure8-track": TaskSpec(PlanarQuadrotor, _quad_figure8_track()),
}


def make(task: str, **overrides) -> ControlEnv:
    """Build a registered :class:`ControlEnv` task; ``overrides`` replace config
    keys (e.g. ``make("pendulum-swingup", max_steps=400))``."""
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; choices: {sorted(TASKS)}")
    spec = TASKS[task]
    cfg = {**spec.kwargs, **overrides}
    return ControlEnv(spec.system_factory(), **cfg)
