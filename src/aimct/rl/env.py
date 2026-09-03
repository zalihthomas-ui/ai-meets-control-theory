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
from ..systems import CartPole, Pendulum

__all__ = ["ControlEnv", "TASKS", "make", "wrap_to_pi"]


def wrap_to_pi(a):
    """Wrap angle(s) to ``(-pi, pi]``."""
    return np.pi - (np.pi - np.asarray(a, dtype=float)) % (2.0 * np.pi)


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
        obs_dim = int(np.asarray(self._obs_fn(np.zeros(self.nx))).size)
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

    def _obs(self) -> np.ndarray:
        return np.asarray(self._obs_fn(self._x), dtype=np.float32).ravel()

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

    def _reward(self, x, u, x_next) -> float:
        if self._reward_fn is not None:
            return float(self._reward_fn(x, u, x_next))
        return float(-(x @ self._Q @ x + u @ self._R @ u))

    def step(self, action):
        u = np.clip(np.atleast_1d(np.asarray(action, dtype=float)),
                    self._a_lo, self._a_hi).reshape(self.nu)
        x = self._x
        x_next = rk4_step(self.system.dynamics, self._t, x, u, self.dt)

        reward = self._reward(x, u, x_next)
        terminated = bool(
            self._s_lo is not None
            and (np.any(x_next < self._s_lo) or np.any(x_next > self._s_hi))
        )
        self._x = x_next
        self._t += self.dt
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


TASKS: dict[str, TaskSpec] = {
    "cartpole-balance": TaskSpec(CartPole, _cartpole_balance()),
    "pendulum-swingup": TaskSpec(Pendulum, _pendulum_swingup()),
}


def make(task: str, **overrides) -> ControlEnv:
    """Build a registered :class:`ControlEnv` task; ``overrides`` replace config
    keys (e.g. ``make("pendulum-swingup", max_steps=400))``."""
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; choices: {sorted(TASKS)}")
    spec = TASKS[task]
    cfg = {**spec.kwargs, **overrides}
    return ControlEnv(spec.system_factory(), **cfg)
