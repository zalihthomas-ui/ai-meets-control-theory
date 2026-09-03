"""Fixed-step simulation of ``system`` under ``controller``.

The controller follows the :class:`aimct.controllers.base.Controller` protocol
(``update(measurement, dt) -> u`` + ``reset()``). Each step the controller is
handed the system's measured output and the step length ``dt``; its input is
held constant across the step (zero-order hold) and the state is advanced with
classical RK4. Returns a :class:`Trajectory` of aligned time / state / input /
output arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .systems.base import DynamicalSystem


class ControllerLike(Protocol):
    def update(self, measurement, dt: float): ...
    def reset(self) -> None: ...


@dataclass
class Trajectory:
    t: np.ndarray  # (N,)            sample times
    x: np.ndarray  # (N, n_states)   state at each sample
    u: np.ndarray  # (N, n_inputs)   input applied over [t[k], t[k+1]); u[-1] repeats
    y: np.ndarray  # (N, n_outputs)  measured output at each sample

    def __len__(self) -> int:
        return len(self.t)


def rk4_step(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = f(t, x, u)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    system: DynamicalSystem,
    controller: ControllerLike | Callable,
    x0: np.ndarray,
    dt: float,
    t_final: float,
    u_bounds: tuple[float, float] | None = None,
) -> Trajectory:
    """Roll out ``system`` from ``x0`` for ``t_final`` seconds at step ``dt``.

    ``controller`` may be a :class:`Controller` (``update(measurement, dt)``) or a
    bare callable with the same ``(measurement, dt) -> u`` signature. If it has a
    ``reset()`` method it is called once before the rollout. ``u_bounds`` is an
    optional ``(low, high)`` saturation applied to every input channel.
    """
    if dt <= 0 or t_final <= 0:
        raise ValueError("dt and t_final must be positive")

    if hasattr(controller, "reset"):
        controller.reset()
    step = controller.update if hasattr(controller, "update") else controller

    n_steps = int(round(t_final / dt))
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    if x.shape != (system.n_states,):
        raise ValueError(f"x0 must have shape ({system.n_states},), got {x.shape}")

    ts = np.zeros(n_steps + 1)
    xs = np.zeros((n_steps + 1, system.n_states))
    us = np.zeros((n_steps + 1, system.n_inputs))
    ys = np.zeros((n_steps + 1, system.n_outputs or system.n_states))
    xs[0] = x
    u_prev = np.zeros(system.n_inputs)

    for k in range(n_steps):
        t = k * dt
        y = np.atleast_1d(np.asarray(system.output(t, x, u_prev), dtype=float))
        ys[k] = y
        u = np.atleast_1d(np.asarray(step(y, dt), dtype=float))
        if u.shape != (system.n_inputs,):
            raise ValueError(
                f"controller returned shape {u.shape}, expected ({system.n_inputs},)"
            )
        if u_bounds is not None:
            u = np.clip(u, u_bounds[0], u_bounds[1])
        x = rk4_step(system.dynamics, t, x, u, dt)
        ts[k + 1] = t + dt
        xs[k + 1] = x
        us[k] = u
        u_prev = u

    us[-1] = us[-2] if n_steps > 0 else us[-1]
    ys[-1] = np.atleast_1d(np.asarray(system.output(ts[-1], xs[-1], us[-1]), dtype=float))
    return Trajectory(t=ts, x=xs, u=us, y=ys)
