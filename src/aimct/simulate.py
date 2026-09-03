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
    measurement_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None,
    input_disturbance: Callable[[float], np.ndarray] | None = None,
) -> Trajectory:
    """Roll out ``system`` from ``x0`` for ``t_final`` seconds at step ``dt``.

    ``controller`` may be a :class:`Controller` (``update(measurement, dt)``) or a
    bare callable with the same ``(measurement, dt) -> u`` signature. If it has a
    ``reset()`` method it is called once before the rollout. ``u_bounds`` is an
    optional ``(low, high)`` saturation applied to every input channel.

    ``measurement_fn(t, x, u_prev) -> measurement`` selects what the controller
    sees. Default: ``system.output`` (full state for the reference systems). Pass
    e.g. ``lambda t, x, u: x[[0]]`` to give an output-feedback controller a single
    channel while the recorded ``Trajectory.y`` still uses ``system.output``.

    ``input_disturbance(t) -> d`` is an additive plant-input disturbance applied
    to the dynamics as ``u_applied = clip(u) + d(t)``. It is **not** included in
    the recorded ``Trajectory.u`` (that stays the controller command), so control
    effort and saturation metrics reflect the actuator, not the disturbance.
    """
    if dt <= 0 or t_final <= 0:
        raise ValueError("dt and t_final must be positive")

    if hasattr(controller, "reset"):
        controller.reset()
    step = controller.update if hasattr(controller, "update") else controller
    measure = measurement_fn if measurement_fn is not None else system.output

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
        ys[k] = np.atleast_1d(np.asarray(system.output(t, x, u_prev), dtype=float))
        meas = np.asarray(measure(t, x, u_prev), dtype=float)
        u = np.atleast_1d(np.asarray(step(meas, dt), dtype=float))
        if u.shape != (system.n_inputs,):
            raise ValueError(
                f"controller returned shape {u.shape}, expected ({system.n_inputs},)"
            )
        if u_bounds is not None:
            u = np.clip(u, u_bounds[0], u_bounds[1])
        us[k] = u
        u_prev = u
        u_applied = u
        if input_disturbance is not None:
            d = np.atleast_1d(np.asarray(input_disturbance(t), dtype=float))
            u_applied = u + d
        x = rk4_step(system.dynamics, t, x, u_applied, dt)
        ts[k + 1] = t + dt
        xs[k + 1] = x

    us[-1] = us[-2] if n_steps > 0 else us[-1]
    ys[-1] = np.atleast_1d(np.asarray(system.output(ts[-1], xs[-1], us[-1]), dtype=float))
    return Trajectory(t=ts, x=xs, u=us, y=ys)
