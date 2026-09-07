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

__all__ = ["simulate", "simulate_batch", "Trajectory", "BatchResult", "rk4_step"]


class ControllerLike(Protocol):
    def update(self, measurement, dt: float): ...
    def reset(self) -> None: ...


@dataclass
class Trajectory:
    t: np.ndarray  # (N,)            sample times
    x: np.ndarray  # (N, n_states)   state at each sample
    u: np.ndarray  # (N, n_inputs)   input applied over [t[k], t[k+1]); u[-1] repeats
    y: np.ndarray  # (N, n_outputs)  measured output at each sample
    diverged: bool = False  # True if the run stopped early on a non-finite state

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
        with np.errstate(over="ignore", invalid="ignore"):
            x = rk4_step(system.dynamics, t, x, u_applied, dt)
        ts[k + 1] = t + dt
        xs[k + 1] = x

        if not np.all(np.isfinite(x)):
            # The run blew up. Keep this first non-finite sample (so downstream
            # divergence checks still fire) and stop -- further RK4 steps would
            # just spew overflow warnings.
            last = k + 2
            return Trajectory(
                t=ts[:last], x=xs[:last], u=us[:last], y=ys[:last], diverged=True
            )

    us[-1] = us[-2] if n_steps > 0 else us[-1]
    ys[-1] = np.atleast_1d(np.asarray(system.output(ts[-1], xs[-1], us[-1]), dtype=float))
    return Trajectory(t=ts, x=xs, u=us, y=ys)


@dataclass
class BatchResult:
    """Stacked output of :func:`simulate_batch` over ``B`` trials.

    ``x`` / ``u`` / ``y`` are ``(B, T, ·)``; ``t`` is the shared ``(T,)`` grid
    (trials that diverged early are right-padded with their last finite sample
    so the stack is rectangular — check ``diverged`` / ``n_valid``).
    """

    t: np.ndarray
    x: np.ndarray                     # (B, T, n_states)
    u: np.ndarray                     # (B, T, n_inputs)
    y: np.ndarray                     # (B, T, n_outputs)
    diverged: np.ndarray             # (B,) bool
    n_valid: np.ndarray              # (B,) int  — finite samples before padding

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def final_x(self) -> np.ndarray:
        """State at the last valid sample of each trial — ``(B, n_states)``."""
        return self.x[np.arange(len(self)), self.n_valid - 1]

    def map_metric(self, fn: Callable[[Trajectory], float]) -> np.ndarray:
        """Apply ``fn`` to each trial as a :class:`Trajectory`; return ``(B,)``."""
        out = np.empty(len(self))
        for i in range(len(self)):
            k = int(self.n_valid[i])
            out[i] = fn(Trajectory(self.t[:k], self.x[i, :k], self.u[i, :k],
                                   self.y[i, :k], bool(self.diverged[i])))
        return out


def simulate_batch(
    system: DynamicalSystem,
    controller: "ControllerLike | Callable | Callable[[], object]",
    x0s: np.ndarray,
    dt: float,
    t_final: float,
    *,
    u_bounds: tuple[float, float] | None = None,
    measurement_fn: Callable | None = None,
    input_disturbances: "Callable | list | None" = None,
    param_overrides: "list[dict] | None" = None,
    fresh_controller: bool = False,
) -> BatchResult:
    r"""Monte-Carlo rollout: run ``simulate`` for every row of ``x0s``.

    Parameters
    ----------
    x0s : ``(B, n_states)`` — one initial state per trial.
    controller : a :class:`Controller` (reset per trial), a bare
        ``(measurement, dt) -> u`` callable, or — with ``fresh_controller=True``
        — a zero-arg factory returning a new controller per trial (use this
        when the controller keeps history that ``reset()`` doesn't fully
        clear, or holds a per-trial seed).
    input_disturbances : one ``t -> d`` callable applied to every trial, or a
        length-``B`` list of them (``None`` entries allowed).
    param_overrides : optional length-``B`` list of ``{attr: value}`` dicts
        temporarily set on ``system`` for that trial (restored after) — for
        sweeping plant parameters. ``system`` must be safe to mutate in place.

    Returns
    -------
    :class:`BatchResult`

    Notes
    -----
    Trials are independent and run sequentially; this is a convenience +
    bookkeeping layer, not a vectorised integrator (the reference systems
    index state as ``x[0]``, ``x[1]`` and do not broadcast over a batch axis).
    For a large sweep, parallelise at the call site over slices of ``x0s``.
    """
    x0s = np.atleast_2d(np.asarray(x0s, dtype=float))
    if x0s.ndim != 2 or x0s.shape[1] != system.n_states:
        raise ValueError(f"x0s must be (B, {system.n_states}), got {x0s.shape}")
    B = x0s.shape[0]
    n_steps = int(round(t_final / dt))
    T = n_steps + 1

    def _dist_for(i):
        if input_disturbances is None:
            return None
        if callable(input_disturbances):
            return input_disturbances
        return input_disturbances[i]

    def _controller_for(i):
        return controller() if fresh_controller else controller

    n_out = system.n_outputs or system.n_states
    x = np.zeros((B, T, system.n_states))
    u = np.zeros((B, T, system.n_inputs))
    y = np.zeros((B, T, n_out))
    diverged = np.zeros(B, dtype=bool)
    n_valid = np.full(B, T, dtype=int)

    for i in range(B):
        ov = (param_overrides or [None] * B)[i]
        saved = {k: getattr(system, k) for k in ov} if ov else {}
        if ov:
            for k, v in ov.items():
                setattr(system, k, v)
        try:
            tr = simulate(system, _controller_for(i), x0=x0s[i], dt=dt,
                          t_final=t_final, u_bounds=u_bounds,
                          measurement_fn=measurement_fn,
                          input_disturbance=_dist_for(i))
        finally:
            for k, v in saved.items():
                setattr(system, k, v)

        k = len(tr)
        x[i, :k], u[i, :k], y[i, :k] = tr.x, tr.u, tr.y
        if k < T:                                # pad a diverged/short run
            x[i, k:], u[i, k:], y[i, k:] = tr.x[-1], tr.u[-1], tr.y[-1]
        diverged[i] = tr.diverged
        n_valid[i] = k
        if i == 0:
            t_grid = tr.t if k == T else np.arange(T) * dt

    return BatchResult(t=t_grid, x=x, u=u, y=y, diverged=diverged, n_valid=n_valid)
