"""Real-time loop runner for hardware-in-the-loop (HIL) testing."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable
import numpy as np

from .emulator import PlantEmulator
from .transport import Transport
from ..simulate import rk4_step


@dataclasses.dataclass
class DeadlineMissInfo:
    """Diagnostic details for a missed control loop deadline."""
    step: int
    target_dt: float
    elapsed_s: float
    overrun_s: float
    timestamp_s: float


@dataclasses.dataclass
class HILResult:
    """Complete telemetry record from an executed HIL loop."""
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    dt_actual: np.ndarray
    deadline_misses: int
    worst_overrun_s: float
    avg_jitter_s: float
    diverged: bool = False
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)


class RealTimeLoop:
    """Fixed-rate real-time loop runner for hardware and HIL testing."""

    def __init__(
        self,
        rate_hz: float,
        controller: Any,
        plant: Any,
        *,
        simulated_time: bool = False,
        on_deadline_miss: Callable[[DeadlineMissInfo], None] | str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.rate_hz = float(rate_hz)
        if self.rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        self.dt = 1.0 / self.rate_hz
        self.controller = controller
        self.plant = plant
        self.simulated_time = simulated_time
        self.on_deadline_miss = on_deadline_miss
        self.transport = transport

    def _handle_deadline_miss(self, info: DeadlineMissInfo) -> None:
        if self.on_deadline_miss == "warn":
            print(
                f"[WARNING: HIL Deadline Miss] Step {info.step}: elapsed {info.elapsed_s * 1e3:.2f} ms "
                f"(budget {info.target_dt * 1e3:.2f} ms, overrun {info.overrun_s * 1e3:.2f} ms)"
            )
        elif self.on_deadline_miss == "raise":
            raise RuntimeError(
                f"HIL RealTimeLoop deadline missed at step {info.step}: "
                f"elapsed {info.elapsed_s * 1e3:.2f} ms exceeded budget {info.target_dt * 1e3:.2f} ms."
            )
        elif callable(self.on_deadline_miss):
            self.on_deadline_miss(info)

    def run(
        self,
        duration: float | None = None,
        max_steps: int | None = None,
        x0: Any = None,
    ) -> HILResult:
        """Execute the real-time loop for duration seconds or max_steps steps."""
        if duration is not None:
            n_steps = int(round(duration * self.rate_hz))
        elif max_steps is not None:
            n_steps = int(max_steps)
        else:
            raise ValueError("Must specify either duration or max_steps")

        if hasattr(self.controller, "reset"):
            self.controller.reset()

        n_states = getattr(self.plant, "n_states", 1)
        n_inputs = getattr(self.plant, "n_inputs", 1)
        n_outputs = getattr(self.plant, "n_outputs", None) or n_states

        if x0 is None:
            x_curr = np.zeros(n_states)
        else:
            x_curr = np.atleast_1d(np.asarray(x0, dtype=float)).copy()

        if hasattr(self.plant, "reset"):
            y_meas = np.atleast_1d(self.plant.reset(x_curr))
        elif hasattr(self.plant, "output"): 
            y_meas = np.atleast_1d(self.plant.output(0.0, x_curr, np.zeros(n_inputs)))
        elif callable(self.plant):
            y_meas = x_curr.copy()
        else:
            y_meas = x_curr.copy()

        if self.transport is not None:
            self.transport.reset()

        t_hist = np.zeros(n_steps)
        x_hist = np.zeros((n_steps, n_states))
        y_hist = np.zeros((n_steps, len(y_meas)))
        u_hist = np.zeros((n_steps, n_inputs))
        dt_hist = np.zeros(n_steps)

        deadline_misses = 0
        worst_overrun = 0.0
        diverged = False

        t_wall_start = time.perf_counter()
        next_deadline = t_wall_start + self.dt

        for k in range(n_steps):
            t_step_start = time.perf_counter()
            t_sim = k * self.dt

            t_hist[k] = t_sim
            y_hist[k] = y_meas.copy()
            if hasattr(self.plant, "x"):
                x_hist[k] = np.asarray(self.plant.x, dtype=float).copy()
            else:
                x_hist[k] = x_curr.copy()

            if hasattr(self.controller, "update"):
                u_cmd = self.controller.update(y_meas, self.dt)
            elif callable(self.controller):
                u_cmd = self.controller(y_meas, self.dt)
            else:
                raise TypeError(f"Invalid controller object: {type(self.controller)}")

            u_cmd = np.atleast_1d(np.asarray(u_cmd, dtype=float))
            u_hist[k] = u_cmd.copy()

            if hasattr(self.transport, "tick"):
                self.transport.tick()

            if hasattr(self.plant, "step"):
                y_meas = np.atleast_1d(self.plant.step(u_cmd, self.dt))
                if hasattr(self.plant, "x"):
                    x_curr = np.asarray(self.plant.x, dtype=float).copy()
            elif hasattr(self.plant, "dynamics"):
                x_next = rk4_step(self.plant.dynamics, t_sim, x_curr, u_cmd, self.dt)
                x_curr = x_next.copy()
                if hasattr(self.plant, "output"): 
                    y_meas = np.atleast_1d(self.plant.output(t_sim + self.dt, x_curr, u_cmd))
                else:
                    y_meas = x_curr.copy()
            else:
                y_meas = np.atleast_1d(self.plant(u_cmd, self.dt))

            if not np.all(np.isfinite(y_meas)):
                diverged = True
                t_hist = t_hist[: k + 1]
                x_hist = x_hist[: k + 1]
                y_hist = y_hist[: k + 1]
                u_hist = u_hist[: k + 1]
                dt_hist = dt_hist[: k + 1]
                break

            t_step_end = time.perf_counter()
            step_elapsed = t_step_end - t_step_start
            dt_hist[k] = step_elapsed

            if not self.simulated_time:
                now = time.perf_counter()
                sleep_time = next_deadline - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    overrun = -sleep_time
                    deadline_misses += 1
                    worst_overrun = max(worst_overrun, overrun)
                    self._handle_deadline_miss(
                        DeadlineMissInfo(
                            step=k,
                            target_dt=self.dt,
                            elapsed_s=step_elapsed,
                            overrun_s=overrun,
                            timestamp_s=now - t_wall_start,
                        )
                    )
                next_deadline += self.dt
            else:
                if step_elapsed > self.dt:
                    overrun = step_elapsed - self.dt
                    deadline_misses += 1
                    worst_overrun = max(worst_overrun, overrun)

        avg_jitter = float(np.std(dt_hist)) if len(dt_hist) > 1 else 0.0

        return HILResult(
            t=t_hist,
            x=x_hist,
            y=y_hist,
            u=u_hist,
            dt_actual=dt_hist,
            deadline_misses=deadline_misses,
            worst_overrun_s=worst_overrun,
            avg_jitter_s=avg_jitter,
            diverged=diverged,
            meta={
                "rate_hz": self.rate_hz,
                "target_dt": self.dt,
                "n_steps": len(t_hist),
                "simulated_time": self.simulated_time,
            },
        )
