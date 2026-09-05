"""Plant emulator with realistic hardware non-idealities."""

from __future__ import annotations

import collections
from typing import Any, Sequence
import numpy as np

from ..systems.base import ArrayLike, DynamicalSystem
from ..simulate import rk4_step


class PlantEmulator:
    """Hardware emulator wrapping a continuous dynamical system."""

    def __init__(
        self,
        system: DynamicalSystem,
        *,
        quantization_bits: int | None = None,
        quantize_channels: Sequence[int] | None = None,
        quantize_ranges: Sequence[tuple[float, float]] | None = None,
        u_min: float | ArrayLike | None = None,
        u_max: float | ArrayLike | None = None,
        slew_rate_max: float | ArrayLike | None = None,
        delay_s: float = 0.0,
        delay_steps: int | None = None,
        sensor_noise_std: float | ArrayLike | None = None,
        sample_jitter_std_s: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.system = system
        self.n_states = system.n_states
        self.n_inputs = system.n_inputs
        self.n_outputs = system.n_outputs or system.n_states

        self.quantization_bits = quantization_bits
        self.quantize_channels = (
            list(quantize_channels)
            if quantize_channels is not None
            else list(range(self.n_outputs))
        )
        self.quantize_ranges = quantize_ranges

        self.u_min = np.atleast_1d(np.asarray(u_min, float)) if u_min is not None else None
        self.u_max = np.atleast_1d(np.asarray(u_max, float)) if u_max is not None else None
        self.slew_rate_max = (
            np.atleast_1d(np.asarray(slew_rate_max, float))
            if slew_rate_max is not None
            else None
        )

        self.delay_s = float(delay_s)
        self.delay_steps = delay_steps
        self.sensor_noise_std = (
            np.atleast_1d(np.asarray(sensor_noise_std, float))
            if sensor_noise_std is not None
            else None
        )
        self.sample_jitter_std_s = float(sample_jitter_std_s)
        self.rng = np.random.default_rng(seed)

        self.t: float = 0.0
        self.x: np.ndarray = np.zeros(self.n_states)
        self.y: np.ndarray = np.zeros(self.n_outputs)
        self.u_prev: np.ndarray = np.zeros(self.n_inputs)
        self.u_applied: np.ndarray = np.zeros(self.n_inputs)

        self._delay_queue: collections.deque[np.ndarray] = collections.deque()
        self.reset()

    def reset(self, x0: ArrayLike | None = None) -> np.ndarray:
        """Reset internal state, delay queues, and actuator slew state."""
        self.t = 0.0
        if x0 is None:
            self.x = np.zeros(self.n_states)
        else:
            self.x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
            if self.x.shape != (self.n_states,):
                raise ValueError(
                    f"x0 shape {self.x.shape} mismatch with system n_states {self.n_states}"
                )

        self.u_prev = np.zeros(self.n_inputs)
        self.u_applied = np.zeros(self.n_inputs)
        self._delay_queue.clear()

        raw_y = np.atleast_1d(self.system.output(self.t, self.x, self.u_applied))
        self.y = self._process_measurement(raw_y)
        return self.y.copy()

    def _quantize(self, val: float, v_min: float, v_max: float, bits: int) -> float:
        """Quantize scalar float into n-bit discrete levels."""
        n_levels = (1 << bits) - 1
        clamped = np.clip(val, v_min, v_max)
        normalized = (clamped - v_min) / (v_max - v_min)
        quantized_level = np.round(normalized * n_levels)
        return float(v_min + (quantized_level / n_levels) * (v_max - v_min))

    def _process_measurement(self, y_raw: np.ndarray) -> np.ndarray:
        """Apply additive sensor noise and encoder quantization."""
        y_proc = y_raw.copy().astype(float)

        if self.sensor_noise_std is not None:
            if len(self.sensor_noise_std) == 1:
                noise = self.rng.normal(0.0, self.sensor_noise_std[0], size=y_proc.shape)
            else:
                noise = self.rng.normal(0.0, self.sensor_noise_std)
            y_proc += noise

        if self.quantization_bits is not None and self.quantization_bits > 0:
            for ch in self.quantize_channels:
                if ch < len(y_proc):
                    if self.quantize_ranges is not None and ch < len(self.quantize_ranges):
                        vmin, vmax = self.quantize_ranges[ch]
                    else:
                        vmin, vmax = -np.pi, np.pi
                    y_proc[ch] = self._quantize(
                        y_proc[ch], vmin, vmax, self.quantization_bits
                    )

        return y_proc

    def step(self, u_cmd: ArrayLike, dt: float) -> np.ndarray:
        """Advance the emulated plant by dt with commanded input u_cmd."""
        u_in = np.atleast_1d(np.asarray(u_cmd, dtype=float)).copy()
        if u_in.shape != (self.n_inputs,):
            if len(u_in) == 1 and self.n_inputs == 1:
                u_in = u_in.reshape((1,))
            else:
                raise ValueError(
                    f"u_cmd shape {u_in.shape} mismatch with plant n_inputs {self.n_inputs}"
                )

        if self.delay_steps is not None and self.delay_steps > 0:
            n_delay = self.delay_steps
        elif self.delay_s > 0.0 and dt > 1e-9:
            n_delay = int(round(self.delay_s / dt))
        else:
            n_delay = 0

        if n_delay > 0:
            self._delay_queue.append(u_in.copy())
            if len(self._delay_queue) <= n_delay:
                u_delayed = self.u_prev.copy()
            else:
                u_delayed = self._delay_queue.popleft()
        else:
            u_delayed = u_in.copy()

        if self.slew_rate_max is not None and dt > 1e-9:
            delta = u_delayed - self.u_prev
            max_delta = self.slew_rate_max * dt
            delta_clamped = np.clip(delta, -max_delta, max_delta)
            u_slewed = self.u_prev + delta_clamped
        else:
            u_slewed = u_delayed

        u_sat = u_slewed.copy()
        if self.u_min is not None:
            u_sat = np.maximum(u_sat, self.u_min)
        if self.u_max is not None:
            u_sat = np.minimum(u_sat, self.u_max)

        self.u_applied = u_sat.copy()
        self.u_prev = self.u_applied.copy()

        if self.sample_jitter_std_s > 0.0:
            dt_actual = max(1e-6, dt + self.rng.normal(0.0, self.sample_jitter_std_s))
        else:
            dt_actual = dt

        self.x = rk4_step(self.system.dynamics, self.t, self.x, self.u_applied, dt_actual)
        self.t += dt_actual

        raw_y = np.atleast_1d(self.system.output(self.t, self.x, self.u_applied))
        self.y = self._process_measurement(raw_y)

        return self.y.copy()
