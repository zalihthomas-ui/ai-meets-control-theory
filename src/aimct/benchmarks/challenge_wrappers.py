"""
Intelligent Control Challenge (ICC) — Track 3 & Track 4 Environment Wrappers.

Conforms to docs/references/challenge-spec.md:
- Track 3: ParamPerturbed (+-30% parameter shifts), ActuatorLag (tau_a delay), ImpulseInjector.
- Track 4: BlackBoxPlant / BlackBoxEnvironment (state/action hiding, interaction budget, safety barriers).
"""

from __future__ import annotations

import copy
from typing import Callable, Sequence

import numpy as np

from ..systems.base import ArrayLike, DynamicalSystem
from ..systems.linear import LinearSystem


class ParamPerturbed(DynamicalSystem):
    """
    Wraps a DynamicalSystem and perturbs its physical parameters or system matrices
    uniformly within +-scale (default +-30%) for Track 3 robustness evaluation.
    """

    def __init__(
        self,
        base_system: DynamicalSystem,
        scale: float = 0.30,
        seed: int | None = None,
        custom_deltas: dict[str, float] | None = None,
    ) -> None:
        self.base_system = copy.deepcopy(base_system)
        self.n_states = base_system.n_states
        self.n_inputs = base_system.n_inputs
        self.n_outputs = base_system.n_outputs
        self.scale = float(scale)
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.param_deltas: dict[str, float] = {}

        if isinstance(self.base_system, LinearSystem):
            delta_A = rng.uniform(-scale, scale, size=self.base_system.A.shape) * np.abs(self.base_system.A + 1e-3)
            delta_B = rng.uniform(-scale, scale, size=self.base_system.B.shape) * np.abs(self.base_system.B + 1e-3)
            self.base_system.A = self.base_system.A + delta_A
            self.base_system.B = self.base_system.B + delta_B
            self.param_deltas["A_perturb_norm"] = float(np.linalg.norm(delta_A))
            self.param_deltas["B_perturb_norm"] = float(np.linalg.norm(delta_B))
        else:
            perturbed_attrs = ["m", "c", "k", "length", "mc", "mp", "l", "b", "I", "g", "d", "J", "mass"]
            for attr in perturbed_attrs:
                if hasattr(self.base_system, attr):
                    val = getattr(self.base_system, attr)
                    if isinstance(val, (int, float)) and val != 0.0:
                        delta_factor = (
                            custom_deltas[attr]
                            if (custom_deltas and attr in custom_deltas)
                            else rng.uniform(-scale, scale)
                        )
                        new_val = float(val * (1.0 + delta_factor))
                        setattr(self.base_system, attr, new_val)
                        self.param_deltas[attr] = float(delta_factor)

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        return self.base_system.dynamics(t, x, u)

    def linearize(self, x_eq: ArrayLike | None = None, u_eq: ArrayLike | None = None, eps: float = 1e-6):
        return self.base_system.linearize(x_eq=x_eq, u_eq=u_eq, eps=eps)

    def output(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        return self.base_system.output(t, x, u)


def perturbed_system(
    system_or_factory: DynamicalSystem | Callable[[], DynamicalSystem],
    rng: np.random.Generator | int | None = None,
    frac: float = 0.30,
) -> DynamicalSystem:
    """
    Functional constructor returning a +-frac perturbed instance of a system.
    """
    sys = system_or_factory() if callable(system_or_factory) else system_or_factory
    seed = rng if isinstance(rng, int) else None
    return ParamPerturbed(sys, scale=frac, seed=seed)


class ActuatorLag(DynamicalSystem):
    """
    Augments a DynamicalSystem with 1st-order actuator lag:
      tau_a * u_dot_applied + u_applied = u_commanded
      
    State is augmented: x_aug = [x_plant (n_states), u_applied (n_inputs)].
    """

    def __init__(self, base_system: DynamicalSystem, tau_a: float = 0.05) -> None:
        if tau_a <= 0.0:
            raise ValueError("Actuator time constant tau_a must be positive")
        self.base_system = base_system
        self.tau_a = float(tau_a)
        self.n_states = base_system.n_states + base_system.n_inputs
        self.n_inputs = base_system.n_inputs
        self.n_outputs = base_system.n_outputs

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        n_p = self.base_system.n_states

        x_plant = x[:n_p]
        u_act = x[n_p:]

        # Plant dynamics driven by filtered actuator state
        x_dot_plant = self.base_system.dynamics(t, x_plant, u_act)

        # 1st-order actuator response: u_dot_act = (u_cmd - u_act) / tau_a
        u_dot_act = (u - u_act) / self.tau_a

        return np.concatenate([x_dot_plant, u_dot_act])

    def output(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        n_p = self.base_system.n_states
        x_plant = x[:n_p]
        u_act = x[n_p:]
        return self.base_system.output(t, x_plant, u_act)


class ImpulseDisturbance:
    """
    Disturbance generator injecting sudden Laplace/Gaussian impulse shocks at random times.
    """

    def __init__(
        self,
        magnitude: float = 2.0,
        rate_hz: float = 0.5,
        duration: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.magnitude = float(magnitude)
        self.rate_hz = float(rate_hz)
        self.duration = float(duration)
        self.rng = np.random.default_rng(seed)
        self.impulse_times: list[float] = []
        self.impulse_strengths: list[float] = []

    def schedule_horizon(self, t_final: float) -> None:
        self.impulse_times.clear()
        self.impulse_strengths.clear()
        t = 0.5
        while t < t_final - self.duration:
            dt = float(self.rng.exponential(1.0 / max(1e-3, self.rate_hz)))
            t += dt
            if t < t_final - self.duration:
                strength = float(self.rng.laplace(0.0, self.magnitude))
                self.impulse_times.append(t)
                self.impulse_strengths.append(strength)

    def __call__(self, t: float) -> np.ndarray:
        val = 0.0
        for t_k, strength in zip(self.impulse_times, self.impulse_strengths):
            if t_k <= t < t_k + self.duration:
                val += strength
        return np.array([val])


def ImpulseInjector(
    rng: np.random.Generator | int | None = None,
    b_scale: float = 2.0,
    rate_hz: float = 0.5,
    duration: float = 0.05,
    t_final: float = 10.0,
) -> Callable[[float], np.ndarray]:
    """
    Functional constructor returning a callable disturbance d(t) -> array.
    """
    seed = rng if isinstance(rng, int) else None
    injector = ImpulseDisturbance(magnitude=b_scale, rate_hz=rate_hz, duration=duration, seed=seed)
    injector.schedule_horizon(t_final=t_final)
    return injector


class BlackBoxPlant:
    """
    Track 4: Safe Black-Box Plant Environment.
    
    Hides internal ODE equations and state matrices ($A, B$).
    Exposes only n_states, n_inputs, action_limit, dt, and interaction budget.
    Monitors hard safety barriers on every step; flags instant disqualification if breached.
    """

    def __init__(
        self,
        system: DynamicalSystem,
        dt: float = 0.001,
        max_step_budget: int = 10000,
        action_limit: float = 20.0,
        state_safety_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        output_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._system = system
        self.dt = float(dt)
        self.max_step_budget = int(max_step_budget)
        self.action_limit = float(action_limit)
        self.state_safety_bounds = state_safety_bounds
        self.output_fn = output_fn

        # Public metadata (no equations or A/B matrices)
        self.n_states: int = system.n_states
        self.n_inputs: int = system.n_inputs
        self.action_bounds: tuple[float, float] = (-self.action_limit, self.action_limit)

        self._state: np.ndarray = np.zeros(self.n_states)
        self._time: float = 0.0
        self._step_count: int = 0
        self._disqualified: bool = False
        self._dq_reasons: list[str] = []

    def reset(self, initial_state: Sequence[float]) -> np.ndarray:
        self._state = np.atleast_1d(np.asarray(initial_state, dtype=float)).copy()
        if len(self._state) != self.n_states:
            raise ValueError(f"initial_state length must match n_states ({self.n_states})")
        self._time = 0.0
        self._step_count = 0
        self._disqualified = False
        self._dq_reasons.clear()
        return self._get_obs()

    def step(self, action: Sequence[float] | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self._disqualified:
            return self._get_obs(), 1e6, True, {"disqualified": True, "reasons": self._dq_reasons}

        self._step_count += 1
        if self._step_count > self.max_step_budget:
            self._disqualified = True
            self._dq_reasons.append(f"Interaction budget exceeded: {self._step_count} > {self.max_step_budget} steps")
            return self._get_obs(), 1e6, True, {"disqualified": True, "reasons": self._dq_reasons}

        u_cmd = np.atleast_1d(np.asarray(action, dtype=float))
        if not np.all(np.isfinite(u_cmd)):
            self._disqualified = True
            self._dq_reasons.append("Non-finite action received (NaN / Inf).")
            return self._get_obs(), 1e6, True, {"disqualified": True, "reasons": self._dq_reasons}

        u_clamped = np.clip(u_cmd, -self.action_limit, self.action_limit)

        # RK4 integration
        k1 = self._system.dynamics(self._time, self._state, u_clamped)
        k2 = self._system.dynamics(self._time + 0.5 * self.dt, self._state + 0.5 * self.dt * k1, u_clamped)
        k3 = self._system.dynamics(self._time + 0.5 * self.dt, self._state + 0.5 * self.dt * k2, u_clamped)
        k4 = self._system.dynamics(self._time + self.dt, self._state + self.dt * k3, u_clamped)
        self._state = self._state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self._time += self.dt

        # Safety barrier validation
        if self.state_safety_bounds is not None:
            s_min, s_max = self.state_safety_bounds
            if np.any(self._state < s_min) or np.any(self._state > s_max):
                self._disqualified = True
                self._dq_reasons.append(
                    f"Safety envelope breached at t={self._time:.3f}s: state {self._state} outside bounds"
                )
                return self._get_obs(), 1e6, True, {"disqualified": True, "reasons": self._dq_reasons}

        if not np.all(np.isfinite(self._state)) or np.max(np.abs(self._state)) > 1e4:
            self._disqualified = True
            self._dq_reasons.append("State trajectory exploded / diverged.")
            return self._get_obs(), 1e6, True, {"disqualified": True, "reasons": self._dq_reasons}

        cost = float(np.sum(self._state**2) + 0.01 * np.sum(u_clamped**2))
        return self._get_obs(), cost, False, {"time": self._time, "step": self._step_count, "disqualified": False}

    def _get_obs(self) -> np.ndarray:
        if self.output_fn is not None:
            return self.output_fn(self._state)
        return self._state.copy()

    @property
    def is_disqualified(self) -> bool:
        return self._disqualified

    @property
    def dq_reasons(self) -> list[str]:
        return list(self._dq_reasons)


# Alias
BlackBoxEnvironment = BlackBoxPlant
