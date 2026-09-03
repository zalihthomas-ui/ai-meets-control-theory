"""Mass-spring-damper:  m x'' + c x' + k x = u  (force input)."""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class MassSpringDamper(DynamicalSystem):
    n_states = 2  # [position, velocity]
    n_inputs = 1  # [force]

    def __init__(self, m: float = 1.0, c: float = 0.4, k: float = 1.0) -> None:
        if m <= 0:
            raise ValueError("mass must be positive")
        self.m, self.c, self.k = float(m), float(c), float(k)

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        pos, vel = x
        acc = (u[0] - self.c * vel - self.k * pos) / self.m
        return np.array([vel, acc])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        A = np.array([[0.0, 1.0], [-self.k / self.m, -self.c / self.m]])
        B = np.array([[0.0], [1.0 / self.m]])
        return A, B
