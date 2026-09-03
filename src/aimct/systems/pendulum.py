"""Simple pendulum with a torque input at the pivot.

State ``[theta, theta_dot]`` with ``theta = 0`` at the stable-down position
(gravity pulls toward ``theta = 0``). Equation:

    theta'' = -(g / L) sin(theta) - (b / (m L^2)) theta_dot + u / (m L^2)
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class Pendulum(DynamicalSystem):
    n_states = 2  # [angle, angular velocity]
    n_inputs = 1  # [torque]

    def __init__(
        self,
        m: float = 1.0,
        L: float = 1.0,
        b: float = 0.1,
        g: float = 9.81,
    ) -> None:
        self.m, self.L, self.b, self.g = float(m), float(L), float(b), float(g)

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        theta, omega = x
        inertia = self.m * self.L**2
        domega = (
            -(self.g / self.L) * np.sin(theta)
            - (self.b / inertia) * omega
            + u[0] / inertia
        )
        return np.array([omega, domega])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        """Analytic Jacobian about ``x_eq`` (default: upright, ``theta = pi``)."""
        if x_eq is None:
            x_eq = np.array([np.pi, 0.0])
        theta = float(np.asarray(x_eq, dtype=float)[0])
        inertia = self.m * self.L**2
        A = np.array(
            [
                [0.0, 1.0],
                [-(self.g / self.L) * np.cos(theta), -self.b / inertia],
            ]
        )
        B = np.array([[0.0], [1.0 / inertia]])
        return A, B
