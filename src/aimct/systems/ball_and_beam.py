r"""Ball and Beam underactuated nonlinear dynamical system.

Coordinates:
- State ``x = [r, r_dot, theta, theta_dot]``
  - ``r``: position of the rolling ball along the beam [m] (``r = 0`` is at the central pivot)
  - ``r_dot``: velocity of the ball along the beam [m/s]
  - ``theta``: beam tilt angle with the horizontal [rad] (``theta = 0`` is horizontal)
  - ``theta_dot``: angular velocity of the beam [rad/s]
- Input ``u = [tau]``: motor torque [N.m] applied to tilt the beam.

Parameters default to the canonical Quanser Ball and Beam apparatus.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem

__all__ = ["BallAndBeam"]


class BallAndBeam(DynamicalSystem):
    """Ball and Beam underactuated nonlinear dynamical system (relative degree 4).

    Parameters
    ----------
    m: float
        Mass of the solid steel ball [kg]. Defaults to 0.064 (64 g).
    R_b: float
        Radius of the ball [m]. Defaults to 0.0127 (12.7 mm).
    J_b: float | None
        Moment of inertia of the ball about its center [kg.m^2]. Defaults to (2/5) * m * R_b^2.
    c_r: float
        Viscous rolling friction coefficient of the ball on the beam [N.s/m]. Defaults to 0.002.
    M: float
        Mass of the rigid beam [kg]. Defaults to 0.20.
    L: float
        Total length of the beam [m]. Defaults to 0.425 (42.5 cm).
    J: float | None
        Moment of inertia of the beam about its central pivot [kg.m^2]. Defaults to (1/12) * M * L^2.
    b: float
        Viscous damping coefficient at the central pivot [N.m.s/rad]. Defaults to 0.05.
    g: float
        Gravitational acceleration [m/s^2]. Defaults to 9.81.
    r_max: float
        Maximum usable rolling track half-distance before bumper [m]. Defaults to 0.20 (20 cm).
    theta_max: float
        Maximum beam tilt angle before mechanical hard-stop [rad]. Defaults to 0.45 (~25.8 deg).
    tau_max: float
        Maximum actuator torque [N.m]. Defaults to 1.50.
    """

    n_states = 4
    n_inputs = 1
    n_outputs = 4

    def __init__(
        self,
        m: float = 0.064,
        R_b: float = 0.0127,
        J_b: float | None = None,
        c_r: float = 0.002,
        M: float = 0.20,
        L: float = 0.425,
        J: float | None = None,
        b: float = 0.05,
        g: float = 9.81,
        r_max: float = 0.20,
        theta_max: float = 0.45,
        tau_max: float = 1.50,
    ) -> None:
        self.m = float(m)
        self.R_b = float(R_b)
        self.J_b = float((2.0 / 5.0) * self.m * self.R_b**2 if J_b is None else J_b)
        self.c_r = float(c_r)

        self.M = float(M)
        self.L = float(L)
        self.J = float((1.0 / 12.0) * self.M * self.L**2 if J is None else J)
        self.b = float(b)

        self.g = float(g)
        self.r_max = float(r_max)
        self.theta_max = float(theta_max)
        self.tau_max = float(tau_max)

        # Effective rolling mass (linear + rotational inertia)
        self.m_eff = self.m + self.J_b / (self.R_b**2)  # = (7/5) * m

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        """Compute state derivative dx/dt = [r_dot, r_ddot, theta_dot, theta_ddot]."""
        x, u = self._prep(x, u)
        r, r_d, th, th_d = x
        tau = float(np.clip(u[0], -self.tau_max, self.tau_max))

        # Constrain physical state to track limits
        r = float(np.clip(r, -self.r_max, self.r_max))
        th = float(np.clip(th, -self.theta_max, self.theta_max))

        sin_th = np.sin(th)
        cos_th = np.cos(th)

        # 1. Ball acceleration: m_eff * r_ddot = m * r * th_dot^2 - m * g * sin(th) - c_r * r_dot
        r_dd = (self.m * r * th_d**2 - self.m * self.g * sin_th - self.c_r * r_d) / self.m_eff

        # 2. Beam acceleration: (J + m * r^2) * th_ddot = tau - 2 * m * r * r_dot * th_dot - m * g * r * cos(th) - b * th_dot
        J_total = self.J + self.m * r**2
        th_dd = (tau - 2.0 * self.m * r * r_d * th_d - self.m * self.g * r * cos_th - self.b * th_d) / J_total

        return np.array([r_d, r_dd, th_d, th_dd], dtype=float)

    def linearize(
        self,
        x_eq: ArrayLike | None = None,
        u_eq: ArrayLike | None = None,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuous state-space linearisation (A, B) about equilibrium.

        When x_eq is None or zeros, uses the exact analytical formula about the center horizontal rest.
        """
        if x_eq is None or np.allclose(x_eq, 0.0):
            # Analytical linearization about center horizontal equilibrium (r = 0, theta = 0)
            A = np.array([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -self.c_r / self.m_eff, -(self.m / self.m_eff) * self.g, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-self.m * self.g / self.J, 0.0, 0.0, -self.b / self.J],
            ], dtype=float)

            B = np.array([
                [0.0],
                [0.0],
                [0.0],
                [1.0 / self.J],
            ], dtype=float)

            return A, B

        # Numerical fallback for custom equilibrium
        return super().linearize(x_eq=x_eq, u_eq=u_eq, eps=eps)

    def total_energy(self, x: ArrayLike) -> float:
        """Total mechanical kinetic + potential energy of the ball and beam."""
        x = np.asarray(x, dtype=float)
        r, r_d, th, th_d = x
        T = 0.5 * self.m_eff * r_d**2 + 0.5 * (self.J + self.m * r**2) * th_d**2
        V = self.m * self.g * r * np.sin(th)
        return float(T + V)
