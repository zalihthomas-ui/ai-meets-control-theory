r"""Furuta Pendulum (Rotary Inverted Pendulum) dynamical system.

Coordinates:
- State ``x = [theta, alpha, theta_dot, alpha_dot]``
  - ``theta``: horizontal rotary arm angle [rad]
  - ``alpha``: vertical pendulum deflection angle [rad] (``alpha = 0`` is upright, ``alpha = pi`` is hanging)
  - ``theta_dot``: rotary arm angular velocity [rad/s]
  - ``alpha_dot``: pendulum angular velocity [rad/s]
- Input ``u = [tau]``: motor torque [N.m] applied to the rotary arm.

Parameters default to the canonical Quanser QUBE-Servo 2 Rotary Inverted Pendulum.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem

__all__ = ["FurutaPendulum"]


class FurutaPendulum(DynamicalSystem):
    """Rotary Inverted Pendulum (Furuta Pendulum) nonlinear dynamical system.

    Parameters
    ----------
    mr: float
        Mass of the rotary arm [kg].
    Lr: float
        Kinematic length of the rotary arm [m].
    Jr: float | None
        Moment of inertia of the rotary arm about vertical axis [kg.m^2].
        Defaults to (1/3) * mr * Lr^2.
    Dr: float
        Viscous damping coefficient of the rotary arm [N.m.s/rad].
    mp: float
        Mass of the pendulum link [kg].
    Lp: float
        Total length of the pendulum link [m].
    lp: float | None
        Distance from pivot to center of mass of the pendulum [m].
        Defaults to Lp / 2.
    Jp: float | None
        Moment of inertia of the pendulum link about its COM [kg.m^2].
        Defaults to (1/12) * mp * Lp^2.
    Dp: float
        Viscous damping coefficient of the pendulum joint [N.m.s/rad].
    g: float
        Gravitational acceleration [m/s^2].
    tau_max: float
        Maximum peak actuator torque [N.m].
    """

    n_states = 4
    n_inputs = 1
    n_outputs = 4

    def __init__(
        self,
        mr: float = 0.095,
        Lr: float = 0.085,
        Jr: float | None = None,
        Dr: float = 5.0e-4,
        mp: float = 0.024,
        Lp: float = 0.129,
        lp: float | None = None,
        Jp: float | None = None,
        Dp: float = 1.0e-4,
        g: float = 9.81,
        tau_max: float = 0.15,
    ) -> None:
        self.mr = float(mr)
        self.Lr = float(Lr)
        self.Jr = float((1.0 / 3.0) * self.mr * self.Lr**2 if Jr is None else Jr)
        self.Dr = float(Dr)

        self.mp = float(mp)
        self.Lp = float(Lp)
        self.lp = float(self.Lp / 2.0 if lp is None else lp)
        self.Jp = float((1.0 / 12.0) * self.mp * self.Lp**2 if Jp is None else Jp)
        self.Dp = float(Dp)

        self.g = float(g)
        self.tau_max = float(tau_max)

        # Derived lumped inertial terms
        self.J_t = self.Jr + self.mp * self.Lr**2
        self.Jp_eff = self.Jp + self.mp * self.lp**2

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        """Compute state derivative dx/dt = [theta_dot, alpha_dot, theta_ddot, alpha_ddot]."""
        x, u = self._prep(x, u)
        th, al, th_d, al_d = x
        tau = float(np.clip(u[0], -self.tau_max, self.tau_max))

        sin_al = np.sin(al)
        cos_al = np.cos(al)
        sin_2al = np.sin(2.0 * al)

        # Inertia matrix elements M(alpha)
        M11 = self.J_t + self.mp * self.lp**2 * sin_al**2
        M12 = self.mp * self.Lr * self.lp * cos_al
        M22 = self.Jp_eff

        det_M = M11 * M22 - M12**2

        # Effective generalized forces (Coriolis, centrifugal, gravity, damping, input)
        tau1 = (
            tau
            - self.Dr * th_d
            - self.mp * self.lp**2 * sin_2al * al_d * th_d
            + self.mp * self.Lr * self.lp * sin_al * al_d**2
        )
        tau2 = (
            -self.Dp * al_d
            + 0.5 * self.mp * self.lp**2 * sin_2al * th_d**2
            + self.mp * self.g * self.lp * sin_al
        )

        # Invert 2x2 inertia matrix analytically
        th_dd = (M22 * tau1 - M12 * tau2) / det_M
        al_dd = (-M12 * tau1 + M11 * tau2) / det_M

        return np.array([th_d, al_d, th_dd, al_dd], dtype=float)

    def linearize(
        self,
        x_eq: ArrayLike | None = None,
        u_eq: ArrayLike | None = None,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuous state-space linearisation (A, B) about equilibrium.

        When x_eq is None or [0, 0, 0, 0], uses the exact analytical formula
        about the unstable upright equilibrium.
        """
        if x_eq is None or np.allclose(x_eq, 0.0):
            # Analytical linearization about upright equilibrium (alpha = 0)
            M12_0 = self.mp * self.Lr * self.lp
            det_M0 = self.J_t * self.Jp_eff - M12_0**2

            A = np.array([
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [
                    0.0,
                    -(self.mp**2 * self.Lr * self.lp**2 * self.g) / det_M0,
                    -(self.Jp_eff * self.Dr) / det_M0,
                    (M12_0 * self.Dp) / det_M0,
                ],
                [
                    0.0,
                    (self.J_t * self.mp * self.g * self.lp) / det_M0,
                    (M12_0 * self.Dr) / det_M0,
                    -(self.J_t * self.Dp) / det_M0,
                ],
            ], dtype=float)

            B = np.array([
                [0.0],
                [0.0],
                [self.Jp_eff / det_M0],
                [-M12_0 / det_M0],
            ], dtype=float)

            return A, B

        # Custom equilibrium fallback via finite difference
        return super().linearize(x_eq=x_eq, u_eq=u_eq, eps=eps)

    def pendulum_energy(self, x: ArrayLike) -> float:
        """Mechanical energy of the pendulum relative to upright equilibrium (0 upright, -2*mp*g*lp hanging)."""
        x = np.asarray(x, dtype=float)
        _, al, _, al_d = x
        return float(
            0.5 * self.Jp_eff * al_d**2 + self.mp * self.g * self.lp * (np.cos(al) - 1.0)
        )

    def total_energy(self, x: ArrayLike) -> float:
        """Total mechanical kinetic + potential energy of the entire system."""
        x = np.asarray(x, dtype=float)
        _, al, th_d, al_d = x
        sin_al = np.sin(al)
        cos_al = np.cos(al)

        M11 = self.J_t + self.mp * self.lp**2 * sin_al**2
        M12 = self.mp * self.Lr * self.lp * cos_al
        M22 = self.Jp_eff

        T = 0.5 * M11 * th_d**2 + M12 * th_d * al_d + 0.5 * M22 * al_d**2
        V = self.mp * self.g * self.lp * np.cos(al)
        return float(T + V)
