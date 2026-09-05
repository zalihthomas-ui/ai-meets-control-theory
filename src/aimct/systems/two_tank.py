r"""Coupled Two-Tank liquid level dynamical system.

Coordinates:
- State ``x = [h1, h2]``
  - ``h1``: liquid level in Tank 1 [m]
  - ``h2``: liquid level in Tank 2 [m]
- Input ``u = [V_p]``: pump voltage [V] (0 to V_max), delivering volumetric flow ``F_in = K_p * V_p``.

Parameters default to the canonical Quanser Coupled Tanks laboratory apparatus.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem

__all__ = ["TwoTank"]


class TwoTank(DynamicalSystem):
    """Coupled Two-Tank liquid level process control system.

    Parameters
    ----------
    A1: float
        Cross-sectional area of Tank 1 [m^2]. Defaults to 1.555e-3 (D = 4.45 cm).
    A2: float
        Cross-sectional area of Tank 2 [m^2]. Defaults to 1.555e-3 (D = 4.45 cm).
    a12: float
        Cross-sectional area of the inter-tank coupling orifice [m^2]. Defaults to 1.81e-5 (d = 4.8 mm).
    a2: float
        Cross-sectional area of the Tank 2 bottom drain orifice [m^2]. Defaults to 1.81e-5 (d = 4.8 mm).
    Kp: float
        Volumetric pump flow coefficient [m^3 / (s * V)]. Defaults to 3.3e-6 (3.3 cm^3/(s*V)).
    g: float
        Gravitational acceleration [m/s^2]. Defaults to 9.81.
    h_max: float
        Maximum physical tank level before overflow [m]. Defaults to 0.30.
    v_max: float
        Maximum pump input voltage [V]. Defaults to 12.0.
    """

    n_states = 2
    n_inputs = 1
    n_outputs = 2

    def __init__(
        self,
        A1: float = 1.555e-3,
        A2: float = 1.555e-3,
        a12: float = 1.81e-5,
        a2: float = 1.81e-5,
        Kp: float = 3.3e-6,
        g: float = 9.81,
        h_max: float = 0.30,
        v_max: float = 12.0,
    ) -> None:
        self.A1 = float(A1)
        self.A2 = float(A2)
        self.a12 = float(a12)
        self.a2 = float(a2)
        self.Kp = float(Kp)
        self.g = float(g)
        self.h_max = float(h_max)
        self.v_max = float(v_max)

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        """Compute rate of change of liquid levels [dh1/dt, dh2/dt]."""
        x, u = self._prep(x, u)
        h1 = float(np.clip(x[0], 0.0, self.h_max))
        h2 = float(np.clip(x[1], 0.0, self.h_max))
        V_p = float(np.clip(u[0], 0.0, self.v_max))

        # Pump inflow (unidirectional flow >= 0)
        Fin = self.Kp * V_p

        # Inter-tank flow (Torricelli law with bi-directional sign handling)
        delta_h = h1 - h2
        if abs(delta_h) > 1e-9:
            q12 = self.a12 * np.sign(delta_h) * np.sqrt(2.0 * self.g * abs(delta_h))
        else:
            q12 = 0.0

        # Tank 2 bottom outflow drain
        if h2 > 1e-9:
            q2 = self.a2 * np.sqrt(2.0 * self.g * h2)
        else:
            q2 = 0.0

        dh1_dt = (Fin - q12) / self.A1
        dh2_dt = (q12 - q2) / self.A2

        if h1 <= 0.0 and dh1_dt < 0.0:
            dh1_dt = 0.0
        if h2 <= 0.0 and dh2_dt < 0.0:
            dh2_dt = 0.0

        return np.array([dh1_dt, dh2_dt], dtype=float)

    def steady_state_operating_point(self, h2_target: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        """Compute the equilibrium state x_eq = [h1_0, h2_0] and input u_eq = [Vp_0] for a given h2."""
        h2_0 = float(h2_target)
        h1_0 = h2_0 + (self.a2 / self.a12) ** 2 * h2_0
        Fin_0 = self.a2 * np.sqrt(2.0 * self.g * h2_0)
        Vp_0 = Fin_0 / self.Kp
        return np.array([h1_0, h2_0], dtype=float), np.array([Vp_0], dtype=float)

    def linearize(
        self,
        x_eq: ArrayLike | None = None,
        u_eq: ArrayLike | None = None,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuous state-space linearisation (A, B) about operating equilibrium.

        When x_eq is None, defaults to the nominal operating level h2_0 = 0.10 m (h1_0 = 0.20 m).
        """
        if x_eq is None:
            x_eq, u_eq = self.steady_state_operating_point(0.10)
        else:
            x_eq = np.asarray(x_eq, dtype=float)
            if u_eq is None:
                _, u_eq = self.steady_state_operating_point(x_eq[1])

        h1_0, h2_0 = x_eq[0], x_eq[1]
        delta_h = max(1e-6, abs(h1_0 - h2_0))
        h2_val = max(1e-6, h2_0)

        # Flow resistances R12 and R2
        R12 = np.sqrt(2.0 * delta_h) / (self.a12 * np.sqrt(self.g))
        R2 = np.sqrt(2.0 * h2_val) / (self.a2 * np.sqrt(self.g))

        A = np.array([
            [-1.0 / (R12 * self.A1), 1.0 / (R12 * self.A1)],
            [1.0 / (R12 * self.A2), -(1.0 / (R12 * self.A2) + 1.0 / (R2 * self.A2))],
        ], dtype=float)

        B = np.array([
            [self.Kp / self.A1],
            [0.0],
        ], dtype=float)

        return A, B
