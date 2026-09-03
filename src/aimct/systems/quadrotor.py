r"""Planar quadrotor -- a real drone flying in a vertical plane.

Parameters are those of the **Bitcraze Crazyflie 2.0** nano-quadrotor. The body
moves in the :math:`x`-:math:`z` plane and rotates about the body :math:`y` axis;
two lumped rotor groups produce thrusts :math:`T_1` (right) and :math:`T_2`
(left), each bounded by ``[0, T_max]``.

State  ``[x, z, theta, xdot, zdot, thetadot]`` (position m, pitch rad, velocities).
Input  ``[T1, T2]`` (N).

.. math::

    \ddot x      &= -(T_1 + T_2)\sin\theta / m \\
    \ddot z      &=  (T_1 + T_2)\cos\theta / m - g \\
    \ddot\theta  &=  (T_1 - T_2)\,\ell / I_{yy}

with a small aerodynamic drag :math:`-(c_d/m)\,\dot{}` on the translational
velocities. ``theta = 0`` is level flight; hover is
:math:`T_1 = T_2 = m g / 2`.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem

# --- Crazyflie 2.0 -----------------------------------------------------------
CF2_MASS = 0.028          # kg
CF2_IYY = 1.4e-5          # kg m^2  (about the body y axis)
CF2_ARM = 0.046           # m       (motor to centre)
CF2_THRUST_MAX = 0.1573   # N per rotor group (4 * ~0.393 N max / 2 groups ~ 2:1 T/W)


class PlanarQuadrotor(DynamicalSystem):
    n_states = 6
    n_inputs = 2
    n_outputs = 6

    def __init__(
        self,
        m: float = CF2_MASS,
        Iyy: float = CF2_IYY,
        arm: float = CF2_ARM,
        g: float = 9.81,
        drag: float = 1e-4,
        thrust_max: float = CF2_THRUST_MAX,
    ) -> None:
        self.m, self.Iyy, self.l, self.g = float(m), float(Iyy), float(arm), float(g)
        self.cd = float(drag)
        self.thrust_max = float(thrust_max)

    # hover trim -----------------------------------------------------------
    @property
    def u_hover(self) -> np.ndarray:
        return np.full(2, self.m * self.g / 2.0)

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        _, _, th, xd, zd, thd = x
        T1, T2 = u
        thrust = T1 + T2
        s, c = np.sin(th), np.cos(th)
        xdd = -thrust * s / self.m - self.cd * xd / self.m
        zdd = thrust * c / self.m - self.g - self.cd * zd / self.m
        thdd = (T1 - T2) * self.l / self.Iyy
        return np.array([xd, zd, thd, xdd, zdd, thdd])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        """Analytic Jacobians about level hover (``x = 0``, ``u = u_hover``)."""
        m, cd = self.m, self.cd
        # d(xdd)/d(theta) = -(T1+T2) cos(th)/m ; at hover = -(m g)/m = -g
        A = np.zeros((6, 6))
        A[0, 3] = A[1, 4] = A[2, 5] = 1.0
        A[3, 2] = -self.g            # tilt -> horizontal accel
        A[3, 3] = -cd / m
        A[4, 4] = -cd / m
        B = np.zeros((6, 2))
        B[4, 0] = B[4, 1] = 1.0 / m           # both rotors -> vertical accel
        B[5, 0] = self.l / self.Iyy           # differential thrust -> pitch accel
        B[5, 1] = -self.l / self.Iyy
        return A, B
