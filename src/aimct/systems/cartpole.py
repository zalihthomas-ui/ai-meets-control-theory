"""Cart-pole (inverted pendulum on a cart), force input on the cart.

State ``[x, x_dot, theta, theta_dot]`` with ``theta = 0`` at the **upright**
position. Standard nonlinear model (e.g. Barto/Sutton, with pole mass and
length), no friction by default.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class CartPole(DynamicalSystem):
    n_states = 4  # [cart pos, cart vel, pole angle from upright, pole ang. vel]
    n_inputs = 1  # [force on cart]

    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        length: float = 0.5,  # distance to pole centre of mass
        g: float = 9.81,
    ) -> None:
        self.mc, self.mp, self.l, self.g = (
            float(m_cart),
            float(m_pole),
            float(length),
            float(g),
        )

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        _, xdot, th, thdot = x
        f = u[0]
        mc, mp, l, g = self.mc, self.mp, self.l, self.g
        total = mc + mp
        sin, cos = np.sin(th), np.cos(th)

        temp = (f + mp * l * thdot**2 * sin) / total
        thddot = (g * sin - cos * temp) / (l * (4.0 / 3.0 - mp * cos**2 / total))
        xddot = temp - mp * l * thddot * cos / total
        return np.array([xdot, xddot, thdot, thddot])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        """Analytic Jacobian about the upright equilibrium ``x = 0``, ``u = 0``."""
        mc, mp, l, g = self.mc, self.mp, self.l, self.g
        denom = l * (4.0 / 3.0 - mp / (mc + mp))
        # d(thddot)/d(theta) and d(thddot)/d(f) at theta = 0, thetadot = 0
        dth_dtheta = g / denom
        dth_df = -1.0 / ((mc + mp) * denom)
        dx_dtheta = -mp * l * dth_dtheta / (mc + mp)
        dx_df = 1.0 / (mc + mp) - mp * l * dth_df / (mc + mp)
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, dx_dtheta, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, dth_dtheta, 0.0],
            ]
        )
        B = np.array([[0.0], [dx_df], [0.0], [dth_df]])
        return A, B
