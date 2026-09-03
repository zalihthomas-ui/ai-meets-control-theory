r"""Armature-controlled DC motor -- a real electromechanical actuator.

State ``[theta, omega, i]`` (rotor angle rad, speed rad/s, armature current A),
input ``v`` (armature voltage V). This is a *linear* plant:

.. math::

    \dot\theta &= \omega \\
    \dot\omega &= (K_t\,i - b\,\omega - \tau_L) / J \\
    \dot i     &= (v - R\,i - K_e\,\omega) / L

with an optional constant load torque ``tau_load``. Default parameters are a
small brushed servo motor (Pittman/Maxon class).

Reduced 2-state model (armature inductance neglected, ``L -> 0``) is available
via :meth:`reduced` for the classic textbook transfer function
:math:`\theta(s)/V(s) = K / (s\,(J s + B_{\rm eff}))`.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class DCMotor(DynamicalSystem):
    n_states = 3          # [angle, speed, current]
    n_inputs = 1          # [voltage]

    def __init__(
        self,
        R: float = 1.2,          # armature resistance [ohm]
        L: float = 5.0e-4,       # armature inductance [H]
        Kt: float = 0.043,       # torque constant [N.m/A]
        Ke: float = 0.043,       # back-EMF constant [V.s/rad]
        J: float = 2.5e-5,       # rotor + load inertia [kg.m^2]
        b: float = 3.0e-5,       # viscous friction [N.m.s]
        tau_load: float = 0.0,   # constant load torque [N.m]
        v_max: float = 24.0,     # supply voltage limit [V]
    ) -> None:
        self.R, self.L, self.Kt, self.Ke = float(R), float(L), float(Kt), float(Ke)
        self.J, self.b, self.tau_load = float(J), float(b), float(tau_load)
        self.v_max = float(v_max)

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        _, omega, i = x
        v = u[0]
        domega = (self.Kt * i - self.b * omega - self.tau_load) / self.J
        di = (v - self.R * i - self.Ke * omega) / self.L
        return np.array([omega, domega, di])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        A = np.array([
            [0.0, 1.0, 0.0],
            [0.0, -self.b / self.J, self.Kt / self.J],
            [0.0, -self.Ke / self.L, -self.R / self.L],
        ])
        B = np.array([[0.0], [0.0], [1.0 / self.L]])
        return A, B

    def reduced(self):
        """Return an equivalent 2-state ``[theta, omega]`` :class:`DCMotor2`
        with the armature inductance neglected."""
        return DCMotor2(self)


class DCMotor2(DynamicalSystem):
    """Inductance-free reduced model, state ``[theta, omega]``."""

    n_states = 2
    n_inputs = 1

    def __init__(self, full: DCMotor):
        self._f = full
        f = full
        self._a = f.b / f.J + f.Kt * f.Ke / (f.J * f.R)   # effective damping / J
        self._k = f.Kt / (f.J * f.R)                        # voltage -> accel gain
        self._d = f.tau_load / f.J
        self.v_max = f.v_max

    def dynamics(self, t, x, u):
        x, u = self._prep(x, u)
        _, omega = x
        return np.array([omega, -self._a * omega + self._k * u[0] - self._d])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        A = np.array([[0.0, 1.0], [0.0, -self._a]])
        B = np.array([[0.0], [self._k]])
        return A, B
