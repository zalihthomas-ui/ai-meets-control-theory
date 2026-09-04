r"""Differential-drive (unicycle) mobile robot with first-order actuator lag.

State ``[x, y, theta, v, omega]`` -- planar pose (m, m, rad) plus the *actual*
body linear / angular speed (m/s, rad/s).  Input ``u = [v_cmd, omega_cmd]`` is
the commanded body twist; the on-board speed controller tracks it with a
first-order lag, so the robot cannot change speed or turn-rate instantly:

.. math::

    \dot x      &= v \cos\theta \\
    \dot y      &= v \sin\theta \\
    \dot\theta  &= \omega \\
    \dot v      &= (v_{\rm cmd} - v) / \tau_v \\
    \dot\omega  &= (\omega_{\rm cmd} - \omega) / \tau_\omega

The kinematic ``[x, y, theta]`` unicycle is the ``tau_v, tau_omega -> 0`` limit;
keeping ``v, omega`` in the state is what makes a path-follower's job realistic.

Default parameters are a small indoor robot of the TurtleBot3-Burger class
(wheel radius 33 mm, 160 mm track, |v| <= 0.22 m/s, |omega| <= 2.84 rad/s).
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class DifferentialDriveRobot(DynamicalSystem):
    n_states = 5      # [x, y, theta, v, omega]
    n_inputs = 2      # [v_cmd, omega_cmd]

    def __init__(
        self,
        wheel_radius: float = 0.033,   # drive-wheel radius [m]
        wheel_base: float = 0.160,     # distance between the wheels (track) [m]
        mass: float = 1.0,             # chassis mass [kg] (informational)
        tau_v: float = 0.05,           # linear-speed loop time constant [s]
        tau_omega: float = 0.05,       # yaw-rate loop time constant [s]
        v_max: float = 0.22,           # linear-speed limit [m/s]
        omega_max: float = 2.84,       # yaw-rate limit [rad/s]
        v_ref: float = 0.15,           # nominal cruise speed for linearize() [m/s]
    ) -> None:
        self.wheel_radius = float(wheel_radius)
        self.wheel_base = float(wheel_base)
        self.mass = float(mass)
        self.tau_v = float(tau_v)
        self.tau_omega = float(tau_omega)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.v_ref = float(v_ref)

    # -- dynamics -----------------------------------------------------------

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        _, _, theta, v, omega = x
        v_cmd = np.clip(u[0], -self.v_max, self.v_max)
        omega_cmd = np.clip(u[1], -self.omega_max, self.omega_max)
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega,
            (v_cmd - v) / self.tau_v,
            (omega_cmd - omega) / self.tau_omega,
        ])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        """Analytic ``(A, B)`` about a constant-speed straight path.

        Default equilibrium: heading ``theta = 0``, cruising at ``v_ref`` with
        zero yaw rate (``x_eq = [0, 0, 0, v_ref, 0]``, ``u_eq = [v_ref, 0]``).
        The ``x``/``y`` rows carry free integrators, exactly like the DC motor's
        angle state.
        """
        if x_eq is None:
            x_eq = np.array([0.0, 0.0, 0.0, self.v_ref, 0.0])
        x_eq = np.asarray(x_eq, dtype=float)
        theta, v = float(x_eq[2]), float(x_eq[3])
        s, c = np.sin(theta), np.cos(theta)
        A = np.array([
            [0.0, 0.0, -v * s, c, 0.0],
            [0.0, 0.0,  v * c, s, 0.0],
            [0.0, 0.0,  0.0,   0.0, 1.0],
            [0.0, 0.0,  0.0, -1.0 / self.tau_v, 0.0],
            [0.0, 0.0,  0.0,   0.0, -1.0 / self.tau_omega],
        ])
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0 / self.tau_v, 0.0],
            [0.0, 1.0 / self.tau_omega],
        ])
        return A, B

    # -- differential-drive wheel kinematics ------------------------------

    def wheel_speeds(self, v: float, omega: float) -> np.ndarray:
        """Body twist ``(v, omega)`` -> wheel angular speeds ``[left, right]``
        (rad/s), using ``v = r (wR + wL) / 2`` and ``omega = r (wR - wL) / b``."""
        r, b = self.wheel_radius, self.wheel_base
        w_right = (v + 0.5 * b * omega) / r
        w_left = (v - 0.5 * b * omega) / r
        return np.array([w_left, w_right])

    def body_twist(self, w_left: float, w_right: float) -> np.ndarray:
        """Wheel angular speeds ``[left, right]`` -> body twist ``[v, omega]``."""
        r, b = self.wheel_radius, self.wheel_base
        v = 0.5 * r * (w_right + w_left)
        omega = r * (w_right - w_left) / b
        return np.array([v, omega])
