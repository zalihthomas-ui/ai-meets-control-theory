r"""Full 3-D quadrotor - a real drone in six degrees of freedom.

State (12): ``[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]``
    position (m, ENU / z up), world-frame velocity (m/s),
    roll-pitch-yaw Euler angles (rad), body angular rates (rad/s).

Input (4): ``[f, tau_x, tau_y, tau_z]``
    collective thrust along the body z axis (N) and the three body torques
    (N.m) - the standard control-allocation abstraction; a real airframe maps
    these to four rotor speeds through a fixed mixer.

.. math::

    \dot p_w      &= v_w \\
    \dot v_w      &= [0,0,-g] + \tfrac{f}{m}\,R(\phi,\theta,\psi)\,e_z
                     - \tfrac{k_d}{m} v_w \\
    [\dot\phi,\dot\theta,\dot\psi]^\top &= W(\phi,\theta)\,[p,q,r]^\top \\
    I\,[\dot p,\dot q,\dot r]^\top &= \tau - [p,q,r]\times I[p,q,r]

Hover trim: ``f = m g``, ``tau = 0``, attitude and rates zero. Default
parameters are the Bitcraze Crazyflie 2.0.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem

CF2_MASS = 0.028
CF2_IXX = 1.4e-5
CF2_IYY = 1.4e-5
CF2_IZZ = 2.17e-5
CF2_ARM = 0.046
CF2_F_MAX = 0.60          # N total thrust (~2.2:1 thrust-to-weight)
CF2_TAU_MAX = 0.010       # N.m per body axis


def rotation_matrix(phi, theta, psi):
    """Body -> world rotation, Z-Y-X (yaw-pitch-roll) convention."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth,       cth * sphi,                       cth * cphi],
    ])


class Quadrotor3D(DynamicalSystem):
    n_states = 12
    n_inputs = 4
    n_outputs = 12

    def __init__(
        self,
        m: float = CF2_MASS,
        Ixx: float = CF2_IXX,
        Iyy: float = CF2_IYY,
        Izz: float = CF2_IZZ,
        arm: float = CF2_ARM,
        g: float = 9.81,
        drag: float = 1e-4,
        f_max: float = CF2_F_MAX,
        tau_max: float = CF2_TAU_MAX,
    ) -> None:
        self.m, self.g, self.arm, self.cd = float(m), float(g), float(arm), float(drag)
        self.I = np.diag([float(Ixx), float(Iyy), float(Izz)])
        self.Iinv = np.linalg.inv(self.I)
        self.f_max, self.tau_max = float(f_max), float(tau_max)

    @property
    def u_hover(self) -> np.ndarray:
        return np.array([self.m * self.g, 0.0, 0.0, 0.0])

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        v = x[3:6]
        phi, theta, psi = x[6:9]
        omega = x[9:12]
        f, tau = u[0], u[1:4]

        R = rotation_matrix(phi, theta, psi)
        acc = np.array([0.0, 0.0, -self.g]) + (f / self.m) * R[:, 2] - (self.cd / self.m) * v

        # Euler-rate kinematics  [phi_dot; theta_dot; psi_dot] = W(phi,theta) omega
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth = np.cos(theta)
        tth = np.tan(theta)
        W = np.array([
            [1.0, sphi * tth,  cphi * tth],
            [0.0, cphi,       -sphi],
            [0.0, sphi / cth,  cphi / cth],
        ])
        euler_dot = W @ omega
        omega_dot = self.Iinv @ (tau - np.cross(omega, self.I @ omega))

        return np.concatenate([v, acc, euler_dot, omega_dot])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        """Numerical Jacobians about hover (default ``x = 0``, ``u = u_hover``).

        The hover linearisation is exactly decoupled into four SISO/低-order
        channels (altitude, yaw, and the two symmetric position-attitude
        subsystems); we compute it by central differences for simplicity.
        """
        if x_eq is None:
            x_eq = np.zeros(12)
        if u_eq is None:
            u_eq = self.u_hover
        return super().linearize(x_eq, u_eq, eps)
