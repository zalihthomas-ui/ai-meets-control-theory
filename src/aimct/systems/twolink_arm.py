r"""Planar two-link revolute arm -- rigid-body manipulator dynamics.

State ``[q1, q2, dq1, dq2]`` (joint angles rad, joint rates rad/s), input
``[tau1, tau2]`` (joint torques N.m).  ``q1`` is measured from the ``+x`` axis,
``q2`` from link 1, so gravity enters through ``cos``.  The Euler-Lagrange
model (Spong & Vidyasagar, *Robot Modeling and Control*):

.. math::

    M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) + F_v\dot q = \tau

A rigid point payload of mass :attr:`payload` at the link-2 tip (distance
``l2`` from joint 2) enters ``M``, ``C`` and ``G`` explicitly.  :meth:`M`,
:meth:`C` and :meth:`G` are public for computed-torque / inverse-dynamics
control.

Default parameters are a table-top 2-DOF arm (Quanser / teaching-arm class);
see ``docs/references/robotics-systems-reference.md``.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class TwoLinkArm(DynamicalSystem):
    n_states = 4      # [q1, q2, dq1, dq2]
    n_inputs = 2      # [tau1, tau2]

    def __init__(
        self,
        m1: float = 1.0, m2: float = 0.8,            # link masses [kg]
        l1: float = 0.50, l2: float = 0.40,          # link lengths [m]
        lc1: float = 0.25, lc2: float = 0.20,        # centre-of-mass offsets [m]
        I1: float = 0.02083, I2: float = 0.01067,    # link inertias about COM [kg.m^2]
        g: float = 9.81,                             # gravity [m/s^2]
        b=0.10,                                      # joint viscous friction [N.m.s], scalar or (2,)
        payload: float = 0.0,                        # point mass at the wrist [kg], 0..0.5
        tau_max=(15.0, 10.0),                        # per-joint torque limit [N.m]
        q2_limit: float = 2.8,                       # |q2| travel bound [rad] (informational)
    ) -> None:
        self.m1, self.m2 = float(m1), float(m2)
        self.l1, self.l2 = float(l1), float(l2)
        self.lc1, self.lc2 = float(lc1), float(lc2)
        self.I1, self.I2 = float(I1), float(I2)
        self.g = float(g)
        self.b = np.broadcast_to(np.asarray(b, float), (2,)).astype(float).copy()
        self.tau_max = np.broadcast_to(np.asarray(tau_max, float), (2,)).astype(float).copy()
        self.payload = float(payload)
        self.q2_limit = float(q2_limit)

    # -- manipulator terms (payload m_p enters explicitly) ----------------

    def M(self, q: ArrayLike) -> np.ndarray:
        """Configuration-space inertia matrix ``M(q)`` (2x2, symmetric PD)."""
        q = np.asarray(q, dtype=float)
        m1, m2, mp = self.m1, self.m2, self.payload
        l1, l2, lc1, lc2 = self.l1, self.l2, self.lc1, self.lc2
        c2 = np.cos(q[1])
        m11 = (m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
               + mp * (l1**2 + l2**2 + 2 * l1 * l2 * c2) + self.I1 + self.I2)
        m12 = (m2 * (lc2**2 + l1 * lc2 * c2) + mp * (l2**2 + l1 * l2 * c2) + self.I2)
        m22 = m2 * lc2**2 + mp * l2**2 + self.I2
        return np.array([[m11, m12], [m12, m22]])

    def C(self, q: ArrayLike, dq: ArrayLike) -> np.ndarray:
        """Coriolis / centrifugal matrix ``C(q, dq)`` -- ``C @ dq`` is the
        velocity-product torque.  ``d/dt M - 2C`` is skew-symmetric."""
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        h = (self.m2 * self.l1 * self.lc2
             + self.payload * self.l1 * self.l2) * np.sin(q[1])
        return np.array([[-h * dq[1], -h * (dq[0] + dq[1])],
                         [ h * dq[0],  0.0]])

    def G(self, q: ArrayLike) -> np.ndarray:
        """Gravity torque vector ``G(q)`` (2,).  Zero if ``g == 0``."""
        q = np.asarray(q, dtype=float)
        m1, m2, mp, g = self.m1, self.m2, self.payload, self.g
        l1, lc1, lc2, l2 = self.l1, self.lc1, self.lc2, self.l2
        q1, q12 = q[0], q[0] + q[1]
        tip = (m2 * lc2 + mp * l2) * g * np.cos(q12)
        g1 = (m1 * lc1 + m2 * l1 + mp * l1) * g * np.cos(q1) + tip
        g2 = tip
        return np.array([g1, g2])

    # -- dynamics --------------------------------------------------------

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        q, dq = x[:2], x[2:]
        tau = np.clip(u, -self.tau_max, self.tau_max)
        rhs = tau - self.C(q, dq) @ dq - self.G(q) - self.b * dq
        ddq = np.linalg.solve(self.M(q), rhs)
        return np.concatenate([dq, ddq])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        """Numeric ``(A, B)`` about ``x_eq`` (default: link 1 up, link 2
        folded, at rest) holding the gravity-compensation torque."""
        if x_eq is None:
            x_eq = np.array([np.pi / 2.0, 0.0, 0.0, 0.0])
        x_eq = np.asarray(x_eq, dtype=float)
        if u_eq is None:
            u_eq = self.G(x_eq[:2])
        return super().linearize(x_eq, u_eq, eps)

    # -- kinematics ----------------------------------------------------

    def forward_kinematics(self, q: ArrayLike) -> np.ndarray:
        """Wrist (link-2 tip) position ``[x, y]`` in the base frame."""
        q = np.asarray(q, dtype=float)
        q1, q12 = q[0], q[0] + q[1]
        return np.array([self.l1 * np.cos(q1) + self.l2 * np.cos(q12),
                         self.l1 * np.sin(q1) + self.l2 * np.sin(q12)])

    def jacobian(self, q: ArrayLike) -> np.ndarray:
        """End-effector position Jacobian ``d[x, y]/dq`` (2x2).  Singular at
        ``q2 = 0`` (extended) and ``q2 = +/-pi`` (folded)."""
        q = np.asarray(q, dtype=float)
        q1, q12 = q[0], q[0] + q[1]
        return np.array([
            [-self.l1 * np.sin(q1) - self.l2 * np.sin(q12), -self.l2 * np.sin(q12)],
            [ self.l1 * np.cos(q1) + self.l2 * np.cos(q12),  self.l2 * np.cos(q12)],
        ])
