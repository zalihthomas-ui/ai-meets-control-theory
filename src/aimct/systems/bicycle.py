r"""Dynamic single-track ("bicycle") ground vehicle -- lateral tire forces,
not just kinematics.

State ``[X, Y, psi, vx, vy, r]`` (global position m, heading rad, body-frame
longitudinal/lateral velocity m/s, yaw rate rad/s). Input ``[delta, ax]``
(front steering angle rad, longitudinal acceleration command m/s^2).

.. math::

    \dot X &= v_x\cos\psi - v_y\sin\psi \\
    \dot Y &= v_x\sin\psi + v_y\cos\psi \\
    \dot\psi &= r \\
    \dot v_x &= a_x + v_y r - F_{yf}\sin\delta / m \\
    \dot v_y &= (F_{yf}\cos\delta + F_{yr}) / m - v_x r \\
    \dot r &= (a\,F_{yf}\cos\delta - b\,F_{yr}) / I_z

with front/rear slip angles :math:`\alpha_f = \operatorname{atan2}(v_y + a r,
v_x) - \delta`, :math:`\alpha_r = \operatorname{atan2}(v_y - b r, v_x)`, and a
tire model producing the lateral force :math:`F_y(\alpha)` per axle -- either
**linear** (:math:`F_y = -C_\alpha\alpha`, valid at small slip) or the
**Pacejka "Magic Formula"** (:math:`F_y = -\mu F_z D\sin(C\arctan(B\alpha -
E(B\alpha - \arctan B\alpha)))`, saturating and eventually *falling* at large
slip -- the nonlinearity that makes an aggressive high-speed manoeuvre a
genuinely nonlinear control problem, the way Experiment 02 showed for the
pendulum's linearisation).

Default parameters are a mid-size sedan (Rajamani, *Vehicle Dynamics and
Control*, 2nd ed., representative values); ``linearize()`` is the inherited
numeric central difference -- freeze ``vx`` at the equilibrium and this is
exactly the textbook 4-state constant-speed lateral model.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class BicycleVehicle(DynamicalSystem):
    n_states = 6      # [X, Y, psi, vx, vy, r]
    n_inputs = 2       # [delta, ax]

    def __init__(
        self,
        m: float = 1600.0,        # vehicle mass [kg]
        Iz: float = 2500.0,       # yaw moment of inertia [kg.m^2]
        a: float = 1.2,           # CG to front axle [m]
        b: float = 1.6,           # CG to rear axle [m]
        tire_model: str = "linear",
        Caf: float = 80_000.0,    # front cornering stiffness [N/rad] (linear tire)
        Car: float = 80_000.0,    # rear cornering stiffness [N/rad] (linear tire)
        mu: float = 1.0,          # road-tire friction coefficient (Pacejka peak scale)
        pacejka_B: float = 10.0,  # stiffness factor
        pacejka_C: float = 1.9,   # shape factor
        pacejka_E: float = 0.97,  # curvature factor
        delta_max: float = np.radians(30.0),   # steering limit [rad]
        ax_max: float = 4.0,                    # longitudinal accel limit [m/s^2]
    ) -> None:
        self.m, self.Iz, self.a, self.b = float(m), float(Iz), float(a), float(b)
        self.tire_model = tire_model
        self.Caf, self.Car = float(Caf), float(Car)
        self.mu = float(mu)
        self.pB, self.pC, self.pE = float(pacejka_B), float(pacejka_C), float(pacejka_E)
        self.delta_max, self.ax_max = float(delta_max), float(ax_max)
        # per-axle static normal load, for the Pacejka peak force mu*Fz
        g = 9.81
        L = self.a + self.b
        self.Fzf = self.m * g * self.b / L
        self.Fzr = self.m * g * self.a / L
        if tire_model not in ("linear", "pacejka"):
            raise ValueError("tire_model must be 'linear' or 'pacejka'")

    # -- tire model -----------------------------------------------------

    def tire_force(self, alpha, Fz: float) -> np.ndarray:
        """Lateral tire force ``Fy(alpha)`` for one axle; ``alpha`` may be a
        scalar or array. Positive slip angle -> negative (restoring) force."""
        alpha = np.asarray(alpha, dtype=float)
        if self.tire_model == "linear":
            Calpha = self.Caf if Fz == self.Fzf else self.Car
            return -Calpha * alpha
        B, C, E = self.pB, self.pC, self.pE
        Bx = B * alpha
        return -self.mu * Fz * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))

    def slip_angles(self, x: ArrayLike, delta: float):
        x = np.asarray(x, dtype=float)
        vx, vy, r = x[3], x[4], x[5]
        alpha_f = np.arctan2(vy + self.a * r, vx) - delta
        alpha_r = np.arctan2(vy - self.b * r, vx)
        return alpha_f, alpha_r

    # -- dynamics --------------------------------------------------------

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        _, _, psi, vx, vy, r = x
        delta = np.clip(u[0], -self.delta_max, self.delta_max)
        ax_cmd = np.clip(u[1], -self.ax_max, self.ax_max)

        alpha_f, alpha_r = self.slip_angles(x, delta)
        Fyf = self.tire_force(alpha_f, self.Fzf)
        Fyr = self.tire_force(alpha_r, self.Fzr)

        Xdot = vx * np.cos(psi) - vy * np.sin(psi)
        Ydot = vx * np.sin(psi) + vy * np.cos(psi)
        psidot = r
        vxdot = ax_cmd + vy * r - Fyf * np.sin(delta) / self.m
        vydot = (Fyf * np.cos(delta) + Fyr) / self.m - vx * r
        rdot = (self.a * Fyf * np.cos(delta) - self.b * Fyr) / self.Iz
        return np.array([Xdot, Ydot, psidot, vxdot, vydot, rdot])

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        """Numeric Jacobian (inherited central difference) about a
        constant-speed straight-line cruise by default (``vx = 20`` m/s,
        everything else zero) -- freezing ``vx`` at the operating point
        recovers the textbook constant-speed lateral-dynamics model."""
        if x_eq is None:
            x_eq = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
        return super().linearize(np.asarray(x_eq, dtype=float),
                                 np.zeros(2) if u_eq is None else u_eq, eps)
