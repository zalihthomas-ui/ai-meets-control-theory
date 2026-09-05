r"""Disturbance-Observer (DOB) Control & Q-Filter Architectures.

A Disturbance Observer estimates internal model mismatch and external disturbances
(including wind, friction, unmodeled loads) from measurable inputs and outputs,
providing two key capabilities:

1. **Matched Disturbance Cancellation**:
   Direct algebraic subtraction in the control channel :math:`u = u_{\\text{base}} - \\hat{d}_{\\text{matched}}`.
2. **Unmatched Disturbance Reallocation**:
   Virtual reference/state modification (e.g., generating wind-opposing tilt
   :math:`\\theta_{\\text{virtual}} = \\hat{d}_x / g` for underactuated quadrotors)
   without requiring position drift or sluggish integrator accumulation.

Theoretical references:
- Ohnishi, K. (1987). "A new servo method in mechatronics."
- Umeno, T., & Hori, Y. (1991). "Robust speed control of DC servomotors using
  modern two degrees-of-freedom controller design."
- See :doc:`/references/disturbance-observer-reference` for detailed proofs.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .base import ArrayLike, Controller, _clip

__all__ = ["QFilter", "DisturbanceObserver"]


class QFilter:
    r"""Discrete-time Low-Pass Q-Filter for Disturbance Observers.

    Implements causal rational low-pass filters discretized via the Tustin
    (bilinear) transform:

    - **Order 1**: :math:`Q(s) = \\frac{\\omega_c}{s + \\omega_c}` (relative degree 1).
    - **Order 2**: :math:`Q(s) = \\frac{\\omega_c^2}{s^2 + 2\\zeta \\omega_c s + \\omega_c^2}` (relative degree 2).
    - **Order 3**: :math:`Q(s) = \\frac{\\omega_c^3}{(s + \\omega_c)^3}` (relative degree 3).

    Parameters
    ----------
    cutoff_freq:
        Cutoff frequency :math:`\\omega_c` in rad/s (default: 10.0).
    damping:
        Damping ratio :math:`\\zeta` for 2nd-order filter (default: 1.0 = critically damped).
    order:
        Filter order (1, 2, or 3; default: 2).
    dt:
        Default discrete sample time in seconds (default: 0.01).
    n_channels:
        Number of parallel filter channels (default: 1).
    """

    def __init__(
        self,
        cutoff_freq: float = 10.0,
        damping: float = 1.0,
        order: int = 2,
        dt: float = 0.01,
        n_channels: int = 1,
    ) -> None:
        self.cutoff_freq = float(cutoff_freq)
        self.damping = float(damping)
        self.order = int(order)
        self.dt = float(dt)
        self.n_channels = int(n_channels)

        if self.cutoff_freq <= 0.0:
            raise ValueError(f"cutoff_freq must be positive, got {cutoff_freq}")
        if self.damping <= 0.0:
            raise ValueError(f"damping must be positive, got {damping}")
        if self.order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2, or 3, got {order}")

        self._compute_coefficients(self.dt)
        self.reset()

    def _compute_coefficients(self, dt: float) -> None:
        """Compute IIR difference equation coefficients via Tustin transform."""
        self.dt = float(dt)
        w = self.cutoff_freq
        zeta = self.damping

        if self.order == 1:
            # Q(s) = w / (s + w)
            # s -> (2/dt) (1 - z^-1)/(1 + z^-1)
            # (2/dt)(1 - z^-1) + w(1 + z^-1) = (2 + w*dt) + (w*dt - 2) z^-1
            a0 = 2.0 + w * dt
            self.b = np.array([w * dt, w * dt]) / a0
            self.a = np.array([1.0, (w * dt - 2.0) / a0])

        elif self.order == 2:
            # Q(s) = w^2 / (s^2 + 2*zeta*w*s + w^2)
            w2 = w * w
            w_dt = w * dt
            a0 = 4.0 + 4.0 * zeta * w_dt + w2 * dt * dt
            a1 = -8.0 + 2.0 * w2 * dt * dt
            a2 = 4.0 - 4.0 * zeta * w_dt + w2 * dt * dt
            b0 = w2 * dt * dt
            b1 = 2.0 * w2 * dt * dt
            b2 = w2 * dt * dt

            self.b = np.array([b0, b1, b2]) / a0
            self.a = np.array([1.0, a1 / a0, a2 / a0])

        elif self.order == 3:
            # Q(s) = w^3 / (s + w)^3 = w^3 / (s^3 + 3w s^2 + 3w^2 s + w^3)
            w3 = w**3
            # Discretize (s+w)^3 using (2/dt)*(1-z^-1)/(1+z^-1)
            k = 2.0 / dt
            p0 = k + w
            p1 = w - k
            # (p0 + p1 z^-1)^3 = p0^3 + 3 p0^2 p1 z^-1 + 3 p0 p1^2 z^-2 + p1^3 z^-3
            a0 = p0**3
            a1 = 3.0 * p0**2 * p1
            a2 = 3.0 * p0 * p1**2
            a3 = p1**3
            # Numerator: w^3 * (1 + z^-1)^3 = w^3 * (1 + 3z^-1 + 3z^-2 + z^-3)
            self.b = w3 * np.array([1.0, 3.0, 3.0, 1.0]) / a0
            self.a = np.array([1.0, a1 / a0, a2 / a0, a3 / a0])

        self.order_len = len(self.a)

    def reset(self, initial_value: ArrayLike | None = None) -> None:
        """Reset internal filter buffers to initial value (or zeros)."""
        if initial_value is None:
            init = np.zeros(self.n_channels)
        else:
            init = np.atleast_1d(np.asarray(initial_value, dtype=float))
            if len(init) != self.n_channels:
                if len(init) == 1:
                    init = np.full(self.n_channels, init[0])
                else:
                    raise ValueError(
                        f"initial_value shape {init.shape} mismatch with n_channels {self.n_channels}"
                    )

        self.x_buf = np.zeros((self.order_len, self.n_channels))
        self.y_buf = np.zeros((self.order_len, self.n_channels))

        for i in range(self.order_len):
            self.x_buf[i] = init
            self.y_buf[i] = init

    def filter(self, x: ArrayLike, dt: float | None = None) -> np.ndarray:
        """Advance the filter by one step with input sample ``x``."""
        if dt is not None and abs(dt - self.dt) > 1e-9:
            self._compute_coefficients(dt)

        x_in = np.atleast_1d(np.asarray(x, dtype=float))
        if len(x_in) != self.n_channels:
            if len(x_in) == 1:
                x_in = np.full(self.n_channels, x_in[0])
            else:
                raise ValueError(
                    f"Input dimension {len(x_in)} does not match n_channels {self.n_channels}"
                )

        # Shift buffers: x_buf[0] is newest input
        self.x_buf[1:] = self.x_buf[:-1]
        self.x_buf[0] = x_in

        # Compute Direct Form I IIR difference equation
        # y[k] = sum(b_i * x[k-i]) - sum(a_j * y[k-j]) for j >= 1
        y_out = np.zeros(self.n_channels)
        for i in range(self.order_len):
            y_out += self.b[i] * self.x_buf[i]
        for j in range(1, self.order_len):
            y_out -= self.a[j] * self.y_buf[j - 1]

        # Shift y buffer
        self.y_buf[1:] = self.y_buf[:-1]
        self.y_buf[0] = y_out

        return y_out.copy()


class DisturbanceObserver(Controller):
    r"""Disturbance Observer (DOB) wrapping a base controller.

    Reconstructs lumped disturbance forces/torques and injects targeted
    matched and unmatched cancellation laws.

    Parameters
    ----------
    base_controller:
        Primary feedback controller (e.g. :class:`LQR`, :class:`PID`,
        :class:`StateFeedback`, :class:`LinearMPC`).
    plant:
        Optional plant instance (e.g. :class:`PlanarQuadrotor`, :class:`MassSpringDamper`).
        If provided, system-specific nonlinear dynamics and kinematic transformations
        are automatically extracted.
    cutoff_freq:
        Cutoff bandwidth :math:`\\omega_Q` in rad/s (default: 10.0 rad/s).
    damping:
        Filter damping ratio :math:`\\zeta_Q` (default: 1.0, critically damped).
    filter_order:
        Order of the Q-filter (default: 2).
    mode:
        Operation mode: ``"auto"``, ``"quadrotor"``, or ``"linear"``.
    A, B, Bd:
        Linear state-space matrices if running in linear mode without a plant instance.
    """

    def __init__(
        self,
        base_controller: Controller,
        plant: Any | None = None,
        *,
        cutoff_freq: float = 10.0,
        damping: float = 1.0,
        filter_order: int = 2,
        mode: str = "auto",
        A: ArrayLike | None = None,
        B: ArrayLike | None = None,
        Bd: ArrayLike | None = None,
    ) -> None:
        self.base_controller = base_controller
        self.plant = plant
        self.cutoff_freq = float(cutoff_freq)
        self.damping = float(damping)
        self.filter_order = int(filter_order)
        self.mode = mode

        # Auto-detect mode
        if self.mode == "auto":
            if plant is not None and plant.__class__.__name__ == "PlanarQuadrotor":
                self.mode = "quadrotor"
            else:
                self.mode = "linear"

        if self.mode == "quadrotor":
            # Quadrotor has 3 disturbance channels:
            # d_hat = [d_x (unmatched m/s^2), d_z (matched m/s^2), d_theta (matched rad/s^2)]
            self.q_filter = QFilter(
                cutoff_freq=self.cutoff_freq,
                damping=self.damping,
                order=self.filter_order,
                n_channels=3,
            )
            self.n_u = 2
        else:
            # Linear state-space mode
            if plant is not None and hasattr(plant, "linearize"):
                A_mat, B_mat = plant.linearize()
            else:
                A_mat, B_mat = A, B

            if A_mat is None or B_mat is None:
                raise ValueError("Linear mode requires A and B matrices or a linearizable plant.")

            self.A = np.asarray(A_mat, dtype=float)
            self.B = np.asarray(B_mat, dtype=float)
            self.n_x = self.A.shape[0]
            self.n_u = self.B.shape[1] if self.B.ndim > 1 else 1

            if Bd is None:
                # Default: disturbance enters matched channels
                self.Bd = self.B.copy()
            else:
                self.Bd = np.asarray(Bd, dtype=float)

            self.n_d = self.Bd.shape[1] if self.Bd.ndim > 1 else 1
            self.q_filter = QFilter(
                cutoff_freq=self.cutoff_freq,
                damping=self.damping,
                order=self.filter_order,
                n_channels=self.n_d,
            )

        self.reset()

    def reset(self) -> None:
        """Reset base controller, Q-filter, and stored history."""
        if hasattr(self.base_controller, "reset"):
            self.base_controller.reset()
        self.q_filter.reset()
        self.u_prev: np.ndarray | None = None
        self.x_prev: np.ndarray | None = None
        self.d_hat = np.zeros(self.q_filter.n_channels)

    def update(self, measurement: ArrayLike, dt: float) -> np.ndarray:
        """Advance the DOB controller one time step."""
        x = np.asarray(measurement, dtype=float)

        if self.mode == "quadrotor":
            return self._update_quadrotor(x, dt)
        else:
            return self._update_linear(x, dt)

    def _update_quadrotor(self, x: np.ndarray, dt: float) -> np.ndarray:
        """DOB update specialized for PlanarQuadrotor with matched/unmatched handling."""
        # State: [x, z, theta, xdot, zdot, thetadot]
        px, pz, th, vx, vz, w = x
        m = getattr(self.plant, "m", 0.028)
        Iyy = getattr(self.plant, "Iyy", 1.4e-5)
        arm = getattr(self.plant, "l", 0.046)
        g = getattr(self.plant, "g", 9.81)
        cd = getattr(self.plant, "cd", 1e-4)

        # 1. Reconstruct disturbances from acceleration mismatch
        if self.x_prev is not None and self.u_prev is not None and dt > 1e-9:
            # Finite-difference measured accelerations
            vx_prev, vz_prev, w_prev = self.x_prev[3], self.x_prev[4], self.x_prev[5]
            ax_meas = (vx - vx_prev) / dt
            az_meas = (vz - vz_prev) / dt
            ath_meas = (w - w_prev) / dt

            # Model expected accelerations with previous input u_prev = [T1, T2]
            T1_prev, T2_prev = self.u_prev
            thrust_prev = T1_prev + T2_prev
            th_prev = self.x_prev[2]

            ax_model = -thrust_prev * np.sin(th_prev) / m - cd * vx_prev / m
            az_model = thrust_prev * np.cos(th_prev) / m - g - cd * vz_prev / m
            ath_model = (T1_prev - T2_prev) * arm / Iyy

            # Discrepancy (apparent disturbance acceleration)
            disc_x = ax_meas - ax_model
            disc_z = az_meas - az_model
            disc_th = ath_meas - ath_model

            raw_d = np.array([disc_x, disc_z, disc_th])
            self.d_hat = self.q_filter.filter(raw_d, dt=dt)
        else:
            self.d_hat = np.zeros(3)

        d_x_hat, d_z_hat, d_th_hat = self.d_hat

        # 2. Unmatched Disturbance Reallocation:
        # Horizontal force disturbance d_x_hat (m/s^2) must be countered by tilting thrust:
        # F_thrust_x = -T * sin(theta_trim) = -m * d_x_hat  => theta_trim ~= d_x_hat / g
        theta_virtual_bias = d_x_hat / g

        # Adjust the measurement / reference seen by the base controller
        # When base controller is state feedback / LQR, biasing x[2] (theta) by -theta_virtual_bias
        # causes it to command the quadrotor to hold theta = +theta_virtual_bias.
        x_base = x.copy()
        x_base[2] -= theta_virtual_bias

        # 3. Base Controller Computation
        u_base = np.asarray(self.base_controller.update(x_base, dt), dtype=float)
        T1_base, T2_base = u_base

        # 4. Matched Disturbance Cancellation:
        # Vertical force cancellation: Delta_T = -m * d_z_hat / cos(theta)
        cos_th = max(np.cos(th), 0.2)
        delta_T = -m * d_z_hat / cos_th

        # Pitch torque cancellation: Delta_tau = -Iyy * d_th_hat
        # Differential thrust: Delta_T1 - Delta_T2 = Delta_tau / arm
        delta_tau = -Iyy * d_th_hat
        delta_T_diff = delta_tau / (2.0 * arm)

        # Combine matched compensations onto rotor thrusts
        T1_cmd = T1_base + 0.5 * delta_T + delta_T_diff
        T2_cmd = T2_base + 0.5 * delta_T - delta_T_diff

        # Enforce actuator limits if available
        thrust_max = getattr(self.plant, "thrust_max", 0.30)
        u_cmd = np.array([
            _clip(T1_cmd, 0.0, thrust_max),
            _clip(T2_cmd, 0.0, thrust_max),
        ])

        self.u_prev = u_cmd.copy()
        self.x_prev = x.copy()

        return u_cmd

    def _update_linear(self, x: np.ndarray, dt: float) -> np.ndarray:
        """DOB update for linear state-space plants."""
        if self.x_prev is not None and self.u_prev is not None and dt > 1e-9:
            # xdot_meas ~= (x - x_prev) / dt
            xdot_meas = (x - self.x_prev) / dt
            xdot_model = self.A @ self.x_prev + (self.B @ self.u_prev if self.B.ndim > 1 else self.B * self.u_prev)

            # Residual in state derivative: res = Bd * d
            res = xdot_meas - xdot_model
            # Least-squares pseudo-inverse to isolate d
            Bd_pinv = np.linalg.pinv(self.Bd)
            raw_d = Bd_pinv @ res
            self.d_hat = self.q_filter.filter(raw_d, dt=dt)
        else:
            self.d_hat = np.zeros(self.n_d)

        # Base controller
        u_base = np.asarray(self.base_controller.update(x, dt), dtype=float)

        # Matched cancellation: u = u_base - B_pinv * Bd * d_hat
        B_pinv = np.linalg.pinv(self.B)
        u_comp = B_pinv @ (self.Bd @ self.d_hat)
        u_cmd = u_base - u_comp

        self.u_prev = np.atleast_1d(u_cmd).copy()
        self.x_prev = x.copy()

        return u_cmd
