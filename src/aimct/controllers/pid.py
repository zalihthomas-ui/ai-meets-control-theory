r"""PID controller — implemented from scratch.

Continuous-time control law
---------------------------

.. math::

    u(t) = K_p\,e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d\,\frac{d}{dt}d(t)

with tracking error :math:`e(t) = r(t) - y(t)`.  The derivative acts on the
signal :math:`d(t)`, which is either the error (``derivative_on='error'``) or the
*measurement* (``derivative_on='measurement'``, the default).  Differentiating the
measurement instead of the error removes the "derivative kick" — the impulsive
spike in ``u`` that a step change in the set-point would otherwise cause — at no
cost when the set-point is constant.

Discretisation
--------------

A fixed step :math:`\Delta t` with a backward-difference derivative and
rectangular integration:

.. math::

    I_k         &= I_{k-1} + e_k\,\Delta t \\
    D^{\text{raw}}_k &= \begin{cases}
        (e_k - e_{k-1}) / \Delta t & \text{derivative on error} \\
        -(y_k - y_{k-1}) / \Delta t & \text{derivative on measurement}
    \end{cases} \\
    D_k         &= D_{k-1} + \frac{\Delta t}{\tau_d + \Delta t}\,
                   \bigl(D^{\text{raw}}_k - D_{k-1}\bigr) \\
    u_k         &= K_p e_k + K_i I_k + K_d D_k

``tau_d`` is the time constant of a first-order low-pass filter on the derivative
term (``0`` disables it).  Real derivative action is almost never used unfiltered
because it amplifies measurement noise.

Anti-windup
-----------

When ``output_limits`` are set, the integrator uses **conditional integration**
(a.k.a. integrator clamping): the running integral is updated only when the
unsaturated command is within limits, or when the new error would drive the
command *back* toward the linear region.  This prevents the integral term from
growing without bound while the actuator is saturated, which would otherwise
cause large overshoot on recovery.  ``integral_limits`` additionally hard-clamps
the integral term.

Scalars or vectors
------------------

Gains, set-point and measurement may be scalars or NumPy arrays of a common
shape; the controller then produces a matching vector command (independent
per-channel PID loops).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ArrayLike, Controller, _clip

__all__ = ["PID"]

_Bounds = tuple[Optional[float], Optional[float]]


class PID(Controller):
    """A single (optionally vector-valued) PID feedback loop.

    Parameters
    ----------
    kp, ki, kd:
        Proportional, integral and derivative gains.  Scalars or arrays.
    setpoint:
        Desired value of the measured signal.  Can also be changed later via the
        :attr:`setpoint` attribute or the ``setpoint=`` argument of
        :meth:`update`.
    dt:
        Default control step in seconds.  If given, :meth:`update` may be called
        as ``update(y)``; otherwise ``dt`` must be passed to every call.
    output_limits:
        ``(low, high)`` saturation applied to the command.  ``None`` on a side
        disables that bound.  Enabling limits also enables anti-windup.
    integral_limits:
        ``(low, high)`` hard clamp on the integral term ``ki * integral``.
    derivative_on:
        ``'measurement'`` (default) or ``'error'`` — see module docstring.
    tau_d:
        Derivative low-pass filter time constant in seconds (``0`` = off).
    """

    def __init__(
        self,
        kp: ArrayLike = 0.0,
        ki: ArrayLike = 0.0,
        kd: ArrayLike = 0.0,
        *,
        setpoint: ArrayLike = 0.0,
        dt: float | None = None,
        output_limits: _Bounds = (None, None),
        integral_limits: _Bounds = (None, None),
        derivative_on: str = "measurement",
        tau_d: float = 0.0,
    ) -> None:
        if derivative_on not in ("measurement", "error"):
            raise ValueError(
                f"derivative_on must be 'measurement' or 'error', got {derivative_on!r}"
            )
        if dt is not None and dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if tau_d < 0:
            raise ValueError(f"tau_d must be non-negative, got {tau_d}")

        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.setpoint = np.asarray(setpoint, dtype=float)
        self.dt = dt
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_on = derivative_on
        self.tau_d = float(tau_d)

        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Zero the integrator, stored past samples and cached term values."""
        self._integral = np.zeros(())
        self._prev_error: np.ndarray | None = None
        self._prev_measurement: np.ndarray | None = None
        self._derivative = np.zeros(())
        #: Last ``(P, I, D)`` contributions, for logging / plots.
        self.terms: tuple[ArrayLike, ArrayLike, ArrayLike] = (0.0, 0.0, 0.0)
        #: Last command actually returned (post-saturation).
        self.output: ArrayLike = 0.0

    # ------------------------------------------------------------------ step

    def update(
        self,
        measurement: ArrayLike,
        dt: float | None = None,
        *,
        setpoint: ArrayLike | None = None,
    ) -> ArrayLike:
        """Advance the loop one step and return the (saturated) command.

        Parameters
        ----------
        measurement:
            Current value of the controlled signal, :math:`y_k`.
        dt:
            Step length in seconds.  Falls back to the ``dt`` given at
            construction; a positive value is required one way or the other.
        setpoint:
            Optional per-call set-point override; also stored on the instance.
        """
        step = self.dt if dt is None else dt
        if step is None:
            raise ValueError("dt was not given at construction; pass it to update()")
        if step <= 0:
            raise ValueError(f"dt must be positive, got {step}")

        if setpoint is not None:
            self.setpoint = np.asarray(setpoint, dtype=float)

        y = np.asarray(measurement, dtype=float)
        error = self.setpoint - y

        # --- proportional -------------------------------------------------
        p_term = self.kp * error

        # --- derivative (on measurement or error), then low-pass ---------
        if self.derivative_on == "measurement":
            if self._prev_measurement is None:
                raw_d = np.zeros_like(y)
            else:
                raw_d = -(y - self._prev_measurement) / step
        else:  # 'error'
            if self._prev_error is None:
                raw_d = np.zeros_like(error)
            else:
                raw_d = (error - self._prev_error) / step

        if self.tau_d > 0.0:
            alpha = step / (self.tau_d + step)
            self._derivative = self._derivative + alpha * (raw_d - self._derivative)
        else:
            self._derivative = raw_d
        d_term = self.kd * self._derivative

        # --- integral with conditional-integration anti-windup ----------
        candidate = self._integral + error * step
        i_term_candidate = _clip(self.ki * candidate, *self.integral_limits)

        unsaturated = p_term + i_term_candidate + d_term
        saturated = _clip(unsaturated, *self.output_limits)

        # Accept the integration step where the command is unsaturated, or
        # where the incoming error points back into the linear region.
        overshoot = unsaturated - saturated  # >0: clamped high, <0: clamped low
        winding_up = overshoot * error > 0
        accept = np.logical_not(winding_up)
        self._integral = np.where(accept, candidate, self._integral)

        i_term = _clip(self.ki * self._integral, *self.integral_limits)
        output = _clip(p_term + i_term + d_term, *self.output_limits)

        # Scalar in -> scalar out, for ergonomics; vectors pass through as arrays.
        if y.ndim == 0:
            p_term, i_term, d_term = float(p_term), float(i_term), float(d_term)
            output = float(output)

        # --- bookkeeping ----------------------------------------------------
        self._prev_error = error
        self._prev_measurement = y
        self.terms = (p_term, i_term, d_term)
        self.output = output
        return output

    # ------------------------------------------------------------------ misc

    @property
    def integral(self) -> ArrayLike:
        """Current value of the raw error integral :math:`\\int e\\,dt`."""
        return self._integral

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PID(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"setpoint={self.setpoint}, derivative_on={self.derivative_on!r}, "
            f"tau_d={self.tau_d}, output_limits={self.output_limits})"
        )
