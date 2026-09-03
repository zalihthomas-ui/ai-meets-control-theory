r"""Extended Kalman filter - from scratch.

For a nonlinear model

.. math::

    x_{k+1} = f(x_k, u_k) + w_k, \quad w_k \sim \mathcal N(0, Q), \qquad
    y_k     = h(x_k) + v_k,       \quad v_k \sim \mathcal N(0, R),

the EKF runs the ordinary Kalman recursion on the first-order Taylor expansion
about the current estimate:

.. math::

    F_k = \left.\frac{\partial f}{\partial x}\right|_{\hat x_k, u_k}, \qquad
    H_k = \left.\frac{\partial h}{\partial x}\right|_{\hat x_k}.

``f`` may be given as continuous dynamics ``f(x, u) -> xdot`` (integrated with
one RK4 step of ``dt`` - matching :func:`aimct.simulate.simulate`) or as a
ready discrete map ``f(x, u) -> x_next`` (``discrete=True``).  Jacobians are
taken analytically when ``F_jac`` / ``H_jac`` are supplied, otherwise by central
finite differences.  The covariance update uses the Joseph form for symmetry,
and a ``residual`` hook lets angle-valued measurements be wrapped.

The predict / update / step / reset surface matches
:class:`~aimct.estimation.DiscreteKalmanFilter`, so an EKF drops straight into
:class:`~aimct.controllers.ObserverFeedback`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..simulate import rk4_step

__all__ = ["ExtendedKalmanFilter", "finite_diff_jacobian"]


def finite_diff_jacobian(fn: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian ``d fn / d x`` (shape ``(len(fn(x)), len(x))``)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    base = np.atleast_1d(np.asarray(fn(x), dtype=float))
    J = np.zeros((base.size, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (np.atleast_1d(np.asarray(fn(x + dx), dtype=float))
                   - np.atleast_1d(np.asarray(fn(x - dx), dtype=float))) / (2.0 * eps)
    return J


class ExtendedKalmanFilter:
    """Discrete-time extended Kalman filter.

    Parameters
    ----------
    f:
        ``f(x, u) -> xdot`` (continuous, default) or ``f(x, u) -> x_next``
        (``discrete=True``).
    h:
        Measurement function ``h(x) -> y``.
    Q, R:
        Process / measurement noise covariances (discrete-time).
    dt:
        Control step; used to integrate a continuous ``f`` and to size things.
    n, p:
        State / measurement dimensions.  ``p`` is inferred from ``R`` if omitted.
    F_jac, H_jac:
        Optional analytic Jacobians: ``F_jac(x, u) -> (n, n)`` of the *discrete*
        transition, ``H_jac(x) -> (p, n)``.  Finite-differenced when omitted.
    residual:
        ``residual(y, y_pred) -> innovation`` (default ``y - y_pred``); use it to
        wrap angle components.
    x0, P0:
        Initial estimate and covariance (zeros / identity by default).
    fd_eps:
        Step for finite-difference Jacobians.
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        *,
        dt: float,
        n: int | None = None,
        discrete: bool = False,
        F_jac: Callable | None = None,
        H_jac: Callable | None = None,
        residual: Callable | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        fd_eps: float = 1e-6,
    ) -> None:
        self._f = f
        self._h = h
        self.dt = float(dt)
        self.discrete = bool(discrete)
        self._F_jac = F_jac
        self._H_jac = H_jac
        self._residual = residual or (lambda y, yp: np.asarray(y, float) - np.asarray(yp, float))
        self._eps = float(fd_eps)

        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.n = int(n) if n is not None else self.Q.shape[0]
        self.p = self.R.shape[0]

        self._x0 = np.zeros(self.n) if x0 is None else np.asarray(x0, float).reshape(self.n)
        self._P0 = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, float))
        self.reset()

    # ------------------------------------------------------------------ model

    def _transition(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.discrete:
            return np.atleast_1d(np.asarray(self._f(x, u), dtype=float))
        return rk4_step(lambda t, xx, uu: np.asarray(self._f(xx, uu), dtype=float),
                        0.0, x, u, self.dt)

    # ------------------------------------------------------------------ state

    def reset(self, x0: np.ndarray | None = None, P0: np.ndarray | None = None) -> None:
        self.x_hat = (self._x0 if x0 is None
                      else np.asarray(x0, float).reshape(self.n)).copy()
        self.P = (self._P0 if P0 is None
                  else np.atleast_2d(np.asarray(P0, float))).copy()

    # ------------------------------------------------------------------ steps

    def predict(self, u=None) -> np.ndarray:
        u = np.zeros(1) if u is None else np.atleast_1d(np.asarray(u, dtype=float))
        x = self.x_hat
        if self._F_jac is not None:
            F = np.atleast_2d(np.asarray(self._F_jac(x, u), dtype=float))
        else:
            F = finite_diff_jacobian(lambda xx: self._transition(xx, u), x, self._eps)
        self.x_hat = self._transition(x, u)
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x_hat

    def update(self, y) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        x = self.x_hat
        if self._H_jac is not None:
            H = np.atleast_2d(np.asarray(self._H_jac(x), dtype=float))
        else:
            H = finite_diff_jacobian(
                lambda xx: np.atleast_1d(np.asarray(self._h(xx), dtype=float)), x, self._eps)

        innov = np.asarray(self._residual(y, np.atleast_1d(np.asarray(self._h(x), float))),
                           dtype=float)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x_hat = x + K @ innov
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return self.x_hat

    def step(self, y, u=None) -> np.ndarray:
        """``predict(u)`` then ``update(y)``."""
        self.predict(u)
        return self.update(y)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ExtendedKalmanFilter(n={self.n}, p={self.p}, dt={self.dt}, "
                f"analytic_F={self._F_jac is not None}, analytic_H={self._H_jac is not None})")
