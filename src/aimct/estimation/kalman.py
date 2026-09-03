r"""Kalman filtering — continuous steady-state (LQE) and discrete-time.

Continuous plant with process/measurement noise
:math:`\dot x = Ax + Bu + w`, :math:`y = Cx + v`,
:math:`w \sim \mathcal N(0, W)`, :math:`v \sim \mathcal N(0, V)`.

The steady-state estimator gain minimising
:math:`\Sigma = \lim_{t\to\infty}\mathbb E[e e^\top]` solves the filter ARE

.. math::

    \Sigma A^\top + A\Sigma - \Sigma C^\top V^{-1} C \Sigma + W = 0,
    \qquad L = \Sigma C^\top V^{-1},

which is the LQR CARE under the duality
:math:`A\to A^\top,\ B\to C^\top,\ Q\to W,\ R\to V`.
"""

from __future__ import annotations

import numpy as np

from ..controllers.lqr import solve_care
from .luenberger import LuenbergerObserver

__all__ = ["solve_fare", "KalmanFilter", "DiscreteKalmanFilter"]


def solve_fare(A, C, W, V) -> np.ndarray:
    r"""Stabilising :math:`\Sigma = \Sigma^\top \succeq 0` of the filter ARE."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    W = np.atleast_2d(np.asarray(W, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    return solve_care(A.T, C.T, W, V)


class KalmanFilter(LuenbergerObserver):
    r"""Continuous steady-state Kalman filter (LQE).

    Computes the optimal ``L`` from ``(A, C, W, V)`` at construction, then runs as
    a :class:`LuenbergerObserver` with that gain. ``Sigma`` (the steady-state
    error covariance) is kept for inspection.
    """

    def __init__(self, A, B, C, W, V, *, x_hat0=None) -> None:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        C = np.atleast_2d(np.asarray(C, dtype=float))
        V = np.atleast_2d(np.asarray(V, dtype=float))
        Sigma = solve_fare(A, C, W, V)
        L = Sigma @ C.T @ np.linalg.inv(V)
        self.Sigma, self.W, self.V = Sigma, np.atleast_2d(np.asarray(W, float)), V
        super().__init__(A, B, C, L=L, x_hat0=x_hat0)

    def fare_residual(self) -> np.ndarray:
        A, C, S, W, V = self.A, self.C, self.Sigma, self.W, self.V
        return S @ A.T + A @ S - S @ C.T @ np.linalg.inv(V) @ C @ S + W


class DiscreteKalmanFilter:
    r"""Textbook discrete-time Kalman filter (predict / update).

    Discretises the continuous model as
    :math:`F = I + A\Delta t + \tfrac12 A^2\Delta t^2`, :math:`G_u = B\Delta t`,
    :math:`H = C`, :math:`Q_d = W\Delta t`, :math:`R_d = V/\Delta t`
    (a fixed ``dt`` given at construction), and uses the Joseph-form covariance
    update for numerical symmetry.
    """

    def __init__(self, A, B, C, W, V, dt: float, *, x_hat0=None, P0=None) -> None:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        C = np.atleast_2d(np.asarray(C, dtype=float))
        n = A.shape[0]
        self.n = n
        self.dt = float(dt)
        self.F = np.eye(n) + A * dt + 0.5 * (A @ A) * dt**2
        self.Gu = B * dt
        self.H = C
        self.Qd = np.atleast_2d(np.asarray(W, dtype=float)) * dt
        self.Rd = np.atleast_2d(np.asarray(V, dtype=float)) / dt
        self._x0 = np.zeros(n) if x_hat0 is None else np.asarray(x_hat0, float).reshape(n)
        self._P0 = np.eye(n) if P0 is None else np.atleast_2d(np.asarray(P0, float))
        self.reset()

    def reset(self, x_hat0=None, P0=None) -> None:
        self.x_hat = (self._x0 if x_hat0 is None else
                      np.asarray(x_hat0, float).reshape(self.n)).copy()
        self.P = (self._P0 if P0 is None else np.atleast_2d(np.asarray(P0, float))).copy()

    def predict(self, u=None) -> np.ndarray:
        u = np.zeros(self.Gu.shape[1]) if u is None else np.atleast_1d(np.asarray(u, float))
        self.x_hat = self.F @ self.x_hat + self.Gu @ u
        self.P = self.F @ self.P @ self.F.T + self.Qd
        return self.x_hat

    def update(self, y) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        innov = y - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.Rd
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ innov
        IKH = np.eye(self.n) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.Rd @ K.T  # Joseph form
        return self.x_hat

    def step(self, y, u=None) -> np.ndarray:
        """Convenience: ``predict(u)`` then ``update(y)``."""
        self.predict(u)
        return self.update(y)
