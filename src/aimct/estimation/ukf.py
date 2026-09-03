r"""Unscented Kalman filter - from scratch.

Instead of linearising :math:`f` and :math:`h` (the EKF), the UKF propagates a
deterministic set of :math:`2n+1` **sigma points** through the true nonlinear
maps and rebuilds the mean and covariance from the transformed points.  It is
exact for affine maps and captures curvature to (at least) second order, so it
tracks strongly nonlinear models where the EKF's single-point linearisation
drifts.

Scaled unscented transform (van der Merwe): with
:math:`\lambda = \alpha^2 (n + \kappa) - n`, :math:`\gamma = \sqrt{n + \lambda}`,
the points are :math:`\mathcal X_0 = \hat x`,
:math:`\mathcal X_i = \hat x \pm \gamma\, [\sqrt P]_i`, weighted by

.. math::

    W^m_0 = \tfrac{\lambda}{n+\lambda},\quad
    W^c_0 = W^m_0 + (1 - \alpha^2 + \beta),\quad
    W^m_i = W^c_i = \tfrac{1}{2(n+\lambda)}.

Defaults :math:`\alpha = 10^{-3}`, :math:`\beta = 2` (Gaussian), :math:`\kappa = 0`.

The ``predict / update / step / reset`` surface matches
:class:`~aimct.estimation.DiscreteKalmanFilter` and
:class:`~aimct.estimation.ExtendedKalmanFilter`, so a UKF is a drop-in for
:class:`~aimct.controllers.ObserverFeedback`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .ekf import step_map

__all__ = ["UnscentedKalmanFilter"]


class UnscentedKalmanFilter:
    """Additive-noise unscented Kalman filter.

    Parameters mirror :class:`~aimct.estimation.ExtendedKalmanFilter` (``f`` is
    continuous ``f(x, u) -> xdot`` unless ``discrete=True``; ``h(x) -> y``;
    ``Q``, ``R`` discrete-time covariances; ``residual(y, y_pred)`` hook for
    angle-valued measurements) plus the sigma-point spread ``alpha``, ``beta``,
    ``kappa``.
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
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        residual: Callable | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> None:
        self._f = f
        self._h = h
        self.dt = float(dt)
        self.discrete = bool(discrete)
        self._residual = residual or (
            lambda y, yp: np.asarray(y, float) - np.asarray(yp, float))

        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.n = int(n) if n is not None else self.Q.shape[0]
        self.p = self.R.shape[0]

        self.alpha, self.beta, self.kappa = float(alpha), float(beta), float(kappa)
        lam = self.alpha**2 * (self.n + self.kappa) - self.n
        self._lambda = lam
        self._gamma = np.sqrt(self.n + lam)
        w = 1.0 / (2.0 * (self.n + lam))
        self.Wm = np.full(2 * self.n + 1, w)
        self.Wc = np.full(2 * self.n + 1, w)
        self.Wm[0] = lam / (self.n + lam)
        self.Wc[0] = self.Wm[0] + (1.0 - self.alpha**2 + self.beta)

        self._x0 = np.zeros(self.n) if x0 is None else np.asarray(x0, float).reshape(self.n)
        self._P0 = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, float))
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self, x0: np.ndarray | None = None, P0: np.ndarray | None = None) -> None:
        self.x_hat = (self._x0 if x0 is None
                      else np.asarray(x0, float).reshape(self.n)).copy()
        self.P = (self._P0 if P0 is None
                  else np.atleast_2d(np.asarray(P0, float))).copy()

    # ------------------------------------------------------------- sigma points

    def _sigma_points(self) -> np.ndarray:
        P = 0.5 * (self.P + self.P.T)
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(P + 1e-9 * np.eye(self.n))
        pts = np.empty((2 * self.n + 1, self.n))
        pts[0] = self.x_hat
        for i in range(self.n):
            pts[1 + i] = self.x_hat + self._gamma * S[:, i]
            pts[1 + self.n + i] = self.x_hat - self._gamma * S[:, i]
        return pts

    # ------------------------------------------------------------------ steps

    def predict(self, u=None) -> np.ndarray:
        u = np.zeros(1) if u is None else np.atleast_1d(np.asarray(u, dtype=float))
        prop = np.array([step_map(self._f, x, u, self.dt, self.discrete)
                         for x in self._sigma_points()])
        x_pred = self.Wm @ prop
        d = prop - x_pred
        P_pred = (d.T * self.Wc) @ d + self.Q
        self.x_hat = x_pred
        self.P = 0.5 * (P_pred + P_pred.T)
        return self.x_hat

    def update(self, y) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        pts = self._sigma_points()
        Z = np.array([np.atleast_1d(np.asarray(self._h(x), dtype=float)) for x in pts])
        z_pred = self.Wm @ Z

        dz = np.array([self._residual(Z[i], z_pred) for i in range(len(Z))])
        dx = pts - self.x_hat
        Pzz = (dz.T * self.Wc) @ dz + self.R
        Pxz = (dx.T * self.Wc) @ dz

        K = Pxz @ np.linalg.inv(Pzz)
        self.x_hat = self.x_hat + K @ np.asarray(self._residual(y, z_pred), dtype=float)
        P = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P + P.T)
        return self.x_hat

    def step(self, y, u=None) -> np.ndarray:
        """``predict(u)`` then ``update(y)``."""
        self.predict(u)
        return self.update(y)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"UnscentedKalmanFilter(n={self.n}, p={self.p}, dt={self.dt}, "
                f"alpha={self.alpha}, beta={self.beta}, kappa={self.kappa})")
