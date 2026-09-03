r"""Deterministic Luenberger observer.

.. math::

    \dot{\hat x} = A\hat x + Bu + L\,(y - C\hat x),
    \qquad \dot e = (A - LC)\,e .

The error decays iff :math:`\lambda(A - LC)` are all in the open left-half plane.
By transpose duality the gain solves a pole-placement problem on
:math:`(A^\top, C^\top)`:

.. math::

    L = \operatorname{place}(A^\top,\ C^\top,\ \text{poles})^\top .

Single-output: from-scratch Ackermann (``controllers.place_poles``).
Multi-output: ``scipy.signal.place_poles`` (cross-check dependency).
"""

from __future__ import annotations

import numpy as np

from .observability import is_observable

__all__ = ["place_observer", "LuenbergerObserver"]


def place_observer(A, C, poles) -> np.ndarray:
    """Observer gain ``L`` (shape ``(n, p)``) placing ``eig(A - LC)`` at ``poles``."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n, p = A.shape[0], C.shape[0]
    if not is_observable(A, C):
        raise ValueError("(A, C) is not observable; observer poles cannot be placed")

    if p == 1:
        from ..controllers.state_feedback import place_poles

        K = place_poles(A.T, C.T, poles)  # (1, n)
        return K.T  # (n, 1)

    from scipy.signal import place_poles as _sp_place

    res = _sp_place(A.T, C.T, np.asarray(poles))
    return res.gain_matrix.T  # (n, p)


class LuenbergerObserver:
    r"""Continuous observer integrated with the simulator's fixed step.

    Construct with an explicit gain ``L`` or with ``poles`` to place
    ``eig(A - LC)``. Call :meth:`update(y, u, dt)` once per control step; it
    advances :math:`\hat x` by one RK4 step of :math:`(A-LC)\hat x + Bu + Ly`
    and returns the new estimate.
    """

    def __init__(self, A, B, C, *, L=None, poles=None, x_hat0=None) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.C = np.atleast_2d(np.asarray(C, dtype=float))
        self.n = self.A.shape[0]
        if (L is None) == (poles is None):
            raise ValueError("pass exactly one of L or poles")
        self.L = place_observer(self.A, self.C, poles) if L is None else \
            np.atleast_2d(np.asarray(L, dtype=float))
        if self.L.shape != (self.n, self.C.shape[0]):
            raise ValueError(f"L must be ({self.n}, {self.C.shape[0]}), got {self.L.shape}")
        self._x_hat0 = np.zeros(self.n) if x_hat0 is None else \
            np.asarray(x_hat0, dtype=float).reshape(self.n)
        self.reset()

    def reset(self, x_hat0=None) -> None:
        self.x_hat = self._x_hat0.copy() if x_hat0 is None else \
            np.asarray(x_hat0, dtype=float).reshape(self.n).copy()

    @property
    def error_poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.A - self.L @ self.C)

    def _xdot(self, x_hat, u, y):
        return (self.A @ x_hat + self.B @ np.atleast_1d(u)
                + self.L @ (np.atleast_1d(y) - self.C @ x_hat))

    def update(self, y, u, dt: float) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        u = np.atleast_1d(np.asarray(u, dtype=float))
        x = self.x_hat
        k1 = self._xdot(x, u, y)
        k2 = self._xdot(x + 0.5 * dt * k1, u, y)
        k3 = self._xdot(x + 0.5 * dt * k2, u, y)
        k4 = self._xdot(x + dt * k3, u, y)
        self.x_hat = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.x_hat
