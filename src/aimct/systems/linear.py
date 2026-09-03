"""Linear time-invariant system  xdot = A x + B u ,  y = C x + D u."""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, DynamicalSystem


class LinearSystem(DynamicalSystem):
    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike | None = None,
        D: ArrayLike | None = None,
    ) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if self.B.shape[0] != n:
            raise ValueError("B must have the same number of rows as A")

        self.n_states = n
        self.n_inputs = self.B.shape[1]

        self.C = np.eye(n) if C is None else np.atleast_2d(np.asarray(C, dtype=float))
        self.D = (
            np.zeros((self.C.shape[0], self.n_inputs))
            if D is None
            else np.atleast_2d(np.asarray(D, dtype=float))
        )
        self.n_outputs = self.C.shape[0]

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        return self.A @ x + self.B @ u

    def output(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        x, u = self._prep(x, u)
        return self.C @ x + self.D @ u

    def linearize(self, x_eq=None, u_eq=None, eps: float = 1e-6):
        return self.A.copy(), self.B.copy()
