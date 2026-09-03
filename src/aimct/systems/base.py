"""Common interface for dynamical-system models.

A system is a first-order ODE  xdot = f(t, x, u)  with an output  y = g(t, x, u).
State and input are 1-D numpy arrays of length ``n_states`` / ``n_inputs``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

ArrayLike = np.ndarray


class DynamicalSystem(ABC):
    """Base class for continuous-time dynamical systems.

    Subclasses set ``n_states`` / ``n_inputs`` (and optionally ``n_outputs``,
    default = ``n_states``) and implement :meth:`dynamics`.
    """

    n_states: int
    n_inputs: int
    n_outputs: int | None = None

    @abstractmethod
    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        """Return ``xdot`` (shape ``(n_states,)``) given time, state, input."""

    def output(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        """Measured output. Defaults to full-state measurement."""
        return np.asarray(x, dtype=float)

    # -- helpers ---------------------------------------------------------------

    def linearize(
        self,
        x_eq: ArrayLike,
        u_eq: ArrayLike,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numerical Jacobians ``(A, B)`` of ``f`` about ``(x_eq, u_eq)``.

        ``A = df/dx`` has shape ``(n_states, n_states)``;
        ``B = df/du`` has shape ``(n_states, n_inputs)``.
        Subclasses with a clean analytic linearization should override this.
        """
        x_eq = np.asarray(x_eq, dtype=float)
        u_eq = np.asarray(u_eq, dtype=float)
        n, m = self.n_states, self.n_inputs

        A = np.zeros((n, n))
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = eps
            A[:, j] = (
                self.dynamics(0.0, x_eq + dx, u_eq)
                - self.dynamics(0.0, x_eq - dx, u_eq)
            ) / (2 * eps)

        B = np.zeros((n, m))
        for j in range(m):
            du = np.zeros(m)
            du[j] = eps
            B[:, j] = (
                self.dynamics(0.0, x_eq, u_eq + du)
                - self.dynamics(0.0, x_eq, u_eq - du)
            ) / (2 * eps)

        return A, B

    def _prep(self, x: ArrayLike, u: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Coerce ``x``/``u`` to float arrays of the expected shape."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if x.shape != (self.n_states,):
            raise ValueError(f"state must have shape ({self.n_states},), got {x.shape}")
        if u.shape != (self.n_inputs,):
            raise ValueError(f"input must have shape ({self.n_inputs},), got {u.shape}")
        return x, u
