r"""Infinite-horizon linear-quadratic regulator — from scratch.

Problem
-------

Minimise :math:`J = \int_0^\infty (x^\top Q\,x + u^\top R\,u)\,dt` for
:math:`\dot x = A x + B u`, with :math:`Q = Q^\top \succeq 0`,
:math:`R = R^\top \succ 0`.  The optimal law is static state feedback

.. math::

    u = -K x, \qquad K = R^{-1} B^\top P,

where :math:`P = P^\top \succeq 0` solves the continuous-time algebraic Riccati
equation (CARE)

.. math::

    A^\top P + P A - P B R^{-1} B^\top P + Q = 0 .

Solver — stable eigenspace of the Hamiltonian
---------------------------------------------

Form the :math:`2n \times 2n` Hamiltonian matrix

.. math::

    H = \begin{bmatrix} A & -B R^{-1} B^\top \\ -Q & -A^\top \end{bmatrix}.

Its spectrum is symmetric about the imaginary axis; for a stabilisable /
detectable problem exactly :math:`n` eigenvalues lie in the open left half
plane.  Stack the corresponding eigenvectors as
:math:`\begin{bmatrix} U_1 \\ U_2 \end{bmatrix}` (each :math:`n \times n`); then
:math:`P = U_2 U_1^{-1}` is the stabilising CARE solution and
:math:`\operatorname{eig}(A - BK) = ` the stable eigenvalues of :math:`H`.

This needs only a dense eigensolver (``numpy.linalg.eig``).  The unit tests
cross-check :func:`solve_care` against :func:`scipy.linalg.solve_continuous_are`
and against the CARE residual directly.

Only the continuous-time regulator is implemented here; the discrete-time
version (DARE) is a Phase-2 item.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike
from .state_feedback import StateFeedback, is_controllable

__all__ = ["LQR", "solve_care"]


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def solve_care(
    A: ArrayLike,
    B: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    r"""Solve :math:`A^\top P + P A - P B R^{-1} B^\top P + Q = 0` for the
    stabilising :math:`P = P^\top`.

    Parameters
    ----------
    A, B:
        System matrices, shapes ``(n, n)`` and ``(n, m)``.
    Q, R:
        Weights; ``Q`` is symmetrised, ``R`` must be symmetric positive
        definite.
    tol:
        Guard tolerance for "an eigenvalue sits on the imaginary axis" and for
        the invertibility of ``U_1``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = _sym(np.atleast_2d(np.asarray(Q, dtype=float)))
    R = _sym(np.atleast_2d(np.asarray(R, dtype=float)))
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if B.shape[0] != n:
        raise ValueError("B must have n rows")
    if Q.shape != (n, n):
        raise ValueError("Q must be n x n")
    if R.shape != (B.shape[1], B.shape[1]):
        raise ValueError("R must be m x m")
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError("R must be symmetric positive definite") from exc

    Rinv = np.linalg.inv(R)
    H = np.block([[A, -B @ Rinv @ B.T],
                  [-Q, -A.T]])

    eigvals, eigvecs = np.linalg.eig(H)
    if np.any(np.abs(eigvals.real) < tol):
        raise ValueError(
            "Hamiltonian has an eigenvalue on the imaginary axis; the LQR "
            "problem is not stabilisable/detectable with these weights"
        )

    stable = np.argsort(eigvals.real)[:n]  # n most-negative real parts
    U = eigvecs[:, stable]
    U1, U2 = U[:n, :], U[n:, :]
    if abs(np.linalg.det(U1)) < tol:
        raise ValueError("stable eigenspace is not a valid Riccati graph subspace")

    P = _sym(np.real(U2 @ np.linalg.inv(U1)))
    return P


class LQR(StateFeedback):
    """LQR regulator/tracker.  Computes ``K`` from ``(A, B, Q, R)`` at
    construction, then behaves exactly like :class:`StateFeedback`
    (``u = u_ref - K (x - x_ref)``).

    Attributes ``P``, ``K``, ``A``, ``B``, ``Q``, ``R`` are kept for inspection
    and :meth:`care_residual`.
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
        *,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
        check_controllable: bool = True,
    ) -> None:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        B = np.atleast_2d(np.asarray(B, dtype=float))
        Q = _sym(np.atleast_2d(np.asarray(Q, dtype=float)))
        R = _sym(np.atleast_2d(np.asarray(R, dtype=float)))

        if check_controllable and not is_controllable(A, B):
            raise ValueError(
                "(A, B) is not controllable; LQR has no stabilising solution. "
                "Pass check_controllable=False to attempt it anyway."
            )

        P = solve_care(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)  # R^{-1} B^T P

        self.A, self.B, self.Q, self.R, self.P = A, B, Q, R, P
        super().__init__(K, x_ref=x_ref, u_ref=u_ref)

    def care_residual(self) -> np.ndarray:
        """``A'P + P A - P B R^{-1} B' P + Q`` — should be ~0."""
        A, B, Q, R, P = self.A, self.B, self.Q, self.R, self.P
        return A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q

    def cost_to_go(self, x: ArrayLike) -> float:
        """Optimal cost-to-go ``x' P x`` from state ``x``."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return float(x @ self.P @ x)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LQR(K={self.K.tolist()})"
