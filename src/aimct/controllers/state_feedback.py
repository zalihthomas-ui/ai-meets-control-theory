r"""Full-state feedback and single-input pole placement — from scratch.

Control law
-----------

.. math::

    u = u_{\text{ref}} - K\,(x - x_{\text{ref}})

With ``x_ref = 0`` and ``u_ref = 0`` this is the regulator :math:`u = -Kx`; a
non-zero reference turns it into a set-point tracker (no integral action, so a
model mismatch still leaves a static error — that is what ``LQR`` + integral
augmentation or a feed-forward ``u_ref`` are for).

Pole placement (Ackermann's formula)
------------------------------------

For a **single-input** controllable pair :math:`(A, b)` and desired closed-loop
poles :math:`\{p_i\}_{i=1}^n`, the unique gain that makes
:math:`\operatorname{eig}(A - bK) = \{p_i\}` is

.. math::

    K = \begin{bmatrix} 0 & \cdots & 0 & 1 \end{bmatrix}\,
        \mathcal{C}^{-1}\,\phi(A),
    \qquad
    \mathcal{C} = \begin{bmatrix} b & Ab & \cdots & A^{n-1}b \end{bmatrix},

where :math:`\phi(s) = \prod_i (s - p_i) = s^n + a_1 s^{n-1} + \dots + a_n` is the
desired characteristic polynomial and :math:`\phi(A)` is that polynomial
evaluated on the matrix :math:`A` (Cayley–Hamilton).

Multi-input placement is deliberately not implemented here (it is
non-unique and needs e.g. the Kautsky–Nichols algorithm); every Phase-0
benchmark system is single-input.  For the multi-input case use :class:`LQR`.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, Controller

__all__ = [
    "StateFeedback",
    "controllability_matrix",
    "is_controllable",
    "place_poles",
]


def controllability_matrix(A: ArrayLike, B: ArrayLike) -> np.ndarray:
    r"""Return :math:`\mathcal{C} = [\,B\ \ AB\ \ \cdots\ \ A^{n-1}B\,]`."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    n = A.shape[0]
    blocks = [B]
    for _ in range(1, n):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def is_controllable(A: ArrayLike, B: ArrayLike, tol: float = 1e-9) -> bool:
    """True iff the controllability matrix has full row rank ``n``."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ctrb = controllability_matrix(A, B)
    return int(np.linalg.matrix_rank(ctrb, tol=tol)) == A.shape[0]


def _assert_conjugate_symmetric(poles: np.ndarray) -> None:
    """Complex poles must occur in conjugate pairs for a real gain."""
    a = np.sort_complex(np.asarray(poles, dtype=complex))
    b = np.sort_complex(np.conj(a))
    if not np.allclose(a, b, atol=1e-12):
        raise ValueError("complex poles must come in conjugate pairs")


def place_poles(A: ArrayLike, B: ArrayLike, poles: ArrayLike) -> np.ndarray:
    """Ackermann single-input pole placement.  Returns ``K`` of shape ``(1, n)``.

    Raises ``ValueError`` if ``(A, B)`` is uncontrollable or the pole list has
    the wrong length / an unpaired complex value, and ``NotImplementedError``
    for a multi-input ``B``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if B.shape != (n, 1):
        raise NotImplementedError(
            f"Ackermann placement is single-input only; B has shape {B.shape}, "
            "expected (n, 1). Use LQR for multi-input systems."
        )
    poles = np.asarray(poles)
    if poles.shape != (n,):
        raise ValueError(f"expected exactly n={n} poles, got shape {poles.shape}")
    _assert_conjugate_symmetric(poles)

    ctrb = controllability_matrix(A, B)
    if abs(np.linalg.det(ctrb)) < 1e-12:
        raise ValueError("(A, B) is not controllable; poles cannot be placed")

    # phi(A) = A^n + a1 A^(n-1) + ... + an I  via Horner on matrices.
    coeffs = np.real(np.poly(poles))  # [1, a1, ..., an]
    phi_A = np.eye(n)
    for a_i in coeffs[1:]:
        phi_A = phi_A @ A + a_i * np.eye(n)

    e_n = np.zeros(n)
    e_n[-1] = 1.0
    K = e_n @ np.linalg.solve(ctrb, phi_A)
    return K.reshape(1, n)


class StateFeedback(Controller):
    """Static full-state feedback ``u = u_ref - K (x - x_ref)``.

    Parameters
    ----------
    K:
        Gain matrix, shape ``(n_u, n_x)`` (a length-``n_x`` vector is treated as
        one input row).
    x_ref, u_ref:
        Reference state and feed-forward input; default to zeros.  Both may be
        reassigned on the instance between steps.
    """

    def __init__(
        self,
        K: ArrayLike,
        *,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> None:
        self.K = np.atleast_2d(np.asarray(K, dtype=float))
        self.n_u, self.n_x = self.K.shape
        self.x_ref = (
            np.zeros(self.n_x) if x_ref is None
            else np.asarray(x_ref, dtype=float).reshape(self.n_x)
        )
        self.u_ref = (
            np.zeros(self.n_u) if u_ref is None
            else np.asarray(u_ref, dtype=float).reshape(self.n_u)
        )
        self.reset()

    # ------------------------------------------------------------- constructors

    @classmethod
    def from_poles(
        cls,
        A: ArrayLike,
        B: ArrayLike,
        poles: ArrayLike,
        *,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> "StateFeedback":
        """Build the controller by placing ``poles`` on ``(A, B)`` (Ackermann)."""
        sf = cls(place_poles(A, B, poles), x_ref=x_ref, u_ref=u_ref)
        sf.A = np.atleast_2d(np.asarray(A, dtype=float))
        sf.B = np.atleast_2d(np.asarray(B, dtype=float))
        return sf

    # -------------------------------------------------------------------- step

    def update(self, measurement: ArrayLike, dt: float | None = None) -> ArrayLike:
        """Return ``u`` for the current measured **state**.  ``dt`` is ignored
        (the law is static)."""
        x = np.atleast_1d(np.asarray(measurement, dtype=float))
        if x.shape != (self.n_x,):
            raise ValueError(
                f"state feedback expects a length-{self.n_x} state, got {x.shape}"
            )
        u = self.u_ref - self.K @ (x - self.x_ref)
        self.output = float(u[0]) if self.n_u == 1 else u
        return self.output

    def reset(self) -> None:
        self.output: ArrayLike = 0.0

    # ------------------------------------------------------------------- introspection

    def closed_loop_poles(self, A: ArrayLike | None = None,
                          B: ArrayLike | None = None) -> np.ndarray:
        """Eigenvalues of ``A - B K`` (uses the ``A``/``B`` stored by
        :meth:`from_poles` if not given)."""
        A = getattr(self, "A", None) if A is None else np.atleast_2d(A)
        B = getattr(self, "B", None) if B is None else np.atleast_2d(B)
        if A is None or B is None:
            raise ValueError("A and B are required (not stored on this instance)")
        return np.linalg.eigvals(np.asarray(A, dtype=float) - np.asarray(B, dtype=float) @ self.K)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"StateFeedback(K={self.K.tolist()})"
