r"""Least-squares / DMDc identification of a linear state-space model.

Given a rollout of states :math:`x_0,\dots,x_T` and inputs :math:`u_0,\dots,u_{T-1}`
sampled at a fixed step, stack

.. math::

    X_{+} = \begin{bmatrix} x_1 & \cdots & x_T \end{bmatrix},\quad
    \Omega = \begin{bmatrix} x_0 & \cdots & x_{T-1} \\
                             u_0 & \cdots & u_{T-1} \end{bmatrix},

and solve :math:`\min_{A_d,B_d}\; \lVert X_{+} - [A_d\ B_d]\,\Omega \rVert_F`
in closed form, :math:`[A_d\ B_d] = X_{+}\,\Omega^{+}` (Moore-Penrose).

DMDc adds a truncated SVD of :math:`\Omega` (rank ``r``) for noise rejection /
reduced order -- Proctor, Brunton & Kutz (2016).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "least_squares_id",
    "dmdc",
    "to_continuous",
    "prediction_error",
    "model_mismatch",
]


def _stack(X: np.ndarray, U: np.ndarray):
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be (T+1, n): one row per sample")
    if U.ndim == 1:
        U = U[:, None]
    T = X.shape[0] - 1
    if T < 1:
        raise ValueError("need at least two state samples")
    if U.shape[0] not in (T, T + 1):
        raise ValueError(f"U must have {T} or {T + 1} rows, got {U.shape[0]}")
    Xm = X[:-1].T          # (n, T)
    Xp = X[1:].T           # (n, T)
    Um = U[:T].T           # (m, T)
    return Xm, Xp, Um


def least_squares_id(X: np.ndarray, U: np.ndarray, *, rcond: float | None = None):
    r"""Ordinary least-squares fit of ``x_{k+1} = A_d x_k + B_d u_k``.

    Parameters
    ----------
    X : ``(T+1, n)`` array of state samples.
    U : ``(T, m)`` (or ``(T+1, m)``, last row ignored) array of inputs; a 1-D
        array is treated as a single input channel.

    Returns
    -------
    (A_d, B_d) : ``(n, n)`` and ``(n, m)`` arrays.
    """
    Xm, Xp, Um = _stack(X, U)
    Omega = np.vstack([Xm, Um])                 # (n+m, T)
    G = Xp @ np.linalg.pinv(Omega, rcond=rcond) if rcond else Xp @ np.linalg.pinv(Omega)
    n = Xm.shape[0]
    return G[:, :n], G[:, n:]


def dmdc(X: np.ndarray, U: np.ndarray, *, rank: int | None = None):
    r"""DMD with control: least squares through a rank-``r`` truncated SVD of
    the stacked ``[X; U]`` snapshot matrix. ``rank=None`` -> full rank (identical
    to :func:`least_squares_id`)."""
    Xm, Xp, Um = _stack(X, U)
    Omega = np.vstack([Xm, Um])
    Uu, s, Vt = np.linalg.svd(Omega, full_matrices=False)
    r = len(s) if rank is None else int(rank)
    r = min(r, len(s))
    Uu, s, Vt = Uu[:, :r], s[:r], Vt[:r]
    G = Xp @ Vt.T @ np.diag(1.0 / s) @ Uu.T
    n = Xm.shape[0]
    return G[:, :n], G[:, n:]


def to_continuous(A_d: np.ndarray, B_d: np.ndarray, dt: float):
    r"""Invert the zero-order-hold map to recover ``(A_c, B_c)``.

    Uses the block-matrix logarithm
    :math:`\log\!\begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix} / \Delta t
    = \begin{bmatrix} A_c & B_c \\ 0 & 0 \end{bmatrix}`,
    which stays well defined when ``A_c`` is singular (e.g. an integrator).
    """
    from scipy.linalg import logm

    A_d = np.atleast_2d(np.asarray(A_d, dtype=float))
    B_d = np.atleast_2d(np.asarray(B_d, dtype=float))
    n, m = A_d.shape[0], B_d.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_d
    M[:n, n:] = B_d
    M[n:, n:] = np.eye(m)
    L = np.real(logm(M)) / dt
    return L[:n, :n], L[:n, n:]


def prediction_error(
    A_d: np.ndarray,
    B_d: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    *,
    horizon: int | None = None,
) -> float:
    r"""RMS state-prediction error of ``(A_d, B_d)`` on a rollout.

    ``horizon=1`` (default) scores one-step-ahead prediction from each true
    state. An integer ``horizon=h`` rolls the model open-loop for ``h`` steps
    from each start; ``horizon=None`` -> 1.
    """
    A_d = np.atleast_2d(np.asarray(A_d, dtype=float))
    B_d = np.atleast_2d(np.asarray(B_d, dtype=float))
    Xm, Xp, Um = _stack(X, U)
    h = 1 if horizon is None else int(horizon)
    T = Xm.shape[1]
    errs = []
    for k in range(T - h + 1):
        x = Xm[:, k].copy()
        for j in range(h):
            x = A_d @ x + B_d @ Um[:, k + j]
        errs.append(x - Xp[:, k + h - 1])
    E = np.array(errs)
    return float(np.sqrt(np.mean(E**2)))


def model_mismatch(A1, B1, A2, B2) -> dict[str, float]:
    """Relative Frobenius distance between two ``(A, B)`` pairs and their
    spectra -- a quick 'how close is the identified model' summary."""
    A1, B1, A2, B2 = (np.atleast_2d(np.asarray(m, float)) for m in (A1, B1, A2, B2))
    eig1 = np.sort_complex(np.linalg.eigvals(A1))
    eig2 = np.sort_complex(np.linalg.eigvals(A2))
    return {
        "A_rel_fro": float(np.linalg.norm(A1 - A2) / max(np.linalg.norm(A1), 1e-12)),
        "B_rel_fro": float(np.linalg.norm(B1 - B2) / max(np.linalg.norm(B1), 1e-12)),
        "eig_max_abs_diff": float(np.max(np.abs(eig1 - eig2))),
    }
