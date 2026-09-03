r"""Linear model-predictive control - condensed dense QP, receding horizon.

At each step the continuous ``(A, B)`` is discretised (ZOH) with the control
period ``dt``,

.. math::

    \begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix}
      = \exp\!\Big( \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} dt \Big),

the :math:`N`-step prediction is condensed onto the input sequence
:math:`U = (u_0,\dots,u_{N-1})`,

.. math::

    X = \Phi\,x_0 + \Gamma\,U,\qquad
    \Phi = \begin{bmatrix} A_d \\ A_d^2 \\ \vdots \\ A_d^N \end{bmatrix},\qquad
    \Gamma_{k,j} = A_d^{\,k-j}B_d \ \ (j\le k),

and

.. math::

    J = \sum_{k=1}^{N-1}\|x_k-r\|_Q^2 + \|x_N-r\|_{Q_f}^2
      + \sum_{k=0}^{N-1}\|u_k-u_{\text{ref}}\|_R^2

is minimised over :math:`U` subject to the input box and (softened) state box,
via the from-scratch active-set QP in :mod:`aimct.controllers._qp`.  Only
:math:`u_0` is applied; the plan is warm-started into the next solve.

With no constraints and :math:`Q_f` the discrete-ARE solution, the first move is
exactly the discrete-LQR move :math:`u_0 = -K_d x_0` for any horizon - so
unconstrained ``LinearMPC`` reproduces LQR.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .base import ArrayLike, Controller
from ._qp import solve_qp

__all__ = ["LinearMPC", "dare"]


def dare(Ad, Bd, Q, R, *, tol=1e-12, max_iter=10_000):
    """Discrete algebraic Riccati solution, iterated from scratch.

    ``P = Q + Ad' P Ad - Ad' P Bd (R + Bd' P Bd)^{-1} Bd' P Ad``.
    """
    Ad = np.atleast_2d(np.asarray(Ad, float))
    Bd = np.atleast_2d(np.asarray(Bd, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    R = np.atleast_2d(np.asarray(R, float))
    P = Q.copy()
    for _ in range(max_iter):
        S = R + Bd.T @ P @ Bd
        K = np.linalg.solve(S, Bd.T @ P @ Ad)
        P_next = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
        P_next = 0.5 * (P_next + P_next.T)
        if np.max(np.abs(P_next - P)) < tol:
            return P_next
        P = P_next
    return P


def _discretize(A, B, dt):
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def _prediction_matrices(Ad, Bd, N):
    n, m = Ad.shape[0], Bd.shape[1]
    Phi = np.zeros((N * n, n))
    Gamma = np.zeros((N * n, N * m))
    Apow = np.eye(n)
    for k in range(N):
        Apow = Apow @ Ad                                       # A_d^{k+1}
        Phi[k * n:(k + 1) * n] = Apow
        if k > 0:                                              # shift previous row down
            Gamma[k * n:(k + 1) * n, :] = Ad @ Gamma[(k - 1) * n:k * n, :]
        Gamma[k * n:(k + 1) * n, k * m:(k + 1) * m] = Bd       # new diagonal block
    return Phi, Gamma


class LinearMPC(Controller):
    """Receding-horizon linear MPC.

    Parameters
    ----------
    A, B:
        Continuous-time system matrices.
    Q, R:
        Stage state / input weights (``R`` positive definite).
    N:
        Prediction/control horizon (steps).
    x_ref, u_ref:
        Reference state / feed-forward input (default zeros).
    u_bounds:
        ``(lo, hi)`` scalar or per-channel input box (hard).
    x_bounds:
        ``(lo, hi)`` per-state box; use ``+/-inf`` (or ``None`` entry) for
        unconstrained components. Enforced **softly**: a quadratic penalty
        ``soft_weight`` on the *active* violation, its active set found by a
        short outer loop (warm-started across control steps). Softness keeps the
        QP solvable when the current state already violates or the constraint is
        infeasible over the horizon.
    Qf:
        Terminal weight; defaults to the discrete-ARE solution (=> unconstrained
        MPC reproduces LQR).
    soft_weight:
        Quadratic penalty weight on state-box violation (larger => tighter).
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        *,
        Q: ArrayLike,
        R: ArrayLike,
        N: int,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
        u_bounds: tuple | None = None,
        x_bounds: tuple | None = None,
        Qf: ArrayLike | None = None,
        soft_weight: float = 1e4,
    ) -> None:
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.n, self.m = self.B.shape
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.N = int(N)
        if self.N < 1:
            raise ValueError("N must be >= 1")
        self.x_ref = (np.zeros(self.n) if x_ref is None
                      else np.asarray(x_ref, float).reshape(self.n))
        self.u_ref = (np.zeros(self.m) if u_ref is None
                      else np.asarray(u_ref, float).reshape(self.m))
        self._Qf_user = None if Qf is None else np.atleast_2d(np.asarray(Qf, float))
        self.soft_weight = float(soft_weight)

        self._u_lo, self._u_hi = self._parse_box(u_bounds, self.m)
        self._x_lo, self._x_hi = self._parse_box(x_bounds, self.n)
        self._cache: dict = {}
        self.reset()

    @staticmethod
    def _parse_box(bounds, size):
        if bounds is None:
            return np.full(size, -np.inf), np.full(size, np.inf)
        lo, hi = bounds
        def vec(v, fill):
            if v is None:
                return np.full(size, fill)
            v = np.asarray([fill if x is None else x for x in np.atleast_1d(v)],
                           dtype=float)
            return np.broadcast_to(v, (size,)).copy()
        return vec(lo, -np.inf), vec(hi, np.inf)

    # ------------------------------------------------------------------ build

    def _build(self, dt: float):
        key = round(float(dt), 12)
        if key in self._cache:
            return self._cache[key]

        Ad, Bd = _discretize(self.A, self.B, dt)
        Phi, Gamma = _prediction_matrices(Ad, Bd, self.N)
        n, m, N = self.n, self.m, self.N

        Qf = dare(Ad, Bd, self.Q, self.R) if self._Qf_user is None else self._Qf_user
        Qbar = np.zeros((N * n, N * n))
        for k in range(N - 1):
            Qbar[k * n:(k + 1) * n, k * n:(k + 1) * n] = self.Q
        Qbar[(N - 1) * n:, (N - 1) * n:] = Qf
        Rbar = np.kron(np.eye(N), self.R)

        H = Gamma.T @ Qbar @ Gamma + Rbar
        H = 0.5 * (H + H.T)

        # discrete LQR gain (for the unconstrained equivalence)
        S = self.R + Bd.T @ Qf @ Bd
        Kd = np.linalg.solve(S, Bd.T @ Qf @ Ad)

        # which state-box rows are actually finite, stacked over k = 1..N
        x_lo_h = np.tile(self._x_lo, N)
        x_hi_h = np.tile(self._x_hi, N)
        srows = np.where(np.isfinite(x_lo_h) | np.isfinite(x_hi_h))[0]

        built = dict(Ad=Ad, Bd=Bd, Phi=Phi, Gamma=Gamma, Qbar=Qbar, Rbar=Rbar,
                     H=H, Qf=Qf, Kd=Kd, srows=srows,
                     x_lo_h=x_lo_h, x_hi_h=x_hi_h)
        self._cache[key] = built
        return built

    # ------------------------------------------------------------------- step

    def reset(self) -> None:
        self._warm = None
        self._pen_hi = None
        self._pen_lo = None
        self.horizon_plan: np.ndarray | None = None
        self.predicted_states: np.ndarray | None = None
        self.last_qp = None
        self.output: ArrayLike = 0.0

    def update(self, measurement: ArrayLike, dt: float) -> ArrayLike:
        x0 = np.atleast_1d(np.asarray(measurement, dtype=float)).reshape(self.n)
        b = self._build(dt)
        n, m, N = self.n, self.m, self.N

        Rref = np.tile(self.x_ref, N)
        Uref = np.tile(self.u_ref, N)
        e = b["Phi"] @ x0 - Rref
        g_U = b["Gamma"].T @ b["Qbar"] @ e - b["Rbar"] @ Uref

        u_lo = np.tile(self._u_lo, N)
        u_hi = np.tile(self._u_hi, N)
        z0 = self._warm if (self._warm is not None and self._warm.size == N * m) else None

        srows = b["srows"]
        if not srows.size:
            res = solve_qp(b["H"], g_U, lb=u_lo, ub=u_hi, z0=z0)
            U = res.x.reshape(N, m)
        else:
            # State box, softened: a fixed quadratic penalty on the *active*
            # violation, its active set re-linearised in a short outer loop
            # (warm-started from the previous control step); each inner solve is
            # a fast box-only QP on U.
            Gs = b["Gamma"][srows]
            Phis = b["Phi"][srows]
            bhi = b["x_hi_h"][srows] - Phis @ x0
            blo = b["x_lo_h"][srows] - Phis @ x0
            hi_ok = np.isfinite(b["x_hi_h"][srows])
            lo_ok = np.isfinite(b["x_lo_h"][srows])
            rho = self.soft_weight

            ns = srows.size
            same = lambda a: a is not None and a.size == ns
            a_hi = self._pen_hi.copy() if same(self._pen_hi) else np.zeros(ns, bool)
            a_lo = self._pen_lo.copy() if same(self._pen_lo) else np.zeros(ns, bool)
            U = z0.copy() if z0 is not None else np.clip(np.zeros(N * m), u_lo, u_hi)
            res = None
            for _ in range(6):
                if a_hi.any() or a_lo.any():
                    Aa = np.vstack([Gs[a_hi], Gs[a_lo]])
                    ba = np.concatenate([bhi[a_hi], blo[a_lo]])
                    Hp = b["H"] + rho * (Aa.T @ Aa)
                    gp = g_U + rho * (Aa.T @ (-ba))
                else:
                    Hp, gp = b["H"], g_U
                res = solve_qp(Hp, gp, lb=u_lo, ub=u_hi, z0=U)
                U = res.x
                xh = Gs @ U
                new_hi = hi_ok & (xh - bhi > 1e-9)
                new_lo = lo_ok & (blo - xh > 1e-9)
                if np.array_equal(new_hi, a_hi) and np.array_equal(new_lo, a_lo):
                    break
                a_hi, a_lo = new_hi, new_lo
            self._pen_hi, self._pen_lo = a_hi, a_lo
            U = U.reshape(N, m)

        self.last_qp = res
        u0 = U[0]
        self._warm = np.vstack([U[1:], U[-1]]).ravel()   # shift the plan one step

        self.horizon_plan = U
        self.predicted_states = (b["Phi"] @ x0 + b["Gamma"] @ U.ravel()).reshape(N, n)
        self.output = float(u0[0]) if m == 1 else u0
        return self.output

    # --------------------------------------------------------------- introspection

    def discrete_lqr_gain(self, dt: float) -> np.ndarray:
        """``K_d`` such that the unconstrained first move is ``-K_d x``."""
        return self._build(dt)["Kd"]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"LinearMPC(n={self.n}, m={self.m}, N={self.N}, "
                f"u_box={np.isfinite(self._u_lo).any()}, "
                f"x_box={np.isfinite(self._x_lo).any() or np.isfinite(self._x_hi).any()})")
