r"""Small dense quadratic-program solver - primal active-set, from scratch.

Solves

.. math::

    \min_z \; \tfrac12 z^\top H z + g^\top z
    \quad\text{s.t.}\quad C z \le d,\qquad \ell \le z \le u ,

with :math:`H = H^\top \succeq 0`.  Box bounds are folded into ``C z \le d`` as
:math:`\pm e_i^\top z \le \pm(\cdot)`.  The method (Nocedal & Wright, Alg. 16.3):
at a feasible point, treat the active constraints as equalities, take the
equality-constrained Newton step from the KKT system, walk to the first blocking
inactive constraint (add it), and when the step vanishes either stop (all
multipliers of :math:`\le` constraints are non-negative) or drop the most
negative one.

Only used by :mod:`aimct.controllers.mpc`; kept dependency-free (NumPy only) and
cross-checked against ``scipy.optimize.minimize`` in the tests.
"""

from __future__ import annotations

import numpy as np

__all__ = ["solve_qp", "QPResult"]


class QPResult:
    __slots__ = ("x", "iterations", "feasible", "active_set")

    def __init__(self, x, iterations, feasible, active_set):
        self.x = x
        self.iterations = iterations
        self.feasible = feasible
        self.active_set = active_set

    def __repr__(self) -> str:  # pragma: no cover
        return (f"QPResult(iterations={self.iterations}, feasible={self.feasible}, "
                f"|active|={len(self.active_set)})")


def _stack_constraints(n, C, d, lb, ub):
    rows, rhs = [], []
    if C is not None and len(C):
        rows.append(np.atleast_2d(np.asarray(C, dtype=float)))
        rhs.append(np.asarray(d, dtype=float).ravel())
    if ub is not None:
        ub = np.asarray(ub, dtype=float).ravel()
        m = np.isfinite(ub)
        if m.any():
            rows.append(np.eye(n)[m])
            rhs.append(ub[m])
    if lb is not None:
        lb = np.asarray(lb, dtype=float).ravel()
        m = np.isfinite(lb)
        if m.any():
            rows.append(-np.eye(n)[m])
            rhs.append(-lb[m])
    if not rows:
        return np.zeros((0, n)), np.zeros(0)
    return np.vstack(rows), np.concatenate(rhs)


def _phase1(C, d, z, box_lb, box_ub, iters=200, step=1.0):
    """Projected-gradient descent on ``||max(Cz - d, 0)||^2`` to reach (near)
    feasibility; box bounds enforced by clipping."""
    lo = -np.inf if box_lb is None else np.asarray(box_lb, float)
    hi = np.inf if box_ub is None else np.asarray(box_ub, float)
    z = np.clip(z, lo, hi)
    if not len(C):
        return z
    L = np.linalg.norm(C, 2) ** 2 + 1e-12
    for _ in range(iters):
        viol = C @ z - d
        active = viol > 0
        if not active.any():
            break
        grad = 2.0 * C[active].T @ viol[active]
        z = np.clip(z - (step / L) * grad, lo, hi)
    return z


def solve_qp(
    H,
    g,
    *,
    C=None,
    d=None,
    lb=None,
    ub=None,
    z0=None,
    max_iter=200,
    tol=1e-9,
) -> QPResult:
    """Minimise ``0.5 z'H z + g'z`` s.t. ``C z <= d`` and ``lb <= z <= ub``."""
    H = np.atleast_2d(np.asarray(H, dtype=float))
    g = np.asarray(g, dtype=float).ravel()
    n = g.size
    H = 0.5 * (H + H.T)
    # mild regularisation: keep H strictly positive definite so every
    # equality-constrained sub-solve has a unique step
    H = H + (1e-11 * (np.trace(H) / max(n, 1)) + 1e-14) * np.eye(n)
    try:
        Hchol = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        H = H + (1e-6 * (np.trace(H) / max(n, 1)) + 1e-10) * np.eye(n)
        Hchol = np.linalg.cholesky(H)

    def _solve_H(rhs):
        return np.linalg.solve(Hchol.T, np.linalg.solve(Hchol, rhs))

    A, b = _stack_constraints(n, C, d, lb, ub)
    m = A.shape[0]

    if z0 is None:
        z = np.zeros(n)
    else:
        z = np.asarray(z0, dtype=float).ravel().copy()

    if m and np.any(A @ z - b > 1e-7):
        z = _phase1(A, b, z, lb, ub)

    feasible = not m or bool(np.all(A @ z - b <= 1e-6))
    W = list(np.where(A @ z - b >= -1e-8)[0]) if m else []

    it = 0
    for it in range(1, max_iter + 1):
        gk = H @ z + g
        if W:
            # EQP  min 1/2 p'Hp + gk'p  s.t.  Cw p = 0  via the Schur complement
            #   (Cw H^-1 Cw') lam = -Cw H^-1 gk ;  p = -H^-1 (gk + Cw' lam)
            Cw = A[W]
            HinvCwT = _solve_H(Cw.T)
            Hinvgk = _solve_H(gk)
            schur = Cw @ HinvCwT
            lam, *_ = np.linalg.lstsq(schur, -(Cw @ Hinvgk), rcond=None)
            p = -(Hinvgk + HinvCwT @ lam)
        else:
            p = -_solve_H(gk)
            lam = np.zeros(0)

        if np.linalg.norm(p) <= tol * (1.0 + np.linalg.norm(z)):
            if W and lam.min(initial=0.0) < -tol:
                # Bland's rule: drop the lowest-index constraint with lam < 0
                neg = [W[k] for k in range(len(W)) if lam[k] < -tol]
                W.remove(min(neg))
                continue
            return QPResult(z, it, feasible, sorted(W))

        Ap = A @ p if m else np.zeros(0)
        alpha, hit = 1.0, None
        for i in range(m):
            if i in W or Ap[i] <= tol:
                continue
            ai = (b[i] - A[i] @ z) / Ap[i]
            if ai < alpha - 1e-12 or (hit is not None and abs(ai - alpha) <= 1e-12 and i < hit):
                alpha, hit = max(ai, 0.0), i          # Bland's rule on ties
        z = z + alpha * p
        if hit is not None:
            W.append(hit)

    return QPResult(z, it, feasible, sorted(W))
