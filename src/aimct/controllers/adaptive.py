r"""Adaptive control — the plant changes, the controller re-tunes itself.

Two classic approaches:

``GainScheduledLQR``
    Pre-compute an LQR gain at a grid of operating points (indexed by a
    measurable scheduling variable) and blend them at run time. Cheap, provably
    stable *at* each design point, no on-line learning.

``MRAC`` (model-reference adaptive control, direct, matched uncertainty)
    The plant is :math:`\dot x = A x + B\big(u + \theta^{*\top}\phi(x)\big)` with
    :math:`\theta^*` unknown (state-dependent uncertainty **and** a constant
    disturbance, via :math:`\phi(x) = [x;\,1]`). A fixed baseline law drives the
    nominal plant onto a Hurwitz reference model
    :math:`\dot x_m = A_m x_m + B_m r`; an adaptive term
    :math:`-\hat\theta^\top\phi(x)` cancels the uncertainty, with

    .. math::
        \dot{\hat\theta} = -\Gamma\,\phi(x)\,e^\top P B,\qquad
        A_m^\top P + P A_m = -Q,\quad e = x - x_m .

    Lyapunov function :math:`V = e^\top P e + \operatorname{tr}(\tilde\theta^\top
    \Gamma^{-1}\tilde\theta)` gives :math:`\dot V = -e^\top Q e \le 0` — the
    tracking error goes to zero even though the plant is never identified.
"""

from __future__ import annotations

import numpy as np

from .base import Controller
from .lqr import LQR

__all__ = ["GainScheduledLQR", "MRAC", "solve_lyapunov"]


def solve_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    r"""Solve :math:`A^\top P + P A + Q = 0` for symmetric ``P`` (``A`` Hurwitz).

    From scratch via the Kronecker form
    :math:`(I\otimes A^\top + A^\top\otimes I)\,\operatorname{vec}(P) =
    -\operatorname{vec}(Q)`; falls back to ``scipy`` if that is ill-conditioned.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    n = A.shape[0]
    K = np.kron(np.eye(n), A.T) + np.kron(A.T, np.eye(n))
    try:
        P = np.linalg.solve(K, -Q.reshape(-1)).reshape(n, n)
    except np.linalg.LinAlgError:  # pragma: no cover
        from scipy.linalg import solve_continuous_lyapunov
        P = solve_continuous_lyapunov(A.T, -Q)
    return 0.5 * (P + P.T)


class GainScheduledLQR(Controller):
    """LQR gains pre-computed over a grid of a scheduling variable, blended live.

    Parameters
    ----------
    linearize_at : ``f(s) -> (A, B)`` — the plant linearisation at scheduling
        value ``s``.
    grid : 1-D array of scheduling values to design at.
    schedule_fn : ``g(x) -> s`` — extract the scheduling value from the state.
    Q, R : LQR weights (shared across the grid).
    x_ref, u_ref : optional set-point / feed-forward.
    """

    name = "gain-scheduled LQR"

    def __init__(self, linearize_at, grid, schedule_fn, Q, R, *,
                 x_ref=None, u_ref=None):
        self.grid = np.asarray(grid, dtype=float)
        self.schedule_fn = schedule_fn
        self._K = []
        for s in self.grid:
            A, B = linearize_at(float(s))
            self._K.append(LQR(A, B, Q, R).K)
        self._K = np.array(self._K)                 # (G, m, n)
        n = self._K.shape[2]
        m = self._K.shape[1]
        self.x_ref = np.zeros(n) if x_ref is None else np.asarray(x_ref, float)
        self.u_ref = np.zeros(m) if u_ref is None else np.asarray(u_ref, float)

    def gain(self, x) -> np.ndarray:
        s = float(self.schedule_fn(np.asarray(x, dtype=float)))
        # linear interpolation between the two bracketing grid gains
        i = int(np.clip(np.searchsorted(self.grid, s) - 1, 0, len(self.grid) - 2))
        s0, s1 = self.grid[i], self.grid[i + 1]
        w = 0.0 if s1 == s0 else np.clip((s - s0) / (s1 - s0), 0.0, 1.0)
        return (1 - w) * self._K[i] + w * self._K[i + 1]

    def reset(self):
        pass

    def update(self, measurement, dt):
        x = np.asarray(measurement, dtype=float)
        K = self.gain(x)
        return self.u_ref - K @ (x - self.x_ref)


class MRAC(Controller):
    """Direct model-reference adaptive control with matched uncertainty.

    Parameters
    ----------
    A_m, B_m : Hurwitz reference-model matrices, ``x_m`` is integrated internally.
    B : plant input matrix (the *nominal* one; the uncertainty absorbs error in it).
    K_x, K_r : fixed baseline gains, ``u_bl = K_x @ x + K_r @ r``. If ``None`` they
        are solved from the matching conditions ``A_nom + B K_x = A_m`` (needs
        ``A_nom``) and ``B K_r = B_m`` (least squares).
    A_nom : nominal ``A`` (only used to derive ``K_x`` when it is not given).
    gamma : adaptation-rate scalar or ``(p, p)`` matrix, ``p = n + 1``.
    Q : Lyapunov weight for ``P`` (default ``I``).
    reference_fn : ``r(t) -> array`` command; or set ``.r`` each step.
    u_bounds : optional ``(lo, hi)`` saturation (adaptation pauses while saturated).
    """

    name = "MRAC"

    def __init__(self, A_m, B_m, B, *, K_x=None, K_r=None, A_nom=None,
                 gamma=1.0, Q=None, reference_fn=None, u_bounds=None,
                 theta_leak=0.0):
        self.A_m = np.atleast_2d(np.asarray(A_m, dtype=float))
        self.B_m = np.atleast_2d(np.asarray(B_m, dtype=float))
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.n, self.m = self.B.shape
        self.p = self.n + 1                         # regressor phi(x) = [x; 1]

        if K_r is None:
            K_r = np.linalg.lstsq(self.B, self.B_m, rcond=None)[0]
        if K_x is None:
            if A_nom is None:
                raise ValueError("give K_x or A_nom")
            A_nom = np.atleast_2d(np.asarray(A_nom, dtype=float))
            K_x = np.linalg.lstsq(self.B, self.A_m - A_nom, rcond=None)[0]
        self.K_x = np.atleast_2d(np.asarray(K_x, dtype=float))
        self.K_r = np.atleast_2d(np.asarray(K_r, dtype=float))

        g = np.asarray(gamma, dtype=float)
        self.Gamma = g * np.eye(self.p) if g.ndim == 0 else g
        self.Q = np.eye(self.n) if Q is None else np.atleast_2d(np.asarray(Q, float))
        self.P = solve_lyapunov(self.A_m, self.Q)
        self.reference_fn = reference_fn
        self.u_bounds = u_bounds
        self.theta_leak = float(theta_leak)
        self.r = np.zeros(self.B_m.shape[1])
        self.reset()

    def reset(self):
        self.x_m = np.zeros(self.n)
        self.theta = np.zeros((self.p, self.m))    # theta_hat
        self._t = 0.0

    def _phi(self, x):
        return np.concatenate([x, [1.0]])

    def update(self, measurement, dt):
        x = np.asarray(measurement, dtype=float)
        r = np.asarray(self.reference_fn(self._t), dtype=float) \
            if self.reference_fn is not None else np.asarray(self.r, dtype=float)

        u_bl = self.K_x @ x + self.K_r @ r
        phi = self._phi(x)
        u = u_bl - self.theta.T @ phi
        u_cmd = u
        if self.u_bounds is not None:
            u_cmd = np.clip(u, self.u_bounds[0], self.u_bounds[1])

        # advance reference model + adaptive law (Euler; dt is small)
        e = x - self.x_m
        self.x_m = self.x_m + dt * (self.A_m @ self.x_m + self.B_m @ r)
        if self.u_bounds is None or np.allclose(u_cmd, u):
            # sign pairs with u = u_bl - theta^T phi; Lyapunov gives Vdot = -e'Qe
            dtheta = self.Gamma @ np.outer(phi, e @ self.P @ self.B)
            self.theta = self.theta + dt * (dtheta - self.theta_leak * self.theta)
        self._t += dt
        return u_cmd
