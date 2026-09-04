r"""Iterative LQR (iLQR) trajectory optimisation and real-time-iteration NMPC.

iLQR is the Gauss-Newton cousin of DDP: it locally approximates a *nonlinear*
optimal-control problem as a time-varying LQR problem, solves that in closed
form (a Riccati-like backward sweep), does a line-searched forward rollout of
the true nonlinear dynamics, and repeats.  Unlike :class:`SamplingMPC` (CEM) it
uses gradients, so it converges quadratically near a solution and produces a
*locally optimal* trajectory plus a stabilising time-varying feedback gain
``K_k`` -- not just an open-loop plan.

Two objects:

``iLQR``
    The optimiser.  ``solve(x0, U_init) -> iLQRResult(X, U, K, kff, cost, ...)``.
    Discrete dynamics come in as ``step(x, u) -> x_next``; use
    :meth:`iLQR.from_system` to wrap a continuous :mod:`aimct.systems` model with
    one RK4 step of length ``dt``.

``ILQR``
    A :class:`~aimct.controllers.base.Controller`: receding-horizon wrapper.
    The first ``update`` runs a full solve; each later step runs only
    ``rti_iters`` iLQR iterations from the shifted warm start (Diehl's
    *real-time iteration* scheme) and applies ``u_0`` with its local feedback.
    ``x_ref`` / ``u_ref`` may be constants or callables of absolute time, so it
    drops straight into :func:`aimct.benchmarks.track_trajectory`.

Input box constraints are handled by *clamping* in the forward pass (clamped
iLQR).  This is the cheap, robust choice; it is not full box-DDP, so the
feedback gain ``K_k`` is the unconstrained one and can over-drive when a channel
is saturated for long stretches.  State constraints are not handled here -- use
:class:`LinearMPC` or add a penalty to a custom ``cost``.

References
----------
* Li & Todorov, "Iterative linear quadratic regulator design for nonlinear
  biological movement systems", ICINCO 2004.
* Tassa, Erez & Todorov, "Synthesis and stabilization of complex behaviors
  through online trajectory optimization", IROS 2012  (regularisation +
  line-search scheme used here).
* Diehl et al., "Real-time iterations" for NMPC.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from .base import Controller

__all__ = ["iLQR", "ILQR", "iLQRResult"]


class iLQRResult(NamedTuple):
    X: np.ndarray          # (H+1, n_x)  optimal state trajectory
    U: np.ndarray          # (H,   n_u)  optimal input trajectory
    K: np.ndarray          # (H, n_u, n_x)  time-varying feedback gains
    kff: np.ndarray        # (H, n_u)    feed-forward corrections (~0 at convergence)
    cost: float            # final total cost
    iters: int             # iLQR iterations actually run
    converged: bool        # feed-forward / cost change fell below tol
    cost_history: list     # cost after each accepted iteration (incl. initial)


def _rk4(dynamics: Callable, dt: float) -> Callable:
    """One classical RK4 step of ``xdot = dynamics(0, x, u)`` (ZOH input)."""

    def step(x, u):
        k1 = dynamics(0.0, x, u)
        k2 = dynamics(0.0, x + 0.5 * dt * k1, u)
        k3 = dynamics(0.0, x + 0.5 * dt * k2, u)
        k4 = dynamics(0.0, x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return step


class iLQR:
    r"""Iterative-LQR optimiser for a discrete-time nonlinear system.

    Minimises

    .. math::

        J = (x_H - x^{\mathrm{ref}}_H)^\top Q_f (x_H - x^{\mathrm{ref}}_H)
          + \sum_{k=0}^{H-1} (x_k - x^{\mathrm{ref}}_k)^\top Q
            (x_k - x^{\mathrm{ref}}_k)
          + (u_k - u^{\mathrm{ref}}_k)^\top R (u_k - u^{\mathrm{ref}}_k)

    subject to ``x_{k+1} = step(x_k, u_k)`` and (by clamping) the input box.

    Parameters
    ----------
    step : ``f(x, u) -> x_next``  (single state / input vectors).
    n_x, n_u : state and input dimensions.
    horizon : number of control intervals ``H``.
    Q, R, Qf : stage-state, input, terminal weights.  Scalars are broadcast to
        ``scalar * I``.  ``Qf`` defaults to ``Q``.
    u_bounds : ``(lo, hi)`` scalar or per-channel input box (``None`` = free).
    x_ref, u_ref : reference.  ``None`` -> zeros; scalar / vector -> held over
        the horizon; ``(H+1, n_x)`` / ``(H, n_u)`` array -> a preview; callable
        ``g(k) -> vector`` -> evaluated per knot.  Settable attributes.
    deriv : optional analytic ``g(X, U) -> (fx, fu)`` with
        ``fx`` ``(H, n_x, n_x)`` and ``fu`` ``(H, n_x, n_u)``.  Default: central
        finite differences on ``step``.
    cost : optional custom terminal+stage cost, see :meth:`_quad_cost` for the
        tuple it must return.  Default: the quadratic tracking cost above.
    max_iter, tol : iteration cap and convergence tolerance on the relative
        feed-forward update ``max_k |kff_k| / (1 + |u_k|)``.
    reg_init, reg_min, reg_max, reg_factor : Levenberg-Marquardt-style state
        regularisation schedule (Tassa 2012).
    alphas : line-search step fractions (default ``1.1**-(0..9)**2``).
    fd_eps : finite-difference step for ``deriv``.
    """

    def __init__(
        self,
        step: Callable,
        *,
        n_x: int,
        n_u: int,
        horizon: int,
        Q,
        R,
        Qf=None,
        u_bounds: tuple | None = None,
        x_ref=None,
        u_ref=None,
        deriv: Callable | None = None,
        cost: Callable | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_init: float = 1e-6,
        reg_min: float = 1e-9,
        reg_max: float = 1e10,
        reg_factor: float = 2.0,
        alphas: np.ndarray | None = None,
        fd_eps: float = 1e-6,
    ) -> None:
        self.step = step
        self.n_x, self.n_u, self.H = int(n_x), int(n_u), int(horizon)
        self.Q = self._as_mat(Q, self.n_x)
        self.R = self._as_mat(R, self.n_u)
        self.Qf = self.Q.copy() if Qf is None else self._as_mat(Qf, self.n_x)
        if u_bounds is None:
            self.u_lo = np.full(self.n_u, -np.inf)
            self.u_hi = np.full(self.n_u, np.inf)
        else:
            self.u_lo = np.broadcast_to(np.asarray(u_bounds[0], float), (self.n_u,)).copy()
            self.u_hi = np.broadcast_to(np.asarray(u_bounds[1], float), (self.n_u,)).copy()
        self.x_ref = x_ref
        self.u_ref = u_ref
        self._deriv = deriv
        self._cost = cost
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_init, self.reg_min, self.reg_max = reg_init, reg_min, reg_max
        self.reg_factor = float(reg_factor)
        if alphas is None:
            alphas = 1.1 ** (-(np.arange(10) ** 2))
        self.alphas = np.asarray(alphas, float)
        self.fd_eps = float(fd_eps)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_system(cls, system, dt: float, **kw) -> "iLQR":
        """Wrap a continuous :mod:`aimct.systems` model (one RK4 step of ``dt``)."""
        kw.setdefault("n_x", system.n_states)
        kw.setdefault("n_u", system.n_inputs)
        return cls(_rk4(system.dynamics, dt), **kw)

    # -- reference plumbing ----------------------------------------------------

    @staticmethod
    def _as_mat(w, n: int) -> np.ndarray:
        w = np.asarray(w, float)
        if w.ndim == 0:
            return float(w) * np.eye(n)
        if w.ndim == 1:
            return np.diag(w)
        return w

    def _ref(self, spec, n: int, length: int) -> np.ndarray:
        """Resolve a reference spec to an ``(length, n)`` array."""
        out = np.zeros((length, n))
        if spec is None:
            return out
        if callable(spec):
            for k in range(length):
                out[k] = np.broadcast_to(np.asarray(spec(k), float), (n,))
            return out
        arr = np.asarray(spec, float)
        if arr.ndim <= 1:
            out[:] = np.broadcast_to(arr, (n,))
        else:
            m = min(length, arr.shape[0])
            out[:m] = arr[:m]
            out[m:] = arr[-1] if arr.shape[0] else 0.0
        return out

    # -- cost ----------------------------------------------------------------

    def _quad_cost(self, X: np.ndarray, U: np.ndarray, Xr: np.ndarray, Ur: np.ndarray):
        """Return ``(total, lx, lu, lxx, luu, lux)`` for the quadratic tracker.

        ``lx``   (H+1, n_x)      ``lxx`` (H+1, n_x, n_x)
        ``lu``   (H,   n_u)      ``luu`` (H,   n_u, n_u)
        ``lux``  (H,   n_u, n_x) -- zero for a separable quadratic.
        """
        H, nx, nu = self.H, self.n_x, self.n_u
        dx, du = X - Xr, U - Ur
        stage = np.einsum("ki,ij,kj->", dx[:H], self.Q, dx[:H]) \
            + np.einsum("ki,ij,kj->", du, self.R, du)
        term = float(dx[H] @ self.Qf @ dx[H])
        lx = np.zeros((H + 1, nx))
        lx[:H] = 2.0 * dx[:H] @ self.Q
        lx[H] = 2.0 * dx[H] @ self.Qf
        lxx = np.broadcast_to(2.0 * self.Q, (H + 1, nx, nx)).copy()
        lxx[H] = 2.0 * self.Qf
        lu = 2.0 * du @ self.R
        luu = np.broadcast_to(2.0 * self.R, (H, nu, nu)).copy()
        lux = np.zeros((H, nu, nx))
        return stage + term, lx, lu, lxx, luu, lux

    def _rollout_cost(self, X: np.ndarray, U: np.ndarray, Xr: np.ndarray, Ur: np.ndarray) -> float:
        if self._cost is not None:
            return float(self._cost(X, U, Xr, Ur)[0])
        return self._quad_cost(X, U, Xr, Ur)[0]

    # -- dynamics linearisation --------------------------------------------

    def _linearize(self, X: np.ndarray, U: np.ndarray):
        if self._deriv is not None:
            fx, fu = self._deriv(X, U)
            return np.asarray(fx, float), np.asarray(fu, float)
        H, nx, nu, e = self.H, self.n_x, self.n_u, self.fd_eps
        fx = np.zeros((H, nx, nx))
        fu = np.zeros((H, nx, nu))
        eye_x, eye_u = np.eye(nx), np.eye(nu)
        for k in range(H):
            xk, uk = X[k], U[k]
            for i in range(nx):
                d = e * eye_x[i]
                fx[k, :, i] = (self.step(xk + d, uk) - self.step(xk - d, uk)) / (2 * e)
            for i in range(nu):
                d = e * eye_u[i]
                fu[k, :, i] = (self.step(xk, uk + d) - self.step(xk, uk - d)) / (2 * e)
        return fx, fu

    # -- forward / backward passes ---------------------------------------------

    def _forward(self, X, U, kff, K, alpha, Xr, Ur):
        H = self.H
        Xn = np.empty_like(X)
        Un = np.empty_like(U)
        Xn[0] = X[0]
        for k in range(H):
            Un[k] = U[k] + alpha * kff[k] + K[k] @ (Xn[k] - X[k])
            Un[k] = np.clip(Un[k], self.u_lo, self.u_hi)
            Xn[k + 1] = self.step(Xn[k], Un[k])
        return Xn, Un, self._rollout_cost(Xn, Un, Xr, Ur)

    def _backward(self, fx, fu, lx, lu, lxx, luu, lux, mu):
        """One regularised backward sweep.  Returns ``(kff, K, dV1, dV2, ok)``."""
        H, nx, nu = self.H, self.n_x, self.n_u
        kff = np.zeros((H, nu))
        K = np.zeros((H, nu, nx))
        Vx = lx[H].copy()
        Vxx = lxx[H].copy()
        dV1 = dV2 = 0.0
        reg = mu * np.eye(nx)
        for k in range(H - 1, -1, -1):
            fxk, fuk = fx[k], fu[k]
            Qx = lx[k] + fxk.T @ Vx
            Qu = lu[k] + fuk.T @ Vx
            Qxx = lxx[k] + fxk.T @ Vxx @ fxk
            Quu = luu[k] + fuk.T @ Vxx @ fuk
            Qux = lux[k] + fuk.T @ Vxx @ fxk
            # state-regularised versions used only for the gains (Tassa 2012)
            Vxx_r = Vxx + reg
            Quu_r = luu[k] + fuk.T @ Vxx_r @ fuk
            Qux_r = lux[k] + fuk.T @ Vxx_r @ fxk
            Quu_r = 0.5 * (Quu_r + Quu_r.T)
            w = np.linalg.eigvalsh(Quu_r)
            if np.min(w) <= 1e-9:
                return kff, K, 0.0, 0.0, False
            L = np.linalg.cholesky(Quu_r)
            rhs = -np.concatenate([Qu[:, None], Qux_r], axis=1)
            sol = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            kff[k] = sol[:, 0]
            K[k] = sol[:, 1:]
            dV1 += kff[k] @ Qu
            dV2 += 0.5 * kff[k] @ Quu @ kff[k]
            Vx = Qx + K[k].T @ Quu @ kff[k] + K[k].T @ Qu + Qux.T @ kff[k]
            Vxx = Qxx + K[k].T @ Quu @ K[k] + K[k].T @ Qux + Qux.T @ K[k]
            Vxx = 0.5 * (Vxx + Vxx.T)
        return kff, K, dV1, dV2, True

    # -- solve ---------------------------------------------------------------

    def solve(self, x0, U_init=None, *, max_iter: int | None = None) -> iLQRResult:
        H, nx, nu = self.H, self.n_x, self.n_u
        x0 = np.asarray(x0, float)
        max_iter = self.max_iter if max_iter is None else int(max_iter)
        Xr = self._ref(self.x_ref, nx, H + 1)
        Ur = self._ref(self.u_ref, nu, H)

        U = np.zeros((H, nu)) if U_init is None else np.array(U_init, float).reshape(H, nu)
        U = np.clip(U, self.u_lo, self.u_hi)
        X = np.empty((H + 1, nx))
        X[0] = x0
        for k in range(H):
            X[k + 1] = self.step(X[k], U[k])
        cost = self._rollout_cost(X, U, Xr, Ur)

        mu = self.reg_init
        dmu = 1.0
        K = np.zeros((H, nu, nx))
        kff = np.zeros((H, nu))
        history = [cost]
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            fx, fu = self._linearize(X, U)
            if self._cost is not None:
                _, lx, lu, lxx, luu, lux = self._cost(X, U, Xr, Ur)
            else:
                _, lx, lu, lxx, luu, lux = self._quad_cost(X, U, Xr, Ur)

            # backward pass, growing mu until it succeeds
            bp_ok = False
            for _ in range(40):
                kff, K, dV1, dV2, bp_ok = self._backward(
                    fx, fu, lx, lu, lxx, luu, lux, mu)
                if bp_ok:
                    break
                dmu = max(dmu * self.reg_factor, self.reg_factor)
                mu = max(mu * dmu, self.reg_min)
                if mu > self.reg_max:
                    break
            if not bp_ok:
                break

            # line search
            accepted = False
            for alpha in self.alphas:
                Xn, Un, cost_new = self._forward(X, U, kff, K, alpha, Xr, Ur)
                if not np.all(np.isfinite(Xn)):
                    continue
                expected = -(alpha * dV1 + alpha ** 2 * dV2)
                reduction = cost - cost_new
                if expected > 0:
                    ratio = reduction / expected
                    ok = ratio > 1e-4
                else:                       # non-positive model improvement
                    ok = reduction > 1e-9
                if ok:
                    X, U, cost = Xn, Un, cost_new
                    accepted = True
                    break

            history.append(cost)
            if accepted:
                dmu = min(dmu / self.reg_factor, 1.0 / self.reg_factor)
                mu = max(mu * dmu, self.reg_min)
                rel = np.max(np.abs(kff) / (1.0 + np.abs(U)))
                if rel < self.tol or abs(history[-2] - cost) / (abs(cost) + 1e-12) < self.tol:
                    converged = True
                    break
            else:
                dmu = max(dmu * self.reg_factor, self.reg_factor)
                mu = max(mu * dmu, self.reg_min)
                if mu > self.reg_max:
                    break

        return iLQRResult(X=X, U=U, K=K, kff=kff, cost=float(cost),
                          iters=it, converged=converged, cost_history=history)


class ILQR(Controller):
    """Receding-horizon iLQR / real-time-iteration NMPC as a :class:`Controller`.

    Parameters mirror :class:`iLQR` plus:

    warm_iters : iterations on the first ``update`` (cold start).  Default
        ``max_iter`` (a full solve).
    rti_iters : iterations per subsequent ``update`` -- ``1`` is Diehl's
        real-time iteration.  Raise for a slower but tighter re-solve.
    feedback : apply the local gain ``u = U0 + K0 (x - X0)`` (default ``True``).
        With a fresh solve each step ``x == X0`` so this only bites between
        re-solves / under model error.
    x_ref, u_ref : constant vector **or callable of absolute time** ``g(t)``.
        A trajectory tracker just passes the ``aimct.trajectories`` object's
        state-space lift here.
    """

    name = "ILQR"

    def __init__(
        self,
        step: Callable,
        *,
        n_x: int,
        n_u: int,
        horizon: int,
        dt: float,
        Q,
        R,
        Qf=None,
        u_bounds: tuple | None = None,
        x_ref=None,
        u_ref=None,
        warm_iters: int | None = None,
        rti_iters: int = 1,
        feedback: bool = True,
        **opt,
    ) -> None:
        self._xref_spec = x_ref
        self._uref_spec = u_ref
        self.dt = float(dt)
        self.feedback = bool(feedback)
        self.rti_iters = int(rti_iters)
        self.opt = iLQR(step, n_x=n_x, n_u=n_u, horizon=horizon,
                        Q=Q, R=R, Qf=Qf, u_bounds=u_bounds, **opt)
        self.warm_iters = self.opt.max_iter if warm_iters is None else int(warm_iters)
        self.n_u = n_u
        self.reset()

    @classmethod
    def from_system(cls, system, dt: float, **kw) -> "ILQR":
        kw.setdefault("n_x", system.n_states)
        kw.setdefault("n_u", system.n_inputs)
        return cls(_rk4(system.dynamics, dt), dt=dt, **kw)

    def reset(self) -> None:
        self._t = 0.0
        self._k = 0
        self.U = None
        self.X = None
        self.K = None
        self.last_result = None

    # -- moving reference -> horizon preview ------------------------------------

    def _previews(self):
        H, dt = self.opt.H, self.dt
        xs = self._xref_spec
        us = self._uref_spec
        if callable(xs):
            xr = np.array([np.asarray(xs(self._t + k * dt), float) for k in range(H + 1)])
        else:
            xr = xs
        if callable(us):
            ur = np.array([np.asarray(us(self._t + k * dt), float) for k in range(H)])
        else:
            ur = us
        return xr, ur

    def update(self, measurement, dt: float):
        x = np.atleast_1d(np.asarray(measurement, float))
        self.opt.x_ref, self.opt.u_ref = self._previews()

        if self.U is None:                                   # cold start
            res = self.opt.solve(x, U_init=None, max_iter=self.warm_iters)
        else:                                                # warm real-time iter
            U_ws = np.vstack([self.U[1:], self.U[-1]])
            res = self.opt.solve(x, U_init=U_ws, max_iter=self.rti_iters)

        self.X, self.U, self.K = res.X, res.U, res.K
        self.last_result = res

        u = self.U[0].copy()
        if self.feedback:
            u = u + self.K[0] @ (x - self.X[0])
        u = np.clip(u, self.opt.u_lo, self.opt.u_hi)

        self._t += dt
        self._k += 1
        return u if self.n_u > 1 else float(u[0])
