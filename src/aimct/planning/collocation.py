r"""Direct trajectory optimisation by Hermite-Simpson collocation.

The finite-horizon optimal-control problem

.. math::

    \min_{x(\cdot),\,u(\cdot)} \;
        \phi\bigl(x(t_f)\bigr)
        + \int_{0}^{t_f} \ell\bigl(x(t), u(t)\bigr)\, dt
    \quad\text{s.t.}\quad
    \dot x = f(x, u),\;
    x(0) = x_0,\;
    u \in [\underline u, \overline u],\; \dots

is *transcribed* into a nonlinear program.  The horizon is split into ``N-1``
intervals by ``N`` knot points; the decision vector carries the state **and**
the input at every knot,

.. math::

    z = \bigl(x_0, \dots, x_{N-1},\; u_0, \dots, u_{N-1}\bigr).

Between consecutive knots the dynamics are enforced by a **Hermite-Simpson**
defect constraint (a third-order implicit Runge-Kutta / Simpson-quadrature
rule).  With :math:`f_k = f(x_k, u_k)`, interval width :math:`h`, the collocation
(mid) point is the Hermite interpolant

.. math::

    x_c = \tfrac12 (x_k + x_{k+1}) + \tfrac{h}{8}\,(f_k - f_{k+1}),
    \qquad
    u_c = \tfrac12 (u_k + u_{k+1}),

and the defect that the NLP drives to zero is

.. math::

    x_k - x_{k+1} + \frac{h}{6}\bigl(f_k + 4 f_c + f_{k+1}\bigr) = 0,
    \qquad f_c = f(x_c, u_c).

The running cost is integrated by the trapezoidal rule over the same knots.
Initial / terminal state equalities and the input / state boxes become simple
bounds on ``z``; an optional ``path_con(X, U) <= 0`` adds inequality
constraints (keep-out regions, etc.).  The NLP is handed to
:func:`scipy.optimize.minimize` (``"SLSQP"`` by default, ``"trust-constr"``
optional) with an **analytic constraint Jacobian** built from the system
linearisation when one is available, and a finite-difference fallback
otherwise.

Unlike :class:`aimct.controllers.iLQR` this returns only an *open-loop*
trajectory ``(X, U)`` -- there is no time-varying feedback gain and no
receding-horizon wrapper; direct collocation here is an **offline** planner.
Feed the result to :mod:`aimct.benchmarks` playback, or track it online with an
LQR / MPC of your choice.

References
----------
* Hargrove & Paris, "Direct trajectory optimization using nonlinear
  programming and collocation", J. Guidance 1987.
* Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear
  Programming*, 2nd ed., SIAM 2010 (Hermite-Simpson, section 4.5).
* Kelly, "An introduction to trajectory optimization: how to do your own
  direct collocation", SIAM Review 2017.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

__all__ = ["DirectCollocation", "CollocationResult"]


class CollocationResult(NamedTuple):
    X: np.ndarray          # (N, n_x)   state at each knot
    U: np.ndarray          # (N, n_u)   input at each knot
    t: np.ndarray          # (N,)       knot times, 0 .. t_final
    cost: float            # objective value at the solution
    defect_norm: float     # max abs Hermite-Simpson defect at the solution
    success: bool           # solver reported convergence
    nit: int               # solver iterations
    nfev: int              # objective evaluations
    message: str           # solver status message


def _as_mat(w, n: int) -> np.ndarray:
    """Broadcast a scalar / vector / matrix weight to an ``(n, n)`` matrix."""
    w = np.asarray(w, float)
    if w.ndim == 0:
        return float(w) * np.eye(n)
    if w.ndim == 1:
        return np.diag(w)
    return w


def _ref(spec, n: int, length: int) -> np.ndarray:
    """Resolve a reference spec to an ``(length, n)`` array.

    ``None`` -> zeros; scalar / vector -> held; ``(*, n)`` array -> preview
    (last row held past its end); callable ``g(k) -> vector`` -> per knot.
    """
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


class DirectCollocation:
    r"""Hermite-Simpson direct-collocation transcription of a finite-horizon OCP.

    Parameters
    ----------
    dynamics : ``f(x, u) -> xdot``
        Continuous-time dynamics on single state / input vectors.
    n_x, n_u : int
        State and input dimension.
    N : int
        Number of knot points (``N - 1`` collocation intervals).
    t_final : float
        Horizon length; knot spacing is ``h = t_final / (N - 1)``.
    x0 : array (n_x,)
        Fixed initial state (an equality via bounds on ``z``).
    x_goal : array (n_x,) or None
        If given, a fixed terminal state (equality); otherwise the terminal
        state is free and only ``Qf`` / ``terminal_cost`` shapes it.
    Q, R, Qf : weights
        Stage-state, stage-input and terminal weights of the default quadratic
        cost :math:`(x-x_r)^\top Q (x-x_r) + (u-u_r)^\top R (u-u_r)` and
        :math:`(x-x_r)^\top Q_f (x-x_r)`.  Scalars / vectors are broadcast.
        ``Qf`` defaults to ``Q``.  Ignored where ``running_cost`` /
        ``terminal_cost`` are supplied.
    x_ref, u_ref : reference specs
        ``None`` / scalar / vector / ``(N, .)`` array / callable ``g(k)``.
    u_bounds, x_bounds : ``(lo, hi)`` or None
        Box constraints; scalar or per-channel.  ``None`` -> unbounded.
    path_con : ``g(X, U) -> array`` or None
        Optional path inequality enforced as ``g(X, U) <= 0`` at every knot
        (``X`` is ``(N, n_x)``, ``U`` is ``(N, n_u)``; return any length).
    running_cost : ``l(x, u) -> float`` or None
        Overrides the quadratic stage cost (finite-differenced for the
        gradient).
    terminal_cost : ``phi(x) -> float`` or None
        Overrides the quadratic terminal cost.
    linearize : ``g(x, u) -> (A, B)`` or None
        Analytic Jacobians of ``dynamics``; enables an analytic constraint
        Jacobian.  ``None`` -> central finite differences.
    method : {"SLSQP", "trust-constr"}
        Backend for :func:`scipy.optimize.minimize`.
    max_iter, tol : int, float
        Passed through to the solver.
    fd_eps : float
        Finite-difference step for the fallback Jacobian / custom-cost gradient.
    """

    def __init__(
        self,
        dynamics: Callable,
        *,
        n_x: int,
        n_u: int,
        N: int,
        t_final: float,
        x0,
        x_goal=None,
        Q=1.0,
        R=1.0,
        Qf=None,
        x_ref=None,
        u_ref=None,
        u_bounds: tuple | None = None,
        x_bounds: tuple | None = None,
        path_con: Callable | None = None,
        running_cost: Callable | None = None,
        terminal_cost: Callable | None = None,
        linearize: Callable | None = None,
        method: str = "SLSQP",
        max_iter: int = 300,
        tol: float = 1e-8,
        fd_eps: float = 1e-6,
    ) -> None:
        self.f = dynamics
        self.n_x, self.n_u, self.N = int(n_x), int(n_u), int(N)
        if self.N < 3:
            raise ValueError("need at least N = 3 knot points")
        self.t_final = float(t_final)
        self.h = self.t_final / (self.N - 1)
        self.x0 = np.asarray(x0, float).reshape(self.n_x)
        self.x_goal = None if x_goal is None else np.asarray(x_goal, float).reshape(self.n_x)

        self.Q = _as_mat(Q, self.n_x)
        self.R = _as_mat(R, self.n_u)
        self.Qf = self.Q.copy() if Qf is None else _as_mat(Qf, self.n_x)
        self.Xr = _ref(x_ref, self.n_x, self.N)
        self.Ur = _ref(u_ref, self.n_u, self.N)

        self._ubox = self._box(u_bounds, self.n_u)
        self._xbox = self._box(x_bounds, self.n_x)
        self.path_con = path_con
        self._running_cost = running_cost
        self._terminal_cost = terminal_cost
        self._lin = linearize
        self.method = str(method)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fd_eps = float(fd_eps)

        self.t = np.linspace(0.0, self.t_final, self.N)
        self._nz = self.N * (self.n_x + self.n_u)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _box(spec, n: int):
        if spec is None:
            return np.full(n, -np.inf), np.full(n, np.inf)
        lo = np.broadcast_to(np.asarray(spec[0], float), (n,)).astype(float)
        hi = np.broadcast_to(np.asarray(spec[1], float), (n,)).astype(float)
        return lo.copy(), hi.copy()

    def _split(self, z: np.ndarray):
        """``z -> (X (N, n_x), U (N, n_u))``."""
        nx = self.N * self.n_x
        return z[:nx].reshape(self.N, self.n_x), z[nx:].reshape(self.N, self.n_u)

    def _merge(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(X, float).ravel(), np.asarray(U, float).ravel()])

    def _AB(self, x, u):
        """Analytic or finite-difference Jacobians of ``f`` at ``(x, u)``."""
        if self._lin is not None:
            A, B = self._lin(x, u)
            return np.asarray(A, float), np.asarray(B, float)
        e = self.fd_eps
        nx, nu = self.n_x, self.n_u
        A = np.empty((nx, nx))
        B = np.empty((nx, nu))
        for i in range(nx):
            d = np.zeros(nx)
            d[i] = e
            A[:, i] = (self.f(x + d, u) - self.f(x - d, u)) / (2 * e)
        for i in range(nu):
            d = np.zeros(nu)
            d[i] = e
            B[:, i] = (self.f(x, u + d) - self.f(x, u - d)) / (2 * e)
        return A, B

    # --------------------------------------------------------------- objective

    def _objective(self, z: np.ndarray) -> float:
        X, U = self._split(z)
        if self._running_cost is not None:
            stage = np.array([self._running_cost(X[k], U[k]) for k in range(self.N)])
        else:
            dX, dU = X - self.Xr, U - self.Ur
            stage = np.einsum("ki,ij,kj->k", dX, self.Q, dX) \
                + np.einsum("ki,ij,kj->k", dU, self.R, dU)
        # trapezoidal quadrature of the running cost
        run = self.h * (stage.sum() - 0.5 * stage[0] - 0.5 * stage[-1])
        if self._terminal_cost is not None:
            term = float(self._terminal_cost(X[-1]))
        else:
            dxf = X[-1] - self.Xr[-1]
            term = float(dxf @ self.Qf @ dxf)
        return float(run + term)

    def _objective_grad(self, z: np.ndarray) -> np.ndarray:
        if self._running_cost is not None or self._terminal_cost is not None:
            return self._fd_grad(self._objective, z)
        X, U = self._split(z)
        dX, dU = X - self.Xr, U - self.Ur
        w = np.full(self.N, self.h)
        w[0] = w[-1] = 0.5 * self.h
        gX = 2.0 * (w[:, None] * (dX @ self.Q))
        gU = 2.0 * (w[:, None] * (dU @ self.R))
        gX[-1] += 2.0 * (X[-1] - self.Xr[-1]) @ self.Qf
        return self._merge(gX, gU)

    def _fd_grad(self, fn, z: np.ndarray) -> np.ndarray:
        e = self.fd_eps
        g = np.empty_like(z)
        for i in range(z.size):
            d = np.zeros_like(z)
            d[i] = e
            g[i] = (fn(z + d) - fn(z - d)) / (2 * e)
        return g

    # -------------------------------------------------------- defect constraint

    def _defects(self, z: np.ndarray) -> np.ndarray:
        X, U = self._split(z)
        h = self.h
        F = np.array([self.f(X[k], U[k]) for k in range(self.N)])
        xk, xk1 = X[:-1], X[1:]
        fk, fk1 = F[:-1], F[1:]
        uk, uk1 = U[:-1], U[1:]
        xc = 0.5 * (xk + xk1) + (h / 8.0) * (fk - fk1)
        uc = 0.5 * (uk + uk1)
        fc = np.array([self.f(xc[k], uc[k]) for k in range(self.N - 1)])
        d = xk - xk1 + (h / 6.0) * (fk + 4.0 * fc + fk1)
        return d.ravel()

    def _defects_jac(self, z: np.ndarray) -> np.ndarray:
        """Dense ``((N-1) n_x, n_z)`` Jacobian of :meth:`_defects`."""
        X, U = self._split(z)
        nx, nu, N, h = self.n_x, self.n_u, self.N, self.h
        eye = np.eye(nx)
        J = np.zeros(((N - 1) * nx, self._nz))
        uoff = N * nx
        AB = [self._AB(X[k], U[k]) for k in range(N)]
        for k in range(N - 1):
            Ak, Bk = AB[k]
            Ak1, Bk1 = AB[k + 1]
            fk = self.f(X[k], U[k])
            fk1 = self.f(X[k + 1], U[k + 1])
            xc = 0.5 * (X[k] + X[k + 1]) + (h / 8.0) * (fk - fk1)
            uc = 0.5 * (U[k] + U[k + 1])
            Ac, Bc = self._AB(xc, uc)

            dxc_dxk = 0.5 * eye + (h / 8.0) * Ak
            dxc_dxk1 = 0.5 * eye - (h / 8.0) * Ak1
            dxc_duk = (h / 8.0) * Bk
            dxc_duk1 = -(h / 8.0) * Bk1

            ddef_dxk = eye + (h / 6.0) * (Ak + 4.0 * Ac @ dxc_dxk)
            ddef_dxk1 = -eye + (h / 6.0) * (Ak1 + 4.0 * Ac @ dxc_dxk1)
            ddef_duk = (h / 6.0) * (Bk + 4.0 * (Ac @ dxc_duk + 0.5 * Bc))
            ddef_duk1 = (h / 6.0) * (Bk1 + 4.0 * (Ac @ dxc_duk1 + 0.5 * Bc))

            r = slice(k * nx, (k + 1) * nx)
            J[r, k * nx:(k + 1) * nx] = ddef_dxk
            J[r, (k + 1) * nx:(k + 2) * nx] = ddef_dxk1
            J[r, uoff + k * nu:uoff + (k + 1) * nu] = ddef_duk
            J[r, uoff + (k + 1) * nu:uoff + (k + 2) * nu] = ddef_duk1
        return J

    # ---------------------------------------------------------- boundary equality

    def _boundary(self, z: np.ndarray) -> np.ndarray:
        """``x_0 - x0`` (always) stacked with ``x_{N-1} - x_goal`` (if set).

        Carried as an explicit linear equality rather than a zero-width variable
        bound: SLSQP's active-set LP goes degenerate when a box constraint is
        also active if the endpoints are pinned by equal lo/hi bounds.
        """
        nx, N = self.n_x, self.N
        b = z[:nx] - self.x0
        if self.x_goal is not None:
            b = np.concatenate([b, z[(N - 1) * nx:N * nx] - self.x_goal])
        return b

    def _boundary_jac(self, z: np.ndarray) -> np.ndarray:
        nx, N = self.n_x, self.N
        rows = nx * (1 if self.x_goal is None else 2)
        J = np.zeros((rows, self._nz))
        J[:nx, :nx] = np.eye(nx)
        if self.x_goal is not None:
            J[nx:2 * nx, (N - 1) * nx:N * nx] = np.eye(nx)
        return J

    # ------------------------------------------------------------ path inequality

    def _path(self, z: np.ndarray) -> np.ndarray:
        X, U = self._split(z)
        return np.atleast_1d(np.asarray(self.path_con(X, U), float)).ravel()

    # --------------------------------------------------------------- initial guess

    def _initial_guess(self, X_init, U_init) -> np.ndarray:
        if X_init is None:
            end = self.x0 if self.x_goal is None else self.x_goal
            s = np.linspace(0.0, 1.0, self.N)[:, None]
            X = (1.0 - s) * self.x0 + s * end
        else:
            X = np.asarray(X_init, float).reshape(self.N, self.n_x)
        U = np.zeros((self.N, self.n_u)) if U_init is None \
            else np.asarray(U_init, float).reshape(self.N, self.n_u)
        return self._merge(X, U)

    # ------------------------------------------------------------------- solve

    def solve(self, *, X_init=None, U_init=None, verbose: bool = False) -> CollocationResult:
        """Transcribe and solve the NLP; return the open-loop trajectory."""
        nx, nu, N = self.n_x, self.n_u, self.N
        z0 = self._initial_guess(X_init, U_init)

        # variable bounds: the state / input boxes only.  The endpoints enter as
        # explicit equality constraints (see _boundary), not zero-width bounds.
        lo = np.empty(self._nz)
        hi = np.empty(self._nz)
        lo[:N * nx] = np.tile(self._xbox[0], N)
        hi[:N * nx] = np.tile(self._xbox[1], N)
        lo[N * nx:] = np.tile(self._ubox[0], N)
        hi[N * nx:] = np.tile(self._ubox[1], N)
        bounds = [(None if not np.isfinite(a) else float(a),
                   None if not np.isfinite(b) else float(b))
                  for a, b in zip(lo, hi)]

        if self.method == "trust-constr":
            tc = [
                NonlinearConstraint(self._defects, 0.0, 0.0, jac=self._defects_jac),
                NonlinearConstraint(self._boundary, 0.0, 0.0, jac=self._boundary_jac),
            ]
            if self.path_con is not None:
                tc.append(NonlinearConstraint(lambda z: self._path(z), -np.inf, 0.0))
            res = minimize(
                self._objective, z0, jac=self._objective_grad, method="trust-constr",
                bounds=bounds, constraints=tc,
                options={"maxiter": self.max_iter, "gtol": self.tol,
                         "xtol": self.tol, "verbose": 2 if verbose else 0},
            )
        else:
            cons = [
                {"type": "eq", "fun": self._defects, "jac": self._defects_jac},
                {"type": "eq", "fun": self._boundary, "jac": self._boundary_jac},
            ]
            if self.path_con is not None:
                # scipy inequality convention is g(z) >= 0; our contract is g <= 0
                cons.append({"type": "ineq", "fun": lambda z: -self._path(z)})
            res = minimize(
                self._objective, z0, jac=self._objective_grad, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"maxiter": self.max_iter, "ftol": self.tol, "disp": verbose},
            )

        X, U = self._split(res.x)
        return CollocationResult(
            X=X.copy(),
            U=U.copy(),
            t=self.t.copy(),
            cost=float(res.fun),
            defect_norm=float(np.max(np.abs(self._defects(res.x)))) if N > 1 else 0.0,
            success=bool(res.success),
            nit=int(getattr(res, "nit", 0)),
            nfev=int(getattr(res, "nfev", 0)),
            message=str(res.message),
        )

    # ---------------------------------------------------------------- construct

    @classmethod
    def from_system(cls, system, t_final: float, N: int, **kw) -> "DirectCollocation":
        """Build from an :mod:`aimct.systems` model.

        Wraps ``system.dynamics(0.0, x, u)``.  The constraint Jacobian is
        finite-differenced by default: several systems override
        :meth:`~aimct.systems.DynamicalSystem.linearize` with a *fixed-point*
        analytic Jacobian that ignores its arguments (e.g. ``CartPole`` about
        the upright), which is wrong along a manoeuvring trajectory.  Pass
        ``linearize=g(x, u) -> (A, B)`` explicitly only if it is genuinely
        valid pointwise.
        """
        kw.setdefault("n_x", system.n_states)
        kw.setdefault("n_u", system.n_inputs)
        return cls(lambda x, u: np.asarray(system.dynamics(0.0, x, u), float),
                   t_final=t_final, N=N, **kw)
