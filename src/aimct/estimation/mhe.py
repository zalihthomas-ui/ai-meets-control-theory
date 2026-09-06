r"""Moving-Horizon Estimation (MHE) - constrained nonlinear optimization.

For a nonlinear dynamical system

.. math::

    x_{k+1} = f(x_k, u_k) + w_k, \quad w_k \sim \mathcal{N}(0, Q), \qquad
    y_k     = h(x_k) + v_k,       \quad v_k \sim \mathcal{N}(0, R),

subject to physical state constraints :math:`x_{\text{lo}} \le x \le x_{\text{hi}}`
and disturbance bounds :math:`w_{\text{lo}} \le w \le w_{\text{hi}}`,
Moving-Horizon Estimation (MHE) solves a constrained Maximum A Posteriori (MAP)
trajectory estimation problem over a sliding finite window of the :math:`N` most
recent measurements:

.. math::

    \min_{x_{k_0}, \{w_i\}_{i=k_0}^{k-1}} \; \frac{1}{2} \|x_{k_0} - \bar{x}_{k_0}\|_{P_{k_0}^{-1}}^2
    + \frac{1}{2} \sum_{i=k_0}^{k-1} \|w_i\|_{Q^{-1}}^2
    + \frac{1}{2} \sum_{i=k_0}^k \|y_i - h(x_i)\|_{R^{-1}}^2

subject to:

.. math::

    x_{i+1} = f(x_i, u_i) + w_i, \quad i = k_0, \dots, k-1 \\
    x_{\text{lo}} \le x_i \le x_{\text{hi}}, \quad i = k_0, \dots, k \\
    w_{\text{lo}} \le w_i \le w_{\text{hi}}, \quad i = k_0, \dots, k-1 \\
    g(x_i) \le 0

Arrival cost :math:`(\bar{x}_{k_0}, P_{k_0})` is propagated using an Extended
Kalman Filter (EKF) Riccati recursion at the trailing edge of the sliding window.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import scipy.optimize

from ..simulate import rk4_step
from .ekf import finite_diff_jacobian, step_map

__all__ = ["MovingHorizonEstimator", "MHE"]


class MovingHorizonEstimator:
    r"""Moving-Horizon Estimator (MHE) with hard state and disturbance constraints.

    Parameters
    ----------
    f:
        Continuous dynamics ``f(x, u) -> xdot`` (default) or discrete transition
        ``f(x, u) -> x_next`` (if ``discrete=True``).
    h:
        Measurement function ``h(x) -> y``.
    Q:
        Process noise covariance matrix (shape ``(n, n)``).
    R:
        Measurement noise covariance matrix (shape ``(p, p)``).
    horizon:
        Sliding window estimation horizon :math:`N \ge 1`. Defaults to 10.
    dt:
        Sampling time interval in seconds.
    n:
        State dimension. Inferred from ``Q`` if omitted.
    p:
        Measurement dimension. Inferred from ``R`` if omitted.
    m:
        Control input dimension. Inferred if omitted.
    discrete:
        If True, ``f`` is a discrete-time transition map; otherwise integrated
        with one RK4 step of ``dt``.
    x_min:
        Lower bound vector on state :math:`x` (shape ``(n,)``).
    x_max:
        Upper bound vector on state :math:`x` (shape ``(n,)``).
    w_min:
        Lower bound vector on process disturbance :math:`w` (shape ``(n,)``).
    w_max:
        Upper bound vector on process disturbance :math:`w` (shape ``(n,)``).
    constraints:
        List of inequality constraint functions ``g(x) <= 0``.
    x0:
        Initial state prior mean :math:`\bar{x}_0`. Defaults to zeros.
    P0:
        Initial arrival cost covariance matrix :math:`P_0`. Defaults to identity.
    arrival_cost_mode:
        Arrival cost propagation mode: ``'ekf'`` (default), ``'fixed'``, or ``'none'``.
    residual:
        Residual function ``residual(y, y_pred) -> innovation`` (default ``y - y_pred``).
    max_iter:
        Maximum optimizer iterations for `scipy.optimize.minimize`. Defaults to 100.
    tol:
        Optimizer tolerance. Defaults to 1e-6.
    method:
        Optimization algorithm (default ``'SLSQP'``).
    fd_eps:
        Finite-difference perturbation step for Jacobians.
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int = 10,
        *,
        dt: float,
        n: int | None = None,
        p: int | None = None,
        m: int | None = None,
        discrete: bool = False,
        x_min: Sequence[float] | np.ndarray | None = None,
        x_max: Sequence[float] | np.ndarray | None = None,
        w_min: Sequence[float] | np.ndarray | None = None,
        w_max: Sequence[float] | np.ndarray | None = None,
        constraints: Sequence[Callable[[np.ndarray], float | np.ndarray]] | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        arrival_cost_mode: str = "ekf",
        residual: Callable | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        method: str = "SLSQP",
        fd_eps: float = 1e-6,
    ) -> None:
        self._f = f
        self._h = h
        self.dt = float(dt)
        self.horizon = int(max(1, horizon))
        self.discrete = bool(discrete)

        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.n = int(n) if n is not None else self.Q.shape[0]
        self.p = int(p) if p is not None else self.R.shape[0]
        self.m = int(m) if m is not None else 1

        self.x_min = None if x_min is None else np.asarray(x_min, dtype=float).reshape(self.n)
        self.x_max = None if x_max is None else np.asarray(x_max, dtype=float).reshape(self.n)
        self.w_min = None if w_min is None else np.asarray(w_min, dtype=float).reshape(self.n)
        self.w_max = None if w_max is None else np.asarray(w_max, dtype=float).reshape(self.n)
        self.constraints = list(constraints) if constraints is not None else []

        self._x0_prior = np.zeros(self.n) if x0 is None else np.asarray(x0, float).reshape(self.n)
        self._P0_prior = np.eye(self.n) if P0 is None else np.atleast_2d(np.asarray(P0, float))

        mode = str(arrival_cost_mode).lower().strip()
        if mode not in {"ekf", "fixed", "none"}:
            raise ValueError(f"Unknown arrival_cost_mode '{arrival_cost_mode}'. Expected 'ekf', 'fixed', or 'none'.")
        self.arrival_cost_mode = mode

        self._residual = residual or (lambda y, yp: np.asarray(y, float) - np.asarray(yp, float))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.method = str(method)
        self._eps = float(fd_eps)

        self.reset()

    # ------------------------------------------------------------------ model

    def _transition(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return step_map(self._f, x, u, self.dt, self.discrete)

    # ------------------------------------------------------------------ state

    def reset(self, x0: np.ndarray | None = None, P0: np.ndarray | None = None) -> None:
        """Reset estimator state, covariance, and history buffers."""
        self.x_hat = (self._x0_prior if x0 is None
                      else np.asarray(x0, float).reshape(self.n)).copy()
        self.P = (self._P0_prior if P0 is None
                  else np.atleast_2d(np.asarray(P0, float))).copy()

        self._x_bar = self.x_hat.copy()
        self._P_bar = self.P.copy()

        self.history_y: list[np.ndarray] = []
        self.history_u: list[np.ndarray] = []
        self.history_x_opt: list[np.ndarray] = []
        self.history_w_opt: list[np.ndarray] = []

        self.trajectory: np.ndarray = self.x_hat.reshape(1, self.n).copy()
        self.disturbances: np.ndarray = np.zeros((0, self.n))
        self._last_sol_z: np.ndarray | None = None

    # ----------------------------------------------------------- EKF arrival

    def _update_arrival_cost(self, x_prev_opt: np.ndarray, u_prev: np.ndarray, y_prev: np.ndarray) -> None:
        """Propagate arrival cost prior (x_bar, P_bar) by one step using EKF Riccati update."""
        if self.arrival_cost_mode == "none":
            self._P_bar = np.eye(self.n) * 1e6
            return
        if self.arrival_cost_mode == "fixed":
            return

        # 1. EKF measurement update at k0 - 1
        H = finite_diff_jacobian(
            lambda xx: np.atleast_1d(np.asarray(self._h(xx), dtype=float)),
            x_prev_opt,
            self._eps,
        )
        P_prior = self._P_bar
        S = H @ P_prior @ H.T + self.R
        try:
            K = P_prior @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P_prior @ H.T @ np.linalg.pinv(S)

        IKH = np.eye(self.n) - K @ H
        P_post = IKH @ P_prior @ IKH.T + K @ self.R @ K.T
        P_post = 0.5 * (P_post + P_post.T)

        # 2. EKF time propagation to k0
        F = finite_diff_jacobian(
            lambda xx: self._transition(xx, u_prev),
            x_prev_opt,
            self._eps,
        )
        self._x_bar = self._transition(x_prev_opt, u_prev)
        P_next = F @ P_post @ F.T + self.Q
        self._P_bar = 0.5 * (P_next + P_next.T)

    # ------------------------------------------------------------- estimation

    def update(self, y: Any, u: Any = None, dt: float | None = None) -> np.ndarray:
        """Incorporate new measurement ``y`` and previous control input ``u``."""
        if dt is not None:
            self.dt = float(dt)

        y_k = np.atleast_1d(np.asarray(y, dtype=float))
        if u is not None:
            u_k = np.atleast_1d(np.asarray(u, dtype=float))
        else:
            u_k = np.zeros(self.m, dtype=float)

        # Record history
        if len(self.history_y) > 0:
            self.history_u.append(u_k)
        self.history_y.append(y_k)

        k = len(self.history_y) - 1
        window_len = min(k, self.horizon)
        k0 = k - window_len

        # If window shifted, advance arrival cost
        if k > self.horizon:
            idx_prev = k0 - 1
            x_prev_opt = self.history_x_opt[idx_prev]
            u_prev = self.history_u[idx_prev]
            y_prev = self.history_y[idx_prev]
            self._update_arrival_cost(x_prev_opt, u_prev, y_prev)

        # Solve MHE optimization over window [k0, k]
        y_seq = self.history_y[k0 : k + 1]
        u_seq = self.history_u[k0:k] if window_len > 0 else []

        x_hat_k, traj, dists, sol_z = self._solve_window(k0, y_seq, u_seq, window_len)

        # Cache results
        self._last_sol_z = sol_z
        self.trajectory = traj
        self.disturbances = dists
        self.x_hat = x_hat_k
        self.P = self._P_bar.copy()

        # Update history of optimal state at k0
        if len(self.history_x_opt) == k0:
            self.history_x_opt.append(traj[0].copy())
        elif len(self.history_x_opt) > k0:
            self.history_x_opt[k0] = traj[0].copy()

        return self.x_hat

    def step(self, y: Any, u: Any = None) -> np.ndarray:
        """Alias for :meth:`update` to match standard filter interface."""
        return self.update(y, u)

    def predict(self, u: Any = None) -> np.ndarray:
        """One-step forward open-loop state prediction from current estimate."""
        u_vec = np.zeros(self.m) if u is None else np.atleast_1d(np.asarray(u, dtype=float))
        return self._transition(self.x_hat, u_vec)

    # ------------------------------------------------------------- optimizer

    def _solve_window(
        self,
        k0: int,
        y_seq: list[np.ndarray],
        u_seq: list[np.ndarray],
        M: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Formulate and solve constrained NLP over window of length M."""
        n = self.n

        # Decision variables: z = [x_k0 (n), w_k0 (n), ..., w_{k-1} (n)]
        z_dim = n + M * n

        # Initial guess z0 with warm start
        z0 = np.zeros(z_dim)
        if self._last_sol_z is not None and len(self._last_sol_z) == z_dim:
            z0[:] = self._last_sol_z
        elif self._last_sol_z is not None and len(self._last_sol_z) == (z_dim - n) and M > 0:
            # Shift previous solution
            z0[:n] = self._x_bar
            z0[n : len(self._last_sol_z)] = self._last_sol_z[n:]
        else:
            z0[:n] = self._x_bar

        # Apply bounds to initial guess
        if self.x_min is not None:
            z0[:n] = np.maximum(z0[:n], self.x_min)
        if self.x_max is not None:
            z0[:n] = np.minimum(z0[:n], self.x_max)

        # Variable bounds
        bounds: list[tuple[float | None, float | None]] = []
        for i in range(n):
            lo = None if self.x_min is None else float(self.x_min[i])
            hi = None if self.x_max is None else float(self.x_max[i])
            bounds.append((lo, hi))
        for j in range(M):
            for i in range(n):
                lo = None if self.w_min is None else float(self.w_min[i])
                hi = None if self.w_max is None else float(self.w_max[i])
                bounds.append((lo, hi))

        # Inverses for quadratic forms
        try:
            inv_P = np.linalg.inv(self._P_bar)
        except np.linalg.LinAlgError:
            inv_P = np.linalg.pinv(self._P_bar)

        try:
            inv_Q = np.linalg.inv(self.Q)
        except np.linalg.LinAlgError:
            inv_Q = np.linalg.pinv(self.Q)

        try:
            inv_R = np.linalg.inv(self.R)
        except np.linalg.LinAlgError:
            inv_R = np.linalg.pinv(self.R)

        x_bar_k0 = self._x_bar

        def unpack_trajectory(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_traj = np.empty((M + 1, n))
            w_seq = z[n:].reshape((M, n)) if M > 0 else np.empty((0, n))
            x_curr = z[:n]
            x_traj[0] = x_curr
            for j in range(M):
                w_j = w_seq[j]
                x_curr = self._transition(x_curr, u_seq[j]) + w_j
                x_traj[j + 1] = x_curr
            return x_traj, w_seq

        def objective(z: np.ndarray) -> float:
            x_traj, w_seq = unpack_trajectory(z)
            # Arrival cost
            dx0 = x_traj[0] - x_bar_k0
            cost = 0.5 * float(dx0.T @ inv_P @ dx0)

            # Disturbance cost
            for j in range(M):
                wj = w_seq[j]
                cost += 0.5 * float(wj.T @ inv_Q @ wj)

            # Measurement cost
            for j in range(M + 1):
                y_pred = np.atleast_1d(np.asarray(self._h(x_traj[j]), dtype=float))
                res = np.asarray(self._residual(y_seq[j], y_pred), dtype=float)
                cost += 0.5 * float(res.T @ inv_R @ res)

            return cost

        # Vectorized constraint functions for high-speed evaluation
        constraints_list: list[dict[str, Any]] = []

        if M > 0 and (self.x_min is not None or self.x_max is not None):
            x_min_vec = self.x_min
            x_max_vec = self.x_max

            def state_bounds_con(z: np.ndarray) -> np.ndarray:
                x_traj, _ = unpack_trajectory(z)
                x_sub = x_traj[1:]  # shape (M, n)
                res_list: list[np.ndarray] = []
                if x_min_vec is not None:
                    res_list.append((x_sub - x_min_vec).ravel())
                if x_max_vec is not None:
                    res_list.append((x_max_vec - x_sub).ravel())
                return np.concatenate(res_list)

            constraints_list.append({"type": "ineq", "fun": state_bounds_con})

        # User-supplied constraints g(x) <= 0  ==> -g(x) >= 0
        if self.constraints:
            for g_fn in self.constraints:
                def user_con(z, fn=g_fn) -> np.ndarray:
                    x_traj, _ = unpack_trajectory(z)
                    vals = [np.atleast_1d(np.asarray(fn(x_traj[j]), dtype=float)) for j in range(M + 1)]
                    return -np.concatenate(vals)

                constraints_list.append({"type": "ineq", "fun": user_con})

        # Solve NLP
        res = scipy.optimize.minimize(
            objective,
            z0,
            method=self.method,
            bounds=bounds,
            constraints=constraints_list,
            tol=self.tol,
            options={"maxiter": self.max_iter, "disp": False},
        )

        z_sol = res.x if (res.success or np.all(np.isfinite(res.x))) else z0
        x_traj_opt, w_seq_opt = unpack_trajectory(z_sol)

        # Enforce hard clipping on output estimate for absolute safety
        if self.x_min is not None:
            x_traj_opt = np.maximum(x_traj_opt, self.x_min)
        if self.x_max is not None:
            x_traj_opt = np.minimum(x_traj_opt, self.x_max)

        x_hat_current = x_traj_opt[-1].copy()
        return x_hat_current, x_traj_opt, w_seq_opt, z_sol

    # ------------------------------------------------------------- factory

    @classmethod
    def from_system(
        cls,
        system: Any,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int = 10,
        *,
        dt: float,
        h: Callable | None = None,
        x_min: Sequence[float] | np.ndarray | None = None,
        x_max: Sequence[float] | np.ndarray | None = None,
        **kwargs,
    ) -> MovingHorizonEstimator:
        """Construct MHE directly from an `aimct.systems.DynamicalSystem` instance."""
        f = lambda x, u: system.dynamics(0.0, x, u)
        if h is None:
            p_dim = getattr(system, "n_outputs", getattr(system, "n_states", 2))
            h = lambda x: np.asarray(x, dtype=float)[:p_dim]

        n_dim = getattr(system, "n_states", Q.shape[0])
        p_dim = getattr(system, "n_outputs", R.shape[0])
        m_dim = getattr(system, "n_inputs", 1)

        return cls(
            f=f,
            h=h,
            Q=Q,
            R=R,
            horizon=horizon,
            dt=dt,
            n=n_dim,
            p=p_dim,
            m=m_dim,
            x_min=x_min,
            x_max=x_max,
            **kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MovingHorizonEstimator(n={self.n}, p={self.p}, horizon={self.horizon}, "
            f"dt={self.dt}, method='{self.method}', arrival_cost='{self.arrival_cost_mode}')"
        )


# Convenient alias
MHE = MovingHorizonEstimator
