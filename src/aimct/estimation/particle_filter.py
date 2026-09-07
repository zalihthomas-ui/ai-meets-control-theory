r"""Particle Filter (Sequential Monte Carlo) - Bootstrap / SIR filter.

For a nonlinear dynamical system

.. math::

    x_{k+1} = f(x_k, u_k) + w_k, \quad w_k \sim \mathcal{N}(0, Q), \qquad
    y_k     = h(x_k) + v_k,       \quad v_k \sim \mathcal{N}(0, R),

the Particle Filter represents the posterior distribution :math:`p(x_k \mid y_{1:k})`
as an empirical set of :math:`N_p` weighted samples (particles):

.. math::

    p(x_k \mid y_{1:k}) \approx \sum_{i=1}^{N_p} w_k^{(i)} \delta(x_k - x_k^{(i)}),
    \quad \sum_{i=1}^{N_p} w_k^{(i)} = 1.

Weight updates use Log-Sum-Exp normalization to prevent numerical floating-point
underflow, and adaptive systematic resampling is triggered when the Effective
Sample Size (ESS) drops below a threshold:

.. math::

    N_{\text{eff}} = \frac{1}{\sum_{i=1}^{N_p} (w_k^{(i)})^2} < \gamma \cdot N_p.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..simulate import rk4_step
from .ekf import step_map

__all__ = ["ParticleFilter", "PF", "systematic_resample"]


def systematic_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform systematic resampling on a weighted particle collection in O(N) time.

    Parameters
    ----------
    particles:
        Array of shape ``(N_p, n)``.
    weights:
        Normalized weights array of shape ``(N_p,)``.
    rng:
        Numpy random Generator.

    Returns
    -------
    new_particles, new_weights:
        Resampled particles of shape ``(N_p, n)`` with uniform weights ``1 / N_p``.
    """
    n_p = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # Prevent rounding truncation

    u1 = rng.uniform(0.0, 1.0 / n_p)
    u_pointers = u1 + np.arange(n_p) / float(n_p)

    indices = np.searchsorted(cumsum, u_pointers)
    indices = np.clip(indices, 0, n_p - 1)

    new_particles = particles[indices].copy()
    new_weights = np.full(n_p, 1.0 / float(n_p))
    return new_particles, new_weights


class ParticleFilter:
    r"""Sequential Importance Resampling (SIR) Bootstrap Particle Filter.

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
    n_particles:
        Number of particles :math:`N_p \ge 10`. Defaults to 500.
    dt:
        Sampling time interval in seconds.
    resample_thresh:
        Fraction of :math:`N_p` for adaptive resampling threshold :math:`\gamma \in (0, 1]`.
        Defaults to 0.5 (resample when :math:`ESS < 0.5 N_p`).
    n:
        State dimension. Inferred from ``Q`` if omitted.
    p:
        Measurement dimension. Inferred from ``R`` if omitted.
    m:
        Control input dimension. Inferred if omitted.
    discrete:
        If True, ``f`` is a discrete-time transition map; otherwise integrated
        with one RK4 step of ``dt``.
    residual:
        Residual function ``residual(y, y_pred) -> innovation`` (default ``y - y_pred``).
    x0:
        Initial state prior mean vector (shape ``(n,)``). Defaults to zeros.
    spread:
        Initial prior dispersion / covariance matrix for particle initialization.
        Defaults to ``P0`` if provided, else identity or ``Q``.
    particles:
        Optional explicit initial particle array (shape ``(n_particles, n)``).
    seed:
        Random seed or ``np.random.Generator`` for reproducible Monte Carlo runs.
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        n_particles: int = 500,
        *,
        dt: float,
        resample_thresh: float = 0.5,
        n: int | None = None,
        p: int | None = None,
        m: int | None = None,
        discrete: bool = False,
        residual: Callable | None = None,
        x0: Sequence[float] | np.ndarray | None = None,
        P0: Sequence[float] | np.ndarray | None = None,
        spread: Sequence[float] | np.ndarray | None = None,
        particles: np.ndarray | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self._f = f
        self._h = h
        self.dt = float(dt)
        self.n_particles = int(max(10, n_particles))
        self.resample_thresh = float(np.clip(resample_thresh, 0.01, 1.0))
        self.discrete = bool(discrete)

        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.n = int(n) if n is not None else self.Q.shape[0]
        self.p = int(p) if p is not None else self.R.shape[0]
        self.m = int(m) if m is not None else 1

        self._residual = residual or (lambda y, yp: np.asarray(y, float) - np.asarray(yp, float))

        if isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            self._rng = np.random.default_rng(seed)

        self._x0 = np.zeros(self.n) if x0 is None else np.asarray(x0, float).reshape(self.n)
        init_spread = spread if spread is not None else (P0 if P0 is not None else np.eye(self.n))
        self._spread = np.atleast_2d(np.asarray(init_spread, dtype=float))

        try:
            self._inv_R = np.linalg.inv(self.R)
        except np.linalg.LinAlgError:
            self._inv_R = np.linalg.pinv(self.R)

        self.particles: np.ndarray = np.empty((self.n_particles, self.n))
        self.weights: np.ndarray = np.empty(self.n_particles)
        self.x_hat: np.ndarray = self._x0.copy()
        self.P: np.ndarray = self._spread.copy()
        self.ess: float = float(self.n_particles)
        self.resample_count: int = 0

        self.reset(x0=self._x0, spread=self._spread, particles=particles)

    # ------------------------------------------------------------------ model

    def _transition(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return step_map(self._f, x, u, self.dt, self.discrete)

    # ------------------------------------------------------------------ state

    def reset(
        self,
        x0: np.ndarray | None = None,
        spread: np.ndarray | None = None,
        particles: np.ndarray | None = None,
    ) -> None:
        """Initialize or reset particle cloud and weights."""
        if particles is not None:
            self.particles = np.asarray(particles, dtype=float).copy()
            self.n_particles = len(self.particles)
        else:
            mean = self._x0 if x0 is None else np.asarray(x0, float).reshape(self.n)
            cov = self._spread if spread is None else np.atleast_2d(np.asarray(spread, dtype=float))
            self.particles = self._rng.multivariate_normal(mean, cov, size=self.n_particles)

        self.weights = np.full(self.n_particles, 1.0 / float(self.n_particles))
        self.ess = float(self.n_particles)
        self.resample_count = 0
        self._update_estimates()

    def _update_estimates(self) -> None:
        """Compute weighted mean and posterior covariance matrix."""
        self.x_hat = np.sum(self.weights[:, None] * self.particles, axis=0)
        dx = self.particles - self.x_hat
        self.P = (dx.T * self.weights) @ dx
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------ steps

    def predict(self, u: Any = None) -> np.ndarray:
        """Propagate particles forward through dynamics with process disturbance."""
        u_vec = np.zeros(self.m) if u is None else np.atleast_1d(np.asarray(u, dtype=float))
        
        # Sample process disturbances
        w_noise = self._rng.multivariate_normal(np.zeros(self.n), self.Q, size=self.n_particles)

        # Vectorized or iterative step_map propagation
        for i in range(self.n_particles):
            self.particles[i] = self._transition(self.particles[i], u_vec) + w_noise[i]

        self._update_estimates()
        return self.x_hat

    def update(self, y: Any) -> np.ndarray:
        """Evaluate measurement likelihoods, update weights via Log-Sum-Exp, and resample if needed."""
        y_vec = np.atleast_1d(np.asarray(y, dtype=float))

        # Evaluate predicted measurement for each particle
        residuals = np.empty((self.n_particles, self.p))
        for i in range(self.n_particles):
            y_pred = np.atleast_1d(np.asarray(self._h(self.particles[i]), dtype=float))
            residuals[i] = self._residual(y_vec, y_pred)

        # Log-likelihoods: log p(y | x_i) = -0.5 * residual^T R^-1 residual
        log_liks = -0.5 * np.sum(residuals @ self._inv_R * residuals, axis=1)

        # Add to prior log-weights
        log_weights = np.log(np.maximum(self.weights, 1e-300)) + log_liks

        # Normalized weights via Log-Sum-Exp
        max_log_w = np.max(log_weights)
        unnorm_w = np.exp(log_weights - max_log_w)
        sum_w = np.sum(unnorm_w)
        if sum_w <= 0.0 or not np.isfinite(sum_w):
            self.weights = np.full(self.n_particles, 1.0 / float(self.n_particles))
        else:
            self.weights = unnorm_w / sum_w

        # Compute Effective Sample Size
        self.ess = float(1.0 / np.sum(self.weights ** 2))

        # Adaptive Systematic Resampling
        thresh = self.resample_thresh * self.n_particles
        if self.ess < thresh:
            self.particles, self.weights = systematic_resample(
                self.particles, self.weights, self._rng
            )
            self.ess = float(self.n_particles)
            self.resample_count += 1

        self._update_estimates()
        return self.x_hat

    def step(self, y: Any, u: Any = None) -> np.ndarray:
        """Sequential predict(u) then update(y) step."""
        self.predict(u)
        return self.update(y)

    def resample(self) -> None:
        """Force manual systematic resampling of the current particle population."""
        self.particles, self.weights = systematic_resample(
            self.particles, self.weights, self._rng
        )
        self.ess = float(self.n_particles)
        self.resample_count += 1
        self._update_estimates()

    # ------------------------------------------------------------- factory

    @classmethod
    def from_system(
        cls,
        system: Any,
        Q: np.ndarray,
        R: np.ndarray,
        n_particles: int = 500,
        *,
        dt: float,
        h: Callable | None = None,
        **kwargs,
    ) -> ParticleFilter:
        """Construct ParticleFilter directly from an `aimct.systems.DynamicalSystem` instance."""
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
            n_particles=n_particles,
            dt=dt,
            n=n_dim,
            p=p_dim,
            m=m_dim,
            **kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ParticleFilter(n={self.n}, p={self.p}, n_particles={self.n_particles}, "
            f"dt={self.dt}, ess={self.ess:.1f}, resamples={self.resample_count})"
        )


# Convenient alias
PF = ParticleFilter
