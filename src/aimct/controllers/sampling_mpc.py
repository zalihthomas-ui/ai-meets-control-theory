r"""Sampling-based receding-horizon control (cross-entropy method).

Model-predictive control that needs only a *callable* one-step model
``step(x, u) -> x_next`` (vectorised over a batch) and a running cost -- no
linearisation, no QP. Each step it refines a Gaussian over action sequences by
keeping the elite (lowest-cost) samples (Rubinstein's CEM), applies the first
action, and warm-starts the next solve by shifting the mean.

Pairs naturally with :class:`aimct.ml.LearnedDynamics` for planning with a
learned model; use the analytic ``system.dynamics`` (wrapped to one RK4 step)
for the true-model reference.
"""

from __future__ import annotations

import numpy as np

from .base import Controller

__all__ = ["SamplingMPC"]


class SamplingMPC(Controller):
    """Cross-entropy-method planner as a :class:`Controller`.

    Parameters
    ----------
    step : ``f(X, U) -> X_next`` where ``X`` is ``(B, n)`` and ``U`` is ``(B, m)``.
    running_cost : ``g(X, U) -> (B,)`` per-step cost (lower is better).
    horizon : planning horizon ``H`` (steps).
    n_samples, n_elite, n_iter : CEM population, elite count, refinement iters.
    u_dim : action dimension ``m``.
    u_bounds : ``(low, high)`` scalar box on every action channel.
    terminal_cost : optional ``h(X) -> (B,)`` added once at the horizon end.
    init_std : initial per-channel action std (default: quarter of the box width).
    seed : RNG seed.
    """

    name = "SamplingMPC"

    def __init__(
        self,
        step,
        running_cost,
        *,
        horizon: int,
        n_samples: int = 256,
        n_elite: int = 32,
        n_iter: int = 4,
        u_dim: int = 1,
        u_bounds: tuple[float, float] = (-np.inf, np.inf),
        terminal_cost=None,
        init_std: float | None = None,
        seed: int = 0,
    ) -> None:
        self.step = step
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.H = int(horizon)
        self.n_samples = int(n_samples)
        self.n_elite = int(n_elite)
        self.n_iter = int(n_iter)
        self.m = int(u_dim)
        self.lo, self.hi = float(u_bounds[0]), float(u_bounds[1])
        if init_std is not None:
            self._init_std = float(init_std)
        elif np.isfinite(self.lo) and np.isfinite(self.hi):
            self._init_std = 0.25 * (self.hi - self.lo)
        else:
            self._init_std = 1.0
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.mu = np.zeros((self.H, self.m))
        self.std = np.full((self.H, self.m), self._init_std)
        self.last_cost = np.nan

    # ------------------------------------------------------------------ solve

    def _plan(self, x0: np.ndarray) -> np.ndarray:
        mu, std = self.mu.copy(), self.std.copy()
        B, H, m = self.n_samples, self.H, self.m
        for _ in range(self.n_iter):
            eps = self._rng.standard_normal((B, H, m))
            samples = np.clip(mu[None] + eps * std[None], self.lo, self.hi)

            X = np.tile(np.asarray(x0, dtype=float), (B, 1))
            cost = np.zeros(B)
            for h in range(H):
                Uh = samples[:, h, :]
                cost += np.asarray(self.running_cost(X, Uh), dtype=float)
                X = np.asarray(self.step(X, Uh), dtype=float)
            if self.terminal_cost is not None:
                cost += np.asarray(self.terminal_cost(X), dtype=float)

            elite = samples[np.argsort(cost)[: self.n_elite]]
            mu = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-6
            self.last_cost = float(np.sort(cost)[: self.n_elite].mean())
        self.mu, self.std = mu, std
        return mu[0].copy()

    def update(self, measurement, dt: float):
        x0 = np.atleast_1d(np.asarray(measurement, dtype=float))
        u0 = self._plan(x0)
        # warm start: shift the elite MEAN forward one step, but re-inflate the
        # search std every control step (a converged elite std ~1e-6 would
        # otherwise freeze the planner on its previous solution).
        self.mu = np.vstack([self.mu[1:], self.mu[-1]])
        self.std = np.full((self.H, self.m), self._init_std)
        return u0 if self.m > 1 else float(u0[0])
