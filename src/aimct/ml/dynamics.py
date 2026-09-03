r"""Learned one-step dynamics model  x_{k+1} = x_k + f_theta(x_k, u_k).

A residual MLP (predict the *increment*, which is small and well-scaled) with
input/output standardisation baked in. Trained on trajectory rollouts; used for
model-predictive *planning* (see :class:`aimct.controllers.SamplingMPC`) and for
open-loop prediction-error evaluation.
"""

from __future__ import annotations

import numpy as np

from .mlp import MLP

__all__ = ["LearnedDynamics"]


class LearnedDynamics:
    def __init__(
        self,
        n_states: int,
        n_inputs: int,
        *,
        hidden=(64, 64),
        activation: str = "tanh",
        residual: bool = True,
        seed: int = 0,
    ) -> None:
        self.n_states = int(n_states)
        self.n_inputs = int(n_inputs)
        self.residual = bool(residual)
        self.net = MLP([n_states + n_inputs, *hidden, n_states],
                       activation=activation, seed=seed)
        self._xu_mu = np.zeros(n_states + n_inputs)
        self._xu_sd = np.ones(n_states + n_inputs)
        self._tg_mu = np.zeros(n_states)
        self._tg_sd = np.ones(n_states)
        self.fitted = False

    # ------------------------------------------------------------------ data

    @staticmethod
    def _pairs(X, U):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U[:, None]
        T = X.shape[0] - 1
        XU = np.hstack([X[:-1], U[:T]])
        return XU, X[1:], X[:-1]

    def fit(self, X, U, *, epochs: int = 500, lr: float = 5e-3,
            batch_size: int = 256, seed: int = 0, verbose: bool = False):
        """Fit on a rollout ``X`` ``(T+1, n)``, ``U`` ``(T, m)``. To use several
        trajectories, concatenate their ``(x_k, u_k, x_{k+1})`` triples upstream.
        """
        XU, Xnext, Xcur = self._pairs(X, U)
        target = (Xnext - Xcur) if self.residual else Xnext

        self._xu_mu, self._xu_sd = XU.mean(0), XU.std(0) + 1e-8
        self._tg_mu, self._tg_sd = target.mean(0), target.std(0) + 1e-8
        Xn = (XU - self._xu_mu) / self._xu_sd
        Yn = (target - self._tg_mu) / self._tg_sd

        hist = self.net.fit(Xn, Yn, epochs=epochs, lr=lr, batch_size=batch_size,
                            seed=seed, verbose=verbose)
        self.fitted = True
        return hist

    # ------------------------------------------------------------------ use

    def step(self, x, u):
        """One-step prediction. Accepts single vectors or batches (``(N, n)``)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        u = np.atleast_2d(np.asarray(u, dtype=float))
        xu = np.hstack([x, u])
        out_n = self.net((xu - self._xu_mu) / self._xu_sd)
        out = out_n * self._tg_sd + self._tg_mu
        nxt = x + out if self.residual else out
        return nxt[0] if nxt.shape[0] == 1 else nxt

    def rollout(self, x0, U):
        """Open-loop prediction for an input sequence ``U`` ``(H, m)``."""
        x = np.asarray(x0, dtype=float)
        xs = [x]
        for u in np.atleast_2d(U):
            x = self.step(x, u)
            xs.append(np.asarray(x))
        return np.array(xs)

    def prediction_error(self, X, U, *, horizon: int = 1) -> float:
        """RMS ``h``-step open-loop state error on a held-out rollout."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        U = np.asarray(U, dtype=float)
        if U.ndim == 1:
            U = U[:, None]
        T = X.shape[0] - 1
        errs = []
        for k in range(T - horizon + 1):
            x = X[k].copy()
            for j in range(horizon):
                x = self.step(x, U[k + j])
            errs.append(x - X[k + horizon])
        return float(np.sqrt(np.mean(np.asarray(errs) ** 2)))
