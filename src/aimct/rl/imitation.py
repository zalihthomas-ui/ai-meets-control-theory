r"""Imitation learning -- behaviour cloning and DAgger, from scratch.

**Behaviour cloning (BC)** fits a policy to a fixed set of
``(observation, expert action)`` pairs by supervised regression. It is only as
good as the states in that set: once the learner's own mistakes take it
somewhere the expert never demonstrated, it has nothing to go on. Experiment
27 shows this concretely -- a perfect clone of an LQR lane-keeper drives off
the road the moment the tyre model changes under it.

**DAgger** (Ross, Gordon & Bagnell, 2011) is the fix: roll the *current
learner*, ask the expert what it would have done at every state the learner
actually visited, add those labels to the dataset, refit, repeat. The
training distribution is dragged toward the learner's own state distribution,
so the labels cover the states that matter.

    from aimct.rl.imitation import BehaviorCloning, dagger

    bc = BehaviorCloning(obs_dim=5, act_dim=2, act_low=lo, act_high=hi)
    bc.fit(obs, expert_actions)                    # plain BC

    bc = dagger(bc, rollout_states=roll, expert=lqr_u, observe=obs_of,
                iterations=5, steps_per_iter=4000)  # DAgger on top

``rollout_states(act_fn) -> (T, n_x)`` is caller-supplied -- a few lines around
whatever simulator the task uses -- so this module needs no ``gym`` env and
works for any plant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .policy_gradient import GaussianPolicy

__all__ = ["BehaviorCloning", "aggregate", "dagger"]


class BehaviorCloning:
    """A deterministic policy fit to ``(obs, action)`` pairs by MSE regression.

    Wraps a :class:`~aimct.rl.policy_gradient.GaussianPolicy`; only its mean
    (the MLP) is trained -- ``act`` returns that mean, clipped to the action
    box. ``log_std`` is left at its init value and is unused for control (it is
    kept so a BC policy can be handed to anything expecting a ``GaussianPolicy``).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden: tuple[int, ...] = (64, 64),
        act_low=-1.0,
        act_high=1.0,
        seed: int = 0,
    ) -> None:
        self.policy = GaussianPolicy(
            obs_dim, act_dim, hidden=hidden, act_low=act_low, act_high=act_high,
            log_std_init=-2.0, seed=seed,
        )
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.loss_history: list[float] = []

    # -- training --------------------------------------------------------

    def fit(
        self,
        obs,
        actions,
        *,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 3e-3,
        seed: int = 0,
        verbose: bool = False,
    ) -> list[float]:
        """Supervised regression of the policy mean onto ``actions``. Returns
        the per-epoch mean-squared-error history (also stored on
        ``.loss_history``)."""
        obs = np.atleast_2d(np.asarray(obs, dtype=float))
        actions = np.atleast_2d(np.asarray(actions, dtype=float))
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(
                f"obs and actions disagree on count: {obs.shape[0]} vs {actions.shape[0]}")
        self.loss_history = list(self.policy.net.fit(
            obs, actions, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
            verbose=verbose))
        return self.loss_history

    # -- use ----------------------------------------------------------

    def act(self, obs) -> np.ndarray:
        """Greedy action -- the policy mean, clipped to the action box."""
        return self.policy.greedy(np.asarray(obs, dtype=float))

    __call__ = act

    # -- persistence ------------------------------------------------------

    def save(self, path) -> Path:
        p = self.policy
        blob = {"act_low": p.act_low, "act_high": p.act_high,
                "n_layers": np.array(len(p.net.W)), "log_std": p.log_std}
        for i, (W, b) in enumerate(zip(p.net.W, p.net.b)):
            blob[f"W{i}"], blob[f"b{i}"] = W, b
        path = Path(path)
        np.savez(path, **blob)
        return path

    @classmethod
    def load(cls, path) -> "BehaviorCloning":
        d = np.load(Path(path))
        n = int(d["n_layers"])
        Ws = [d[f"W{i}"] for i in range(n)]
        bs = [d[f"b{i}"] for i in range(n)]
        hidden = tuple(W.shape[1] for W in Ws[:-1])
        bc = cls(Ws[0].shape[0], Ws[-1].shape[1], hidden=hidden,
                 act_low=d["act_low"], act_high=d["act_high"])
        bc.policy.net.W = [W.copy() for W in Ws]
        bc.policy.net.b = [b.copy() for b in bs]
        bc.policy.log_std = d["log_std"].copy()
        return bc


def aggregate(*datasets):
    """Concatenate ``(obs, actions)`` datasets; empty ones are skipped."""
    xs = [np.atleast_2d(np.asarray(o, float)) for o, _ in datasets if len(o)]
    us = [np.atleast_2d(np.asarray(a, float)) for _, a in datasets if len(a)]
    if not xs:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.vstack(xs), np.vstack(us)


def dagger(
    bc: BehaviorCloning,
    *,
    rollout_states: Callable[[Callable], np.ndarray],
    expert: Callable[[np.ndarray], np.ndarray],
    observe: Callable[[np.ndarray], np.ndarray],
    iterations: int = 5,
    steps_per_iter: int | None = None,
    beta_schedule: Callable[[int], float] | None = None,
    fit_kwargs: dict | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> BehaviorCloning:
    r"""DAgger on top of an already-BC-initialised ``bc``.

    Each iteration ``i``:

    1. roll a policy that mixes expert and learner -- action from the expert
       with probability :math:`\beta_i`, else from ``bc`` (``beta_schedule(i)``;
       default :math:`\beta_0 = 1`, :math:`\beta_{i\ge1} = 0` -- the common
       practical choice, "pure learner rollouts after the first"),
    2. label **every visited state** with ``expert(x)``,
    3. aggregate into the running dataset and refit ``bc`` on all of it.

    Parameters
    ----------
    rollout_states:
        ``rollout_states(act_fn) -> (T, n_x)`` -- run one episode driving the
        plant with ``act_fn(x) -> u`` and return the states visited. Caller
        supplies this (wrap your ``simulate``); it is the only plant coupling.
    expert:
        ``expert(x) -> u`` -- the demonstrator, queried at arbitrary states.
    observe:
        ``observe(x) -> obs`` -- the same state → observation map ``bc`` was
        trained on.
    iterations, steps_per_iter:
        DAgger outer loop count; ``steps_per_iter`` is informational only (the
        episode length is decided by ``rollout_states``).

    Returns the same ``bc``, refit in place; its ``.loss_history`` is the last
    fit's history and ``.dagger_datasets`` holds the per-iteration
    ``(obs, action)`` pairs for inspection.
    """
    rng = np.random.default_rng(seed)
    beta_schedule = beta_schedule or (lambda i: 1.0 if i == 0 else 0.0)
    fit_kwargs = dict(fit_kwargs or {})

    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    bc.dagger_datasets = datasets

    for i in range(iterations):
        beta = float(beta_schedule(i))

        def mixed(x, _beta=beta):
            u_e = np.asarray(expert(x), dtype=float)
            if _beta >= 1.0 or (_beta > 0.0 and rng.random() < _beta):
                return u_e
            return bc.act(observe(x))

        X = np.atleast_2d(np.asarray(rollout_states(mixed), dtype=float))
        obs_i = np.array([observe(x) for x in X])
        act_i = np.array([np.asarray(expert(x), dtype=float) for x in X])
        datasets.append((obs_i, act_i))

        obs_all, act_all = aggregate(*datasets)
        bc.fit(obs_all, act_all, seed=seed, **fit_kwargs)
        if verbose:
            print(f"  DAgger iter {i}  beta={beta:.2f}  "
                  f"states+={len(X)}  |D|={len(obs_all)}  "
                  f"fit_loss={bc.loss_history[-1]:.3e}")

    return bc
