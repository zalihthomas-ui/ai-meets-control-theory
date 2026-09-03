r"""Output-feedback: a state observer in the loop with a static gain.

.. math::

    \hat x_k = \text{observer}(y_k,\ u_{k-1}),
    \qquad u_k = u_{\text{ref}} - K\,(\hat x_k - x_{\text{ref}}) .

The controller only ever sees the measurement ``y`` (whatever ``measurement_fn``
feeds it - a subset of states, noisy, ...); it reconstructs an estimate ``x_hat``
and applies the same law as :class:`StateFeedback` to that estimate.  With
``K`` from :class:`LQR` and the observer a steady-state Kalman filter this is
**LQG**; with pole-placement gains on both it is a Luenberger compensator.

Timing.  Each step the observer is advanced with the measurement just received
and the input applied over the *previous* interval (the first step uses
``u = 0``).  This one-step input delay is the honest closed-loop causality and is
what the separation principle assumes.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, Controller

__all__ = ["ObserverFeedback"]


class ObserverFeedback(Controller):
    """Wrap any state estimator + a feedback gain into a :class:`Controller`.

    Parameters
    ----------
    observer:
        A state estimator exposing ``x_hat`` and ``reset()``, plus either
        ``update(y, u, dt) -> x_hat`` (continuous - :class:`LuenbergerObserver`,
        :class:`KalmanFilter`) or ``step(y, u) -> x_hat``
        (:class:`DiscreteKalmanFilter`).
    K:
        Feedback gain of shape ``(n_u, n_x)``, or any object with a ``.K``
        attribute (an :class:`LQR` / :class:`StateFeedback` instance).
    x_ref, u_ref:
        Reference state and feed-forward input (default zeros); may be
        reassigned between steps.
    """

    def __init__(
        self,
        observer,
        K: ArrayLike,
        *,
        x_ref: ArrayLike | None = None,
        u_ref: ArrayLike | None = None,
    ) -> None:
        self.observer = observer
        K = K.K if hasattr(K, "K") else K            # accept an LQR / StateFeedback
        self.K = np.atleast_2d(np.asarray(K, dtype=float))
        self.n_u, self.n_x = self.K.shape
        if getattr(observer, "n", self.n_x) != self.n_x:
            raise ValueError(
                f"observer state dim {observer.n} != K columns {self.n_x}"
            )
        self.x_ref = (
            np.zeros(self.n_x) if x_ref is None
            else np.asarray(x_ref, dtype=float).reshape(self.n_x)
        )
        self.u_ref = (
            np.zeros(self.n_u) if u_ref is None
            else np.asarray(u_ref, dtype=float).reshape(self.n_u)
        )
        # DiscreteKalmanFilter drives with step(y, u); the continuous observers
        # with update(y, u, dt).
        self._discrete = hasattr(observer, "predict")
        self.reset()

    # ------------------------------------------------------------- constructors

    @classmethod
    def luenberger(cls, A, B, C, K, *, observer_poles, x_ref=None, u_ref=None):
        """Build a :class:`LuenbergerObserver` (poles placed at ``observer_poles``)
        and wrap it with gain ``K``."""
        from ..estimation import LuenbergerObserver

        obs = LuenbergerObserver(A, B, C, poles=observer_poles)
        return cls(obs, K, x_ref=x_ref, u_ref=u_ref)

    @classmethod
    def lqg(cls, A, B, C, Q, R, W, V, *, x_ref=None, u_ref=None):
        """LQG: :class:`LQR` gain from ``(A, B, Q, R)`` + steady-state
        :class:`KalmanFilter` from ``(A, C, W, V)``."""
        from ..estimation import KalmanFilter
        from .lqr import LQR

        K = LQR(A, B, Q, R).K
        kf = KalmanFilter(A, B, C, W, V)
        return cls(kf, K, x_ref=x_ref, u_ref=u_ref)

    # -------------------------------------------------------------------- step

    def reset(self) -> None:
        self.observer.reset()
        self._u_prev = np.zeros(self.n_u)
        self.x_hat = np.asarray(self.observer.x_hat, dtype=float).copy()
        self.output: ArrayLike = 0.0

    def update(self, measurement: ArrayLike, dt: float) -> ArrayLike:
        y = np.atleast_1d(np.asarray(measurement, dtype=float))
        if self._discrete:
            x_hat = np.asarray(self.observer.step(y, self._u_prev), dtype=float)
        else:
            x_hat = np.asarray(self.observer.update(y, self._u_prev, dt), dtype=float)

        u = self.u_ref - self.K @ (x_hat - self.x_ref)
        self._u_prev = u
        self.x_hat = x_hat
        self.output = float(u[0]) if self.n_u == 1 else u
        return self.output

    # ------------------------------------------------------------------- misc

    def estimation_error(self, x_true: ArrayLike) -> np.ndarray:
        """``x_true - x_hat`` for the most recent step."""
        return np.asarray(x_true, dtype=float).reshape(self.n_x) - self.x_hat

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ObserverFeedback(observer={type(self.observer).__name__}, "
                f"K={self.K.tolist()})")
