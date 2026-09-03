r"""Safety shield: run a (learned / RL) policy while a classical fallback keeps
the state inside a safe set.

.. math::

    u_k =
    \begin{cases}
        \pi_{\text{base}}(x_k) & \text{if } \operatorname{safe}(x_k) \\
        \pi_{\text{fallback}}(x_k) & \text{otherwise (switch)}
    \end{cases}

``safe`` is any predicate ``x -> bool`` - a state box
(:func:`box_predicate`) or a control-barrier-style margin ``h(x) >= 0``
(:func:`barrier_predicate`).  ``blend="filter"`` instead keeps the base action
whenever a one-step model prediction stays safe, and otherwise returns the
smallest deviation from it (toward the fallback action) that is predicted safe.

Every step is logged: :attr:`intervention_log` (bool per step, ``True`` when the
shield changed the base action) and :attr:`intervention_rate`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..controllers.base import ArrayLike, Controller

__all__ = ["ShieldedController", "box_predicate", "barrier_predicate"]


def box_predicate(low, high) -> Callable[[ArrayLike], bool]:
    """``x -> lo <= x <= hi`` (elementwise, ``+/-inf`` entries allowed)."""
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)

    def safe(x) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(np.all(x >= lo) and np.all(x <= hi))

    return safe


def barrier_predicate(h: Callable[[ArrayLike], float], margin: float = 0.0):
    """``x -> h(x) >= margin`` for a control-barrier-style scalar margin."""
    def safe(x) -> bool:
        return bool(float(h(np.asarray(x, dtype=float))) >= margin)

    return safe


def _step(controller, x, dt):
    if hasattr(controller, "update"):
        return controller.update(x, dt)
    return controller(x, dt)                     # bare callable(x, dt) -> u


class ShieldedController(Controller):
    """Wrap ``base`` with a safety ``fallback``.

    Parameters
    ----------
    base:
        The primary controller (an RL / learned policy).  ``Controller`` or a
        bare ``callable(x, dt) -> u``.
    fallback:
        The safety controller (typically ``LQR`` / ``StateFeedback``).
    is_safe:
        Predicate ``x -> bool``.  ``True`` -> the base controller drives.
    blend:
        ``"switch"`` (default): hand control entirely to ``fallback`` while
        unsafe.  ``"filter"``: keep the base action when a one-step prediction
        is safe, otherwise return the minimal deviation toward the fallback
        action that is predicted safe (needs ``predict``).
    predict:
        ``predict(x, u, dt) -> x_next`` for ``blend="filter"``.
    n_filter:
        Bisection steps for the minimal-deviation projection.
    keep_warm:
        Also step the *inactive* controller each call (discarding its output) so
        a stateful controller stays synchronised.  Off by default.
    """

    def __init__(
        self,
        base,
        fallback,
        *,
        is_safe: Callable[[ArrayLike], bool],
        blend: str = "switch",
        predict: Callable | None = None,
        n_filter: int = 12,
        keep_warm: bool = False,
    ) -> None:
        if blend not in ("switch", "filter"):
            raise ValueError("blend must be 'switch' or 'filter'")
        if blend == "filter" and predict is None:
            raise ValueError("blend='filter' needs a predict(x, u, dt) callable")
        self.base = base
        self.fallback = fallback
        self.is_safe = is_safe
        self.blend = blend
        self.predict = predict
        self.n_filter = int(n_filter)
        self.keep_warm = bool(keep_warm)
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        for c in (self.base, self.fallback):
            if hasattr(c, "reset"):
                c.reset()
        self.intervention_log: list[bool] = []
        self.mode: str = "base"
        self.output: ArrayLike = 0.0

    @property
    def intervention_rate(self) -> float:
        return float(np.mean(self.intervention_log)) if self.intervention_log else 0.0

    # ------------------------------------------------------------------- step

    def update(self, measurement: ArrayLike, dt: float) -> ArrayLike:
        x = np.atleast_1d(np.asarray(measurement, dtype=float))
        safe_now = bool(self.is_safe(x))

        if self.blend == "switch":
            if safe_now:
                u = np.atleast_1d(np.asarray(_step(self.base, x, dt), dtype=float))
                if self.keep_warm:
                    _step(self.fallback, x, dt)
                intervened = False
                self.mode = "base"
            else:
                u = np.atleast_1d(np.asarray(_step(self.fallback, x, dt), dtype=float))
                if self.keep_warm:
                    _step(self.base, x, dt)
                intervened = True
                self.mode = "fallback"
        else:  # "filter"
            u_base = np.atleast_1d(np.asarray(_step(self.base, x, dt), dtype=float))
            u_fb = np.atleast_1d(np.asarray(_step(self.fallback, x, dt), dtype=float))
            if self.is_safe(self.predict(x, u_base, dt)):
                u, intervened, self.mode = u_base, False, "base"
            else:
                u = self._project(x, u_base, u_fb, dt)
                intervened = not np.allclose(u, u_base)
                self.mode = "filter" if intervened else "base"

        self.intervention_log.append(intervened)
        self.output = float(u[0]) if u.size == 1 else u
        return self.output

    def _project(self, x, u_base, u_fb, dt):
        """Smallest move from ``u_base`` toward ``u_fb`` whose one-step
        prediction is safe (bisection).  Falls back to ``u_fb`` if even that is
        unsafe."""
        if not self.is_safe(self.predict(x, u_fb, dt)):
            return u_fb
        lo, hi = 0.0, 1.0                        # blend factor: 0 = base, 1 = fallback
        for _ in range(self.n_filter):
            mid = 0.5 * (lo + hi)
            u_mid = (1.0 - mid) * u_base + mid * u_fb
            if self.is_safe(self.predict(x, u_mid, dt)):
                hi = mid
            else:
                lo = mid
        return (1.0 - hi) * u_base + hi * u_fb

    def __repr__(self) -> str:  # pragma: no cover
        return (f"ShieldedController(blend={self.blend!r}, mode={self.mode!r}, "
                f"intervention_rate={self.intervention_rate:.2f})")
