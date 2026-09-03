"""Common controller interface.

Every controller in :mod:`aimct.controllers` is a callable state machine:

    u = controller.update(measurement, dt)      # advance one control step
    controller.reset()                          # clear internal state

The signature is deliberately small so the benchmark harness can drive any
controller through the same rollout loop.  Controllers that need the *full*
state vector (``StateFeedback``, ``LQR``) treat ``measurement`` as that vector;
output-feedback controllers (``PID``) treat it as the measured output(s).

This module intentionally has **no** dependency on :mod:`aimct.systems` or the
integrator — a controller only ever sees measurements and time steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

ArrayLike = Any  # float | np.ndarray; kept loose on purpose


class Controller(ABC):
    """Abstract base class for a discrete-time feedback controller.

    Subclasses implement :meth:`update` (the control law) and :meth:`reset`
    (restore the controller to its initial internal state).  ``__call__`` is a
    thin alias for :meth:`update` so a controller can be used directly as
    ``u = controller(y, dt)``.
    """

    #: Number of scalar control channels the controller produces.  ``None`` means
    #: "inferred from the first call" / not fixed.
    n_u: int | None = None

    @abstractmethod
    def update(self, measurement: ArrayLike, dt: float) -> ArrayLike:
        """Return the control signal for one step of length ``dt`` seconds."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state (integrators, stored past samples, ...)."""

    def __call__(self, measurement: ArrayLike, dt: float) -> ArrayLike:
        return self.update(measurement, dt)


def _clip(value: ArrayLike, low, high) -> ArrayLike:
    """``np.clip`` that tolerates ``None`` bounds on either side."""
    if low is None and high is None:
        return value
    return np.clip(value, -np.inf if low is None else low,
                   np.inf if high is None else high)
