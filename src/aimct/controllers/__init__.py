"""Controllers implemented from scratch, then cross-checked against libraries.

Available
---------
``PID``           : classical proportional-integral-derivative output feedback.
``StateFeedback`` : static full-state feedback ``u = u_ref - K (x - x_ref)``,
                    with single-input Ackermann pole placement
                    (:func:`place_poles` / :meth:`StateFeedback.from_poles`).
``LQR``           : infinite-horizon continuous-time linear-quadratic regulator;
                    CARE solved from scratch via the Hamiltonian stable
                    eigenspace (:func:`solve_care`).
"""

from .base import Controller
from .lqr import LQR, solve_care
from .observer_feedback import ObserverFeedback
from .pid import PID
from .state_feedback import (
    StateFeedback,
    controllability_matrix,
    is_controllable,
    place_poles,
)

__all__ = [
    "Controller",
    "PID",
    "StateFeedback",
    "LQR",
    "ObserverFeedback",
    "place_poles",
    "solve_care",
    "controllability_matrix",
    "is_controllable",
]
