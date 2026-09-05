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

from .adaptive import MRAC, GainScheduledLQR, solve_lyapunov
from .base import Controller
from .disturbance_observer import DisturbanceObserver, QFilter
from .ilqr import ILQR, iLQR, iLQRResult
from .lqr import LQR, solve_care
from .mpc import LinearMPC, dare
from .observer_feedback import ObserverFeedback
from .pid import PID
from .sampling_mpc import SamplingMPC
from .state_feedback import (
    StateFeedback,
    controllability_matrix,
    is_controllable,
    place_poles,
)
from .swingup import EnergyShapingSwingUp, HybridSwingUpLQR, wrap_angle

__all__ = [
    "Controller",
    "PID",
    "StateFeedback",
    "LQR",
    "ObserverFeedback",
    "DisturbanceObserver",
    "QFilter",
    "EnergyShapingSwingUp",
    "HybridSwingUpLQR",
    "wrap_angle",
    "LinearMPC",
    "SamplingMPC",
    "ILQR",
    "iLQR",
    "iLQRResult",
    "GainScheduledLQR",
    "MRAC",
    "solve_lyapunov",
    "dare",
    "place_poles",
    "solve_care",
    "controllability_matrix",
    "is_controllable",
]

