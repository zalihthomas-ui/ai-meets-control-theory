"""Controllers implemented from scratch, then cross-checked against libraries.

Currently available
-------------------
``PID``   : classical proportional-integral-derivative output feedback.

Planned (see docs/TASKS.md)
---------------------------
``StateFeedback`` : pole-placement full-state feedback.
``LQR``           : infinite-horizon linear-quadratic regulator.
"""

from .base import Controller
from .pid import PID

__all__ = ["Controller", "PID"]
