"""Machine learning for dynamical systems (from-scratch NumPy).

- :class:`MLP` - fully-connected net with hand-written backprop + Adam.
- :class:`LearnedDynamics` - residual MLP one-step model ``x_{k+1} = x_k + f(x_k, u_k)``
  with standardisation, for model-predictive *planning* and prediction-error study.

See also :class:`aimct.controllers.SamplingMPC` (CEM planner over a learned model).
"""

from .dynamics import LearnedDynamics
from .mlp import MLP
from .planning import batched_rk4, system_step

__all__ = ["MLP", "LearnedDynamics", "system_step", "batched_rk4"]
