"""System identification from trajectory data.

Two families:

* **Linear state-space** (:mod:`aimct.sysid.linear`) -- fit
  ``x_{k+1} = A_d x_k + B_d u_k`` from state/input rollouts, with an optional
  conversion back to continuous time.
* **Rigid-body manipulator** (:mod:`aimct.sysid.manipulator`) -- fit a two-link
  arm's base inertial parameters from a ``(q, dq, ddq, tau)`` motion log using
  the linear-in-parameters regressor, then materialise a
  :class:`aimct.systems.TwoLinkArm`.

The emphasis, as everywhere in this project, is on knowing *what the fitted
model represents physically* and *how wrong it is* -- see
:func:`prediction_error`, :func:`model_mismatch`, and
:class:`ManipulatorID`'s condition-number / validation-residual report.
"""

from .linear import (
    dmdc,
    least_squares_id,
    model_mismatch,
    prediction_error,
    to_continuous,
)
from .manipulator import (
    ManipulatorID,
    finite_difference_derivatives,
    identify_manipulator,
    manipulator_regressor,
)

__all__ = [
    "least_squares_id",
    "dmdc",
    "to_continuous",
    "prediction_error",
    "model_mismatch",
    "identify_manipulator",
    "manipulator_regressor",
    "finite_difference_derivatives",
    "ManipulatorID",
]
