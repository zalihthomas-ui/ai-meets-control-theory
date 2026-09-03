"""Linear system identification from trajectory data.

Fit  x_{k+1} = A_d x_k + B_d u_k  (discrete) from measured state/input rollouts,
with an optional conversion back to continuous time. The emphasis, as everywhere
in this project, is on knowing *what the fitted model represents physically* and
*how wrong it is* -- see :func:`prediction_error` and :func:`model_mismatch`.
"""

from .linear import (
    dmdc,
    least_squares_id,
    model_mismatch,
    prediction_error,
    to_continuous,
)

__all__ = [
    "least_squares_id",
    "dmdc",
    "to_continuous",
    "prediction_error",
    "model_mismatch",
]
