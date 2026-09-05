"""Direct trajectory optimisation.

Where :mod:`aimct.controllers.ilqr` solves an optimal-control problem by
*indirect / single shooting* (a Riccati sweep over the true rollout) and
:class:`aimct.controllers.SamplingMPC` solves it by *sampling* (cross-entropy
over action sequences), this package solves it by *direct transcription*: the
continuous OCP is discretised into a finite-dimensional nonlinear program whose
decision variables are the state **and** input at every knot, and whose
equality constraints make the discretised dynamics hold.  A general-purpose NLP
solver (:func:`scipy.optimize.minimize`) then finds the trajectory.

``collocation`` : :class:`DirectCollocation` -- Hermite-Simpson collocation.
"""

from .collocation import CollocationResult, DirectCollocation

__all__ = ["DirectCollocation", "CollocationResult"]
