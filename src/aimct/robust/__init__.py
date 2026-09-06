r"""Structured-uncertainty analysis: the structured singular value :math:`\mu`.

Where :mod:`aimct.controllers.hinf` shapes a loop for robustness against an
*unstructured* ball of dynamics (:math:`\|\Delta\|_\infty \le 1`, one full
block), :math:`\mu` analysis answers the sharper question: given that the
uncertainty has **known structure** -- a handful of independent parametric or
dynamic perturbations, each with its own bound -- exactly how much of it can the
loop absorb before it goes unstable, or before performance is lost?

The interconnection is pulled into the standard :math:`M`-:math:`\Delta` form
(``F_u(M, \Delta)``): ``M(s)`` is the known part, ``\Delta`` a block-diagonal
perturbation drawn from a structure :math:`\mathbf{\Delta}`.  Robust stability
holds for every ``\Delta`` in the (scaled) structure iff

.. math::

    \sup_\omega\; \mu_{\mathbf{\Delta}}\big(M(j\omega)\big) \;<\; 1,

and ``1 / \sup_\omega \mu`` is the *robust stability margin* -- the factor by
which every uncertainty bound can be inflated before stability is lost.

Contents
--------
``mu``
    Structured singular value of a complex matrix -- a lower bound by power
    iteration and an upper bound by D-scaling.
``robust_stability_margin`` / ``robust_performance_margin``
    Frequency sweeps of :math:`\mu` over an ``M``-grid, with the margin and the
    worst-case frequency.
``dk_iterate``
    A constant-D approximation of D-K iteration: alternate :math:`\mu` analysis
    (fit block D-scales) and :func:`~aimct.controllers.hinf.hinf_syn`.

References
----------
* J. Doyle, "Analysis of feedback systems with structured uncertainties",
  *IEE Proc. D* 129(6), 1982.
* A. Packard & J. Doyle, "The complex structured singular value",
  *Automatica* 29(1), 1993.
* Zhou, Doyle & Glover, *Robust and Optimal Control*, 1996, ch. 10-11.
"""

from .mu import (BlockStructure, dk_iterate, mu, robust_performance_margin,
                 robust_stability_margin)

__all__ = [
    "BlockStructure",
    "mu",
    "robust_stability_margin",
    "robust_performance_margin",
    "dk_iterate",
]
