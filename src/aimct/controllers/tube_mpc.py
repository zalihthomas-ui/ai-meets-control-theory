r"""Tube MPC -- robust constrained control against a bounded disturbance.

Nominal :class:`LinearMPC` assumes the model is exact.  Under a persistent
additive disturbance ``x^+ = A_d x + B_d u + w``, ``w`` in a box ``W``, its
state-box constraints are violated whenever the disturbance pushes the state
past the boundary.  **Tube MPC** (Mayne, Seron & Rakovic 2005) fixes this:

* an *ancillary* feedback ``u = u_nom + K (x - x_nom)`` keeps the true state
  inside a **tube** ``x_nom + Z`` around a nominal trajectory, where ``Z`` is a
  *robust positively invariant* (RPI) set for ``A_K = A_d + B_d K`` and ``W``
  (``A_K Z \oplus W \subseteq Z``);
* the nominal MPC is solved on **tightened** constraints -- the state box shrunk
  by ``Z`` and the input box shrunk by ``K Z`` -- so that *nominal* feasibility
  implies *robust* constraint satisfaction for the true state and input.

The mРPI set is outer-approximated here by its **interval hull** (a box), the
one representation :class:`LinearMPC` consumes directly.  For a symmetric box
``W = {|w| <= \hat w}`` the support of ``A_K^i W`` on axis ``j`` is
``(|A_K^i|\hat w)_j``, so the interval hull of
``F_\infty = \bigoplus_{i\ge 0} A_K^i W`` has half-widths

.. math::

    \hat z \;=\; \Big(\textstyle\sum_{i \ge 0} |A_K^i|\Big)\,\hat w ,

a series that converges for any Schur ``A_K`` (summed until ``\|A_K^k\|_\infty``
is negligible, with a geometric-rate tail bound).  It is conservative -- the
exact mРPI is a tighter polytope -- but purely from-scratch, works for a
disturbance on any subset of the channels, and its width scales *linearly*
with ``\hat w``, so it vanishes as ``W`` shrinks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mpc import LinearMPC, _discretize, dare  # noqa: F401

__all__ = ["TubeMPC", "MRPISet", "mrpi_box"]


@dataclass
class MRPISet:
    """Box that contains the minimal robust positively invariant (mРPI) set for
    the Schur map ``A_K`` and the symmetric disturbance box ``W``.

    ``half_widths`` are the axis half-extents of the interval hull of
    :math:`F_\\infty = \\bigoplus_{i \\ge 0} A_K^i W`, i.e.
    :math:`\\big(\\sum_{i \\ge 0} |A_K^i|\\big)\\hat w`.  It contains
    :math:`F_\\infty` by construction, so tightening constraints by this box is
    robustly sound (conservatively -- the exact mРPI is a tighter polytope).
    ``rpi_residual > 0`` only means the *box hull* is not itself invariant,
    which is expected for a position/velocity plant and does not affect
    soundness.
    """

    half_widths: np.ndarray     # (n,)  Z = {|x| <= half_widths}
    A_K: np.ndarray             # (n, n) closed-loop map used
    n_terms: int                # series terms summed
    alpha: float                # observed per-step contraction ||A_K^k||_inf ** (1/k)
    rpi_residual: float         # max_j (|A_K| z + w - z)_j  (informational)
    contains_mrpi: bool = True

    def contains(self, x) -> bool:
        x = np.abs(np.atleast_1d(np.asarray(x, float)))
        return bool(np.all(x <= self.half_widths + 1e-9))


def _box_halfwidths(W, n: int) -> np.ndarray:
    """Coerce a disturbance-set spec to symmetric box half-widths ``(n,)``.

    Accepts an ``(n,)`` half-width array, a scalar, or a ``(lo, hi)`` pair
    (its half-extent ``(hi - lo) / 2`` is used).
    """
    if isinstance(W, (tuple, list)) and len(W) == 2 and np.ndim(W[0]) <= 1:
        lo = np.broadcast_to(np.asarray(W[0], float), (n,))
        hi = np.broadcast_to(np.asarray(W[1], float), (n,))
        hw = 0.5 * (hi - lo)
    else:
        hw = np.broadcast_to(np.abs(np.asarray(W, float)), (n,)).astype(float)
    if np.any(hw < 0):
        raise ValueError("disturbance half-widths must be non-negative")
    return hw.copy()


def mrpi_box(A_K, W, *, term_tol: float = 1e-12, max_terms: int = 5000) -> MRPISet:
    r"""Box that contains the minimal robust positively invariant set for the
    Schur map ``A_K`` and the symmetric disturbance box ``W``.

    ``F_\infty = \bigoplus_{i=0}^{\infty} A_K^i W`` is bounded for a Schur
    ``A_K``; for a box ``W = \{|w| \le \hat w\}`` the support of ``A_K^i W`` on
    axis ``j`` is ``(|A_K^i|\,\hat w)_j``, so the interval hull of
    ``F_\infty`` has half-widths ``\hat z = \big(\sum_{i\ge 0}|A_K^i|\big)\hat
    w``.  The series is summed until ``\|A_K^k\|_\infty`` drops below
    ``term_tol`` (a geometric-rate bound covers the negligible tail), so this
    works for **any** ``W`` -- including one that is degenerate on some axis
    (a disturbance on only a few channels).  ``alpha`` reports the observed
    per-step contraction ``\|A_K^k\|_\infty^{1/k}``.
    """
    A_K = np.atleast_2d(np.asarray(A_K, float))
    n = A_K.shape[0]
    w_hw = _box_halfwidths(W, n)
    rho = float(np.max(np.abs(np.linalg.eigvals(A_K))))
    if rho >= 1.0 - 1e-9:
        raise ValueError(f"A_K must be Schur; spectral radius {rho:.4f}")
    if not np.any(w_hw > 0):
        return MRPISet(np.zeros(n), A_K, 0, rho, 0.0)

    partial = np.zeros((n, n))
    Ai = np.eye(n)
    s, last = max_terms, 1.0
    for k in range(1, max_terms + 1):
        partial = partial + np.abs(Ai)
        Ai = Ai @ A_K
        last = float(np.max(np.sum(np.abs(Ai), axis=1)))   # ||A_K^k||_inf
        if last < term_tol:
            s = k
            break
    # geometric-rate tail bound:  sum_{i>=s} ||A_K^i|| <= last / (1 - r),  r ~ rho
    r = min(max(rho, 0.0), 0.999)
    tail = (last / (1.0 - r)) * float(np.max(w_hw))
    hw = partial @ w_hw + tail
    resid = float(np.max(np.abs(A_K) @ hw + w_hw - hw))
    alpha = last ** (1.0 / max(s, 1))
    return MRPISet(hw, A_K, s, alpha, resid)


class TubeMPC(LinearMPC):
    r"""Robust MPC with a disturbance-invariant tube.

    All of :class:`LinearMPC`'s parameters, plus:

    Parameters
    ----------
    W : array (n,) / scalar / ``(lo, hi)``
        The additive process-disturbance set ``x^+ = A_d x + B_d u + w``,
        ``w`` in this (symmetric) box.
    K : array (m, n) or None
        The ancillary feedback (``u = u_nom - K (x - x_nom)``).  ``None`` (the
        default) uses the discrete-LQR gain, so the tube shrinks the RPI set as
        much as the ``(Q, R)`` weights allow.
    mrpi_term_tol, mrpi_max_terms :
        Passed to :func:`mrpi_box`.

    Notes
    -----
    ``x_bounds`` / ``u_bounds`` are the constraints on the **true** state and
    input; the nominal problem is solved on their tightening by ``Z`` and
    ``K Z``.  The tube quantities are built lazily on the first :meth:`update`
    (they depend on the control period ``dt``) and exposed as :attr:`mrpi`,
    :attr:`tightened_x_bounds`, :attr:`tightened_u_bounds`.
    """

    def __init__(self, A, B, *, W, K=None, mrpi_term_tol: float = 1e-12,
                 mrpi_max_terms: int = 5000, **linmpc_kw) -> None:
        super().__init__(A, B, **linmpc_kw)
        self._W = _box_halfwidths(W, self.n)
        self._K_user = None if K is None else np.atleast_2d(np.asarray(K, float))
        self._mrpi_term_tol = float(mrpi_term_tol)
        self._mrpi_max_terms = int(mrpi_max_terms)
        self._tube_cache: dict = {}
        self.mrpi: MRPISet | None = None
        self.tightened_x_bounds: tuple | None = None
        self.tightened_u_bounds: tuple | None = None
        self.x_nom: np.ndarray | None = None

    # -- tube build (per dt) ------------------------------------------------
    def _tube_build(self, dt: float):
        key = round(float(dt), 12)
        if key in self._tube_cache:
            return self._tube_cache[key]

        Ad, Bd = _discretize(self.A, self.B, dt)
        if self._K_user is not None:
            K = self._K_user
        else:
            Qf = dare(Ad, Bd, self.Q, self.R)
            S = self.R + Bd.T @ Qf @ Bd
            K = -np.linalg.solve(S, Bd.T @ Qf @ Ad)          # u = +K x  (K is -Kd)
        A_K = Ad + Bd @ K
        Z = mrpi_box(A_K, self._W, term_tol=self._mrpi_term_tol,
                     max_terms=self._mrpi_max_terms)

        kz = np.abs(K) @ Z.half_widths                        # box hull of K Z
        x_lo_t = self._x_lo + Z.half_widths
        x_hi_t = self._x_hi - Z.half_widths
        u_lo_t = self._u_lo + kz
        u_hi_t = self._u_hi - kz
        fin_x = np.isfinite(self._x_lo) | np.isfinite(self._x_hi)
        fin_u = np.isfinite(self._u_lo) | np.isfinite(self._u_hi)
        if np.any(fin_x & (x_lo_t > x_hi_t + 1e-12)) or \
           np.any(fin_u & (u_lo_t > u_hi_t + 1e-12)):
            raise ValueError(
                "disturbance set too large: tightening by the RPI tube leaves "
                "an empty nominal constraint set")

        self.mrpi = Z
        self.tightened_x_bounds = (x_lo_t, x_hi_t)
        self.tightened_u_bounds = (u_lo_t, u_hi_t)
        built = dict(Ad=Ad, Bd=Bd, K=K,
                     x_lo_t=x_lo_t, x_hi_t=x_hi_t, u_lo_t=u_lo_t, u_hi_t=u_hi_t)
        self._tube_cache[key] = built
        return built

    def reset(self) -> None:
        super().reset()
        self.x_nom = None

    # -- step ------------------------------------------------------------
    def update(self, measurement, dt: float):
        x = np.atleast_1d(np.asarray(measurement, float)).reshape(self.n)
        tb = self._tube_build(dt)
        if self.x_nom is None:
            self.x_nom = x.copy()

        # nominal MPC solve on the tightened boxes, seeded from x_nom
        saved = (self._x_lo, self._x_hi, self._u_lo, self._u_hi, self._cache)
        self._x_lo, self._x_hi = tb["x_lo_t"], tb["x_hi_t"]
        self._u_lo, self._u_hi = tb["u_lo_t"], tb["u_hi_t"]
        self._cache = {}                                      # tightened rebuild
        try:
            u_nom0 = np.atleast_1d(np.asarray(
                super().update(self.x_nom, dt), float)).reshape(self.m)
            plan = self.horizon_plan
        finally:
            (self._x_lo, self._x_hi, self._u_lo, self._u_hi,
             self._cache) = saved

        # ancillary feedback + propagate the nominal state
        u = u_nom0 + tb["K"] @ (x - self.x_nom)
        self.x_nom = tb["Ad"] @ self.x_nom + tb["Bd"] @ u_nom0

        u = np.clip(u, self._u_lo, self._u_hi)               # hard true-input box
        self.horizon_plan = plan
        self.output = float(u[0]) if self.m == 1 else u
        return self.output

    # -- construct -------------------------------------------------------
    @classmethod
    def from_system(cls, system, *, W, dt=None, **kw) -> "TubeMPC":
        """Build from an :mod:`aimct.systems` model, linearising about the
        origin (or ``kw['x_ref']`` / ``kw['u_ref']`` if given)."""
        xe = np.zeros(system.n_states)
        ue = np.zeros(system.n_inputs)
        A, B = system.linearize(xe, ue)
        return cls(A, B, W=W, **kw)
