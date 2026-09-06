r"""H-infinity mixed-sensitivity (S / KS / T) loop shaping -- from scratch.

Where :class:`LQR` minimises a *quadratic integral* of the state and input,
H-infinity control minimises the **worst-case gain** (the :math:`H_\infty` norm,
i.e. the peak over frequency of the largest singular value) of a chosen set of
closed-loop transfer functions.  For *mixed-sensitivity* design those are the
weighted sensitivity, control-sensitivity and complementary-sensitivity:

.. math::

    \min_{K \text{ stabilising}} \;
    \left\| \begin{bmatrix} W_S\, S \\ W_{KS}\, K S \\ W_T\, T \end{bmatrix}
    \right\|_\infty,
    \qquad
    S = (I + G K)^{-1},\; T = G K (I + G K)^{-1},\; KS = K S .

Shaping the three weights trades off, across frequency: reference tracking and
output-disturbance rejection (:math:`W_S` large at low frequency), control
effort and actuator bandwidth (:math:`W_{KS}`), and robustness to multiplicative
plant uncertainty / sensor-noise rejection (:math:`W_T` large past crossover).

Contents
--------
``StateSpace``
    A minimal continuous-time LTI realisation ``(A, B, C, D)`` with series /
    parallel / feedback / stacking algebra, a frequency response, and an
    :math:`H_\infty`-norm by the bounded-real Hamiltonian bisection.
``weight_S`` / ``weight_KS`` / ``weight_T``
    First-order shaping filters (Skogestad & Postlethwaite forms), scalar or
    ``I_k``-blocked.
``augment_plant``
    Build the generalised plant :math:`P` of the mixed-sensitivity problem from
    ``(G, W_S, W_KS, W_T)``.
``hinf_syn``
    Sub-optimal :math:`H_\infty` synthesis by :math:`\gamma`-bisection on the
    two DGKF Hamiltonian Riccati equations (Doyle-Glover-Khargonekar-Francis
    1989 / Zhou-Doyle-Glover ch. 17, general :math:`D_{11}\neq 0` form).
``mixsyn``
    Convenience: ``augment_plant`` + ``hinf_syn`` in one call, returning the
    controller and the achieved :math:`\gamma`.
``HinfController``
    :class:`~aimct.controllers.base.Controller` wrapper around a synthesised
    ``StateSpace`` controller (output feedback, its own internal state).

References
----------
* Doyle, Glover, Khargonekar & Francis, "State-space solutions to standard
  H2 and H-infinity control problems", IEEE TAC 34(8), 1989.
* Zhou, Doyle & Glover, *Robust and Optimal Control*, Prentice Hall 1996,
  chapters 16-17.
* Skogestad & Postlethwaite, *Multivariable Feedback Control*, 2nd ed., Wiley
  2005, chapters 2-3 and 9 (weight selection, the S/KS/T stack).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import matrix_balance, schur

from .base import Controller

__all__ = [
    "StateSpace",
    "weight_S",
    "weight_KS",
    "weight_T",
    "augment_plant",
    "lft_lower",
    "hinf_syn",
    "mixsyn",
    "HinfSynResult",
    "HinfController",
]

_EPS = 1e-10


def _mat(x, rows=None, cols=None) -> np.ndarray:
    a = np.atleast_2d(np.asarray(x, dtype=float))
    if rows is not None and cols is not None and a.shape == (1, 1):
        a = a[0, 0] * np.eye(max(rows, cols))[:rows, :cols] if rows == cols else \
            np.full((rows, cols), a[0, 0])
    return a


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


# ======================================================================
# StateSpace
# ======================================================================
@dataclass
class StateSpace:
    r"""Continuous-time LTI system ``xdot = A x + B u``, ``y = C x + D u``.

    ``A (n,n)``, ``B (n,m)``, ``C (p,n)``, ``D (p,m)``.  Scalars / 1-D arrays are
    promoted.  A pure static gain has ``n = 0`` (empty ``A``).
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    def __post_init__(self):
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float)) if np.size(self.A) else np.zeros((0, 0))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=float)) if np.size(self.B) else np.zeros((0, 0))
        self.C = np.atleast_2d(np.asarray(self.C, dtype=float)) if np.size(self.C) else np.zeros((0, 0))
        self.D = np.atleast_2d(np.asarray(self.D, dtype=float))
        n = self.A.shape[0]
        p, m = self.D.shape
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if n:
            if self.B.shape != (n, m):
                raise ValueError(f"B must be ({n},{m}), got {self.B.shape}")
            if self.C.shape != (p, n):
                raise ValueError(f"C must be ({p},{n}), got {self.C.shape}")
        else:
            self.B = np.zeros((0, m))
            self.C = np.zeros((p, 0))

    # -- dimensions ------------------------------------------------------
    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.D.shape[1]

    @property
    def ny(self) -> int:
        return self.D.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nu)

    # -- constructors --------------------------------------------------------
    @classmethod
    def gain(cls, D) -> "StateSpace":
        """A static gain ``y = D u`` (no states)."""
        D = np.atleast_2d(np.asarray(D, dtype=float))
        return cls(np.zeros((0, 0)), np.zeros((0, D.shape[1])),
                   np.zeros((D.shape[0], 0)), D)

    @classmethod
    def eye(cls, k: int) -> "StateSpace":
        return cls.gain(np.eye(k))

    @classmethod
    def from_tf(cls, num: Iterable[float], den: Iterable[float]) -> "StateSpace":
        """SISO transfer function -> controllable canonical realisation."""
        num = np.atleast_1d(np.asarray(num, dtype=float))
        den = np.atleast_1d(np.asarray(den, dtype=float))
        if den[0] == 0:
            raise ValueError("den[0] must be non-zero")
        num = num / den[0]
        den = den / den[0]
        n = den.size - 1
        num = np.concatenate([np.zeros(n + 1 - num.size), num]) if num.size < n + 1 else num
        d0 = num[0]
        b = num[1:] - d0 * den[1:]                       # strictly-proper part
        if n == 0:
            return cls.gain([[d0]])
        A = np.zeros((n, n))
        A[:-1, 1:] = np.eye(n - 1)
        A[-1, :] = -den[:0:-1]
        B = np.zeros((n, 1))
        B[-1, 0] = 1.0
        C = b[::-1].reshape(1, n)
        return cls(A, B, C, [[d0]])

    # -- algebra -----------------------------------------------------------
    def __mul__(self, other) -> "StateSpace":
        """Series ``self * other``: ``other`` first, then ``self`` (``u -> other -> self``)."""
        if np.isscalar(other):
            return StateSpace(self.A, self.B, self.C * other, self.D * other)
        if not isinstance(other, StateSpace):
            return NotImplemented
        if self.nu != other.ny:
            raise ValueError(f"series dim mismatch: {self.shape} * {other.shape}")
        A1, B1, C1, D1 = self.A, self.B, self.C, self.D
        A2, B2, C2, D2 = other.A, other.B, other.C, other.D
        n1, n2 = self.nx, other.nx
        A = np.block([[A1, B1 @ C2], [np.zeros((n2, n1)), A2]])
        B = np.vstack([B1 @ D2, B2])
        C = np.hstack([C1, D1 @ C2])
        D = D1 @ D2
        return StateSpace(A, B, C, D)

    __rmul__ = __mul__

    def __add__(self, other) -> "StateSpace":
        """Parallel: outputs add, shared input."""
        if not isinstance(other, StateSpace):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"parallel dim mismatch: {self.shape} + {other.shape}")
        A = _blkdiag(self.A, other.A)
        B = np.vstack([self.B, other.B])
        C = np.hstack([self.C, other.C])
        return StateSpace(A, B, C, self.D + other.D)

    def __neg__(self) -> "StateSpace":
        return StateSpace(self.A, self.B, -self.C, -self.D)

    def __sub__(self, other) -> "StateSpace":
        return self + (-other)

    def feedback(self, other: "StateSpace | None" = None, sign: int = -1) -> "StateSpace":
        r"""Close the loop: ``self`` (``G``) forward, ``other`` (``K``, default
        identity) in the feedback path.  Returns the map ``r -> y`` for
        ``y = G u``, ``u = r + sign * K y``.  ``sign = -1`` is negative feedback.
        """
        G = self
        K = StateSpace.eye(G.nu) if other is None else other
        if K.shape != (G.nu, G.ny):
            raise ValueError("feedback dim mismatch")
        Ag, Bg, Cg, Dg = G.A, G.B, G.C, G.D
        Ak, Bk, Ck, Dk = K.A, K.B, K.C, K.D
        s = float(sign)
        ng, nk = G.nx, K.nx
        # algebraic loop:  y = Cg xg + Dg u ,  u = r + s (Ck xk + Dk y)
        #   => (I - s Dg Dk) y = Cg xg + s Dg Ck xk + Dg r
        Minv = np.linalg.inv(np.eye(G.ny) - s * Dg @ Dk)
        Cy_g = Minv @ Cg
        Cy_k = s * Minv @ Dg @ Ck
        Dy = Minv @ Dg
        # u  = r + s (Ck xk + Dk y)
        Cu_g = s * Dk @ Cy_g
        Cu_k = s * (Ck + s * Dk @ Cy_k)
        Du = np.eye(G.nu) + s * Dk @ Dy
        A = np.zeros((ng + nk, ng + nk))
        A[:ng, :ng] = Ag + Bg @ Cu_g
        if nk:
            A[:ng, ng:] = Bg @ Cu_k
            A[ng:, :ng] = Bk @ Cy_g
            A[ng:, ng:] = Ak + Bk @ Cy_k
        B = np.vstack([Bg @ Du, Bk @ Dy]) if nk else Bg @ Du
        C = np.hstack([Cy_g, Cy_k]) if nk else Cy_g
        return StateSpace(A, B, C, Dy)

    # -- analysis --------------------------------------------------------
    def poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.A) if self.nx else np.array([])

    def is_stable(self, tol: float = 1e-9) -> bool:
        p = self.poles()
        return bool(p.size == 0 or np.all(p.real < -tol))

    def freqresp(self, w) -> np.ndarray:
        """Frequency response ``G(jw)`` -> array ``(len(w), ny, nu)`` (complex)."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        out = np.empty((w.size, self.ny, self.nu), dtype=complex)
        if self.nx == 0:
            out[:] = self.D
            return out
        eye = np.eye(self.nx)
        for i, wi in enumerate(w):
            out[i] = self.C @ np.linalg.solve(1j * wi * eye - self.A, self.B) + self.D
        return out

    def sigma_max(self, w) -> np.ndarray:
        """Largest singular value of ``G(jw)`` over ``w``."""
        H = self.freqresp(w)
        return np.linalg.svd(H, compute_uv=False)[..., 0]

    def hinf_norm(self, *, tol: float = 1e-6, w_grid: np.ndarray | None = None) -> float:
        r"""``||G||_inf`` = sup_w sigma_max(G(jw)).

        A frequency-grid maximum seeds a bisection that is tightened with the
        bounded-real Hamiltonian test: ``||G||_inf < gamma`` iff
        ``gamma > sigma_max(D)`` and the Hamiltonian
        ``M(gamma)`` has no eigenvalue on the imaginary axis.
        """
        if w_grid is None:
            w_grid = np.concatenate([[0.0], np.logspace(-4, 6, 800)])
        lo = float(np.max(self.sigma_max(w_grid)))
        if self.nx == 0:
            return lo
        dmax = float(np.linalg.svd(self.D, compute_uv=False)[0]) if self.D.size else 0.0
        lo = max(lo, dmax)
        hi = lo * 2.0 + 1.0
        # grow hi until it is an upper bound
        for _ in range(60):
            if self._bounded_real(hi * (1 + 1e-9)):
                break
            hi *= 2.0
        for _ in range(200):
            if hi - lo <= tol * max(1.0, lo):
                break
            mid = 0.5 * (lo + hi)
            if self._bounded_real(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def _bounded_real(self, gamma: float) -> bool:
        """True if ``||G||_inf < gamma`` (bounded-real Hamiltonian has no jw-axis eig)."""
        A, B, C, D = self.A, self.B, self.C, self.D
        m, p = self.nu, self.ny
        R = gamma ** 2 * np.eye(m) - D.T @ D
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            return False
        if np.min(np.linalg.eigvalsh(_sym(R))) <= 0:
            return False
        H = np.block([
            [A + B @ Rinv @ D.T @ C, B @ Rinv @ B.T],
            [-C.T @ (np.eye(p) + D @ Rinv @ D.T) @ C, -(A + B @ Rinv @ D.T @ C).T],
        ])
        ev = np.linalg.eigvals(H)
        return bool(np.all(np.abs(ev.real) > 1e-9))


def _blkdiag(*mats: np.ndarray) -> np.ndarray:
    mats = [np.atleast_2d(m) for m in mats if np.size(m)]
    if not mats:
        return np.zeros((0, 0))
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols))
    r = c = 0
    for m in mats:
        out[r:r + m.shape[0], c:c + m.shape[1]] = m
        r += m.shape[0]
        c += m.shape[1]
    return out


def _vstack_ss(systems: list[StateSpace]) -> StateSpace:
    """Stack outputs of systems that share the same input: ``y = [G1 u; G2 u; ...]``."""
    systems = [s for s in systems]
    m = systems[0].nu
    for s in systems:
        if s.nu != m:
            raise ValueError("vstack: all systems must share the input dimension")
    A = _blkdiag(*[s.A for s in systems])
    B = np.vstack([s.B for s in systems]) if any(s.nx for s in systems) else np.zeros((0, m))
    C = _blkdiag(*[s.C for s in systems]) if any(s.nx for s in systems) else np.zeros((sum(s.ny for s in systems), 0))
    D = np.vstack([s.D for s in systems])
    return StateSpace(A, B, C, D)


def _as_ss(w, k: int, name: str) -> StateSpace:
    """Coerce a weight argument to a ``k x k`` :class:`StateSpace`."""
    if w is None:
        return StateSpace.gain(np.zeros((0, k)))          # 0-row: channel dropped
    if isinstance(w, StateSpace):
        if w.shape != (k, k):
            raise ValueError(f"{name} must be {k}x{k}, got {w.shape}")
        return w
    arr = np.atleast_2d(np.asarray(w, dtype=float))
    if arr.shape == (1, 1):
        arr = arr[0, 0] * np.eye(k)
    if arr.shape != (k, k):
        raise ValueError(f"{name} must be scalar or {k}x{k}")
    return StateSpace.gain(arr)


def _block_weight(a_ss: float, b: float, c: float, d: float, blocks: int) -> StateSpace:
    """Replicate a scalar first-order filter ``blocks`` times on the diagonal."""
    if blocks == 1:
        return StateSpace([[a_ss]], [[b]], [[c]], [[d]])
    A = a_ss * np.eye(blocks)
    B = b * np.eye(blocks)
    C = c * np.eye(blocks)
    D = d * np.eye(blocks)
    return StateSpace(A, B, C, D)


# ======================================================================
# mixed-sensitivity weighting filters  (Skogestad & Postlethwaite, 2nd ed.)
# ======================================================================
def weight_S(wb: float, *, A: float = 1e-4, M: float = 2.0, blocks: int = 1) -> StateSpace:
    r"""Performance weight on the sensitivity ``S``.

    ``W_S(s) = (s/M + wb) / (s + wb*A)`` -- low-frequency gain ``~1/A`` (tight
    tracking / disturbance rejection below ``wb``), high-frequency gain ``1/M``
    (``M`` bounds the sensitivity peak), crossover near ``wb``.
    """
    if not (0 < A < 1 < M):
        raise ValueError("need 0 < A < 1 < M")
    return _block_weight(-wb * A, 1.0, wb * (1.0 - A / M), 1.0 / M, blocks)


def weight_T(wb: float, *, A: float = 1e-3, M: float = 2.0, blocks: int = 1) -> StateSpace:
    r"""Robustness weight on the complementary sensitivity ``T``.

    ``W_T(s) = (s + wb/M) / (A*s + wb)`` -- low-frequency gain ``1/M``,
    high-frequency gain ``~1/A`` (forces ``T`` to roll off past ``wb``, which is
    what buys robust stability against multiplicative plant uncertainty).
    """
    if not (0 < A < 1 < M):
        raise ValueError("need 0 < A < 1 < M")
    return _block_weight(-wb / A, 1.0, (wb / A) * (1.0 / M - 1.0 / A), 1.0 / A, blocks)


def weight_KS(wb: float, *, A: float = 1e-3, M: float = 1e3, blocks: int = 1) -> StateSpace:
    r"""Weight on the control sensitivity ``KS`` (actuator effort / bandwidth).

    Same first-order high-pass shape as :func:`weight_T`: gain ``1/M`` at low
    frequency (control essentially unpenalised in band) rising to ``~1/A`` past
    ``wb`` (limits control action and closed-loop actuator bandwidth).  Pass a
    plain scalar to :func:`augment_plant` instead for a flat effort penalty.
    """
    if not (0 < A and 0 < M):
        raise ValueError("need A > 0 and M > 0")
    return _block_weight(-wb / A, 1.0, (wb / A) * (1.0 / M - 1.0 / A), 1.0 / A, blocks)


# ======================================================================
# generalised plant for the S / KS / T mixed-sensitivity problem
# ======================================================================
def augment_plant(G: StateSpace, W_S=None, W_KS=None, W_T=None):
    r"""Assemble the generalised plant ``P`` of the mixed-sensitivity problem.

    With exogenous input ``w`` (reference), control ``u``, error ``v = w - G u``:

    ``z1 = W_S  v`` , ``z2 = W_KS u`` , ``z3 = W_T (G u)`` , ``y = v`` .

    The lower LFT ``F_l(P, K)`` of the returned plant with any stabilising ``K``
    (``u = K y``, negative feedback folded in) equals
    ``[W_S S ; W_KS K S ; W_T T]``.

    Returns
    -------
    P : StateSpace
        Inputs ``[w (p) ; u (m)]`` -> outputs ``[z (nz) ; y (p)]``.
    nmeas : int
        Size of ``y`` (``= G.ny``).
    ncon : int
        Size of ``u`` (``= G.nu``).
    """
    p, m = G.ny, G.nu
    if p != m:
        raise ValueError("mixed-sensitivity augment_plant expects a square plant")
    WS = _as_ss(W_S, p, "W_S")
    WK = _as_ss(W_KS, m, "W_KS")
    WT = _as_ss(W_T, p, "W_T")
    if WS.ny + WK.ny + WT.ny == 0:
        raise ValueError("at least one weight must be provided")

    Ag, Bg, Cg, Dg = G.A, G.B, G.C, G.D
    ng = G.nx
    blocks = [("S", WS), ("K", WK), ("T", WT)]
    nw_states = {tag: w.nx for tag, w in blocks}
    ntot = ng + sum(nw_states.values())

    A = np.zeros((ntot, ntot))
    A[:ng, :ng] = Ag
    off = {"g": 0}
    cur = ng
    for tag, w in blocks:
        off[tag] = cur
        if w.nx:
            A[cur:cur + w.nx, cur:cur + w.nx] = w.A
        cur += w.nx
    # weight state couplings:  x1' gets -B_S Cg xg ; x3' gets +B_T Cg xg
    if WS.nx:
        A[off["S"]:off["S"] + WS.nx, :ng] = -WS.B @ Cg
    if WT.nx:
        A[off["T"]:off["T"] + WT.nx, :ng] = WT.B @ Cg

    # B1 (w channel, size p) and B2 (u channel, size m)
    B1 = np.zeros((ntot, p))
    B2 = np.zeros((ntot, m))
    B2[:ng, :] = Bg
    if WS.nx:
        B1[off["S"]:off["S"] + WS.nx, :] = WS.B
        B2[off["S"]:off["S"] + WS.nx, :] = -WS.B @ Dg
    if WK.nx:
        B2[off["K"]:off["K"] + WK.nx, :] = WK.B
    if WT.nx:
        B2[off["T"]:off["T"] + WT.nx, :] = WT.B @ Dg

    # outputs z = [z1; z2; z3], then y
    Crows, D1rows, D2rows = [], [], []
    # z1 = W_S v = -D_S Cg xg + C_S x1 + D_S w - D_S Dg u
    if WS.ny:
        row = np.zeros((WS.ny, ntot))
        row[:, :ng] = -WS.D @ Cg
        if WS.nx:
            row[:, off["S"]:off["S"] + WS.nx] = WS.C
        Crows.append(row)
        D1rows.append(WS.D)
        D2rows.append(-WS.D @ Dg)
    # z2 = W_KS u = C_K x2 + D_K u
    if WK.ny:
        row = np.zeros((WK.ny, ntot))
        if WK.nx:
            row[:, off["K"]:off["K"] + WK.nx] = WK.C
        Crows.append(row)
        D1rows.append(np.zeros((WK.ny, p)))
        D2rows.append(WK.D)
    # z3 = W_T (G u) = D_T Cg xg + C_T x3 + D_T Dg u
    if WT.ny:
        row = np.zeros((WT.ny, ntot))
        row[:, :ng] = WT.D @ Cg
        if WT.nx:
            row[:, off["T"]:off["T"] + WT.nx] = WT.C
        Crows.append(row)
        D1rows.append(np.zeros((WT.ny, p)))
        D2rows.append(WT.D @ Dg)

    Cz = np.vstack(Crows)
    D11 = np.vstack(D1rows)
    D12 = np.vstack(D2rows)
    # y = v = w - G u = -Cg xg + w - Dg u
    Cy = np.zeros((p, ntot))
    Cy[:, :ng] = -Cg
    D21 = np.eye(p)
    D22 = -Dg

    C = np.vstack([Cz, Cy])
    B = np.hstack([B1, B2])
    D = np.block([[D11, D12], [D21, D22]])
    P = StateSpace(A, B, C, D)
    P._mixsyn = dict(nz=Cz.shape[0], nw=p, nmeas=p, ncon=m)   # partition hint
    return P, p, m


def lft_lower(P: StateSpace, K: StateSpace, nmeas: int, ncon: int) -> StateSpace:
    r"""Lower linear-fractional transform ``F_l(P, K)``: close ``u = K y`` using
    the last ``ncon`` inputs / ``nmeas`` outputs of ``P``, return the map from
    the remaining inputs ``w`` to the remaining outputs ``z``.
    """
    nz = P.ny - nmeas
    nw = P.nu - ncon
    A, B, C, D = P.A, P.B, P.C, P.D
    B1, B2 = B[:, :nw], B[:, nw:]
    C1, C2 = C[:nz, :], C[nz:, :]
    D11, D12 = D[:nz, :nw], D[:nz, nw:]
    D21, D22 = D[nz:, :nw], D[nz:, nw:]
    Ak, Bk, Ck, Dk = K.A, K.B, K.C, K.D
    np_, nk = P.nx, K.nx
    Minv = np.linalg.inv(np.eye(nmeas) - D22 @ Dk)
    # y = Minv (C2 xP + D22 Ck xK + D21 w)
    Cy_P = Minv @ C2
    Cy_K = Minv @ D22 @ Ck
    Dy_w = Minv @ D21
    # u = Ck xK + Dk y
    Cu_P = Dk @ Cy_P
    Cu_K = Ck + Dk @ Cy_K
    Du_w = Dk @ Dy_w
    Acl = np.zeros((np_ + nk, np_ + nk))
    Acl[:np_, :np_] = A + B2 @ Cu_P
    if nk:
        Acl[:np_, np_:] = B2 @ Cu_K
        Acl[np_:, :np_] = Bk @ Cy_P
        Acl[np_:, np_:] = Ak + Bk @ Cy_K
    Bcl = np.vstack([B1 + B2 @ Du_w, Bk @ Dy_w]) if nk else B1 + B2 @ Du_w
    Ccl = np.hstack([C1 + D12 @ Cu_P, D12 @ Cu_K]) if nk else C1 + D12 @ Cu_P
    Dcl = D11 + D12 @ Du_w
    return StateSpace(Acl, Bcl, Ccl, Dcl)


# ======================================================================
# H-infinity synthesis  (gamma-iteration on the two DGKF Riccati equations)
# ======================================================================
@dataclass
class HinfSynResult:
    K: StateSpace              # the synthesised (central) controller, y -> u
    gamma: float               # achieved closed-loop H-infinity norm bound
    CL: StateSpace             # F_l(P, K), the closed loop w -> z
    X: np.ndarray              # stabilising solution of the X Riccati
    Y: np.ndarray              # stabilising solution of the Y Riccati
    iterations: int            # gamma bisection steps
    gamma_lb: float            # last infeasible gamma (lower bracket)


def _ric_stable(M: np.ndarray, tol: float = 1e-8):
    """Stabilising solution ``P`` of the Riccati whose Hamiltonian-structured
    matrix is ``M`` (``2n x 2n``): ``P = U2 U1^-1`` from the stable invariant
    subspace, extracted by an ordered real Schur factorisation (numerically
    stable for stiff / badly-scaled ``M``).  Returns ``(P, ok)``; ``ok=False``
    if ``M`` has a ~jw eigenvalue, the stable subspace is not ``n``-dimensional,
    or the graph subspace is degenerate.
    """
    n = M.shape[0] // 2
    ev = np.linalg.eigvals(M)
    scale = max(1.0, float(np.max(np.abs(ev))))
    if np.any(np.abs(ev.real) < tol * scale):
        return None, False
    try:
        Mb, (s, _perm) = matrix_balance(M, permute=False, separate=True)
        _T, Z, sdim = schur(Mb, output="real", sort="lhp")
        Z = np.asarray(s)[:, None] * Z          # undo the diagonal balancing
    except Exception:
        return None, False
    if sdim != n:
        return None, False
    U1, U2 = Z[:n, :n], Z[n:, :n]
    if np.linalg.cond(U1) > 1e12:
        return None, False
    try:
        P = _sym(U2 @ np.linalg.solve(U1, np.eye(n)))
    except np.linalg.LinAlgError:
        return None, False
    if not np.all(np.isfinite(P)):
        return None, False
    return P, True


def _normalise_d(D: np.ndarray, by_columns: bool):
    """SVD scaling making ``Dn^T Dn = I`` (``by_columns``) or ``Dn Dn^T = I``.

    Returns ``(Dn, S)`` where ``S`` maps back: for columns, ``u_orig = S u_new``;
    for rows, ``y_new = S y_orig``.
    """
    if by_columns:
        U, s, Vt = np.linalg.svd(D, full_matrices=False)     # D (p,m), m<=p
        if np.min(s) < 1e-9:
            raise ValueError("D12 is not full column rank (assumption A2 fails)")
        Dn = U                                                # p x m, orthonormal cols
        S = Vt.T @ np.diag(1.0 / s)                           # u_orig = S u_new
        return Dn, S
    U, s, Vt = np.linalg.svd(D, full_matrices=False)          # D (p,m), p<=m
    if np.min(s) < 1e-9:
        raise ValueError("D21 is not full row rank (assumption A2 fails)")
    Dn = Vt                                                   # p x m, orthonormal rows
    S = np.diag(1.0 / s) @ U.T                                # y_new = S y_orig
    return Dn, S


def hinf_syn(P: StateSpace, nmeas: int, ncon: int, *,
             gamma_max: float = 1e4, tol: float = 1e-4,
             max_iter: int = 100) -> HinfSynResult:
    r"""Sub-optimal :math:`H_\infty` controller for the generalised plant ``P``.

    ``P`` has inputs ``[w ; u]`` and outputs ``[z ; y]`` with ``u`` the last
    ``ncon`` inputs and ``y`` the last ``nmeas`` outputs.  Bisects
    :math:`\gamma` and, at each trial, solves the two DGKF Hamiltonian Riccati
    equations (Doyle-Glover-Khargonekar-Francis 1989 / Skogestad & Postlethwaite
    ch. 9), checking the sub-optimality conditions

    .. math:: X_\infty \succeq 0, \quad Y_\infty \succeq 0, \quad
              \rho(X_\infty Y_\infty) < \gamma^2 .

    Requires the standard assumptions A1-A4 and ``D11 = 0``, ``D22 = 0`` after
    the ``[w;u] / [z;y]`` partition (:func:`mixsyn` arranges this).  Returns the
    central controller and the achieved ``gamma``.
    """
    nx = P.nx
    nz = P.ny - nmeas
    nw = P.nu - ncon
    # balance the generalised-plant state -- F_l(P, K) is unchanged by a state
    # coordinate transform, and a well-scaled A keeps the 2n x 2n Hamiltonian
    # eigenproblems accurate when the weights sit far from the plant band.
    if nx:
        _, (sbal, _p) = matrix_balance(P.A, permute=False, separate=True)
        sbal = np.asarray(sbal, dtype=float)
        Tinv, T = 1.0 / sbal, sbal
        A = (Tinv[:, None] * P.A) * T[None, :]
        Bfull = Tinv[:, None] * P.B
        Cfull = P.C * T[None, :]
    else:
        A, Bfull, Cfull = P.A, P.B, P.C
    B1, B2 = Bfull[:, :nw], Bfull[:, nw:]
    C1, C2 = Cfull[:nz, :], Cfull[nz:, :]
    D11 = P.D[:nz, :nw]
    D12 = P.D[:nz, nw:]
    D21 = P.D[nz:, :nw]
    D22 = P.D[nz:, nw:]

    if np.linalg.norm(D11) > 1e-6:
        raise ValueError(
            "hinf_syn needs D11 = 0 after the [w;u]/[z;y] partition "
            f"(||D11|| = {np.linalg.norm(D11):.2e}). Use mixsyn(), or a "
            "strictly-proper W_S.")
    d22_shift = np.linalg.norm(D22) > 1e-12

    # --- normalise D12^T D12 = I and D21 D21^T = I via SVD scalings ---------
    D12n, Su = _normalise_d(D12, by_columns=True)     # u_orig = Su @ u_new
    D21n, Sy = _normalise_d(D21, by_columns=False)    # y_new  = Sy @ y_orig
    B2n = B2 @ Su
    C2n = Sy @ C2
    D21n = D21n                                       # already p2 x m1, rows orthonormal

    I_nx = np.eye(nx)
    g_lo, g_hi = 0.0, float(gamma_max)
    best = None
    it = 0
    for it in range(1, max_iter + 1):
        if g_hi - g_lo <= tol * max(1.0, g_lo):
            break
        gamma = 0.5 * (g_lo + g_hi) if best is not None else g_hi
        res = _hinf_try(A, B1, B2n, C1, C2n, D12n, D21n, gamma, I_nx)
        if res is None:
            g_lo = gamma
        else:
            best = (gamma, res)
            g_hi = gamma
        if best is None and gamma >= g_hi:      # never feasible even at gamma_max
            if it == 1:
                raise ValueError(
                    "no H-infinity controller below gamma_max; the problem may "
                    "be infeasible or the weights ill-posed")
    if best is None:
        raise ValueError("gamma bisection failed to find a feasible controller")

    gamma, (X, Y, F, L, Zc) = best
    # central controller (DGKF / Zhou-Doyle-Glover Thm 17.1, output-injection
    # form).  The gamma^-2 disturbance-feedforward uses B1 *projected off* the
    # measurement-noise directions (B1t); the B1 D21^T C2 output injection is
    # carried in the state matrix, so L contributes only its Y C2^T part here.
    B1t = B1 @ (np.eye(B1.shape[1]) - D21n.T @ D21n)
    Ac = (A - B1 @ D21n.T @ C2n
          + (1.0 / gamma ** 2) * B1t @ B1t.T @ X
          + B2n @ F
          - Zc @ (Y @ C2n.T) @ C2n)
    Kn = StateSpace(Ac, -Zc @ L, F, np.zeros((ncon, nmeas)))
    # undo the input / output scalings:  u_orig = Su u_new,  y_new = Sy y_orig
    K = StateSpace(Kn.A, Kn.B @ Sy, Su @ Kn.C, Su @ Kn.D @ Sy)
    if d22_shift:
        K = _absorb_d22(K, D22)

    CL = lft_lower(P, K, nmeas, ncon)
    return HinfSynResult(K=K, gamma=gamma, CL=CL, X=X, Y=Y,
                         iterations=it, gamma_lb=g_lo)


def _hinf_try(A, B1, B2, C1, C2, D12, D21, gamma, I_nx):
    """One gamma trial with normalised ``D12,D21``.  Returns ``(X,Y,F,L,Z)`` or
    ``None`` if this gamma is infeasible."""
    g2 = gamma ** 2
    Dc = D12.T @ C1                     # (m2 x nx)   (not assumed zero)
    Bd = B1 @ D21.T                     # (nx x p2)   (not assumed zero)
    Ax = A - B2 @ Dc
    Hx = np.block([
        [Ax, (1.0 / g2) * B1 @ B1.T - B2 @ B2.T],
        [-C1.T @ (np.eye(C1.shape[0]) - D12 @ D12.T) @ C1, -Ax.T],
    ])
    X, okx = _ric_stable(Hx)
    if not okx or np.min(np.linalg.eigvalsh(X)) < -1e-7:
        return None
    Ay = A - Bd @ C2
    Hy = np.block([
        [Ay.T, (1.0 / g2) * C1.T @ C1 - C2.T @ C2],
        [-B1 @ (np.eye(B1.shape[1]) - D21.T @ D21) @ B1.T, -Ay],
    ])
    Y, oky = _ric_stable(Hy)
    if not oky or np.min(np.linalg.eigvalsh(Y)) < -1e-7:
        return None
    rho = np.max(np.abs(np.linalg.eigvals(X @ Y)))
    if rho >= g2:
        return None
    F = -(Dc + B2.T @ X)
    L = -(Bd + Y @ C2.T)
    Zc = np.linalg.inv(I_nx - (1.0 / g2) * Y @ X)
    return X, Y, F, L, Zc


def _absorb_d22(K: StateSpace, D22: np.ndarray) -> StateSpace:
    """Return ``K (I + D22 K)^-1`` -- undo a ``D22 -> 0`` plant shift."""
    Ak, Bk, Ck, Dk = K.A, K.B, K.C, K.D
    M = np.linalg.inv(np.eye(Dk.shape[0]) + Dk @ D22)
    Ck2 = M @ Ck
    Dk2 = M @ Dk
    Ak2 = Ak - Bk @ D22 @ Ck2
    Bk2 = Bk - Bk @ D22 @ Dk2
    return StateSpace(Ak2, Bk2, Ck2, Dk2)


def mixsyn(G: StateSpace, W_S=None, W_KS=None, W_T=None, **kw) -> HinfSynResult:
    r"""Mixed-sensitivity :math:`H_\infty` loop shaping in one call.

    Builds the generalised plant with :func:`augment_plant` (using the
    *strictly-proper part* of ``W_S`` so ``D11 = 0`` -- the high-frequency
    sensitivity bound is then carried by ``W_T``), then :func:`hinf_syn`.
    Returns a :class:`HinfSynResult`; ``result.K`` is the controller,
    ``result.gamma`` the achieved ``||[W_S S; W_KS K S; W_T T]||_inf`` bound.
    """
    WS = _as_ss(W_S, G.ny, "W_S")
    if WS.nx and np.linalg.norm(WS.D) > 1e-12:
        WS = StateSpace(WS.A, WS.B, WS.C, np.zeros_like(WS.D))    # drop feedthrough
    P, nmeas, ncon = augment_plant(G, WS, W_KS, W_T)
    return hinf_syn(P, nmeas, ncon, **kw)


# ======================================================================
# Controller wrapper
# ======================================================================
class HinfController(Controller):
    r"""Drive a synthesised :class:`StateSpace` controller as an
    :class:`~aimct.controllers.base.Controller` (output feedback).

    ``update(y, dt)`` integrates ``xk' = Ak xk + Bk (r - y)`` one step (RK4) and
    returns ``u = Ck xk + Dk (r - y)``.  Set :attr:`r` for a non-zero reference.
    """

    def __init__(self, K: StateSpace, *, r=None, u_bounds=None):
        self.K = K
        self.r = np.zeros(K.nu) if r is None else np.atleast_1d(np.asarray(r, float))
        self.u_bounds = u_bounds
        self.reset()

    def reset(self) -> None:
        self._xk = np.zeros(self.K.nx)

    def update(self, measurement, dt: float):
        y = np.atleast_1d(np.asarray(measurement, dtype=float))
        e = self.r - y
        Ak, Bk, Ck, Dk = self.K.A, self.K.B, self.K.C, self.K.D
        if self.K.nx:
            def f(x):
                return Ak @ x + Bk @ e
            k1 = f(self._xk)
            k2 = f(self._xk + 0.5 * dt * k1)
            k3 = f(self._xk + 0.5 * dt * k2)
            k4 = f(self._xk + dt * k3)
            self._xk = self._xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u = Ck @ self._xk + Dk @ e
        if self.u_bounds is not None:
            u = np.clip(u, self.u_bounds[0], self.u_bounds[1])
        return u
