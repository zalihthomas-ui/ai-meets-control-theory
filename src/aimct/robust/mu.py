r"""The structured singular value :math:`\mu` -- bounds and frequency sweeps.

For a complex matrix :math:`M \in \mathbb{C}^{n\times n}` and a block structure
:math:`\mathbf{\Delta}`,

.. math::

    \mu_{\mathbf{\Delta}}(M) \;=\;
    \Big( \min\{\, \bar\sigma(\Delta) : \Delta \in \mathbf{\Delta},\;
                 \det(I - M\Delta) = 0 \,\} \Big)^{-1},

with :math:`\mu = 0` when no such :math:`\Delta` exists.  Two classical bounds
bracket it:

.. math::

    \rho(M) \;\le\; \max_{U\in\mathbf{U}} \rho(UM)
    \;\le\; \mu_{\mathbf{\Delta}}(M)
    \;\le\; \inf_{D\in\mathbf{D}} \bar\sigma(D M D^{-1})
    \;\le\; \bar\sigma(M),

the lower one found here by **power iteration** (Packard-Doyle), the upper one
by **D-scaling** (diagonal, one positive scale per block -- exact for three or
fewer blocks, mildly conservative beyond).  Real blocks are handled exactly in
the lower bound; the upper bound treats them as complex (a valid but
conservative over-bound -- ``G``-scaling for tight real-:math:`\mu` is a later
item).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["BlockStructure", "mu", "robust_stability_margin",
           "robust_performance_margin", "dk_iterate"]

_KINDS = {"R", "C", "F"}   # repeated-real scalar, repeated-complex scalar, full complex


@dataclass(frozen=True)
class BlockStructure:
    """A :math:`\\mu` uncertainty structure: an ordered list of ``(size, kind)``
    blocks with ``kind`` in ``{"R", "C", "F"}`` (repeated real scalar,
    repeated complex scalar, full complex)."""

    blocks: tuple[tuple[int, str], ...]

    def __post_init__(self):
        for sz, kind in self.blocks:
            if int(sz) < 1 or kind not in _KINDS:
                raise ValueError(f"bad block {(sz, kind)}; kind in {sorted(_KINDS)}")
        object.__setattr__(self, "blocks",
                           tuple((int(sz), str(k)) for sz, k in self.blocks))

    @classmethod
    def parse(cls, spec) -> "BlockStructure":
        """Coerce a spec to a :class:`BlockStructure`.

        Accepts a :class:`BlockStructure`, a list of ``(size, kind)`` pairs, or
        an int ``n`` (a single ``n x n`` full complex block).
        """
        if isinstance(spec, BlockStructure):
            return spec
        if isinstance(spec, int):
            return cls(((spec, "F"),))
        return cls(tuple((int(sz), str(k)) for sz, k in spec))

    @property
    def n(self) -> int:
        return sum(sz for sz, _ in self.blocks)

    def slices(self):
        out, i = [], 0
        for sz, kind in self.blocks:
            out.append((slice(i, i + sz), sz, kind))
            i += sz
        return out


# ======================================================================
# lower bound  --  mu >= sup { rho(Delta M) : Delta structured, ||Delta|| <= 1 }
# ======================================================================
def _struct_norm(Delta: np.ndarray, S: BlockStructure) -> float:
    m = 0.0
    for sl, _sz, _k in S.slices():
        blk = Delta[sl, sl]
        m = max(m, float(np.linalg.svd(blk, compute_uv=False)[0]) if blk.size else 0.0)
    return m


def _normalise_struct(Delta: np.ndarray, S: BlockStructure) -> np.ndarray:
    nd = _struct_norm(Delta, S)
    return Delta / nd if nd > 1e-14 else Delta


def _align_delta(M: np.ndarray, x: np.ndarray, w: np.ndarray,
                 S: BlockStructure) -> np.ndarray:
    """Blockwise structured ``Delta`` (norm <= 1) that maximises
    ``w^H Delta M x`` -- i.e. drives ``rho(Delta M)`` up, using the right (``x``)
    and left (``w``) dominant eigenvectors of the current ``Delta M``."""
    n = S.n
    Mx = M @ x
    Delta = np.zeros((n, n), complex)
    for sl, sz, kind in S.slices():
        xi, yi, wi = x[sl], Mx[sl], w[sl]
        ny2 = float(np.vdot(yi, yi).real)
        if ny2 < 1e-28:
            continue
        if kind == "F":
            # Delta_i maps yi -> xi (rank-1), scaled to spectral norm <= 1
            Di = np.outer(xi, yi.conj()) / ny2
            s0 = float(np.linalg.svd(Di, compute_uv=False)[0])
            if s0 > 1.0:
                Di /= s0
            Delta[sl, sl] = Di
        else:
            phase = np.sum(np.conj(wi) * yi)             # w_i^H (M x)_i
            if kind == "C":
                di = np.conj(phase) / abs(phase) if abs(phase) > 1e-300 else 0.0
            else:                                        # "R"
                di = 1.0 if phase.real >= 0 else -1.0
            Delta[sl, sl] = di * np.eye(sz)
    return Delta


def _rho_deltaM(M: np.ndarray, Delta: np.ndarray) -> float:
    ev = np.linalg.eigvals(Delta @ M)
    return float(np.max(np.abs(ev))) if ev.size else 0.0


def _mu_lower(M: np.ndarray, S: BlockStructure, *, iters: int = 80,
              restarts: int = 6, seed: int = 0) -> tuple[float, np.ndarray]:
    n = S.n
    rng = np.random.default_rng(seed)

    # rho(M) itself is admissible whenever I is in the structure (always true
    # for R/C scalar blocks; for pure full blocks it is still a valid bound).
    I = np.eye(n, dtype=complex)
    best_mu = _rho_deltaM(M, I)
    best_delta = best_mu * I if best_mu > 0 else I.copy()

    starts = [I]
    for _ in range(restarts):
        d0 = np.zeros((n, n), complex)
        for sl, sz, kind in S.slices():
            if kind == "F":
                g = rng.standard_normal((sz, sz)) + 1j * rng.standard_normal((sz, sz))
                d0[sl, sl] = g / max(np.linalg.svd(g, compute_uv=False)[0], 1e-12)
            elif kind == "C":
                d0[sl, sl] = np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.eye(sz)
            else:
                d0[sl, sl] = (1.0 if rng.random() < 0.5 else -1.0) * np.eye(sz)
        starts.append(d0)

    for Delta in starts:
        Delta = _normalise_struct(Delta, S)
        prev = -1.0
        for _ in range(iters):
            DM = Delta @ M
            ev, V = np.linalg.eig(DM)
            k = int(np.argmax(np.abs(ev)))
            x = V[:, k]
            evL, W = np.linalg.eig(DM.conj().T)          # left eigenvectors
            kl = int(np.argmax(np.abs(evL)))
            w = W[:, kl]
            nx, nw = np.linalg.norm(x), np.linalg.norm(w)
            if nx < 1e-14 or nw < 1e-14:
                break
            x, w = x / nx, w / nw
            Delta = _normalise_struct(_align_delta(M, x, w, S), S)
            cur = _rho_deltaM(M, Delta)
            if cur > best_mu:
                best_mu, best_delta = cur, Delta.copy()
            if abs(cur - prev) < 1e-12 * max(1.0, cur):
                break
            prev = cur
    return best_mu, best_delta


# ======================================================================
# upper bound -- D-scaling (diagonal, per block)
# ======================================================================
def _mu_upper(M: np.ndarray, S: BlockStructure, *, iters: int = 120) -> tuple[float, np.ndarray]:
    sl = S.slices()
    b = len(sl)

    def dvec(logd):
        d = np.exp(logd)
        return np.concatenate([np.full(sz, d[p]) for p, (_s, sz, _k) in enumerate(sl)])

    def sigma(logd):
        dv = dvec(logd)
        return float(np.linalg.svd((dv[:, None] * M) / dv[None, :],
                                   compute_uv=False)[0])

    logd = np.zeros(b)
    best_ub, best_logd = sigma(logd), logd.copy()        # D = I is always feasible
    for _ in range(iters):
        dv = dvec(logd)
        Ms = (dv[:, None] * M) / dv[None, :]
        agg = np.array([[np.linalg.norm(np.abs(Ms[sp, sq]))
                         for sq, _q, _kq in sl] for sp, _p, _kp in sl])
        row, col = agg.sum(axis=1), agg.sum(axis=0)
        step = 0.5 * np.log(np.maximum(col, 1e-300) / np.maximum(row, 1e-300))
        if np.max(np.abs(step)) < 1e-10:
            break
        logd = logd + step
        logd -= logd.mean()
        s = sigma(logd)
        if s < best_ub:
            best_ub, best_logd = s, logd.copy()

    return best_ub, dvec(best_logd)


# ======================================================================
# public: mu of a single matrix
# ======================================================================
def mu(M, structure, *, lower: bool = True, upper: bool = True,
       seed: int = 0) -> dict:
    r"""Structured singular value of ``M`` for the given block ``structure``.

    ``structure`` is a :class:`BlockStructure`, a list of ``(size, kind)`` pairs,
    or an int ``n`` (one ``n x n`` full block, for which ``mu = sigma_max(M)``).

    Returns a dict with ``lower_bound``, ``upper_bound`` (either may be omitted
    if the corresponding flag is ``False``), ``rho`` (the spectral radius, an
    unconditional lower bound), ``sigma_max`` (an unconditional upper bound), and
    ``worst_delta`` (a structured perturbation achieving the lower bound).
    """
    M = np.asarray(M, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")
    S = BlockStructure.parse(structure)
    if S.n != M.shape[0]:
        raise ValueError(f"structure sums to {S.n}, M is {M.shape[0]}x{M.shape[0]}")

    out = {
        "rho": float(np.max(np.abs(np.linalg.eigvals(M)))),
        "sigma_max": float(np.linalg.svd(M, compute_uv=False)[0]),
    }
    # single full block: mu == sigma_max, exactly
    if len(S.blocks) == 1 and S.blocks[0][1] == "F":
        v = out["sigma_max"]
        if lower:
            out["lower_bound"] = v
        if upper:
            out["upper_bound"] = v
        U, s, Vh = np.linalg.svd(M)
        out["worst_delta"] = (s[0]) ** -1 * np.outer(Vh[0].conj(), U[:, 0].conj()) \
            if s[0] > 0 else np.zeros_like(M)
        return out
    if lower:
        lb, dl = _mu_lower(M, S, seed=seed)
        out["lower_bound"] = lb
        out["worst_delta"] = dl
    if upper:
        ub, _dv = _mu_upper(M, S)
        out["upper_bound"] = ub
    return out


# ======================================================================
# frequency sweeps
# ======================================================================
def _sweep(M_of_omega, omega, structure, seed, *, restarts=4, iters=50):
    S = BlockStructure.parse(structure)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    lb = np.empty(omega.size)
    ub = np.empty(omega.size)
    single_full = len(S.blocks) == 1 and S.blocks[0][1] == "F"
    for i, w in enumerate(omega):
        M = np.asarray(M_of_omega(w), dtype=complex)
        if single_full:
            v = float(np.linalg.svd(M, compute_uv=False)[0])
            lb[i] = ub[i] = v
            continue
        ub[i], _ = _mu_upper(M, S)
        lb[i], _ = _mu_lower(M, S, restarts=restarts, iters=iters, seed=seed + i)
    return omega, lb, ub, S


def robust_stability_margin(M_of_omega, structure, omega, *, seed: int = 0) -> dict:
    r"""Sweep :math:`\mu(M(j\omega))` and report the robust-stability margin.

    ``M_of_omega(w) -> (n, n)`` is the ``M``-:math:`\Delta` interconnection at
    frequency ``w`` (the closed loop with the controller already absorbed).
    The margin ``1 / max_w mu_upper`` is the factor by which every block's
    bound can be scaled up while stability is still guaranteed; ``mu_lower``
    gives a destabilising perturbation of size ``1 / max_w mu_lower``.
    """
    w, lb, ub, _S = _sweep(M_of_omega, omega, structure, seed)
    i_ub = int(np.argmax(ub))
    i_lb = int(np.argmax(lb))
    return dict(
        omega=w, mu_lower=lb, mu_upper=ub,
        peak_upper=float(ub[i_ub]), peak_lower=float(lb[i_lb]),
        omega_peak=float(w[i_ub]),
        margin=float(1.0 / ub[i_ub]) if ub[i_ub] > 0 else np.inf,
        robust=bool(ub[i_ub] < 1.0),
    )


def robust_performance_margin(M_of_omega, structure, perf_shape, omega, *,
                              seed: int = 0) -> dict:
    r"""Robust *performance* via the main-loop theorem: append a fictitious
    full-complex performance block of size ``perf_shape = (n_z, n_w)`` and run
    :func:`robust_stability_margin` on the augmented structure.

    ``M_of_omega(w)`` must return the full ``(n_delta + n_z) x (n_delta + n_w)``
    interconnection (uncertainty channels first, then the performance channel).
    Robust performance holds iff ``peak_upper < 1``.
    """
    S = BlockStructure.parse(structure)
    n_z, n_w = perf_shape
    if n_z != n_w:
        # pad the performance block to square with the larger dimension
        raise ValueError("performance block must be square for this mu form; "
                         "pad M so n_z == n_w")
    aug = BlockStructure(S.blocks + ((n_z, "F"),))
    return robust_stability_margin(M_of_omega, aug, omega, seed=seed)


# ======================================================================
# constant-D  D-K iteration (analysis-grade)
# ======================================================================
def dk_iterate(resynthesise, mu_matrix, structure, omega, *, iterations: int = 4,
               seed: int = 0):
    r"""A constant-scaling approximation of D-K iteration.

    Alternates the two half-steps of D-K:

    * **K-step** -- ``resynthesise(d) -> K`` returns an H-infinity controller for
      the plant whose robustness weight has been scaled by the frequency-flat
      block factor ``d`` (a length-``len(structure.blocks)`` positive array;
      ``d`` is all-ones on the first call).
    * **D-step** -- ``mu_matrix(K, w) -> (n, n)`` returns the ``M``-:math:`\Delta`
      matrix at frequency ``w`` for that ``K``; :func:`robust_stability_margin`
      sweeps it, and the D-scaling that minimises the peak upper bound at the
      worst frequency becomes the next ``d``.

    Returns ``(K, history)`` with ``history[i]`` the
    :func:`robust_stability_margin` dict of round ``i`` (round 0 = the plain
    design).  The full method fits a *rational* ``D(s)`` each round; a constant
    ``d`` still tightens a peak that lives in one frequency band -- the common
    case -- and shows the mechanism.
    """
    S = BlockStructure.parse(structure)
    b = len(S.blocks)
    d = np.ones(b)
    K = resynthesise(d)
    history = []

    for _ in range(iterations):
        rs = robust_stability_margin(lambda w, K=K: np.asarray(mu_matrix(K, w),
                                                               dtype=complex),
                                     S, omega, seed=seed)
        history.append(rs)
        Mw = np.asarray(mu_matrix(K, rs["omega_peak"]), dtype=complex)
        _ub, dv = _mu_upper(Mw, S)
        # collapse the per-entry scaling dv to one factor per block, normalise
        newd = np.empty(b)
        for p, (sl, _sz, _k) in enumerate(S.slices()):
            newd[p] = float(np.mean(np.real(dv[sl])))
        newd = newd / np.exp(np.mean(np.log(np.maximum(newd, 1e-12))))
        if np.max(np.abs(np.log(newd / d))) < 1e-3:
            break
        d = newd
        K = resynthesise(d)
    return K, history
