r"""Identify a two-link arm's inertial parameters from a motion log.

A rigid manipulator's dynamics are *linear in a set of base inertial
parameters* :math:`\pi`:

.. math::

    M(q)\ddot q + C(q,\dot q)\dot q + G(q) + F(\dot q)
        \;=\; Y(q,\dot q,\ddot q)\,\pi \;=\; \tau .

So identification is a linear least-squares fit: stack the regressor
:math:`Y` and the measured torque :math:`\tau` over every logged sample and
solve :math:`\min_\pi \lVert Y\pi - \tau\rVert`.  This module builds
:math:`Y` for :class:`aimct.systems.TwoLinkArm`, fits :math:`\pi`, and --
crucially -- reports *how trustworthy the fit is*: the regressor condition
number (was the trajectory exciting enough?), the validation-set torque
residual (does it predict held-out data?), and an internal consistency
check (:math:`p_2 = l_1 p_5` must hold for this model structure, so
:math:`p_2/p_5` is an independent estimate of link 1's length).

Base parameters for this model (columns of :math:`Y`, in order):

===== ============================================ ===========================
 sym   definition                                   appears in
===== ============================================ ===========================
 p1    m1 lc1^2 + m2 (l1^2+lc2^2) + mp(l1^2+l2^2)    M11 constant term
        + I1 + I2
 p2    m2 l1 lc2 + mp l1 l2                          M off-diagonal, Coriolis
 p3    m2 lc2^2 + mp l2^2 + I2                       M22
 p4    m1 lc1 + m2 l1 + mp l1                        gravity on joint 1
 p5    m2 lc2 + mp l2                                gravity on the wrist
 b1    joint-1 viscous friction                     F(dq)
 b2    joint-2 viscous friction                     F(dq)
===== ============================================ ===========================

With ``friction="coulomb"`` two more columns ``c1, c2`` multiplying
``sign(dq)`` are appended.

See ``docs/hardware/two-link-arm.md`` for the end-to-end build workflow this
feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "manipulator_regressor",
    "identify_manipulator",
    "finite_difference_derivatives",
    "ManipulatorID",
]


def finite_difference_derivatives(
    t: np.ndarray,
    q: np.ndarray,
    *,
    smooth: int = 0,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Estimate ``(dq, ddq)`` from a position log ``q`` sampled at ``t``.

    Real logs rarely record acceleration; often not even velocity.
    Differentiating position twice amplifies sensor noise viciously
    (:math:`\sigma_{\ddot q}\sim\sigma_q/\Delta t^2`), so:

    * ``smooth = 0`` -- plain central differences (``np.gradient``, 2nd-order
      edges).  Only for clean / simulated data.
    * ``smooth = w`` (odd, ``>= 5``) -- a Savitzky-Golay fit of window ``w``
      and degree ``polyorder`` whose analytic derivative gives ``dq`` and
      ``ddq`` directly.  This is the right tool for measured data; pick ``w``
      to span roughly a fifth of the fastest motion's period.  Requires a
      uniform time grid (``scipy`` fallback to smoothed central differences
      if the grid is non-uniform).

    Returns ``(dq, ddq)`` with the same shape as ``q``.
    """
    t = np.asarray(t, dtype=float).ravel()
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    if q.shape[0] != t.shape[0]:
        raise ValueError(f"q has {q.shape[0]} rows, t has {t.shape[0]}")

    if not smooth or smooth < 5:
        dq = np.gradient(q, t, axis=0, edge_order=2)
        ddq = np.gradient(dq, t, axis=0, edge_order=2)
        return dq, ddq

    w = int(smooth) | 1
    dt = np.diff(t)
    uniform = np.allclose(dt, dt[0], rtol=1e-3, atol=1e-12)
    if uniform:
        try:
            from scipy.signal import savgol_filter

            h = float(dt[0])
            dq = savgol_filter(q, w, polyorder, deriv=1, delta=h, axis=0)
            ddq = savgol_filter(q, w, polyorder, deriv=2, delta=h, axis=0)
            return dq, ddq
        except ImportError:
            pass

    # non-uniform grid or no scipy: moving-average the position, then difference
    pad = w // 2
    kern = np.ones(w) / w
    qp = np.pad(q, ((pad, pad), (0, 0)), mode="edge")
    qs = np.stack([np.convolve(qp[:, j], kern, mode="valid") for j in range(q.shape[1])], axis=1)
    dq = np.gradient(qs, t, axis=0, edge_order=2)
    ddq = np.gradient(dq, t, axis=0, edge_order=2)
    return dq, ddq


def manipulator_regressor(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    *,
    g: float = 9.81,
    friction: str = "viscous",
) -> np.ndarray:
    r"""Stacked regressor ``Y`` such that ``Y @ pi`` is the joint torque.

    Parameters
    ----------
    q, dq, ddq : ``(N, 2)`` arrays -- joint angle / rate / acceleration logs.
    g : gravity [m/s^2]; pass ``0.0`` for a log taken in a horizontal plane.
    friction : ``"viscous"`` (7 params) or ``"coulomb"`` (9 -- adds
        ``sign(dq)`` columns) or ``"none"`` (5).

    Returns
    -------
    Y : ``(2N, P)`` array.  Row ``2k`` is joint 1 at sample ``k``, row
        ``2k+1`` is joint 2.  ``P`` is 5, 7 or 9 per ``friction``.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    dq = np.atleast_2d(np.asarray(dq, dtype=float))
    ddq = np.atleast_2d(np.asarray(ddq, dtype=float))
    if not (q.shape == dq.shape == ddq.shape) or q.shape[1] != 2:
        raise ValueError("q, dq, ddq must all be (N, 2) and identical shape")
    if friction not in ("viscous", "coulomb", "none"):
        raise ValueError("friction must be 'viscous', 'coulomb' or 'none'")

    n = q.shape[0]
    q1, q2 = q[:, 0], q[:, 1]
    dq1, dq2 = dq[:, 0], dq[:, 1]
    ddq1, ddq2 = ddq[:, 0], ddq[:, 1]
    c2, s2 = np.cos(q2), np.sin(q2)
    cg1 = g * np.cos(q1)
    cg12 = g * np.cos(q1 + q2)
    z = np.zeros(n)

    # rows for joint 1 and joint 2 (see module docstring for the derivation)
    row1 = np.stack([
        ddq1,                                                   # p1
        2 * c2 * ddq1 + c2 * ddq2 - s2 * (2 * dq1 * dq2 + dq2**2),  # p2
        ddq2,                                                   # p3
        cg1,                                                    # p4
        cg12,                                                   # p5
    ], axis=1)
    row2 = np.stack([
        z,                                                      # p1
        c2 * ddq1 + s2 * dq1**2,                                # p2
        ddq1 + ddq2,                                            # p3
        z,                                                      # p4
        cg12,                                                   # p5
    ], axis=1)

    if friction in ("viscous", "coulomb"):
        row1 = np.concatenate([row1, np.stack([dq1, z], axis=1)], axis=1)  # b1, b2
        row2 = np.concatenate([row2, np.stack([z, dq2], axis=1)], axis=1)
    if friction == "coulomb":
        row1 = np.concatenate([row1, np.stack([np.sign(dq1), z], axis=1)], axis=1)  # c1, c2
        row2 = np.concatenate([row2, np.stack([z, np.sign(dq2)], axis=1)], axis=1)

    # interleave: [j1(k=0), j2(k=0), j1(k=1), j2(k=1), ...]
    Y = np.empty((2 * n, row1.shape[1]))
    Y[0::2] = row1
    Y[1::2] = row2
    return Y


@dataclass
class ManipulatorID:
    r"""Result of :func:`identify_manipulator`.

    Attributes
    ----------
    pi : the fitted base-parameter vector (length 5/7/9).
    names : the parameter names, aligned with ``pi``.
    params : ``dict`` view of ``{name: value}``.
    friction : the friction model used.
    g : gravity used in the fit.
    cond : regressor condition number on the *training* split -- above ~1e4
        the trajectory did not excite every parameter and the fit is soft.
    resid_rms_train, resid_rms_val : per-joint RMS torque residual
        ``(2,)`` [N.m] on the training and held-out splits.
    r2_val : coefficient of determination on the held-out torque.
    implied_l1 : ``p2 / p5`` -- an estimate of link 1's length that this
        model structure forces to equal the true ``l1``; compare it to your
        ruler measurement as an independent sanity check.
    n_train, n_val : sample counts.
    """

    pi: np.ndarray
    names: list[str]
    friction: str
    g: float
    cond: float
    resid_rms_train: np.ndarray
    resid_rms_val: np.ndarray
    r2_val: float
    implied_l1: float
    n_train: int
    n_val: int
    params: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.params = {n: float(v) for n, v in zip(self.names, self.pi)}

    def predict_torque(self, q, dq, ddq) -> np.ndarray:
        """Predicted joint torque ``(N, 2)`` for new ``q, dq, ddq`` logs."""
        Y = manipulator_regressor(q, dq, ddq, g=self.g, friction=self.friction)
        return (Y @ self.pi).reshape(-1, 2)

    def summary(self) -> str:
        lines = [
            f"ManipulatorID  ({self.friction} friction, g={self.g:g})",
            f"  samples:            {self.n_train} train / {self.n_val} val",
            f"  regressor cond:     {self.cond:.3g}"
            + ("   <-- POORLY EXCITED (>1e4)" if self.cond > 1e4 else ""),
            f"  torque resid RMS:   train {np.array2string(self.resid_rms_train, precision=4)}"
            f"  val {np.array2string(self.resid_rms_val, precision=4)}  N.m",
            f"  val R^2:            {self.r2_val:.5f}",
            f"  implied l1 (p2/p5): {self.implied_l1:.4f} m",
            "  base parameters:",
        ]
        lines += [f"    {n:>3} = {v:+.6g}" for n, v in self.params.items()]
        return "\n".join(lines)

    def to_twolink_arm(self, l1: float, l2: float, *, lc1=None, lc2=None):
        r"""Build a :class:`~aimct.systems.TwoLinkArm` from the fit.

        The base parameters do not uniquely split into
        ``(m1, m2, lc*, I*)``; supplying the two link **lengths** (a ruler
        measurement) plus the COM offsets closes the system.  Defaults put
        each COM at the link mid-point (``lc = l/2``) and fold any wrist
        payload into link 2 (``payload = 0``).  The returned arm reproduces
        the *identified* ``M/C/G`` -- it is the model to hand to a
        computed-torque or LQR design, not a claim about the true geometry.

        Negative inertias from a noisy fit are clipped to ``1e-6`` with a
        ``RuntimeWarning``.
        """
        import warnings

        from aimct.systems import TwoLinkArm

        lc1 = 0.5 * l1 if lc1 is None else float(lc1)
        lc2 = 0.5 * l2 if lc2 is None else float(lc2)
        p = self.params
        p1, p2, p3, p4, p5 = (p["p1"], p["p2"], p["p3"], p["p4"], p["p5"])

        m2 = p5 / lc2
        I2 = p3 - m2 * lc2**2
        m1 = (p4 - m2 * l1) / lc1
        I1 = p1 - m1 * lc1**2 - m2 * (l1**2 + lc2**2) - I2

        if min(I1, I2, m1, m2) <= 0:
            warnings.warn(
                f"non-physical fit (m1={m1:.3g}, m2={m2:.3g}, I1={I1:.3g}, "
                f"I2={I2:.3g}); clipping to 1e-6 -- trajectory likely under-exciting",
                RuntimeWarning,
                stacklevel=2,
            )
        b = (p.get("b1", 0.0), p.get("b2", 0.0))
        return TwoLinkArm(
            m1=max(m1, 1e-6), m2=max(m2, 1e-6),
            l1=l1, l2=l2, lc1=lc1, lc2=lc2,
            I1=max(I1, 1e-6), I2=max(I2, 1e-6),
            g=self.g, b=b, payload=0.0,
        )


_NAME_SETS = {
    "none": ["p1", "p2", "p3", "p4", "p5"],
    "viscous": ["p1", "p2", "p3", "p4", "p5", "b1", "b2"],
    "coulomb": ["p1", "p2", "p3", "p4", "p5", "b1", "b2", "c1", "c2"],
}


def identify_manipulator(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    tau: np.ndarray,
    *,
    g: float = 9.81,
    friction: str = "viscous",
    val_fraction: float = 0.3,
    ridge: float = 0.0,
    shuffle: bool = True,
    seed: int | None = 0,
) -> ManipulatorID:
    r"""Least-squares fit of a two-link arm's base inertial parameters.

    Parameters
    ----------
    q, dq, ddq : ``(N, 2)`` joint angle / rate / acceleration logs.  If you
        only logged ``q``, get ``dq, ddq`` from
        :func:`finite_difference_derivatives` first.
    tau : ``(N, 2)`` commanded (or measured) joint torque log.
    g : gravity [m/s^2]; ``0.0`` for a horizontal-plane log.
    friction : ``"viscous"`` (default), ``"coulomb"`` or ``"none"``.
    val_fraction : portion of samples held out to score generalisation.
    ridge : Tikhonov ``lambda`` on ``pi`` (``0`` = plain least squares); a
        small value (``1e-6``) stabilises a poorly-excited fit.
    shuffle, seed : shuffle samples before the train/val split (on by
        default so a val split isn't just the trajectory's tail).

    Returns
    -------
    :class:`ManipulatorID`
    """
    q, dq, ddq, tau = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (q, dq, ddq, tau))
    n = q.shape[0]
    if not (q.shape == dq.shape == ddq.shape == tau.shape) or q.shape[1] != 2:
        raise ValueError("q, dq, ddq, tau must all be (N, 2)")
    if n < 8:
        raise ValueError("need at least 8 samples")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    n_val = int(round(val_fraction * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    if tr_i.size < 4:
        tr_i, val_i = idx, idx[:0]

    names = _NAME_SETS[friction]

    def _rows(sample_idx):
        rr = np.concatenate([[2 * k, 2 * k + 1] for k in sample_idx]) if sample_idx.size else np.array([], int)
        return rr

    Y_all = manipulator_regressor(q, dq, ddq, g=g, friction=friction)
    tau_flat = tau.reshape(-1)                      # [j1(0), j2(0), j1(1), ...]

    Ytr, ttr = Y_all[_rows(tr_i)], tau_flat[_rows(tr_i)]
    if ridge > 0.0:
        p = Ytr.shape[1]
        Ytr_aug = np.vstack([Ytr, np.sqrt(ridge) * np.eye(p)])
        ttr_aug = np.concatenate([ttr, np.zeros(p)])
        pi, *_ = np.linalg.lstsq(Ytr_aug, ttr_aug, rcond=None)
    else:
        pi, *_ = np.linalg.lstsq(Ytr, ttr, rcond=None)

    cond = float(np.linalg.cond(Ytr))

    def _resid_rms(sample_idx):
        if sample_idx.size == 0:
            return np.full(2, np.nan)
        rr = _rows(sample_idx)
        e = (Y_all[rr] @ pi - tau_flat[rr]).reshape(-1, 2)
        return np.sqrt(np.mean(e**2, axis=0))

    rms_tr = _resid_rms(tr_i)
    rms_val = _resid_rms(val_i)

    if val_i.size:
        rr = _rows(val_i)
        yv = tau_flat[rr]
        ss_res = float(np.sum((Y_all[rr] @ pi - yv) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2)) or 1e-12
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")

    p = dict(zip(names, pi))
    implied_l1 = float(p["p2"] / p["p5"]) if abs(p["p5"]) > 1e-9 else float("nan")

    return ManipulatorID(
        pi=np.asarray(pi, dtype=float),
        names=list(names),
        friction=friction,
        g=float(g),
        cond=cond,
        resid_rms_train=rms_tr,
        resid_rms_val=rms_val,
        r2_val=float(r2),
        implied_l1=implied_l1,
        n_train=int(tr_i.size),
        n_val=int(val_i.size),
    )
