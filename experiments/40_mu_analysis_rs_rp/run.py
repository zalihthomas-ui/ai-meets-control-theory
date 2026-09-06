"""Experiment 40 - mu analysis: structured RS/RP where single-loop margins miss.

Experiment 35 showed LQG losing stability to an unmodelled resonance and
H-infinity surviving.  Here the same plant is put in the standard M-Delta form
and the structured singular value (`aimct.robust.mu`) is used to *predict* the
failure boundary, and D-K iteration to push it back.

Two output-multiplicative uncertainties on the Exp-35 plant
``G(s) = 12/((s+1)(s+3))`` (controlled by a moderate LQG, ~5 rad/s crossover):

* ``delta_r`` (real, +/-1) scaled by ``w_g = 0.45`` -- a +/-45% loop-gain error;
* ``Delta_c`` (complex, |.|<=1) scaled by ``W_m(s) = R(s) - 1`` -- a dynamic
  uncertainty that exactly covers the lightly-damped mode ``R(s)``
  (``omega_r = 15`` rad/s) at ``Delta_c = 1``, and the disk around it.

Both enter at the plant output, so the interconnection is rank one:
``M(jw) = -T(jw) [w_g  W_m(jw); w_g  W_m(jw)]`` with ``T = GK/(1+GK)`` -- so
``mu = max_w |T|(w_g + |W_m|)`` in closed form, and the ``aimct.robust.mu``
solver is checked against it to machine precision.

Part (a): sweep the resonance damping ``zeta_r`` down.  Neither uncertainty on
its own destabilises anywhere in the range, the nominal ``G*R`` pole stays in
the left half plane, and the LQG gain / phase margins (on ``G`` alone) are dead
flat.  The structured ``mu`` is the *only* quantity that climbs as the
resonance grows, and it crosses 1 slightly *before* the resonance-alone
small-gain boundary -- the +/-45% gain tolerance genuinely costs margin against
the mode.

Part (b): one or two constant-D D-K rounds on the Exp-35 H-infinity design
lower the peak ``mu`` of the same structure versus plain ``mixsyn``.

Run:   python experiments/40_mu_analysis_rs_rp/run.py
       AIMCT_EXP_FULL=1 python experiments/40_mu_analysis_rs_rp/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aimct.controllers import solve_care
from aimct.controllers.hinf import StateSpace, mixsyn, weight_S, weight_T
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.robust import BlockStructure, dk_iterate, robust_stability_margin

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

# ---- Exp-35 plant ---------------------------------------------------------
G_NUM = [12.0]
G_DEN = np.polymul([1.0, 1.0], [1.0, 3.0])
OMEGA_R = 15.0
W_G = 0.45                                   # +/-45% loop-gain error
G = StateSpace.from_tf(G_NUM, G_DEN)

W_GRID = np.logspace(-2, 3, 600 if FULL else 260)
ZETA_GRID = np.linspace(0.40, 0.015, 24 if FULL else 14)
BLOCKS = BlockStructure([(1, "R"), (1, "C")])


def _R_of(zeta_r, w):
    s = 1j * w
    return OMEGA_R ** 2 / (s * s + 2.0 * zeta_r * OMEGA_R * s + OMEGA_R ** 2)


# ---- controllers --------------------------------------------------------
def lqg_controller():
    """A moderate LQG (nominal crossover ~5 rad/s) that is robust to the
    resonance *and* to the gain error taken one at a time - so any RS failure
    that mu reports is a genuinely *coupled* one."""
    A, B, C = G.A, G.B, G.C
    P = solve_care(A, B, 10.0 * (C.T @ C), np.array([[1.0]]))
    Klqr = B.T @ P
    Sig = solve_care(A.T, C.T, 25.0 * (B @ B.T), np.array([[1.0]]))
    Lkf = Sig @ C.T
    return StateSpace(A - B @ Klqr - Lkf @ C, Lkf, Klqr, np.zeros((1, 1)))


def hinf_design(dscale=1.0):
    WS = weight_S(wb=1.0, A=1e-3, M=2.0)
    WK = StateSpace.gain([[3e-2]])
    WT = weight_T(wb=1.2, A=1.0 / (60.0 * dscale), M=2.0)
    return mixsyn(G, WS, WK, WT, tol=1e-4)


# ---- M-Delta (rank-1, output-multiplicative) --------------------------
def _T_of(K, w):
    L = (G * K).freqresp(w)[:, 0, 0]
    return L / (1.0 + L)


def _M_builder(K, zeta_r):
    def M(w):
        t = _T_of(K, np.atleast_1d(w))[0]
        wm = _R_of(zeta_r, w) - 1.0
        return -t * np.array([[W_G, wm], [W_G, wm]], dtype=complex)
    return M


def _peaks(K, zeta_r):
    """Analytic peaks for the rank-1 structure:
      resonance only      max_w |W_m T|
      gain only           max_w |w_g T| = w_g ||T||_inf
      structured mu       max_w |T| (w_g + |W_m|)   (exact for M = a b^T)
    """
    L = (G * K).freqresp(W_GRID)[:, 0, 0]
    T = np.abs(L / (1.0 + L))
    wm = np.abs(_R_of(zeta_r, W_GRID) - 1.0)
    return (float(np.max(wm * T)),
            float(W_G * np.max(T)),
            float(np.max(T * (W_G + wm))))


def _nominal_margins(K):
    w = W_GRID
    L = (G * K).freqresp(w)[:, 0, 0]
    mag = np.abs(L)
    ph = np.degrees(np.unwrap(np.angle(L)))

    def _wrap(a):
        return ((a + 180.0) % 360.0) - 180.0

    d = _wrap(ph - 180.0)
    xs = np.where(np.diff(np.sign(d)) != 0)[0]
    gm = min((abs(20 * np.log10(1.0 / (mag[i] * (mag[i + 1] / mag[i])
             ** (d[i] / (d[i] - d[i + 1])))))
             for i in xs), default=np.inf)
    lm = mag - 1.0
    xs = np.where(np.diff(np.sign(lm)) != 0)[0]
    pm = min((abs(_wrap(ph[i] + (lm[i] / (lm[i] - lm[i + 1]))
             * (ph[i + 1] - ph[i]) + 180.0)) for i in xs), default=np.inf)
    return float(gm), float(pm)


def _nominal_pole_on_true_plant(K, zeta_r):
    rn = [OMEGA_R ** 2]
    rd = [1.0, 2.0 * zeta_r * OMEGA_R, OMEGA_R ** 2]
    Gp = StateSpace.from_tf(np.polymul(G_NUM, rn), np.polymul(G_DEN, rd))
    return float((Gp * K).feedback().poles().real.max())


# ======================================================================
def part_a():
    K = lqg_controller()
    gm0, pm0 = _nominal_margins(K)
    rows = []
    for zr in ZETA_GRID:
        rs = robust_stability_margin(_M_builder(K, zr), BLOCKS, W_GRID)
        wmT, wgT, mu_an = _peaks(K, zr)
        rows.append(dict(
            zeta_r=float(zr),
            mu_solver=rs["peak_upper"], mu_lb=rs["peak_lower"],
            mu_analytic=mu_an, wmT=wmT, wgT=wgT,
            rs_margin=1.0 / mu_an if mu_an > 0 else np.inf,
            nom_pole=_nominal_pole_on_true_plant(K, zr),
            gm_db=gm0, pm_deg=pm0,
        ))
    return rows, (gm0, pm0)


def part_b():
    base = hinf_design(1.0)
    zr_eval = 0.04
    last = {"res": base}

    def resynth(d):
        r = hinf_design(float(max(d[1], 1e-3)))
        last["res"] = r
        return r.K

    def mu_matrix(K, w):
        return _M_builder(K, zr_eval)(w)

    K_dk, hist = dk_iterate(resynth, mu_matrix, BLOCKS, W_GRID,
                            iterations=3 if FULL else 2)

    def peak_mu(K):
        rs = robust_stability_margin(_M_builder(K, zr_eval), BLOCKS, W_GRID)
        return rs["peak_upper"], rs["margin"]

    def s_peak(K):
        L = (G * K).freqresp(W_GRID)[:, 0, 0]
        return float(np.max(np.abs(1.0 / (1.0 + L))))

    bp, bm = peak_mu(base.K)
    dp, dm = peak_mu(K_dk)
    return dict(zr_eval=zr_eval, rounds=len(hist),
                base_peak=bp, base_margin=bm, base_Speak=s_peak(base.K),
                base_gamma=base.gamma,
                dk_peak=dp, dk_margin=dm, dk_Speak=s_peak(K_dk),
                dk_gamma=last["res"].gamma, history=hist)


def _cross(x, y):
    s = np.sign(y)
    xs = np.where(np.diff(s) != 0)[0]
    if not xs.size:
        return float("nan")
    i = xs[0]
    f = y[i] / (y[i] - y[i + 1])
    return float(x[i] + f * (x[i + 1] - x[i]))


def main():
    rows, (gm0, pm0) = part_a()
    b = part_b()

    zr = np.array([r["zeta_r"] for r in rows])
    zr_mu = _cross(zr, np.array([r["mu_analytic"] for r in rows]) - 1.0)
    zr_c = _cross(zr, np.array([r["wmT"] for r in rows]) - 1.0)
    zr_g = _cross(zr, np.array([r["wgT"] for r in rows]) - 1.0)
    zr_nom = _cross(zr, np.array([r["nom_pole"] for r in rows]))
    solver_err = max(abs(r["mu_solver"] - r["mu_analytic"]) /
                     max(r["mu_analytic"], 1e-9) for r in rows)

    def _fz(v):
        return "n/a" if not np.isfinite(v) else f"{v:.3f}"

    lines = [
        "# Experiment 40 - mu analysis: structured robust stability / performance",
        "",
        f"Exp-35 plant `G = 12/((s+1)(s+3))`, a moderate LQG (nominal crossover "
        f"~5 rad/s). Output-multiplicative uncertainty "
        f"`Delta = blkdiag(delta_r * {W_G:.2f} loop gain, Delta_c * (R(s)-1))`, "
        f"`omega_r = {OMEGA_R:.0f}` rad/s. Both enter at the plant output, so "
        f"`M = -T [w_g  W_m ; w_g  W_m]` is rank one and "
        f"`mu = max_w |T|(w_g + |W_m|)` exactly.",
        "",
        f"`aimct.robust.mu` solver peak vs that analytic value: "
        f"max relative error **{solver_err:.1e}** across the sweep.",
        "",
        "## Part (a) - as the resonance grows (zeta_r down)",
        "",
        f"LQG nominal single-loop margins (on `G` alone): **GM {gm0:.1f} dB, "
        f"PM {pm0:.1f} deg** - identical on every row below.",
        "",
        "| checked one at a time | crosses 1 at zeta_r |",
        "| --- | --- |",
        f"| gain error alone  `w_g ||T||_inf` | {_fz(zr_g)} (never - "
        f"`{rows[0]['wgT']:.2f}` and falling) |",
        f"| resonance alone   `||W_m T||_inf` | **{_fz(zr_c)}** |",
        f"| nominal mode `Delta_c = 1` destabilises `G*R` | **{_fz(zr_nom)}** |",
        "",
        f"| checked *together* | crosses 1 at zeta_r |",
        f"| --- | --- |",
        f"| structured `mu(delta_r, Delta_c)` | **{_fz(zr_mu)}** |",
        "",
        f"Neither uncertainty on its own takes the loop down anywhere in this "
        f"range - each single-loop check, and the nominal `G*R` pole, stays on "
        f"the safe side until `zeta_r ~ {_fz(zr_c)}` or below. `mu` says the "
        f"*combination* is already unsafe at `zeta_r ~ {_fz(zr_mu)}`: there is a "
        f"`||Delta|| <= 1` mixing the {W_G:.0%} gain error with a fraction of the "
        f"resonance that destabilises the loop. The gain and phase margins, "
        f"blind to structure, never move.",
        "",
        "| zeta_r | mu (analytic) | mu (solver ub) | mu (solver lb) | ||W_m T|| | w_g||T|| | RS margin 1/mu | nom `G*R` pole | GM dB | PM deg |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['zeta_r']:.3f} | {r['mu_analytic']:.3f} | {r['mu_solver']:.3f} "
            f"| {r['mu_lb']:.3f} | {r['wmT']:.3f} | {r['wgT']:.3f} | "
            f"{r['rs_margin']:.3f} | {r['nom_pole']:+.3f} | {r['gm_db']:.1f} | "
            f"{r['pm_deg']:.1f} |")

    lines += [
        "",
        "## Part (b) - constant-D D-K on the Exp-35 H-infinity design",
        "",
        f"Same structure at a severe `zeta_r = {b['zr_eval']}` (where plain "
        f"`mixsyn` peak mu is above 1), {b['rounds']} constant-D round(s):",
        "",
        "| design | peak mu | RS margin | ||S||_inf | gamma |",
        "| --- | --- | --- | --- | --- |",
        f"| plain mixsyn (Exp 35) | {b['base_peak']:.3f} | {b['base_margin']:.3f} "
        f"| {b['base_Speak']:.2f} | {b['base_gamma']:.3f} |",
        f"| + constant-D D-K | {b['dk_peak']:.3f} | {b['dk_margin']:.3f} "
        f"| {b['dk_Speak']:.2f} | {b['dk_gamma']:.3f} |",
        "",
    ]
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["zeta_r", "mu_analytic", "mu_solver_ub", "mu_solver_lb",
                     "wmT", "wgT", "rs_margin", "nom_pole", "gm_db", "pm_deg"])
        for r in rows:
            wr.writerow([r["zeta_r"], r["mu_analytic"], r["mu_solver"],
                        r["mu_lb"], r["wmT"], r["wgT"], r["rs_margin"],
                        r["nom_pole"], r["gm_db"], r["pm_deg"]])
        wr.writerow([])
        wr.writerow(["boundary", "zeta_r"])
        wr.writerow(["wgT=1", zr_g]); wr.writerow(["wmT=1", zr_c])
        wr.writerow(["GR_unstable", zr_nom]); wr.writerow(["mu=1", zr_mu])
        wr.writerow(["solver_vs_analytic_rel_err", solver_err])
        wr.writerow([])
        for k in ("zr_eval", "base_peak", "base_margin", "base_Speak",
                  "base_gamma", "dk_peak", "dk_margin", "dk_Speak", "dk_gamma"):
            wr.writerow([k, b[k]])

    _figure(rows, b, zr_mu, zr_c, zr_nom, gm0, pm0)
    print((HERE / "table.md").read_text())


def _figure(rows, b, zr_mu, zr_c, zr_nom, gm0, pm0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    zr = np.array([r["zeta_r"] for r in rows])
    mup = np.array([r["mu_analytic"] for r in rows])
    mulb = np.array([r["mu_lb"] for r in rows])
    wmT = np.array([r["wmT"] for r in rows])
    pole = np.array([r["nom_pole"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].plot(zr, mup, "-o", color=PALETTE["mpc"], lw=1.9, ms=4,
                  label=r"structured $\mu$ ($\delta_r + \Delta_c$)")
    ax[0, 0].plot(zr, mulb, ":", color=PALETTE["mpc"], lw=1.0)
    ax[0, 0].plot(zr, wmT, "-s", color=PALETTE["lqr"], lw=1.6, ms=3.5,
                  label=r"$\|W_m T\|_\infty$ (resonance only)")
    ax[0, 0].axhline(1.0, ls=":", color="0.4")
    for xv, lab, c in ((zr_mu, r"$\mu=1$", PALETTE["saturation"]),
                       (zr_c, r"$\|W_mT\|=1$", PALETTE["lqr"])):
        if np.isfinite(xv):
            ax[0, 0].axvline(xv, ls="--", color=c, lw=1.2)
    ax[0, 0].set(title="(a) structured vs resonance-only robustness",
                 xlabel=r"$\zeta_r$", ylabel="peak value")
    ax[0, 0].invert_xaxis()
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(zr, pole, "-o", color=PALETTE["lqr"], lw=1.9, ms=4)
    ax[0, 1].axhline(0.0, ls=":", color="0.4")
    if np.isfinite(zr_nom):
        ax[0, 1].axvline(zr_nom, ls="--", color=PALETTE["saturation"], lw=1.3,
                         label=fr"$\zeta_r$ unstable: {zr_nom:.3f}")
    ax[0, 1].set(title=r"(b) nominal LQG pole on $G\cdot R$", xlabel=r"$\zeta_r$",
                 ylabel="max Re(pole)")
    ax[0, 1].invert_xaxis()
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].axhline(gm0 / 12.0, color=PALETTE["lqr"], lw=2,
                     label=f"GM {gm0:.1f} dB  (/12)")
    ax[1, 0].axhline(pm0 / 60.0, color=PALETTE["mpc"], lw=2,
                     label=f"PM {pm0:.1f} deg  (/60)")
    ax[1, 0].plot(zr, 1.0 / np.maximum(mup, 1e-6), "-o", color=PALETTE["rl"],
                  lw=1.6, ms=3.5, label=r"$\mu$ RS margin $1/\mu$")
    ax[1, 0].axhline(1.0, ls=":", color="0.4")
    ax[1, 0].set(title="(c) single-loop margins are flat; the mu margin collapses",
                 xlabel=r"$\zeta_r$", ylabel="normalised margin")
    ax[1, 0].invert_xaxis()
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].bar([0, 1], [b["base_peak"], b["dk_peak"]],
                 color=[PALETTE["reference"], PALETTE["rl"]], alpha=0.9)
    ax[1, 1].axhline(1.0, ls=":", color="0.4")
    ax[1, 1].set_xticks([0, 1])
    ax[1, 1].set_xticklabels(["plain\nmixsyn", "+ D-K"])
    ax[1, 1].set(title=fr"(d) D-K on the H-$\infty$ design ($\zeta_r$="
                       fr"{b['zr_eval']})", ylabel=r"peak $\mu$")
    for i, v in enumerate([b["base_peak"], b["dk_peak"]]):
        ax[1, 1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(r"Exp 40 - $\mu$ catches the coupled uncertainty that "
                 r"gain / phase margins miss", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
