"""Experiment 35 - H-infinity vs LQG when a resonance is left out of the model.

The design model is a smooth second-order servo ``G(s)``.  The *true* plant is
``G_p(s) = G(s) * R(s)`` where ``R(s)`` is a lightly-damped high-frequency mode
(``omega_r`` well above crossover, ``zeta_r ~ 0.03``) that the designer never
put in the model - a structural / flexible-mode resonance.

* **LQG** shapes only the nominal loop.  It puts no explicit ceiling on ``|T|``
  past crossover, so when the resonance is actually there it sits inside the
  loop and the margins collapse.
* **H-infinity mixed-sensitivity** with a ``W_T`` whose high-frequency level
  exceeds the multiplicative-error size forces ``|T|`` to roll off *before*
  ``omega_r``.  By the small-gain theorem the loop is then robustly stable for
  every ``||Delta||_inf <= 1`` - the resonance included - at the cost of a
  little nominal bandwidth.

We report, nominal and perturbed, for each controller: closed-loop stability,
classical gain / phase margin, the skew-0 **disk margin**, and the ``||S||_inf``
/ ``||T||_inf`` peaks.

Run:   python experiments/35_hinf_vs_lqg/run.py
       AIMCT_EXP_FULL=1 python experiments/35_hinf_vs_lqg/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aimct.controllers import solve_care
from aimct.controllers.hinf import StateSpace, mixsyn, weight_S, weight_T
from aimct.plot_style import PALETTE, set_aimct_style

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

# ---- plant: nominal design model + the unmodelled resonance --------------
# nominal  G(s) = 12 / ((s + 1)(s + 3))     (DC gain 4, poles -1 and -3)
G_NUM = [12.0]
G_DEN = np.polymul([1.0, 1.0], [1.0, 3.0])          # (s+1)(s+3)
OMEGA_R = 12.0                                       # rad/s, a few x the H-inf crossover
ZETA_R = 0.08                                        # lightly damped (peak mult. error ~6)
R_NUM = [OMEGA_R ** 2]
R_DEN = [1.0, 2.0 * ZETA_R * OMEGA_R, OMEGA_R ** 2]

G = StateSpace.from_tf(G_NUM, G_DEN)
GP = StateSpace.from_tf(np.polymul(G_NUM, R_NUM), np.polymul(G_DEN, R_DEN))

W_GRID = np.logspace(-2, 3, 4000 if FULL else 1500)


# ======================================================================
# controllers
# ======================================================================
def design_lqg():
    """Textbook LQG on the nominal plant, tuned AGGRESSIVELY (nominal loop
    crossover ~7 rad/s): tight nominal tracking, but nothing constrains |T| out
    where the resonance lives.  Folded into a y -> u controller in the
    S = (I + G K)^-1 sign convention."""
    A, B, C = G.A, G.B, G.C
    q, r = 60.0, 1.0
    P = solve_care(A, B, q * (C.T @ C), np.array([[r]]))
    Klqr = (B.T @ P) / r                                    # u = -Klqr x
    wv, vv = 200.0, 1.0                                     # fast observer
    S = solve_care(A.T, C.T, wv * (B @ B.T), np.array([[vv]]))
    Lkf = (S @ C.T) / vv
    Ak = A - B @ Klqr - Lkf @ C
    return StateSpace(Ak, Lkf, Klqr, np.zeros((1, 1))), dict(q=q, r=r, wv=wv, vv=vv)


def design_hinf():
    """H-infinity mixed-sensitivity, tuned CONSERVATIVELY: W_S asks for only a
    ~1 rad/s bandwidth and W_T rolls |T| off hard from ~1.2 rad/s up to a
    high-frequency level (1/A = 60) well above the resonance's multiplicative
    error - trading nominal bandwidth for robust stability past omega_r."""
    wb_S, wb_T, inv_A_T = 1.0, 1.2, 60.0
    WS = weight_S(wb=wb_S, A=1e-3, M=2.0)
    WK = StateSpace.gain([[3e-2]])
    WT = weight_T(wb=wb_T, A=1.0 / inv_A_T, M=2.0)
    res = mixsyn(G, WS, WK, WT, tol=1e-4)
    return res.K, res, dict(wb_S=wb_S, wb_T=wb_T, invA_T=inv_A_T, gamma=res.gamma)


# ======================================================================
# frequency-domain margins (from scratch, SISO)
# ======================================================================
def _loop(Gp: StateSpace, K: StateSpace, w):
    return (Gp * K).freqresp(w)[:, 0, 0]


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def margins(Gp: StateSpace, K: StateSpace, w):
    L = _loop(Gp, K, w)
    S = 1.0 / (1.0 + L)
    T = L / (1.0 + L)
    mag = np.abs(L)
    ph_deg = np.degrees(np.unwrap(np.angle(L)))

    cl = (Gp * K).feedback()                 # T = L/(1+L) as a state space
    stable = cl.is_stable(tol=1e-7)

    gm_db, pm_deg = np.nan, np.nan
    if stable:
        # gain margin: smallest |1/|L|| over the phase = -180 (mod 360) crossings
        d = _wrap_deg(ph_deg - 180.0)
        xs = np.where(np.diff(np.sign(d)) != 0)[0]
        gms = []
        for i in xs:
            f = d[i] / (d[i] - d[i + 1])
            magc = mag[i] * (mag[i + 1] / mag[i]) ** f
            if magc > 1e-9:
                gms.append(abs(20.0 * np.log10(1.0 / magc)))
        gm_db = min(gms) if gms else np.inf
        # phase margin: smallest |180 + angle| over the |L| = 1 crossings
        lm = mag - 1.0
        xs = np.where(np.diff(np.sign(lm)) != 0)[0]
        pms = []
        for i in xs:
            f = lm[i] / (lm[i] - lm[i + 1])
            phc = ph_deg[i] + f * (ph_deg[i + 1] - ph_deg[i])
            pms.append(abs(_wrap_deg(phc + 180.0)))
        pm_deg = min(pms) if pms else np.inf

    # skew-0 disk margin (Seiler-Packard-Gahinet):  alpha = 1 / max_w |S - 0.5|
    alpha = 1.0 / float(np.max(np.abs(S - 0.5)))
    if not stable:
        dm_gain_db, dm_phase_deg = 0.0, 0.0
    elif alpha < 1.0:
        dm_gain_db = 20.0 * np.log10((1.0 + alpha) / (1.0 - alpha))
        dm_phase_deg = 2.0 * np.degrees(np.arctan(alpha))
    else:
        dm_gain_db, dm_phase_deg = np.inf, 180.0

    return dict(
        stable=bool(stable),
        gm_db=float(gm_db),
        pm_deg=float(pm_deg),
        disk_alpha=float(alpha) if stable else 0.0,
        dm_gain_db=float(dm_gain_db),
        dm_phase_deg=float(dm_phase_deg),
        Sinf=float(np.max(np.abs(S))),
        Tinf=float(np.max(np.abs(T))),
        _L=L, _S=S, _T=T,
    )


# ======================================================================
# main
# ======================================================================
def _fmt(v, unit=""):
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if not np.isfinite(v):
        return "inf"
    return f"{v:.2f}{unit}"


def main():
    K_lqg, lqg_cfg = design_lqg()
    K_hinf, hinf_res, hinf_cfg = design_hinf()
    print(f"H-infinity synthesis: gamma = {hinf_res.gamma:.3f}")

    rows = []
    for name, K in (("LQG", K_lqg), ("H-infinity", K_hinf)):
        nom = margins(G, K, W_GRID)
        per = margins(GP, K, W_GRID)
        rows.append((name, nom, per))

    cols = ["stable", "gm_db", "pm_deg", "disk_alpha", "dm_gain_db",
            "dm_phase_deg", "Sinf", "Tinf"]
    hdr = ["controller", "plant"] + cols
    lines = [
        "# Experiment 35 - H-infinity vs LQG under an unmodelled resonance",
        "",
        f"nominal `G(s) = 12 / ((s+1)(s+3))`; true plant `G_p = G * R` with "
        f"`R(s)` a mode at `omega_r = {OMEGA_R:.0f}` rad/s, `zeta_r = {ZETA_R}` "
        f"(peak multiplicative error ~ {1/(2*ZETA_R):.0f}).",
        "",
        f"H-infinity achieved `gamma = {hinf_res.gamma:.3f}`.  "
        f"`disk_alpha` = skew-0 disk margin `1 / max_w |S - 0.5|`; "
        f"`dm_gain_db` / `dm_phase_deg` are the simultaneous gain/phase "
        f"variation it guarantees.",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join(["---"] * len(hdr)) + "|",
    ]
    for name, nom, per in rows:
        for tag, m in (("nominal", nom), ("perturbed", per)):
            if m["stable"]:
                cells = [name, tag, "yes",
                         _fmt(m["gm_db"], " dB"), _fmt(m["pm_deg"], " deg"),
                         _fmt(m["disk_alpha"]), _fmt(m["dm_gain_db"], " dB"),
                         _fmt(m["dm_phase_deg"], " deg"),
                         _fmt(m["Sinf"]), _fmt(m["Tinf"])]
            else:
                cells = [name, tag, "**NO**", "--", "--", "--", "--", "--",
                         _fmt(m["Sinf"]), _fmt(m["Tinf"])]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(hdr)
        for name, nom, per in rows:
            for tag, m in (("nominal", nom), ("perturbed", per)):
                wr.writerow([name, tag] + [m[c] for c in cols])

    _figure(rows, K_lqg, K_hinf, hinf_cfg)
    print((HERE / "table.md").read_text())


def _figure(rows, K_lqg, K_hinf, hinf_cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    w = W_GRID
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    col = {"LQG": PALETTE["lqr"], "H-infinity": PALETTE["mpc"]}

    # (a) open-loop gain |L| on the TRUE plant
    for name, _nom, per in rows:
        ax[0, 0].loglog(w, np.abs(per["_L"]), color=col[name], lw=1.8,
                        label=f"{name} (on $G_p$)")
    ax[0, 0].axhline(1.0, ls=":", color="0.5", lw=1.0)
    ax[0, 0].axvline(OMEGA_R, ls="--", color=PALETTE["saturation"], lw=1.2,
                     label=r"$\omega_r$")
    ax[0, 0].set(title="(a) loop gain $|L(j\\omega)|$ on the true plant",
                 xlabel="rad/s", ylabel="|L|")
    ax[0, 0].legend(fontsize=8)

    # (b) sensitivity |S|
    for name, nom, per in rows:
        ax[0, 1].loglog(w, np.abs(nom["_S"]), color=col[name], lw=1.2, ls=":")
        ax[0, 1].loglog(w, np.abs(per["_S"]), color=col[name], lw=1.8,
                        label=f"{name} on $G_p$")
    ax[0, 1].axvline(OMEGA_R, ls="--", color=PALETTE["saturation"], lw=1.2)
    ax[0, 1].set(title="(b) sensitivity $|S|$  (dotted = nominal)",
                 xlabel="rad/s", ylabel="|S|")
    ax[0, 1].legend(fontsize=8)

    # (c) complementary sensitivity |T| with the W_T^-1 bound
    WT = weight_T(wb=hinf_cfg["wb_T"], A=1.0 / hinf_cfg["invA_T"], M=2.0)
    inv_wt = 1.0 / np.abs(WT.freqresp(w)[:, 0, 0])
    for name, nom, per in rows:
        ax[1, 0].loglog(w, np.abs(per["_T"]), color=col[name], lw=1.8,
                        label=f"{name} on $G_p$")
    ax[1, 0].loglog(w, inv_wt, color=PALETTE["reference"], lw=1.4, ls="--",
                    label=r"$|W_T|^{-1}$ bound")
    ax[1, 0].axvline(OMEGA_R, ls="--", color=PALETTE["saturation"], lw=1.2)
    ax[1, 0].set(title="(c) complementary sensitivity $|T|$",
                 xlabel="rad/s", ylabel="|T|", ylim=(1e-3, 5))
    ax[1, 0].legend(fontsize=8)

    # (d) margin bars: disk gain / phase margin, nominal vs perturbed
    labels = ["GM (dB)", "PM (deg)", "disk GM (dB)", "disk PM (deg)"]
    keys = ["gm_db", "pm_deg", "dm_gain_db", "dm_phase_deg"]
    x = np.arange(len(labels))
    width = 0.2
    for i, (name, nom, per) in enumerate(rows):
        for j, (tag, m) in enumerate((("nom", nom), ("pert", per))):
            vals = [min(m[k], 60.0) for k in keys]      # clip inf for the bar
            off = (i * 2 + j - 1.5) * width
            ax[1, 1].bar(x + off, vals, width,
                         color=col[name], alpha=0.9 if tag == "pert" else 0.4,
                         label=f"{name} {tag}")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(labels, fontsize=8)
    ax[1, 1].set(title="(d) margins: nominal (pale) vs perturbed (solid)",
                 ylabel="margin (clipped at 60)")
    ax[1, 1].legend(fontsize=7, ncol=2)

    fig.suptitle("Exp 35 - H-infinity keeps the margin LQG loses to an "
                 "unmodelled resonance", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
