"""Experiment 32 - direct collocation vs shooting vs sampling, OFFLINE.

Experiments 24-26 raced iLQR (indirect / single shooting) against CEM (sampling)
as *online* receding-horizon controllers.  This one steps back to the *offline*
planning problem they all rest on: compute one open-loop minimum-effort
cart-pole swing-up

    min  integral_0^T  u(t)^2 dt
    s.t. xdot = f(x, u),   x(0) = [0, 0, pi, 0],   x(T) = [0, 0, 0, 0],
         |u| <= 20 N

three ways, and compare what each *paradigm* actually delivers:

* **Direct collocation** (`aimct.planning.DirectCollocation`) - transcribe to an
  NLP (state + input at every knot, Hermite-Simpson defect equalities) and hand
  it to SLSQP.  The terminal condition is a hard constraint.
* **iLQR / single shooting** (`aimct.controllers.iLQR`) - a Riccati sweep over
  the true rollout.  It has no mechanism for a hard terminal constraint, so the
  boundary condition enters as a large terminal *penalty* ``Qf``.
* **CEM / sampling** (`aimct.controllers.SamplingMPC`, run open-loop over the
  whole horizon) - derivative-free; the terminal condition is again a penalty.

Scored on: achieved effort ``integral(u^2)``, **terminal-state error** (planned,
and after re-integrating the returned ``u`` through the true dynamics with a
fine RK4 - the honest "does the plan actually fly" check), max inter-knot
dynamics drift, whether the input box is respected, and wall-clock to solve.

Run:   python experiments/32_direct_collocation_vs_ilqr/run.py
       AIMCT_EXP_FULL=1 python experiments/32_direct_collocation_vs_ilqr/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from aimct.controllers import iLQR
from aimct.ml.planning import system_step
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.planning import DirectCollocation
from aimct.systems import CartPole

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"
SEED = 0

# ---- the shared optimal-control problem ---------------------------------
CP = CartPole()
X0 = np.array([0.0, 0.0, np.pi, 0.0])       # hanging, at rest, cart at origin
XG = np.array([0.0, 0.0, 0.0, 0.0])         # upright, at rest, cart at origin
T_FINAL = 2.0
F_MAX = 20.0
R_EFFORT = 1.0                              # weight on integral(u^2)

DT = 0.02                                   # iLQR / CEM discretisation
H = int(round(T_FINAL / DT))               # 100 steps
N_KNOTS = 41 if FULL else 21               # collocation knots


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _term_err(x):
    """Terminal-state error to XG, with the pole angle wrapped to (-pi, pi]."""
    e = np.array(x, float) - XG
    e[2] = _wrap(x[2] - XG[2])
    return float(np.linalg.norm(e))


def _simulate(X0v, U, t, hold, sub=40):
    """Re-integrate the plan on the TRUE dynamics with a fine RK4.

    ``hold`` selects the inter-knot input model the planner itself assumed:
    ``"foh"`` (linear between knots - direct collocation) or ``"zoh"``
    (piecewise-constant per step - iLQR / CEM).  Returns the state at the same
    knot times as ``t`` so it lines up with the planned ``X``.
    """
    f = lambda x, u: np.asarray(CP.dynamics(0.0, x, u), float)
    x = np.array(X0v, float)
    out = [x.copy()]
    for k in range(len(t) - 1):
        h = (t[k + 1] - t[k]) / sub
        for j in range(sub):
            if hold == "foh":
                frac = (j + 0.5) / sub
                u = (1.0 - frac) * U[k] + frac * U[k + 1]
            else:
                u = U[k]
            k1 = f(x, u)
            k2 = f(x + 0.5 * h * k1, u)
            k3 = f(x + 0.5 * h * k2, u)
            k4 = f(x + h * k3, u)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x.copy())
    return np.array(out)


# ======================================================================
# planners
# ======================================================================
def plan_collocation():
    dc = DirectCollocation.from_system(
        CP, t_final=T_FINAL, N=N_KNOTS, x0=X0, x_goal=XG,
        Q=0.0, R=R_EFFORT, Qf=0.0, u_bounds=(-F_MAX, F_MAX),
        max_iter=500, tol=1e-9,
    )
    t0 = time.perf_counter()
    res = dc.solve()
    ms = (time.perf_counter() - t0) * 1e3
    return dict(name="Direct collocation (HS)", t=res.t, X=res.X, U=res.U, hold="foh",
                solve_ms=ms, converged=res.success, defect_norm=res.defect_norm,
                message=res.message)


def plan_ilqr():
    # No hard terminal constraint in shooting: the boundary condition is a
    # large terminal penalty.  Q ~ 0 keeps the *running* objective effort-only.
    Qf = np.diag([400.0, 40.0, 800.0, 40.0])
    opt = iLQR.from_system(
        CP, DT, horizon=H, Q=1e-4, R=R_EFFORT * DT, Qf=Qf,
        x_ref=XG, u_bounds=(-F_MAX, F_MAX), max_iter=600,
    )
    U_init = np.zeros((H, 1))
    t0 = time.perf_counter()
    res = opt.solve(X0, U_init)
    ms = (time.perf_counter() - t0) * 1e3
    t = np.arange(H + 1) * DT
    U = np.vstack([res.U, res.U[-1]])           # (H+1, 1): last control held
    return dict(name="iLQR / single shooting", t=t, X=res.X, U=U, hold="zoh",
                solve_ms=ms, converged=res.converged, defect_norm=np.nan,
                message=f"{res.iters} iters")


def plan_cem():
    from aimct.controllers import SamplingMPC

    step = system_step(CP, DT)
    Qf = np.diag([400.0, 40.0, 800.0, 40.0])

    def running_cost(X, U):
        return R_EFFORT * DT * U[:, 0] ** 2

    def terminal_cost(X):
        e = np.stack([X[:, 0], X[:, 1], _wrap(X[:, 2]), X[:, 3]], axis=1)
        return np.einsum("bi,ij,bj->b", e, Qf, e)

    cem = SamplingMPC(step, running_cost, terminal_cost=terminal_cost, horizon=H,
                      n_samples=2000 if FULL else 800, n_elite=120 if FULL else 60,
                      n_iter=6, u_dim=1, u_bounds=(-F_MAX, F_MAX), seed=SEED)
    n_outer = 40 if FULL else 20
    t0 = time.perf_counter()
    for _ in range(n_outer):
        cem._plan(X0)
    ms = (time.perf_counter() - t0) * 1e3

    U = np.vstack([cem.mu, cem.mu[-1]])          # (H+1, 1), last held
    X = np.empty((H + 1, 4))
    X[0] = X0
    for k in range(H):
        X[k + 1] = step(X[k][None], U[k][None])[0]
    t = np.arange(H + 1) * DT
    return dict(name="CEM / sampling", t=t, X=X, U=U, solve_ms=ms, hold="zoh",
                converged=True, defect_norm=np.nan,
                message=f"{n_outer * cem.n_iter} CEM iters")


# ======================================================================
# score + report
# ======================================================================
def _score(p):
    t, X, U = p["t"], p["X"], p["U"]
    Xsim = _simulate(X[0], U, t, p["hold"])
    effort = float(np.trapezoid(U[:, 0] ** 2, t))
    peak_u = float(np.max(np.abs(U)))
    drift = np.linalg.norm(Xsim - X, axis=1)
    return dict(
        name=p["name"],
        effort=effort,
        term_err_planned=_term_err(X[-1]),
        term_err_rolled=_term_err(Xsim[-1]),
        max_dyn_drift=float(np.max(drift)),
        peak_u_N=peak_u,
        box_ok=bool(peak_u <= F_MAX + 1e-6),
        solve_ms=p["solve_ms"],
        converged=bool(p["converged"]),
        defect_norm=p["defect_norm"],
        knots=len(t),
        _plan=p,
        _Xsim=Xsim,
    )


COLS = ["effort", "term_err_planned", "term_err_rolled", "max_dyn_drift",
        "peak_u_N", "box_ok", "knots", "solve_ms", "converged"]


def main():
    plans = [plan_collocation(), plan_ilqr(), plan_cem()]
    rows = [_score(p) for p in plans]

    lines = [
        "# Experiment 32 - direct collocation vs shooting vs sampling (offline)",
        "",
        f"minimum-effort cart-pole swing-up: `min integral(u^2) dt` s.t. the "
        f"dynamics, `x(0)=[0,0,pi,0]`, `x(T)=[0,0,0,0]`, `|u|<={F_MAX:.0f}` N, "
        f"`T={T_FINAL:.1f}` s.",
        "",
        "`term_err_planned` = ||x(T) - goal|| from the planner's own knots; "
        "`term_err_rolled` = same after re-integrating the returned `u` through "
        "the true dynamics (fine RK4, first-order hold). `max_dyn_drift` = worst "
        "knot mismatch between the plan and that re-integration.",
        "",
        "| planner | " + " | ".join(COLS) + " |",
        "| --- |" + " --- |" * len(COLS),
    ]
    for r in rows:
        cells = []
        for c in COLS:
            v = r[c]
            if isinstance(v, bool):
                cells.append("yes" if v else "**no**")
            elif c == "solve_ms":
                cells.append(f"{v:.0f}")
            elif c == "knots":
                cells.append(str(v))
            else:
                cells.append(f"{v:.4g}")
        lines.append("| " + r["name"] + " | " + " | ".join(cells) + " |")
    lines += ["",
              f"Collocation Hermite-Simpson defect norm at the solution: "
              f"{rows[0]['defect_norm']:.2e} (the NLP equalities are satisfied to "
              f"solver tolerance; the residual drift is the inter-knot "
              f"quadrature error).", ""]

    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["planner", *COLS, "defect_norm"])
        for r in rows:
            w.writerow([r["name"], *(r[c] for c in COLS), r["defect_norm"]])

    _figure(rows)
    print((HERE / "table.md").read_text())


def _figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    colors = [PALETTE["mpc"], PALETTE["lqr"], PALETTE["rl"]]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    for i, r in enumerate(rows):
        p = r["_plan"]
        t, X, U = p["t"], p["X"], p["U"]
        c = colors[i]
        ax[0, 0].plot(t, np.degrees(_wrap(X[:, 2])), color=c, lw=1.8, label=r["name"])
        ax[0, 1].plot(t, X[:, 0], color=c, lw=1.8, label=r["name"])
        ax[1, 0].plot(t, U[:, 0], color=c, lw=1.8, label=r["name"])
        # planned vs re-integrated pole angle
        Xs = r["_Xsim"]
        ax[0, 0].plot(t, np.degrees(_wrap(Xs[:, 2])), color=c, lw=1.0, ls=":")

    ax[0, 0].axhline(0.0, ls="--", color=PALETTE["reference"], lw=1.0)
    ax[0, 0].set(title="(a) pole angle  (solid = plan, dotted = re-integrated)",
                 xlabel="t [s]", ylabel="angle from upright [deg]")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(0.0, ls="--", color=PALETTE["reference"], lw=1.0)
    ax[0, 1].set(title="(b) cart position", xlabel="t [s]", ylabel="x [m]")
    ax[1, 0].axhline(F_MAX, ls=":", color=PALETTE["saturation"], lw=1.2)
    ax[1, 0].axhline(-F_MAX, ls=":", color=PALETTE["saturation"], lw=1.2)
    ax[1, 0].set(title="(c) control input", xlabel="t [s]", ylabel="force [N]")
    ax[1, 0].legend(fontsize=8)

    # (d) effort vs terminal error, bars
    names = [r["name"].split(" / ")[0].split(" (")[0] for r in rows]
    xs = np.arange(len(rows))
    eff = [r["effort"] for r in rows]
    ax[1, 1].bar(xs - 0.2, eff, width=0.4, color=colors, alpha=0.85,
                 label="effort integral(u^2)")
    ax[1, 1].set_ylabel("effort  integral(u^2) [N^2 s]")
    ax[1, 1].set_xticks(xs)
    ax[1, 1].set_xticklabels(names, fontsize=8)
    axr = ax[1, 1].twinx()
    axr.bar(xs + 0.2, [r["term_err_rolled"] for r in rows], width=0.4,
            color="0.4", alpha=0.7)
    axr.set_ylabel("terminal error (re-integrated)  [norm]")
    axr.set_yscale("log")
    ax[1, 1].set_title("(d) effort (colour) vs terminal error (grey, log)")

    fig.suptitle("Exp 32 - offline swing-up: direct collocation vs iLQR vs CEM",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
