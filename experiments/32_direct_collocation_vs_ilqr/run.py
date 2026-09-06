"""Experiment 32 - direct collocation vs shooting vs sampling, OFFLINE.

Experiments 24-26 raced iLQR (indirect / single shooting) against CEM (sampling)
as *online* receding-horizon controllers.  This one steps back to the *offline*
planning problem they all rest on and asks what each *paradigm* delivers when
the constraint that matters is expressed as a **hard constraint** (direct
transcription can) versus a **penalty** (shooting and sampling must).

Task A - minimum-effort cart-pole swing-up with a hard TERMINAL constraint::

    min integral u^2 dt   s.t.  xdot = f(x, u),  x(0) = [0,0,pi,0],
                                x(T) = [0,0,0,0],  |u| <= 20 N

Task B - a planar point mass from A to B around a circular KEEP-OUT disk, a
hard PATH constraint::

    min integral ||a||^2 dt   s.t.  double integrator,  x(0) = A,  x(T) = B,
                                    (x-c_x)^2 + (y-c_y)^2 >= r^2,  |a| <= a_max

Three planners on each:

* **Direct collocation** (`aimct.planning.DirectCollocation`) - transcribe to an
  NLP (state + input at every knot, Hermite-Simpson defect equalities); the
  terminal / path condition is a **hard constraint**.
* **iLQR / single shooting** (`aimct.controllers.iLQR`) - a Riccati sweep over
  the true rollout; the condition is a large **penalty** (terminal ``Qf`` /
  a quartic barrier in a custom cost).
* **CEM / sampling** (`aimct.controllers.SamplingMPC`, open-loop over the whole
  horizon) - derivative-free; the condition is again a **penalty**.

Scored on: achieved objective, whether the constraint is actually met (terminal
error / minimum disk clearance), whether the plan survives re-integration
through the true dynamics, and wall-clock to solve.

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

COLORS = [PALETTE["mpc"], PALETTE["lqr"], PALETTE["rl"]]


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _simulate(f, X0v, U, t, hold, sub=40):
    """Re-integrate a plan on the TRUE dynamics ``f(x, u)`` with a fine RK4.

    ``hold`` picks the inter-knot input model the planner assumed: ``"foh"``
    (linear between knots - direct collocation) or ``"zoh"`` (piecewise constant
    per step - iLQR / CEM).  Returns the state at the same knot times as ``t``.
    """
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
# TASK A - minimum-effort cart-pole swing-up (hard terminal constraint)
# ======================================================================
CP = CartPole()
A_X0 = np.array([0.0, 0.0, np.pi, 0.0])
A_XG = np.array([0.0, 0.0, 0.0, 0.0])
A_T = 2.0
A_FMAX = 20.0
A_DT = 0.02
A_H = int(round(A_T / A_DT))
A_N = 41 if FULL else 21
A_QF = np.diag([400.0, 40.0, 800.0, 40.0])

_cp_f = lambda x, u: np.asarray(CP.dynamics(0.0, x, u), float)


def _a_term_err(x):
    e = np.array(x, float) - A_XG
    e[2] = _wrap(x[2] - A_XG[2])
    return float(np.linalg.norm(e))


def plan_a_collocation():
    dc = DirectCollocation.from_system(
        CP, t_final=A_T, N=A_N, x0=A_X0, x_goal=A_XG,
        Q=0.0, R=1.0, Qf=0.0, u_bounds=(-A_FMAX, A_FMAX), max_iter=500, tol=1e-9,
    )
    t0 = time.perf_counter()
    res = dc.solve()
    ms = (time.perf_counter() - t0) * 1e3
    return dict(name="Direct collocation (HS)", t=res.t, X=res.X, U=res.U,
                hold="foh", f=_cp_f, solve_ms=ms, converged=res.success,
                defect_norm=res.defect_norm)


def plan_a_ilqr():
    opt = iLQR.from_system(
        CP, A_DT, horizon=A_H, Q=1e-4, R=1.0 * A_DT, Qf=A_QF,
        x_ref=A_XG, u_bounds=(-A_FMAX, A_FMAX), max_iter=600,
    )
    t0 = time.perf_counter()
    res = opt.solve(A_X0, np.zeros((A_H, 1)))
    ms = (time.perf_counter() - t0) * 1e3
    U = np.vstack([res.U, res.U[-1]])
    return dict(name="iLQR / single shooting", t=np.arange(A_H + 1) * A_DT,
                X=res.X, U=U, hold="zoh", f=_cp_f, solve_ms=ms,
                converged=res.converged, defect_norm=np.nan)


def plan_a_cem():
    from aimct.controllers import SamplingMPC

    step = system_step(CP, A_DT)

    def running_cost(X, U):
        return 1.0 * A_DT * U[:, 0] ** 2

    def terminal_cost(X):
        e = np.stack([X[:, 0], X[:, 1], _wrap(X[:, 2]), X[:, 3]], axis=1)
        return np.einsum("bi,ij,bj->b", e, A_QF, e)

    cem = SamplingMPC(step, running_cost, terminal_cost=terminal_cost, horizon=A_H,
                      n_samples=2000 if FULL else 800, n_elite=120 if FULL else 60,
                      n_iter=6, u_dim=1, u_bounds=(-A_FMAX, A_FMAX), seed=SEED)
    n_outer = 40 if FULL else 20
    t0 = time.perf_counter()
    for _ in range(n_outer):
        cem._plan(A_X0)
    ms = (time.perf_counter() - t0) * 1e3
    U = np.vstack([cem.mu, cem.mu[-1]])
    X = np.empty((A_H + 1, 4))
    X[0] = A_X0
    for k in range(A_H):
        X[k + 1] = step(X[k][None], U[k][None])[0]
    return dict(name="CEM / sampling", t=np.arange(A_H + 1) * A_DT, X=X, U=U,
                hold="zoh", f=_cp_f, solve_ms=ms, converged=True, defect_norm=np.nan)


def score_a(p):
    t, X, U = p["t"], p["X"], p["U"]
    Xsim = _simulate(p["f"], X[0], U, t, p["hold"])
    return dict(
        name=p["name"],
        effort=float(np.trapezoid(U[:, 0] ** 2, t)),
        term_err_planned=_a_term_err(X[-1]),
        term_err_rolled=_a_term_err(Xsim[-1]),
        max_dyn_drift=float(np.max(np.linalg.norm(Xsim - X, axis=1))),
        peak_u=float(np.max(np.abs(U))),
        box_ok=bool(np.max(np.abs(U)) <= A_FMAX + 1e-6),
        knots=len(t),
        solve_ms=p["solve_ms"],
        converged=bool(p["converged"]),
        defect_norm=p["defect_norm"],
        _plan=p, _Xsim=Xsim,
    )


A_COLS = ["effort", "term_err_planned", "term_err_rolled", "max_dyn_drift",
          "peak_u", "box_ok", "knots", "solve_ms", "converged"]


# ======================================================================
# TASK B - planar point mass around a keep-out disk (hard path constraint)
# ======================================================================
B_A = np.array([-2.0, 0.0, 0.0, 0.0])       # start [x, y, vx, vy]
B_B = np.array([2.0, 0.0, 0.0, 0.0])        # goal
B_CX, B_CY, B_RK = 0.0, 0.0, 0.7            # keep-out disk centre + radius
B_T = 4.0
B_AMAX = 3.0
B_DT = 0.05
B_H = int(round(B_T / B_DT))
B_N = 41 if FULL else 31
B_QF = np.diag([120.0, 120.0, 12.0, 12.0])  # iLQR / CEM soft terminal
B_WOBS = 3.0e3                              # iLQR / CEM disk-barrier weight


def _pm_f(x, u):
    return np.array([x[2], x[3], u[0], u[1]])


def _pm_step(x, u):                         # exact constant-accel discretisation
    x = np.atleast_1d(x); u = np.atleast_1d(u)
    return np.array([x[0] + B_DT * x[2] + 0.5 * B_DT ** 2 * u[0],
                     x[1] + B_DT * x[3] + 0.5 * B_DT ** 2 * u[1],
                     x[2] + B_DT * u[0],
                     x[3] + B_DT * u[1]])


def _pm_step_batched(X, U):
    X = np.asarray(X, float); U = np.asarray(U, float)
    return np.stack([
        X[:, 0] + B_DT * X[:, 2] + 0.5 * B_DT ** 2 * U[:, 0],
        X[:, 1] + B_DT * X[:, 3] + 0.5 * B_DT ** 2 * U[:, 1],
        X[:, 2] + B_DT * U[:, 0],
        X[:, 3] + B_DT * U[:, 1],
    ], axis=1)


def _pm_path_con(X, U):
    return B_RK ** 2 - ((X[:, 0] - B_CX) ** 2 + (X[:, 1] - B_CY) ** 2)  # <= 0


def _disk_penalty_batch(X):
    d2 = (X[:, 0] - B_CX) ** 2 + (X[:, 1] - B_CY) ** 2
    return B_WOBS * np.maximum(0.0, B_RK ** 2 - d2) ** 2


def _disk_cost_terms(px, py):
    """Quartic keep-out barrier value + exact grad / Hessian w.r.t. (px, py)."""
    dx, dy = px - B_CX, py - B_CY
    g = B_RK ** 2 - (dx ** 2 + dy ** 2)
    h = max(0.0, g)
    active = 1.0 if g > 0.0 else 0.0
    total = B_WOBS * h ** 2
    gx = -4.0 * B_WOBS * h * dx
    gy = -4.0 * B_WOBS * h * dy
    hxx = 8.0 * B_WOBS * active * dx ** 2 - 4.0 * B_WOBS * h
    hyy = 8.0 * B_WOBS * active * dy ** 2 - 4.0 * B_WOBS * h
    hxy = 8.0 * B_WOBS * active * dx * dy
    return total, gx, gy, hxx, hyy, hxy


def _check_disk_cost_terms(eps=1e-6, tol=1e-4):
    rng = np.random.default_rng(0)
    for _ in range(20):
        px, py = rng.uniform(-1.2, 1.2, size=2)
        _, gx, gy, hxx, hyy, hxy = _disk_cost_terms(px, py)
        fx = lambda a, b: _disk_cost_terms(a, b)[0]
        ngx = (fx(px + eps, py) - fx(px - eps, py)) / (2 * eps)
        ngy = (fx(px, py + eps) - fx(px, py - eps)) / (2 * eps)
        nhxx = (fx(px + eps, py) - 2 * fx(px, py) + fx(px - eps, py)) / eps ** 2
        nhxy = (fx(px + eps, py + eps) - fx(px + eps, py - eps)
                - fx(px - eps, py + eps) + fx(px - eps, py - eps)) / (4 * eps ** 2)
        for a, b in ((gx, ngx), (gy, ngy)):
            assert abs(a - b) <= tol * (1 + abs(b)), (a, b)
        # Hessian: skip points within ~eps of the barrier edge (kink)
        if abs(B_RK ** 2 - ((px - B_CX) ** 2 + (py - B_CY) ** 2)) > 1e-2:
            assert abs(hxx - nhxx) <= 1e-2 * (1 + abs(nhxx)), (hxx, nhxx)
            assert abs(hxy - nhxy) <= 1e-2 * (1 + abs(nhxy)), (hxy, nhxy)


_check_disk_cost_terms()


def _pm_warm(N):
    """A detour arc over the top of the disk, with the acceleration that traces
    it - the shared warm start for all three planners.  (A knot sitting exactly
    on the disk centre would zero the path-constraint gradient there and make
    the SLSQP LSQ subproblem singular, so the guess must bulge.)"""
    tt = np.linspace(0.0, B_T, N)
    w = np.pi / B_T
    amp = B_RK + 0.4
    X = np.zeros((N, 4))
    X[:, 0] = (1 - tt / B_T) * B_A[0] + (tt / B_T) * B_B[0]
    X[:, 1] = amp * np.sin(w * tt)
    X[:, 2] = (B_B[0] - B_A[0]) / B_T
    X[:, 3] = amp * w * np.cos(w * tt)
    U = np.zeros((N, 2))
    U[:, 1] = -amp * w ** 2 * np.sin(w * tt)          # a_y that traces the arc
    return X, U


def plan_b_collocation():
    dc = DirectCollocation(
        _pm_f, n_x=4, n_u=2, N=B_N, t_final=B_T, x0=B_A, x_goal=B_B,
        Q=0.0, R=np.eye(2), Qf=0.0, u_bounds=(-B_AMAX, B_AMAX),
        path_con=_pm_path_con, max_iter=600, tol=1e-9,
    )
    Xg, Ug = _pm_warm(B_N)
    t0 = time.perf_counter()
    res = dc.solve(X_init=Xg, U_init=Ug)
    ms = (time.perf_counter() - t0) * 1e3
    return dict(name="Direct collocation (HS)", t=res.t, X=res.X, U=res.U,
                hold="foh", f=_pm_f, solve_ms=ms, converged=res.success,
                defect_norm=res.defect_norm)


def plan_b_ilqr():
    H = B_H
    Qf = B_QF

    def cost(X, U, Xr, Ur):
        du = U - Ur
        total = B_DT * float(np.sum(du ** 2)) + float((X[H] - Xr[H]) @ Qf @ (X[H] - Xr[H]))
        lx = np.zeros((H + 1, 4))
        lxx = np.zeros((H + 1, 4, 4))
        lx[H] = 2.0 * (X[H] - Xr[H]) @ Qf
        lxx[H] = 2.0 * Qf
        lu = 2.0 * B_DT * du
        luu = np.broadcast_to(2.0 * B_DT * np.eye(2), (H, 2, 2)).copy()
        lux = np.zeros((H, 2, 4))
        for k in range(H + 1):
            ot, gx, gy, hxx, hyy, hxy = _disk_cost_terms(X[k, 0], X[k, 1])
            total += ot
            lx[k, 0] += gx
            lx[k, 1] += gy
            lxx[k, 0, 0] += hxx
            lxx[k, 1, 1] += hyy
            lxx[k, 0, 1] += hxy
            lxx[k, 1, 0] += hxy
        return total, lx, lu, lxx, luu, lux

    opt = iLQR(_pm_step, n_x=4, n_u=2, horizon=H, Q=0.0, R=np.eye(2),
               x_ref=B_B, u_bounds=(-B_AMAX, B_AMAX), cost=cost, max_iter=400)
    Xg, Ug = _pm_warm(H + 1)
    t0 = time.perf_counter()
    res = opt.solve(B_A, Ug[:H])
    ms = (time.perf_counter() - t0) * 1e3
    U = np.vstack([res.U, res.U[-1]])
    return dict(name="iLQR / single shooting", t=np.arange(H + 1) * B_DT,
                X=res.X, U=U, hold="zoh", f=_pm_f, solve_ms=ms,
                converged=res.converged, defect_norm=np.nan)


def plan_b_cem():
    from aimct.controllers import SamplingMPC

    H = B_H

    def running_cost(X, U):
        return B_DT * np.sum(U ** 2, axis=1) + _disk_penalty_batch(X)

    def terminal_cost(X):
        e = X - B_B
        return np.einsum("bi,ij,bj->b", e, B_QF, e)

    cem = SamplingMPC(_pm_step_batched, running_cost, terminal_cost=terminal_cost,
                      horizon=H, n_samples=3000 if FULL else 1200,
                      n_elite=150 if FULL else 80, n_iter=6, u_dim=2,
                      u_bounds=(-B_AMAX, B_AMAX), seed=SEED)
    n_outer = 40 if FULL else 20
    t0 = time.perf_counter()
    for _ in range(n_outer):
        cem._plan(B_A)
    ms = (time.perf_counter() - t0) * 1e3
    U = np.vstack([cem.mu, cem.mu[-1]])
    X = np.empty((H + 1, 4))
    X[0] = B_A
    for k in range(H):
        X[k + 1] = _pm_step(X[k], U[k])
    return dict(name="CEM / sampling", t=np.arange(H + 1) * B_DT, X=X, U=U,
                hold="zoh", f=_pm_f, solve_ms=ms, converged=True, defect_norm=np.nan)


def score_b(p):
    t, X, U = p["t"], p["X"], p["U"]
    Xsim = _simulate(p["f"], X[0], U, t, p["hold"])
    d = np.sqrt((X[:, 0] - B_CX) ** 2 + (X[:, 1] - B_CY) ** 2)
    dsim = np.sqrt((Xsim[:, 0] - B_CX) ** 2 + (Xsim[:, 1] - B_CY) ** 2)
    plen = float(np.sum(np.hypot(np.diff(X[:, 0]), np.diff(X[:, 1]))))
    return dict(
        name=p["name"],
        effort=float(np.trapezoid(np.sum(U ** 2, axis=1), t)),
        term_err_planned=float(np.linalg.norm(X[-1] - B_B)),
        term_err_rolled=float(np.linalg.norm(Xsim[-1] - B_B)),
        min_clear_planned=float(d.min() - B_RK),
        min_clear_rolled=float(dsim.min() - B_RK),
        path_len=plen,
        peak_a=float(np.max(np.abs(U))),
        box_ok=bool(np.max(np.abs(U)) <= B_AMAX + 1e-6),
        knots=len(t),
        solve_ms=p["solve_ms"],
        converged=bool(p["converged"]),
        defect_norm=p["defect_norm"],
        _plan=p, _Xsim=Xsim,
    )


B_COLS = ["effort", "term_err_planned", "min_clear_planned", "min_clear_rolled",
          "path_len", "peak_a", "box_ok", "knots", "solve_ms", "converged"]


# ======================================================================
# report
# ======================================================================
def _fmt(v, col):
    if isinstance(v, bool):
        return "yes" if v else "**no**"
    if col in ("solve_ms",):
        return f"{v:.0f}"
    if col in ("knots",):
        return str(v)
    if col.startswith("min_clear"):
        return f"{v * 1e3:+.2f} mm"
    return f"{v:.4g}"


def _table(title, blurb, cols, rows):
    out = [f"## {title}", "", blurb, "",
           "| planner | " + " | ".join(cols) + " |",
           "| --- |" + " --- |" * len(cols)]
    for r in rows:
        out.append("| " + r["name"] + " | "
                   + " | ".join(_fmt(r[c], c) for c in cols) + " |")
    out.append("")
    return out


def main():
    a_rows = [score_a(p) for p in (plan_a_collocation(), plan_a_ilqr(), plan_a_cem())]
    b_rows = [score_b(p) for p in (plan_b_collocation(), plan_b_ilqr(), plan_b_cem())]

    lines = ["# Experiment 32 - direct collocation vs shooting vs sampling (offline)", ""]
    lines += _table(
        "Task A - minimum-effort cart-pole swing-up (hard TERMINAL constraint)",
        f"`min integral(u^2) dt` s.t. the dynamics, `x(0)=[0,0,pi,0]`, "
        f"`x(T)=[0,0,0,0]`, `|u|<={A_FMAX:.0f}` N, `T={A_T:.1f}` s. "
        f"`term_err_rolled` re-integrates the returned `u` through the true "
        f"dynamics (FOH for collocation, ZOH for iLQR/CEM).",
        A_COLS, a_rows)
    lines += [f"Collocation HS defect norm: {a_rows[0]['defect_norm']:.2e}.", ""]
    lines += _table(
        "Task B - planar point mass around a keep-out disk (hard PATH constraint)",
        f"`min integral(||a||^2) dt` s.t. the double integrator, `x(0)={B_A[:2].tolist()}`, "
        f"`x(T)={B_B[:2].tolist()}`, stay outside the disk centred "
        f"`({B_CX:.0f},{B_CY:.0f})` radius `{B_RK:.2f}`, `|a|<={B_AMAX:.0f}`, "
        f"`T={B_T:.1f}` s. `min_clear` = min distance to the disk minus its "
        f"radius; **negative = inside the keep-out zone**.",
        B_COLS, b_rows)
    lines += [f"Collocation HS defect norm: {b_rows[0]['defect_norm']:.2e}.", ""]

    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "planner", *A_COLS, *[c for c in B_COLS if c not in A_COLS]])
        for r in a_rows:
            w.writerow(["A", r["name"], *(r[c] for c in A_COLS),
                        *["" for c in B_COLS if c not in A_COLS]])
        extra = [c for c in B_COLS if c not in A_COLS]
        for r in b_rows:
            w.writerow(["B", r["name"], *[r.get(c, "") for c in A_COLS],
                        *(r[c] for c in extra)])

    _figure(a_rows, b_rows)
    print((HERE / "table.md").read_text())


def _figure(a_rows, b_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # --- Task A: pole angle, cart position, input ---
    for i, r in enumerate(a_rows):
        p, c = r["_plan"], COLORS[i]
        t, X, U = p["t"], p["X"], p["U"]
        ax[0, 0].plot(t, np.degrees(_wrap(X[:, 2])), color=c, lw=1.8, label=r["name"])
        ax[0, 0].plot(t, np.degrees(_wrap(r["_Xsim"][:, 2])), color=c, lw=1.0, ls=":")
        ax[0, 1].plot(t, X[:, 0], color=c, lw=1.8)
        ax[0, 2].plot(t, U[:, 0], color=c, lw=1.8, label=r["name"])
    ax[0, 0].axhline(0, ls="--", color=PALETTE["reference"], lw=1.0)
    ax[0, 0].set(title="A (a) pole angle (solid=plan, dotted=re-integrated)",
                 xlabel="t [s]", ylabel="deg from upright")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(0, ls="--", color=PALETTE["reference"], lw=1.0)
    ax[0, 1].set(title="A (b) cart position", xlabel="t [s]", ylabel="x [m]")
    ax[0, 2].axhline(A_FMAX, ls=":", color=PALETTE["saturation"], lw=1.1)
    ax[0, 2].axhline(-A_FMAX, ls=":", color=PALETTE["saturation"], lw=1.1)
    ax[0, 2].set(title="A (c) control force", xlabel="t [s]", ylabel="F [N]")
    ax[0, 2].legend(fontsize=8)

    # --- Task B: xy paths, speed, clearance bars ---
    th = np.linspace(0, 2 * np.pi, 100)
    for j in (0,):
        ax[1, j].fill(B_CX + B_RK * np.cos(th), B_CY + B_RK * np.sin(th),
                      color=PALETTE["saturation"], alpha=0.25)
        ax[1, j].plot(B_CX + B_RK * np.cos(th), B_CY + B_RK * np.sin(th),
                      color=PALETTE["saturation"], lw=1.2)
    for i, r in enumerate(b_rows):
        p, c = r["_plan"], COLORS[i]
        X, U, t = p["X"], p["U"], p["t"]
        ax[1, 0].plot(X[:, 0], X[:, 1], color=c, lw=1.8, label=r["name"])
        ax[1, 1].plot(t, np.hypot(X[:, 2], X[:, 3]), color=c, lw=1.8, label=r["name"])
    ax[1, 0].plot(*B_A[:2], "o", color="0.2", ms=6)
    ax[1, 0].plot(*B_B[:2], "*", color="0.2", ms=11)
    ax[1, 0].set(title="B (a) path around the keep-out disk", xlabel="x [m]",
                 ylabel="y [m]")
    ax[1, 0].set_aspect("equal", "box")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].set(title="B (b) speed", xlabel="t [s]", ylabel="|v| [m/s]")
    ax[1, 1].legend(fontsize=8)

    names = [r["name"].split(" / ")[0].split(" (")[0] for r in b_rows]
    xs = np.arange(len(b_rows))
    ax[1, 2].bar(xs, [r["min_clear_rolled"] for r in b_rows], color=COLORS, alpha=0.85)
    ax[1, 2].axhline(0, color="0.2", lw=1.2)
    ax[1, 2].set_xticks(xs)
    ax[1, 2].set_xticklabels(names, fontsize=8)
    ax[1, 2].set(title="B (c) min disk clearance (re-integrated)\n<0 = inside keep-out",
                 ylabel="clearance [m]")

    fig.suptitle("Exp 32 - offline planning: hard constraint (collocation) vs "
                 "penalty (iLQR, CEM)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
