"""Experiment 37 - tube MPC vs nominal MPC under a persistent bounded disturbance.

A mass on a spring must stay inside a slot ``|x| <= x_max`` while an unmodelled
force pushes on it every step (a bounded process disturbance ``w``, worst-case
adversarial here).  The controller is asked to hold the mass near the wall.

* **Nominal `LinearMPC`** plans as if ``w = 0``: it parks the nominal state at
  the setpoint near the wall and the disturbance carries the *true* mass
  **through** it - a sustained constraint violation.
* **`TubeMPC`** shrinks the state box by the robust-invariant tube ``Z`` (and
  the input box by ``K Z``), solves the nominal problem on that tightened set,
  and applies ``u = u_nom + K (x - x_nom)``.  The true mass stays inside the
  slot for every disturbance realisation - at the cost of a nominal setpoint
  backed off from the wall by the tube width and a slower approach.

Run:   python experiments/37_tube_mpc/run.py
       AIMCT_EXP_FULL=1 python experiments/37_tube_mpc/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aimct.controllers import LinearMPC, TubeMPC
from aimct.controllers.mpc import _discretize
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.systems import MassSpringDamper

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

SYS = MassSpringDamper(m=1.0, k=1.0, c=4.0)             # a well-damped mass-spring
A, B = SYS.linearize(np.zeros(2), np.zeros(1))
DT = 0.05
N = 30
X_MAX = 0.40                                            # slot  |x| <= X_MAX
U_MAX = 6.0
W_HW = np.array([0.0, 0.07])                            # persistent force / accel push
T_SIM = 10.0 if FULL else 7.0
STEPS = int(round(T_SIM / DT))

Q = np.diag([60.0, 2.0])
R = np.array([[0.15]])
X_REF = np.array([0.38, 0.0])                           # hold near the wall
VIOL_TOL = 2e-3                                         # box-hull mРPI rounding

COMMON = dict(Q=Q, R=R, N=N, x_ref=X_REF,
              x_bounds=(np.array([-X_MAX, -np.inf]),
                        np.array([X_MAX, np.inf])),
              u_bounds=(-U_MAX, U_MAX))

Ad, Bd = _discretize(A, B, DT)


def _worst_w(k):
    return np.array([0.0, W_HW[1]])                     # constant push toward the wall


def _rollout(ctrl):
    ctrl.reset()
    x = np.array([0.0, 0.0])
    xs, us = [x.copy()], []
    for k in range(STEPS):
        u = np.atleast_1d(np.asarray(ctrl.update(x, DT), float)).reshape(1)
        x = Ad @ x + Bd @ u + _worst_w(k)
        xs.append(x.copy())
        us.append(float(u[0]))
    return np.array(xs), np.array(us)


def _x_ss(xc):
    """Steady-state position (mean over the last quarter of the run)."""
    return float(np.mean(xc[-max(1, len(xc) // 4):]))


def main():
    nom = LinearMPC(A, B, **COMMON)
    tube = TubeMPC(A, B, W=W_HW, **COMMON)

    xn, un = _rollout(nom)
    xt, ut = _rollout(tube)
    Z = tube.mrpi
    xlo_t, xhi_t = tube.tightened_x_bounds
    ulo_t, uhi_t = tube.tightened_u_bounds

    rows = []
    for name, xs, us in (("nominal LinearMPC", xn, un), ("TubeMPC", xt, ut)):
        xc = xs[:, 0]
        rows.append(dict(
            name=name,
            max_pos=float(np.max(np.abs(xc))),
            box_violation=float(max(0.0, np.max(np.abs(xc)) - X_MAX)),
            violates=bool(np.max(np.abs(xc)) > X_MAX + VIOL_TOL),
            steps_over=int(np.sum(np.abs(xc) > X_MAX + VIOL_TOL)),
            x_ss=_x_ss(xc),
            peak_u=float(np.max(np.abs(us))),
            rms_u=float(np.sqrt(np.mean(us ** 2))),
        ))

    cols = ["max_pos", "box_violation", "violates", "steps_over", "x_ss",
            "peak_u", "rms_u"]
    lines = [
        "# Experiment 37 - tube MPC vs nominal MPC under a bounded disturbance",
        "",
        f"mass-spring-damper (m=1, k=1, c=4), slot `|x| <= {X_MAX:.2f}` m, "
        f"persistent force push `|w_v| = {W_HW[1]:.2f}`, setpoint "
        f"`x = {X_REF[0]:.2f}` (near the wall).",
        "",
        f"mРPI box half-widths `z` (built at dt = {DT}): "
        f"`[{Z.half_widths[0]:.4f}, {Z.half_widths[1]:.4f}]`  "
        f"(series terms {Z.n_terms}, contraction ~{Z.alpha:.3f}).",
        f"Tightened position box: `|x_nom| <= {xhi_t[0]:.3f}` "
        f"(wall {X_MAX:.2f} minus tube width {Z.half_widths[0]:.4f}).",
        f"Tightened input box: `|u_nom| <= {uhi_t[0]:.2f}` (of {U_MAX:.0f}).",
        "",
        "| controller | " + " | ".join(cols) + " |",
        "| --- |" + " --- |" * len(cols),
    ]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, bool):
                cells.append("**YES**" if v else "no")
            elif c == "steps_over":
                cells.append(str(v))
            else:
                cells.append("n/a" if (isinstance(v, float) and np.isnan(v))
                             else f"{v:.3f}")
        lines.append("| " + r["name"] + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Nominal MPC sits {rows[0]['box_violation'] * 1e3:.1f} mm outside the "
        f"slot for {rows[0]['steps_over']} of {STEPS} steps; tube MPC never "
        f"leaves it. The cost: the tube's nominal setpoint is "
        f"{xhi_t[0]:.3f} m, {(X_REF[0] - xhi_t[0]) * 1e3:.0f} mm short of the "
        f"{X_REF[0]:.2f} m the nominal controller aims for.",
        "",
    ]
    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["controller", *cols])
        for r in rows:
            w.writerow([r["name"], *(r[c] for c in cols)])
        w.writerow([])
        w.writerow(["mrpi_half_widths", *Z.half_widths])
        w.writerow(["tightened_x_hi", xhi_t[0]])
        w.writerow(["tightened_u_hi", uhi_t[0]])

    _figure(xn, un, xt, ut, Z, xhi_t[0], uhi_t[0], rows)
    print((HERE / "table.md").read_text())


def _figure(xn, un, xt, ut, Z, xnom_hi, unom_hi, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    t = np.arange(xn.shape[0]) * DT
    tu = np.arange(un.shape[0]) * DT
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    cn, ct = PALETTE["lqr"], PALETTE["mpc"]

    ax[0, 0].axhline(X_MAX, ls="--", color=PALETTE["saturation"], lw=1.5,
                     label=f"slot wall {X_MAX}")
    ax[0, 0].axhline(xnom_hi, ls=":", color=ct, lw=1.3,
                     label=f"tightened box {xnom_hi:.3f}")
    ax[0, 0].fill_between(t, xt[:, 0] - Z.half_widths[0], xt[:, 0] + Z.half_widths[0],
                          color=ct, alpha=0.15, label=r"tube $x_{nom}\pm z$")
    ax[0, 0].plot(t, xn[:, 0], color=cn, lw=2.0, label="nominal LinearMPC (true)")
    ax[0, 0].plot(t, xt[:, 0], color=ct, lw=2.0, label="TubeMPC (true)")
    ax[0, 0].set(title="(a) position - nominal MPC is carried through the wall",
                 xlabel="t [s]", ylabel="x [m]")
    ax[0, 0].legend(fontsize=7, loc="lower right")

    ax[0, 1].plot(t, xn[:, 1], color=cn, lw=1.8, label="nominal")
    ax[0, 1].plot(t, xt[:, 1], color=ct, lw=1.8, label="tube")
    ax[0, 1].axhline(0.0, ls="--", color=PALETTE["reference"], lw=1.0)
    ax[0, 1].set(title="(b) velocity", xlabel="t [s]", ylabel="v [m/s]")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].axhline(unom_hi, ls=":", color=ct, lw=1.3)
    ax[1, 0].axhline(-unom_hi, ls=":", color=ct, lw=1.3)
    ax[1, 0].axhline(U_MAX, ls="--", color=PALETTE["saturation"], lw=1.2)
    ax[1, 0].axhline(-U_MAX, ls="--", color=PALETTE["saturation"], lw=1.2)
    ax[1, 0].plot(tu, un, color=cn, lw=1.7, label="nominal")
    ax[1, 0].plot(tu, ut, color=ct, lw=1.7, label="tube")
    ax[1, 0].set(title="(c) control force (dotted = tightened input box)",
                 xlabel="t [s]", ylabel="F [N]")
    ax[1, 0].legend(fontsize=8)

    labels = ["max |x| [m]", "box violation [m]", "x_ss [m]"]
    x = np.arange(3)
    for i, r in enumerate(rows):
        vals = [r["max_pos"], r["box_violation"],
                r["x_ss"]]
        ax[1, 1].bar(x + (i - 0.5) * 0.35, vals, 0.35, color=[cn, ct][i],
                     label=r["name"])
    ax[1, 1].axhline(X_MAX, ls="--", color=PALETTE["saturation"], lw=1.2)
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(labels, fontsize=8)
    ax[1, 1].set(title="(d) constraint & performance")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("Exp 37 - tube MPC keeps the state box under a persistent "
                 "disturbance, at a conservatism cost", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
