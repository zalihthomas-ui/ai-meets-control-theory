"""Experiment 08 - Constrained MPC vs LQR on the cart-pole.

Balancing the cart-pole from a moderate tilt (theta0 = 0.35 rad) with a +/-20 N
actuator, the LQR recovery swings the cart out past +/-0.5 m of track.  Linear
MPC with the same Q, R but a hard-ish cart-position constraint |x_cart| <= 0.5 m
plans a recovery that rides the limit instead of busting it - the textbook case
for MPC over LQR.

Controllers (same Q = diag(10,1,100,10), R = 0.1):
  * LQR                 - the analytic optimum; no notion of the track limit.
  * MPC (unconstrained) - condensed N-step QP; matches LQR here.
  * MPC (|x_cart|<=0.5) - same MPC plus the state box, softened.

Run:  python experiments/08_mpc_vs_lqr_constrained_cartpole/run.py
Outputs (next to this file): table.md, table.csv, metrics_full.csv,
  figure.png, cart_constraint.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks import compare
from aimct.controllers import LQR, LinearMPC
from aimct.systems import CartPole

HERE = Path(__file__).parent

X0 = np.array([0.0, 0.0, 0.35, 0.0])
F_MAX = 20.0
X_CART_LIMIT = 0.5
DT, T_FINAL = 0.01, 6.0
N_HORIZON = 75

_CP = CartPole()
_A, _B = _CP.linearize()
_Q = np.diag([10.0, 1.0, 100.0, 10.0])
_R = np.array([[0.1]])


def build_controllers() -> dict:
    return {
        "LQR": LQR(_A, _B, _Q, _R),
        "MPC (unconstrained)": LinearMPC(_A, _B, Q=_Q, R=_R, N=N_HORIZON,
                                         u_bounds=(-F_MAX, F_MAX)),
        f"MPC (|x|<={X_CART_LIMIT})": LinearMPC(
            _A, _B, Q=_Q, R=_R, N=N_HORIZON, u_bounds=(-F_MAX, F_MAX),
            x_bounds=([-X_CART_LIMIT, None, None, None],
                      [X_CART_LIMIT, None, None, None]),
            soft_weight=1e5,
        ),
    }


def cart_table(result) -> list[dict]:
    rows = []
    for name in result.names:
        tr = result.trajectories[name]
        xc = tr.x[:, 0]
        rows.append({
            "controller": name,
            "cart_peak_m": round(float(np.max(np.abs(xc))), 4),
            "constraint_violation_m": round(
                float(max(0.0, np.max(np.abs(xc)) - X_CART_LIMIT)), 4),
            "theta_settle_s": round(float(result.metrics[name]["settling_time"]), 3),
            "control_energy": round(float(result.metrics[name]["control_energy"]), 1),
            "peak_force_N": round(float(np.max(np.abs(tr.u))), 2),
        })
    return rows


def write_tables(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    (HERE / "table.csv").write_text(
        ",".join(cols) + "\n"
        + "\n".join(",".join(str(r[c]) for c in cols) for r in rows) + "\n",
        encoding="utf-8", newline="\n",
    )
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (HERE / "table.md").write_text(
        "# Experiment 08 - constrained MPC vs LQR (cart-pole)\n\n"
        f"Balance from theta0 = 0.35 rad, |F| <= {F_MAX:g} N, track limit "
        f"|x_cart| <= {X_CART_LIMIT:g} m.\n\n"
        + "\n".join([head, sep, *body]) + "\n",
        encoding="utf-8", newline="\n",
    )


def cart_figure(result) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aimct.plot_style import set_aimct_style

    set_aimct_style()
    fig, (ax_x, ax_th, ax_u) = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)
    colors = {"LQR": "#D55E00", "MPC (unconstrained)": "#999999",
              f"MPC (|x|<={X_CART_LIMIT})": "#0072B2"}

    for name in result.names:
        tr = result.trajectories[name]
        c = colors.get(name, None)
        ax_x.plot(tr.t, tr.x[:, 0], color=c, lw=2.0, label=name)
        ax_th.plot(tr.t, tr.x[:, 2], color=c, lw=2.0, label=name)
        ax_u.plot(tr.t, tr.u[:, 0], color=c, lw=1.3, label=name)

    for lim in (X_CART_LIMIT, -X_CART_LIMIT):
        ax_x.axhline(lim, color="#D62728", ls="--", lw=1.5)
    ax_x.axhspan(-X_CART_LIMIT, X_CART_LIMIT, color="#000000", alpha=0.05)
    ax_x.set_ylabel("cart position $x$ [m]")
    ax_x.set_title("(a) Cart position - dashed = track limit $\\pm 0.5$ m")
    ax_th.axhline(0.0, color="#555555", ls="--", lw=1.0)
    ax_th.set_ylabel(r"pole angle $\theta$ [rad]")
    ax_th.set_title("(b) Pole angle")
    for lim in (F_MAX, -F_MAX):
        ax_u.axhline(lim, color="#D62728", ls=":", lw=1.3)
    ax_u.set_ylabel("force $u$ [N]")
    ax_u.set_xlabel("time [s]")
    ax_u.set_title("(c) Cart force")
    for ax in (ax_x, ax_th, ax_u):
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Exp 08 - Constrained MPC vs LQR: cart-pole with a track limit",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "cart_constraint.png", dpi=150)
    plt.close(fig)


def main() -> None:
    result = compare(
        _CP, build_controllers(), x0=X0, dt=DT, t_final=T_FINAL, reference=0.0,
        output_index=2, u_bounds=(-F_MAX, F_MAX),
        title="Exp 08 - constrained MPC vs LQR (cart-pole)",
    )

    rows = cart_table(result)
    write_tables(rows)
    (HERE / "metrics_full.csv").write_text(result.full_metrics_csv(),
                                           encoding="utf-8", newline="\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, _ = result.figure(state_label=r"Pole angle $\theta$ [rad]",
                           control_label=r"Cart force $u$ [N]")
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)
    cart_figure(result)

    print("wrote:", ", ".join(p.name for p in sorted(HERE.glob("*"))
                              if p.suffix in {".md", ".csv", ".png"}))
    print()
    print((HERE / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
