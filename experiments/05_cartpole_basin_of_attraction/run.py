"""Experiment 05 - Cart-pole basin of attraction for a linear LQR balance law.

A linear LQR controller ``u = -K x`` (K designed on the upright linearisation) is
run against the true nonlinear ``CartPole`` with no swing-up assistance and the
canonical +/-20 N actuator.  Two studies:

1. **theta0 sweep from rest** (thetadot0 = 0): does the pole recover, and how fast
   does the angle settle, as theta0 grows toward horizontal?
2. **theta0 x thetadot0 basin map** per tuning: the shape of the recoverable set,
   with the measured edges compared against the reference envelope in
   ``docs/references/swingup-and-basin.md`` section 3.1.

Run:  python experiments/05_cartpole_basin_of_attraction/run.py
Outputs (next to this file): sweep.csv, sweep_summary.md, robustness_sweep.png/.svg,
  basin_map.png, basin_edges.md
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from aimct.benchmarks import compare, sweep
from aimct.controllers import LQR
from aimct.systems import CartPole

HERE = Path(__file__).parent

# ----------------------------------------------------------------- configuration
TUNINGS = {
    "balanced":   (np.diag([10.0, 1.0, 100.0, 10.0]), np.array([[0.10]])),
    "aggressive": (np.diag([1.0, 0.1, 1000.0, 10.0]), np.array([[0.01]])),
    "soft":       (np.diag([1.0, 0.1, 10.0, 1.0]),    np.array([[1.00]])),
}
# swingup-and-basin.md sec 3.1: (|theta0|_max rad, |thetadot0|_max rad/s)
PREDICTED = {"balanced": (0.800, 4.00), "aggressive": (0.800, 4.00),
             "soft": (0.334, 1.31)}
U_BOUNDS = (-20.0, 20.0)                     # canonical F_max (cartpole-lqr-reference sec 1)
CAUGHT_TOL = 0.10                            # |theta| over the last 10 % of the run [rad]

# The committed figures/tables are the high-resolution run (AIMCT_EXP_FULL=1, ~45 s).
# The default is a coarse grid so CI's per-experiment smoke-run stays quick; it
# still exercises every code path.
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

if FULL:
    DT_1D, T_1D = 2e-3, 8.0
    THETA0_GRID = np.round(np.arange(0.05, 1.561, 0.05), 3)     # 31 pts, to ~pi/2
    DT_2D, T_2D = 4e-3, 6.0
    THETA0_2D = np.round(np.linspace(0.0, 1.50, 19), 3)
    THETADOT0_2D = np.round(np.linspace(-8.0, 8.0, 19), 3)
else:
    DT_1D, T_1D = 4e-3, 6.0
    THETA0_GRID = np.round(np.arange(0.1, 1.51, 0.2), 3)        # 8 pts
    DT_2D, T_2D = 6e-3, 4.0
    THETA0_2D = np.round(np.linspace(0.0, 1.50, 9), 3)
    THETADOT0_2D = np.round(np.linspace(-8.0, 8.0, 9), 3)

_CP = CartPole()
_A, _B = _CP.linearize()
_LQR = {name: LQR(_A, _B, Q, R) for name, (Q, R) in TUNINGS.items()}


def _caught(traj, tol: float = CAUGHT_TOL) -> bool:
    """The pole is *caught* if the run stays finite and the mean |theta| over the
    last 10 % of the trajectory is within ``tol`` of upright."""
    x = traj.x
    if traj.diverged or not np.all(np.isfinite(x)):
        return False
    tail = x[max(1, int(0.9 * len(x))):, 2]
    return bool(np.mean(np.abs(tail)) < tol)


# --------------------------------------------------------------- 1-D sweep (rest)
def make_case(theta0: float) -> dict:
    return dict(
        system=_CP,
        controllers=dict(_LQR),
        x0=np.array([0.0, 0.0, float(theta0), 0.0]),
        dt=DT_1D, t_final=T_1D, reference=0.0, output_index=2, u_bounds=U_BOUNDS,
    )


def run_1d():
    result = sweep(THETA0_GRID, make_case, param_name="theta0_rad")
    result.save(
        HERE, metric="settling_time",
        plot_kwargs=dict(
            right="control_energy",
            title="Exp 05 - Cart-pole: pole-angle settling vs theta0 (from rest)",
            left_label="Pole-angle settling time $t_s$ [s]",
            right_label="Control energy $E_u$ [N$^2$s] (dotted)",
        ),
    )
    return result


# ------------------------------------------------------- 2-D basin map per tuning
def run_basin_map() -> dict[str, np.ndarray]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    names = list(_LQR)
    caught = {n: np.zeros((len(THETADOT0_2D), len(THETA0_2D)), dtype=bool) for n in names}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, thd in enumerate(THETADOT0_2D):
            for j, th in enumerate(THETA0_2D):
                res = compare(
                    _CP, dict(_LQR), x0=np.array([0.0, 0.0, th, thd]),
                    dt=DT_2D, t_final=T_2D, reference=0.0, output_index=2,
                    u_bounds=U_BOUNDS,
                )
                for n in names:
                    caught[n][i, j] = _caught(res.trajectories[n])

    cmap = ListedColormap(["#D62728", "#009E73"])          # lost / caught
    ext = [THETA0_2D[0], THETA0_2D[-1], THETADOT0_2D[0], THETADOT0_2D[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharey=True)
    for ax, n in zip(axes, names):
        ax.imshow(caught[n].astype(int), origin="lower", aspect="auto",
                  cmap=cmap, vmin=0, vmax=1, extent=ext)
        p_th, p_thd = PREDICTED[n]
        ax.add_patch(plt.Rectangle((-p_th, -p_thd), 2 * p_th, 2 * p_thd,
                                   fill=False, ec="black", lw=1.4, ls="--"))
        ax.set_title(f"{n}  (R={float(TUNINGS[n][1].ravel()[0]):g})")
        ax.set_xlabel(r"$\theta_0$ [rad]")
    axes[0].set_ylabel(r"$\dot\theta_0$ [rad/s]")
    fig.suptitle("Exp 05 - Cart-pole LQR basin of attraction (green = caught, "
                 "dashed = reference envelope, |F| <= 20 N)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "basin_map.png", dpi=150)
    plt.close(fig)
    return caught


def write_basin_edges(caught: dict[str, np.ndarray]) -> None:
    j0 = int(np.argmin(np.abs(THETA0_2D - 0.0)))           # thetadot0 axis at theta0 ~ 0
    i0 = int(np.argmin(np.abs(THETADOT0_2D - 0.0)))        # theta0 axis at thetadot0 = 0
    lines = [
        "# Experiment 05 - measured basin edges vs reference envelope",
        "",
        f"Nonlinear CartPole, RK4 dt={DT_2D}, T={T_2D} s, actuator |F| <= "
        f"{U_BOUNDS[1]:g} N. *Caught* = state stays finite and mean |theta| over "
        f"the last 10 % of the run is < {CAUGHT_TOL} rad. Edges read off the "
        f"{len(THETADOT0_2D)}x{len(THETA0_2D)} grid along the axes; predicted "
        "values from `docs/references/swingup-and-basin.md` section 3.1.",
        "",
        "| Tuning | R | theta0 edge (thetadot0=0) | predicted | thetadot0 edge "
        "(theta0=0) | predicted |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for n in caught:
        row = caught[n][i0, :]
        col = caught[n][:, j0]
        th_edge = THETA0_2D[row][-1] if row.any() else float("nan")
        thd_pos = THETADOT0_2D[(THETADOT0_2D >= 0) & col]
        thd_edge = thd_pos[-1] if thd_pos.size else float("nan")
        p_th, p_thd = PREDICTED[n]
        lines.append(
            f"| {n} | {float(TUNINGS[n][1].ravel()[0]):g} | {th_edge:.2f} rad "
            f"({np.degrees(th_edge):.0f} deg) | {p_th:.2f} rad "
            f"({np.degrees(p_th):.0f} deg) | {thd_edge:.1f} rad/s | {p_thd:.1f} rad/s |"
        )
    lines += [
        "",
        "The reference envelope is a conservative Lyapunov sub-level-set estimate; "
        "the simulated recoverable set is larger, and with a +/-20 N actuator the "
        "linear law catches the pole from rest almost to the horizontal. The "
        "tunings separate mainly in the thetadot0 direction and in transient cost "
        "(see robustness_sweep.png), not in the from-rest angle limit.",
    ]
    (HERE / "basin_edges.md").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8", newline="\n")


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run_1d()
        caught = run_basin_map()
    write_basin_edges(caught)

    made = [p.name for p in sorted(HERE.glob("*"))
            if p.suffix in {".md", ".csv", ".png", ".svg"}]
    print("wrote:", ", ".join(made))
    print()
    print((HERE / "basin_edges.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
