"""Experiment 25 - a moving obstacle on the differential-drive path.

Four controllers follow the Exp-22 waypoint path; two are the Exp-22 blind
trackers (pure pursuit, path LQR - no notion an obstacle exists), two are
receding-horizon planners with an obstacle-penalty term in their cost:
Sampling MPC (CEM, derivative-free) and iLQR/RTI-NMPC (gradient, re-linearised
every step). The obstacle field is three disks: two static, straddling the
path, and one **moving** disk that drifts across the path mid-route at a
speed comparable to the robot's own cruise speed - the case where a planner
has to react to a target that is not there yet when it starts planning.

Run:   python experiments/25_diffdrive_moving_obstacle/run.py
       AIMCT_EXP_FULL=1 python experiments/25_diffdrive_moving_obstacle/run.py
Outputs (next to this file): table.md, table.csv, figure.png
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from aimct.controllers import ILQR, LQR, SamplingMPC, wrap_angle
from aimct.ml.planning import system_step
from aimct.plot_style import PALETTE, set_aimct_style
from aimct.simulate import simulate
from aimct.systems import DifferentialDriveRobot
from aimct.trajectories import Spline

HERE = Path(__file__).parent
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"
SEED = 0

V = 0.15                         # cruise speed [m/s]
DT = 0.05
WAYPOINTS = np.array([[0.0, 0.0], [1.0, 0.6], [2.2, -0.2], [3.2, 0.8], [4.5, 0.0]])

robot = DifferentialDriveRobot(v_ref=V)
U_BOUNDS = (np.array([-robot.v_max, -robot.omega_max]),
           np.array([robot.v_max, robot.omega_max]))


def timed_spline(waypoints, v):
    """Spline re-timed so ``traj(t)`` is where the robot should be at time
    ``t`` (see Exp 22)."""
    P = np.asarray(waypoints, dtype=float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    knots = np.concatenate([[0.0], np.cumsum(seg) / v])
    return Spline(P, knots=knots)


PATH = timed_spline(WAYPOINTS, V)
T_FINAL = PATH.duration

# --------------------------------------------------------------- obstacle field

R_EFF = 0.15 + 0.03           # disk radius + robot/margin allowance

# Each obstacle centre is offset ~0.15 m off the path *normal* at the point the
# blind trackers pass closest - close enough that a blind trajectory still
# clips it, far enough that a ~0.2-0.3 m lateral deviation clears it (this is a
# manoeuvring test, not "can you plan around a wall on the road").
W_OBS = 1.0e4


def _static(c):
    c = np.asarray(c, float)
    return lambda t: c


# path point / unit normal at the three encounter times (computed once from
# PATH(t) - pos, tangent, normal - at the blind trackers' closest-approach
# times), offset 0.15 m along the normal
_OBS1 = np.array([1.029, 0.592]) + 0.15 * np.array([0.292, 0.956])
_OBS2 = np.array([3.164, 0.780]) + 0.15 * np.array([-0.534, 0.845])
_CROSS_P = np.array([2.268, -0.184])
_CROSS_N = np.array([-0.344, 0.939])
_CROSS_T = 18.0


def _moving_center(t):
    return _CROSS_P + _CROSS_N * V * (t - _CROSS_T)


OBSTACLES = [
    (_static(_OBS1), R_EFF),      # static, offset off the first hump (~t=8s)
    (_static(_OBS2), R_EFF),      # static, offset off the second hump (~t=26.5s)
    (_moving_center, R_EFF),      # moving: crosses the path near t~18s at speed V
]


def _obstacle_penalty_batch(X, t):
    """Vectorised soft quartic barrier, summed over every obstacle -
    ``(batch,)``."""
    pen = np.zeros(X.shape[0])
    for center_fn, r in OBSTACLES:
        cx, cy = center_fn(t)
        d2 = (X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2
        pen += W_OBS * np.maximum(0.0, r ** 2 - d2) ** 2
    return pen


def _obstacle_cost_terms(px, py, t):
    """Scalar barrier total + exact gradient / Hessian w.r.t. (px, py) at time
    ``t`` - the analytic terms iLQR's backward pass needs (verified against
    finite differences)."""
    total = gx = gy = hxx = hyy = hxy = 0.0
    for center_fn, r in OBSTACLES:
        cx, cy = center_fn(t)
        dx, dy = px - cx, py - cy
        g = r ** 2 - (dx ** 2 + dy ** 2)
        h = max(0.0, g)
        active = 1.0 if g > 0.0 else 0.0
        total += W_OBS * h ** 2
        gx += -4.0 * W_OBS * h * dx
        gy += -4.0 * W_OBS * h * dy
        hxx += 8.0 * W_OBS * active * dx ** 2 - 4.0 * W_OBS * h
        hyy += 8.0 * W_OBS * active * dy ** 2 - 4.0 * W_OBS * h
        hxy += 8.0 * W_OBS * active * dx * dy
    return total, gx, gy, hxx, hyy, hxy


def _check_obstacle_cost_derivatives(eps=1e-5, tol=1e-3):
    """Finite-difference-verify ``_obstacle_cost_terms`` at a handful of
    points, some inside and some outside each obstacle's active radius."""
    def total_at(px, py, t):
        return sum(W_OBS * max(0.0, r ** 2 - ((px - c(t)[0]) ** 2 + (py - c(t)[1]) ** 2)) ** 2
                  for c, r in OBSTACLES)

    pts = [(1.073, 0.735, 0.0), (1.10, 0.76, 0.0), (0.5, 0.2, 0.0),
          (2.268, -0.184, 18.0), (2.30, -0.20, 18.0), (2.268, -0.184, 14.0),
          (3.084, 0.907, 0.0)]
    for px, py, t in pts:
        _, gx, gy, hxx, hyy, hxy = _obstacle_cost_terms(px, py, t)
        f = lambda x, y: total_at(x, y, t)
        ngx = (f(px + eps, py) - f(px - eps, py)) / (2 * eps)
        ngy = (f(px, py + eps) - f(px, py - eps)) / (2 * eps)
        nhxx = (f(px + eps, py) - 2 * f(px, py) + f(px - eps, py)) / eps ** 2
        nhyy = (f(px, py + eps) - 2 * f(px, py) + f(px, py - eps)) / eps ** 2
        nhxy = (f(px + eps, py + eps) - f(px + eps, py - eps)
                - f(px - eps, py + eps) + f(px - eps, py - eps)) / (4 * eps ** 2)
        scale = max(abs(ngx), abs(ngy), abs(nhxx), abs(nhyy), 1.0)
        for a, n in ((gx, ngx), (gy, ngy), (hxx, nhxx), (hyy, nhyy), (hxy, nhxy)):
            assert abs(a - n) / scale < tol, (
                f"obstacle cost derivative mismatch at ({px},{py},t={t}): "
                f"analytic {a:.4g} vs numeric {n:.4g}")


_check_obstacle_cost_derivatives()


# --------------------------------------------------------------- timed reference

def timed_ref(t):
    """Absolute-time reference state ``[x, y, theta, v, omega]`` for the
    receding-horizon planners: position/heading/curvature-feedforward yaw
    rate from the path, held at the cruise speed."""
    p, vel, acc = PATH(min(t, PATH.duration))
    theta = np.arctan2(vel[1], vel[0])
    sp = np.hypot(vel[0], vel[1]) + 1e-9
    kappa = (vel[0] * acc[1] - vel[1] * acc[0]) / sp ** 3
    return np.array([p[0], p[1], theta, V, kappa * V])


# ------------------------------------------------------------- path-frame (blind)

_TS, _PL = PATH._polyline(1200)
_SL = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(_PL, axis=0), axis=1))])


def _nearest_idx(p):
    return int(np.argmin(np.hypot(_PL[:, 0] - p[0], _PL[:, 1] - p[1])))


def _path_frame(p):
    i = _nearest_idx(p)
    j = min(i + 1, len(_PL) - 1)
    tang = _PL[j] - _PL[max(i - 1, 0)]
    th_path = np.arctan2(tang[1], tang[0])
    d = np.asarray(p)[:2] - _PL[i]
    e_cross = -(np.sin(th_path) * d[0] - np.cos(th_path) * d[1])
    _, v, a = PATH(_TS[i])
    sp = np.hypot(v[0], v[1]) + 1e-9
    kappa = (v[0] * a[1] - v[1] * a[0]) / sp ** 3
    return _PL[i], th_path, e_cross, kappa, _SL[i]


def _lookahead_point(p, ld):
    i = _nearest_idx(p)
    k = int(np.searchsorted(_SL, min(_SL[i] + ld, _SL[-1])))
    return _PL[min(k, len(_PL) - 1)]


# --------------------------------------------------------- blind baselines (Exp 22)

class PurePursuit:
    name = "Pure pursuit (blind)"

    def __init__(self, ld=0.45):
        self.ld = ld

    def reset(self):
        pass

    def update(self, x, dt):
        p, theta = x[:2], x[2]
        tgt = _lookahead_point(p, self.ld)
        alpha = wrap_angle(np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - theta)
        return np.array([V, 2.0 * V * np.sin(alpha) / self.ld])


class PathLQR:
    name = "Path LQR (blind)"

    def __init__(self):
        A, B = robot.linearize()
        idx = np.array([1, 2, 3, 4])
        self.K = LQR(A[np.ix_(idx, idx)], B[idx, :],
                     np.diag([30.0, 8.0, 2.0, 0.5]), np.diag([2.0, 1.0])).K

    def reset(self):
        pass

    def update(self, x, dt):
        _, th_path, e_cross, kappa, _ = _path_frame(x[:2])
        psi = wrap_angle(x[2] - th_path)
        z = np.array([e_cross, psi, x[3] - V, x[4]])
        return np.array([V, kappa * V]) - self.K @ z


# ----------------------------------------------------------- obstacle-aware CEM

_U_SCALE = np.array([robot.v_max, robot.omega_max])   # SamplingMPC only takes a
                                                       # scalar box, so CEM plans
                                                       # in [-1, 1]-normalised
                                                       # actions and this rescales
                                                       # to [v_cmd, omega_cmd]


class CEMObstacle:
    name = "Sampling MPC (CEM, obstacle-aware)"

    def __init__(self, H=70):
        Q = np.diag([12.0, 12.0, 4.0, 1.0, 0.3])
        R = np.diag([1.0, 0.5])
        raw_step = system_step(robot, DT)

        def step(X, U_norm):
            return raw_step(X, U_norm * _U_SCALE)

        def running_cost(X, U_norm, h):
            t = self.mpc.k * DT + h * DT
            xr = timed_ref(t)
            ur_norm = np.array([V, xr[4]]) / _U_SCALE
            e, du = X - xr, U_norm - ur_norm
            return (np.einsum("bi,ij,bj->b", e, Q, e)
                    + np.einsum("bi,ij,bj->b", du, R, du)
                    + _obstacle_penalty_batch(X, t))

        def terminal_cost(X):
            t = self.mpc.k * DT + H * DT
            e = X - timed_ref(t)
            return 3.0 * np.einsum("bi,ij,bj->b", e, Q, e) + _obstacle_penalty_batch(X, t)

        self.mpc = SamplingMPC(step, running_cost, terminal_cost=terminal_cost,
                               horizon=H, n_samples=500 if FULL else 300,
                               n_elite=50 if FULL else 30, n_iter=4 if FULL else 3,
                               u_dim=2, u_bounds=(-1.0, 1.0), seed=SEED)

    def reset(self):
        self.mpc.reset()

    def update(self, x, dt):
        return self.mpc.update(x, dt) * _U_SCALE


# ---------------------------------------------------------- obstacle-aware iLQR

class ILQRObstacle:
    """iLQR/RTI-NMPC with the path-tracking quadratic + the same soft barrier
    as CEM, supplied as an exact analytic cost. ``_obstacle_cost_terms``'s
    gradient/Hessian is finite-difference-verified against
    ``_obstacle_penalty_batch`` by ``_check_obstacle_cost_derivatives()``
    below (run once at import time - iLQR gives no useful error message for a
    wrong Hessian, it just converges badly, so this is worth catching early)."""

    name = "iLQR / RTI-NMPC (obstacle-aware)"

    def __init__(self, H=70):
        Q = np.diag([12.0, 12.0, 4.0, 1.0, 0.3])
        R = np.diag([1.0, 0.5])
        self._clock = {"t": 0.0}

        def x_ref(t):
            return timed_ref(t)

        def u_ref(t):
            return np.array([V, timed_ref(t)[4]])

        def cost(X, U, Xr, Ur):
            Hh = U.shape[0]
            dx, du = X - Xr, U - Ur
            total = (np.einsum("ki,ij,kj->", dx[:Hh], Q, dx[:Hh])
                     + np.einsum("ki,ij,kj->", du, R, du))
            total += 3.0 * float(dx[Hh] @ Q @ dx[Hh])
            lx = np.zeros((Hh + 1, 5))
            lx[:Hh] = 2.0 * dx[:Hh] @ Q
            lx[Hh] = 6.0 * dx[Hh] @ Q
            lxx = np.broadcast_to(2.0 * Q, (Hh + 1, 5, 5)).copy()
            lxx[Hh] = 6.0 * Q
            lu = 2.0 * du @ R
            luu = np.broadcast_to(2.0 * R, (Hh, 2, 2)).copy()
            lux = np.zeros((Hh, 2, 5))

            t0 = self._clock["t"]
            for k in range(Hh + 1):
                tk = t0 + k * DT
                ot, gx, gy, hxx, hyy, hxy = _obstacle_cost_terms(X[k, 0], X[k, 1], tk)
                total += ot
                lx[k, 0] += gx; lx[k, 1] += gy
                lxx[k, 0, 0] += hxx; lxx[k, 1, 1] += hyy
                lxx[k, 0, 1] += hxy; lxx[k, 1, 0] += hxy
            return total, lx, lu, lxx, luu, lux

        self.ilqr = ILQR.from_system(
            robot, DT, horizon=H, Q=Q, R=R, u_bounds=U_BOUNDS,
            x_ref=x_ref, u_ref=u_ref, cost=cost,
            warm_iters=60 if FULL else 40, rti_iters=1, max_iter=60)

    def reset(self):
        self._clock["t"] = 0.0
        self.ilqr.reset()

    def update(self, x, dt):
        u = self.ilqr.update(x, dt)
        self._clock["t"] += dt
        return u


# ---------------------------------------------------------------------- timing

class Timed:
    def __init__(self, ctrl):
        self.c = ctrl
        self.name = ctrl.name
        self.dt_ms: list[float] = []

    def reset(self):
        self.dt_ms.clear()
        self.c.reset()

    def update(self, x, dt):
        t0 = time.perf_counter()
        u = self.c.update(x, dt)
        self.dt_ms.append((time.perf_counter() - t0) * 1e3)
        return u

    def stats(self):
        a = np.asarray(self.dt_ms[1:] or self.dt_ms)
        return float(np.median(a)), float(np.percentile(a, 95))


# -------------------------------------------------------------------- metrics

def _collision_steps(tr):
    steps = 0
    for k, t in enumerate(tr.t):
        for center_fn, r in OBSTACLES:
            cx, cy = center_fn(t)
            if np.hypot(tr.x[k, 0] - cx, tr.x[k, 1] - cy) < r:
                steps += 1
                break
    return steps


def run_one(ctrl):
    ctrl.reset()
    x0 = np.array([0.0, 0.0, np.arctan2(*(WAYPOINTS[1] - WAYPOINTS[0])[::-1]), V, 0.0])
    tr = simulate(robot, ctrl, x0=x0, dt=DT, t_final=T_FINAL, u_bounds=U_BOUNDS)
    n = len(tr.t)
    ref = np.array([timed_ref(t)[:2] for t in tr.t])
    err = np.hypot(tr.x[:, 0] - ref[:, 0], tr.x[:, 1] - ref[:, 1])
    completion = 100.0 * min(1.0, float(tr.t[-1]) / T_FINAL)
    du = tr.u - np.array([V, 0.0])
    med, p95 = ctrl.stats() if hasattr(ctrl, "stats") else (0.0, 0.0)
    return dict(
        trajectory=tr,
        rms_err_mm=float(np.sqrt(np.mean(err ** 2)) * 1e3),
        max_err_mm=float(np.max(err) * 1e3),
        completion_pct=completion,
        collision_steps=_collision_steps(tr),
        ctrl_energy=float(np.trapezoid(np.sum(du ** 2, axis=1), tr.t[:n])),
        lat_median_ms=med, lat_p95_ms=p95,
    )


# ------------------------------------------------------------------------ main

def build():
    return [Timed(PurePursuit()), Timed(PathLQR()),
           Timed(CEMObstacle()), Timed(ILQRObstacle())]


def main():
    results = {ctrl.name: run_one(ctrl) for ctrl in build()}

    cols = ["rms_err_mm", "max_err_mm", "completion_pct", "collision_steps",
           "ctrl_energy", "lat_median_ms", "lat_p95_ms"]
    lines = ["# Experiment 25 - a moving obstacle on the differential-drive path", "",
            f"3 obstacles (2 static + 1 moving, crosses the path near t~18 s), "
            f"{T_FINAL:.0f} s run, {DT * 1e3:.0f} ms step.", "",
            "| controller | " + " | ".join(cols) + " |",
            "| --- |" + " --- |" * len(cols)]
    for name, m in results.items():
        lines.append("| " + name + " | " + " | ".join(
            f"{m[c]:.4g}" for c in cols) + " |")
    lines.append("")

    (HERE / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv
    with open(HERE / "table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["controller", *cols])
        for name, m in results.items():
            w.writerow([name, *(m[c] for c in cols)])

    _figure(results)
    print((HERE / "table.md").read_text())


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_aimct_style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    ts, pl = PATH._polyline(400)
    ax[0].plot(pl[:, 0], pl[:, 1], "--", color=PALETTE["reference"], lw=1.3, label="path")
    cyc = [PALETTE["state_feedback"], PALETTE["lqr"], PALETTE["mpc"], PALETTE["hybrid"]]
    for i, (name, m) in enumerate(results.items()):
        tr = m["trajectory"]
        ax[0].plot(tr.x[:, 0], tr.x[:, 1], color=cyc[i % len(cyc)], lw=1.5, label=name)
    for center_fn, r in OBSTACLES[:2]:
        ax[0].add_patch(plt.Circle(center_fn(0.0), r, color=PALETTE["saturation"], alpha=0.25))
        ax[0].add_patch(plt.Circle(center_fn(0.0), r, color=PALETTE["saturation"], fill=False, lw=1.3))
    mv_c, mv_r = OBSTACLES[2]
    for t in np.linspace(14.0, T_FINAL, 6):
        c = mv_c(t)
        ax[0].add_patch(plt.Circle(c, mv_r, color="0.4", alpha=0.12))
    ax[0].add_patch(plt.Circle(mv_c(18.0), mv_r, color="crimson", fill=False, lw=1.6,
                               label="moving obstacle @t~18s"))
    ax[0].set(title="(a) path + obstacles (grey ghosts = moving obstacle over time)",
             xlabel="x [m]", ylabel="y [m]")
    ax[0].set_aspect("equal", "box")
    ax[0].legend(fontsize=7, loc="lower right")

    labels = list(results)
    x = np.arange(len(labels))
    rms = [results[n]["rms_err_mm"] for n in labels]
    coll = [results[n]["collision_steps"] for n in labels]
    ax2 = ax[1].twinx()
    b1 = ax[1].bar(x - 0.2, rms, 0.4, color=PALETTE["lqr"], label="RMS err [mm]")
    b2 = ax2.bar(x + 0.2, coll, 0.4, color=PALETTE["saturation"], label="collision steps")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.split(" (")[0] for n in labels],
                                               fontsize=8, rotation=10)
    ax[1].set_ylabel("RMS tracking error [mm]")
    ax2.set_ylabel("collision steps")
    ax[1].set_title("(b) tracking error vs collision count")
    ax[1].legend([b1, b2], ["RMS err [mm]", "collision steps"], fontsize=8)

    fig.suptitle("Exp 25 - moving-obstacle diffdrive: blind vs obstacle-aware planners",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
