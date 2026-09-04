"""Experiment 22 - differential-drive robot follows a waypoint path.

Four path-followers on the same spline through five waypoints, scored by the
trajectory-tracking harness:

* **Pure pursuit** - geometric: steer at a look-ahead point on the path.
* **Stanley** - null the heading error plus an arctan cross-track term.
* **Path LQR** - LQR on the error state of the model linearised about the
  path, with a curvature feed-forward on the yaw rate.
* **Kinematic MPC** - condensed linear MPC on the same error model with a
  curvature-preview feed-forward.

Run:  python experiments/22_diffdrive_path_following/run.py
Outputs (next to this file): tracking.md, tracking.csv, tracking.png, README data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks.tracking import track_trajectory
from aimct.controllers import LQR, LinearMPC, wrap_angle
from aimct.systems import DifferentialDriveRobot
from aimct.trajectories import Spline

HERE = Path(__file__).parent

V = 0.15                         # cruise speed [m/s]
DT = 0.02
WAYPOINTS = np.array([[0.0, 0.0], [1.0, 0.6], [2.2, -0.2],
                      [3.2, 0.8], [4.5, 0.0]])

robot = DifferentialDriveRobot(v_ref=V)


def timed_spline(waypoints, v):
    """A natural cubic spline whose time parametrisation is traversed at the
    robot's cruise speed ``v`` -- ``traj(t)`` is then where the robot *should*
    be at time ``t`` (so the along-track metric is meaningful).  ``Spline``
    already memoises its polyline / arc length on the instance."""
    P = np.asarray(waypoints, dtype=float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    knots = np.concatenate([[0.0], np.cumsum(seg) / v])       # duration = length / v
    return Spline(P, knots=knots)


PATH = timed_spline(WAYPOINTS, V)
T_FINAL = PATH.duration                  # stop when the reference reaches the last waypoint
U_BOUNDS = (np.array([-robot.v_max, -robot.omega_max]),
            np.array([robot.v_max, robot.omega_max]))

# a dense polyline of the path + its cumulative arc length, for look-ahead and
# curvature queries
_TS, _PL = PATH._polyline(1200)
_SL = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(_PL, axis=0), axis=1))])


def _nearest_idx(p):
    return int(np.argmin(np.hypot(_PL[:, 0] - p[0], _PL[:, 1] - p[1])))


def _path_frame(p):
    """Nearest point, tangent heading, signed cross-track error (+ = robot left
    of the path) and path curvature at the nearest point."""
    i = _nearest_idx(p)
    j = min(i + 1, len(_PL) - 1)
    tang = _PL[j] - _PL[max(i - 1, 0)]
    th_path = np.arctan2(tang[1], tang[0])
    d = np.asarray(p)[:2] - _PL[i]
    e_cross = np.sin(th_path) * d[0] - np.cos(th_path) * d[1]   # left-positive... see sign
    e_cross = -e_cross
    # curvature from the analytic derivatives
    _, v, a = PATH(_TS[i])
    sp = np.hypot(v[0], v[1]) + 1e-9
    kappa = (v[0] * a[1] - v[1] * a[0]) / sp ** 3
    return _PL[i], th_path, e_cross, kappa, _SL[i]


def _lookahead_point(p, ld):
    i = _nearest_idx(p)
    target_s = _SL[i] + ld
    k = int(np.searchsorted(_SL, min(target_s, _SL[-1])))
    return _PL[min(k, len(_PL) - 1)]


# --------------------------------------------------------------- controllers

class PurePursuit:
    name = "Pure pursuit"

    def __init__(self, ld=0.45):
        self.ld = ld

    def reset(self):
        pass

    def update(self, x, dt):
        p, theta = x[:2], x[2]
        tgt = _lookahead_point(p, self.ld)
        alpha = wrap_angle(np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - theta)
        omega = 2.0 * V * np.sin(alpha) / self.ld
        return np.array([V, omega])


class Stanley:
    name = "Stanley"

    def __init__(self, k_e=1.5, k_stanley=2.2):
        self.k_e, self.k_stanley = k_e, k_stanley

    def reset(self):
        pass

    def update(self, x, dt):
        _, th_path, e_cross, _, _ = _path_frame(x[:2])
        psi = wrap_angle(th_path - x[2])                 # heading error
        # left of the path (e_cross > 0) -> steer right: subtract the arctan term
        delta = psi - np.arctan2(self.k_e * e_cross, V)
        return np.array([V, self.k_stanley * delta])


class PathLQR:
    name = "Path LQR"

    def __init__(self):
        A, B = robot.linearize()                       # straight path at v_ref
        idx = np.array([1, 2, 3, 4])                   # [y, theta, v, omega] error
        self.K = LQR(A[np.ix_(idx, idx)], B[idx, :],
                     np.diag([30.0, 8.0, 2.0, 0.5]), np.diag([2.0, 1.0])).K

    def reset(self):
        pass

    def update(self, x, dt):
        _, th_path, e_cross, kappa, _ = _path_frame(x[:2])
        psi = wrap_angle(x[2] - th_path)
        z = np.array([e_cross, psi, x[3] - V, x[4]])
        u_ff = np.array([V, kappa * V])
        return u_ff - self.K @ z


class KinematicMPC:
    name = "Kinematic MPC"

    def __init__(self, N=25):
        A, B = robot.linearize()
        idx = np.array([1, 2, 3, 4])
        self.mpc = LinearMPC(A[np.ix_(idx, idx)], B[idx, :],
                             Q=np.diag([60.0, 12.0, 2.0, 0.5]),
                             R=np.diag([1.0, 0.4]), N=N,
                             u_bounds=(np.array([-robot.v_max, -robot.omega_max]),
                                       np.array([robot.v_max, robot.omega_max])))
        self.N = N

    def reset(self):
        self.mpc.reset()

    def update(self, x, dt):
        p = x[:2]
        _, th_path, e_cross, _, s0 = _path_frame(p)
        psi = wrap_angle(x[2] - th_path)
        z = np.array([e_cross, psi, x[3] - V, x[4]])
        # curvature preview: sample the path ahead by V*dt per step
        uref = np.zeros((self.N, 2))
        for j in range(self.N):
            k = int(np.searchsorted(_SL, min(s0 + V * dt * (j + 1), _SL[-1])))
            _, vj, aj = PATH(_TS[min(k, len(_TS) - 1)])
            spj = np.hypot(vj[0], vj[1]) + 1e-9
            uref[j] = [0.0, (vj[0] * aj[1] - vj[1] * aj[0]) / spj ** 3 * V]
        self.mpc.x_ref = np.zeros((self.N, 4))
        self.mpc.u_ref = uref                            # [0, kappa*V] curvature ff
        u0 = np.asarray(self.mpc.update(z, dt)).reshape(2)  # ~ u_ref[0] - K z
        return np.array([V, 0.0]) + u0


# ----------------------------------------------------------------- run

def build():
    return {c.name: c for c in (PurePursuit(), Stanley(), PathLQR(), KinematicMPC())}


def main():
    x0 = np.array([0.0, 0.0, np.arctan2(*(WAYPOINTS[1] - WAYPOINTS[0])[::-1]),
                   V, 0.0])
    res = track_trajectory(robot, build(), PATH, x0, dt=DT, t_final=T_FINAL,
                           pos_index=(0, 1), u_bounds=U_BOUNDS,
                           title="Exp 22 - differential-drive path following")
    res.save(HERE)
    print(res.to_markdown())


if __name__ == "__main__":
    main()
