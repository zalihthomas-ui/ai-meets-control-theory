"""Experiment 23 - two-link arm follows a joint-space trajectory.

Part A - nominal model, four torque-level controllers on the same joint spline:
  PD + gravity compensation - computed torque - joint LQR - joint linear MPC.

Part B - the *true* arm picks up an unknown 0.5 kg wrist payload while every
controller still believes the nominal (0 kg) model.  Fixed computed torque
degrades; a one-parameter adaptive computed torque (Slotine-Li) estimates the
payload on-line and recovers the tracking - the Experiment-17 lesson on a real
manipulator.

Run:  python experiments/23_twolink_arm_tracking/run.py
Outputs (next to this file): tracking.md/.csv/.png (Part A),
                             payload.md/.csv/.png (Part B).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aimct.benchmarks.tracking import TrackingResult, track_trajectory
from aimct.controllers import LQR, LinearMPC
from aimct.systems import TwoLinkArm
from aimct.trajectories import Spline

HERE = Path(__file__).parent

DT = 2e-3
JOINT_WAYPOINTS = np.array([[0.40, 0.30], [1.15, -0.55], [1.65, 0.45],
                            [0.70, -0.20], [1.30, 0.15]])
T_TOTAL = 7.0
MID_Q = np.array([1.0, 0.0, 0.0, 0.0])          # linearisation config for LQR / MPC

NOM = TwoLinkArm(payload=0.0)
PAYLOAD = 0.5


def _joint_spline():
    P = JOINT_WAYPOINTS
    knots = np.linspace(0.0, T_TOTAL, len(P))
    return Spline(P, knots=knots)


REF = _joint_spline()
X0 = np.array([JOINT_WAYPOINTS[0, 0], JOINT_WAYPOINTS[0, 1], 0.0, 0.0])
TAU_BOUNDS = (-NOM.tau_max, NOM.tau_max)


# ------------------------------------------------- payload regressor (Y1 column)

def _payload_terms(arm, q, dq):
    """The ``m_p`` coefficient of (M, C, G) - a point mass at the link-2 tip."""
    l1, l2 = arm.l1, arm.l2
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    M1 = np.array([[l1**2 + l2**2 + 2 * l1 * l2 * c2, l2**2 + l1 * l2 * c2],
                   [l2**2 + l1 * l2 * c2,             l2**2]])
    h1 = l1 * l2 * s2
    C1 = np.array([[-h1 * dq[1], -h1 * (dq[0] + dq[1])],
                   [ h1 * dq[0],  0.0]])
    q12 = q[0] + q[1]
    G1 = np.array([l1 * arm.g * np.cos(q[0]) + l2 * arm.g * np.cos(q12),
                   l2 * arm.g * np.cos(q12)])
    return M1, C1, G1


# ------------------------------------------------------------- controllers

class _Tracker:
    """Holds a clock and samples ``REF`` each step."""

    def reset(self):
        self._t = 0.0

    def _ref(self):
        p, v, a = REF(min(self._t, REF.duration))
        return np.asarray(p), np.asarray(v), np.asarray(a)

    def update(self, x, dt):
        u = self._law(np.asarray(x, float))
        self._t += dt
        return u


class PDGravComp(_Tracker):
    name = "PD + gravity comp"
    Kp, Kd = np.diag([120.0, 90.0]), np.diag([22.0, 16.0])

    def _law(self, x):
        q, dq = x[:2], x[2:]
        qr, dqr, _ = self._ref()
        return self.Kp @ (qr - q) + self.Kd @ (dqr - dq) + NOM.G(q)


class ComputedTorque(_Tracker):
    name = "Computed torque"
    Kp, Kd = np.diag([160.0, 140.0]), np.diag([25.0, 22.0])

    def __init__(self, model=NOM):
        self.model = model

    def _law(self, x):
        q, dq = x[:2], x[2:]
        qr, dqr, ddqr = self._ref()
        a = ddqr + self.Kp @ (qr - q) + self.Kd @ (dqr - dq)
        m = self.model
        return m.M(q) @ a + m.C(q, dq) @ dq + m.G(q) + m.b * dq


class JointLQR(_Tracker):
    name = "Joint LQR"

    def __init__(self):
        A, B = NOM.linearize(MID_Q)
        self.K = LQR(A, B, np.diag([80.0, 80.0, 6.0, 6.0]), np.diag([1.0, 1.0])).K

    def _law(self, x):
        q, dq = x[:2], x[2:]
        qr, dqr, ddqr = self._ref()
        z = x - np.concatenate([qr, dqr])
        return NOM.M(q) @ ddqr + NOM.G(q) + NOM.b * dq - self.K @ z


class JointMPC(_Tracker):
    """Inverse-dynamics feed-forward (M ddq_r + C dq + G + b dq) plus a linear
    MPC that regulates the tracking-error state on the model linearised about a
    mid-trajectory configuration - a receding-horizon counterpart of JointLQR."""

    name = "Joint MPC"

    def __init__(self, N=20):
        # decoupled double-integrator error model (feedback linearisation): e'' = a
        Ae = np.block([[np.zeros((2, 2)), np.eye(2)], [np.zeros((2, 4))]])
        Be = np.block([[np.zeros((2, 2))], [np.eye(2)]])
        self.mpc = LinearMPC(Ae, Be, Q=np.diag([400.0, 400.0, 20.0, 20.0]),
                             R=np.diag([0.2, 0.2]), N=N)
        self.N = N

    def reset(self):
        super().reset()
        self.mpc.reset()

    def _law(self, x):
        q, dq = x[:2], x[2:]
        qr, dqr, ddqr = self._ref()
        z = np.concatenate([q - qr, dq - dqr])       # error state for the MPC
        self.mpc.x_ref = np.zeros((self.N, 4))
        a = np.asarray(self.mpc.update(z, DT)).reshape(2)     # virtual accel command
        return NOM.M(q) @ (ddqr + a) + NOM.C(q, dq) @ dq + NOM.G(q) + NOM.b * dq


class AdaptiveComputedTorque(_Tracker):
    """Slotine-Li: computed torque on the nominal model plus ``mhat * Y1`` with
    a one-parameter payload estimate driven by the sliding variable."""

    name = "Adaptive computed torque"
    lam, gamma = 12.0, 3.0
    Kd_s = np.diag([26.0, 22.0])

    def reset(self):
        super().reset()
        self.mhat = 0.0
        self._last = None

    def update(self, x, dt):
        x = np.asarray(x, float)
        q, dq = x[:2], x[2:]
        qr, dqr, ddqr = self._ref()
        Lam = self.lam * np.eye(2)
        e, de = q - qr, dq - dqr
        s = de + Lam @ e
        dqr_ = dqr - Lam @ e
        ddqr_ = ddqr - Lam @ de

        tau0 = NOM.M(q) @ ddqr_ + NOM.C(q, dq) @ dqr_ + NOM.G(q) + NOM.b * dqr_
        M1, C1, G1 = _payload_terms(NOM, q, dq)
        Y1 = M1 @ ddqr_ + C1 @ dqr_ + G1
        tau = tau0 + self.mhat * Y1 - self.Kd_s @ s

        self.mhat = max(0.0, self.mhat - dt * self.gamma * float(Y1 @ s))
        self._t += dt
        return tau


# ----------------------------------------------------------------- run

def _table(res: TrackingResult, title: str) -> str:
    r = TrackingResult(metrics=res.metrics, trajectories=res.trajectories,
                       trajectory=res.trajectory, pos_index=(0, 1), title=title)
    return r.to_markdown()


def main():
    # Part A - nominal
    ctrls_a = {c.name: c for c in (PDGravComp(), ComputedTorque(), JointLQR(),
                                   JointMPC())}
    res_a = track_trajectory(NOM, ctrls_a, REF, X0, dt=DT, t_final=T_TOTAL,
                             pos_index=(0, 1), u_bounds=TAU_BOUNDS,
                             title="Exp 23A - two-link arm joint tracking (nominal)")
    res_a.space = "joint"                            # q1/q2 [rad], mrad cross-track
    res_a.save(HERE)

    # Part B - true arm carries an unknown 0.5 kg payload
    true_arm = TwoLinkArm(payload=PAYLOAD)
    ctrls_b = {
        "PD + gravity comp": PDGravComp(),
        "Computed torque (nominal model)": ComputedTorque(model=NOM),
        "Adaptive computed torque": AdaptiveComputedTorque(),
    }
    res_b = track_trajectory(true_arm, ctrls_b, REF, X0, dt=DT, t_final=T_TOTAL,
                             pos_index=(0, 1), u_bounds=TAU_BOUNDS,
                             title="Exp 23B - unknown 0.5 kg wrist payload")
    res_b.space = "joint"
    (HERE / "payload.md").write_text(res_b.to_markdown(), encoding="utf-8", newline="\n")
    (HERE / "payload.csv").write_text(res_b.to_csv(), encoding="utf-8", newline="\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, _ = res_b.figure()
        fig.savefig(HERE / "payload.png", dpi=150)
        plt.close(fig)
    except Exception as exc:               # pragma: no cover
        print("payload figure skipped:", exc)

    mhat = ctrls_b["Adaptive computed torque"].mhat
    print(res_a.to_markdown())
    print(res_b.to_markdown())
    print(f"\nadaptive payload estimate: mhat = {mhat:.3f} kg  (true {PAYLOAD} kg)")


if __name__ == "__main__":
    main()
