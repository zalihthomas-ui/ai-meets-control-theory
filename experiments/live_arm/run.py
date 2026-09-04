r"""Live 2-link-arm sandbox — computed torque vs. an unknown payload.

The interactive counterpart of Experiment 23B.  A planar
:class:`aimct.systems.TwoLinkArm` holds its wrist on a target you can move
(click, or arrow keys) while **you** hang an unknown payload on it with the
slider.  Switch controllers on the fly and watch:

* **PD + gravity comp** — high-gain, models only gravity; barely notices the
  payload (robustness by *not* modelling it), but always leaves a static droop.
* **Computed torque (0 kg model)** — cancels the nominal dynamics exactly, so it
  is the tightest with no payload and the *worst* once the wrist is loaded: it
  inverts a model that is now wrong.
* **Adaptive computed torque (Slotine-Li)** — the same law plus a one-parameter
  on-line payload estimate ``m̂``; give it a few seconds and it recovers.

Run:  python experiments/live_arm/run.py           (or:  python -m aimct live arm)
      python experiments/live_arm/run.py --headless
"""

from __future__ import annotations

import sys

import numpy as np

from aimct.systems import TwoLinkArm
from aimct.viz import Disturbance, Sandbox

ARM = TwoLinkArm()                      # the true plant (payload set by the slider)
NOM = TwoLinkArm(payload=0.0)           # the model every controller believes
REACH = ARM.l1 + ARM.l2


def ik(target, elbow="down"):
    """Elbow-down/up inverse kinematics for the wrist position ``target``."""
    x, y = np.asarray(target, float)[:2]
    r2 = float(np.clip(x * x + y * y, 1e-6, (0.999 * REACH) ** 2))
    c2 = np.clip((r2 - ARM.l1 ** 2 - ARM.l2 ** 2) / (2 * ARM.l1 * ARM.l2), -1, 1)
    q2 = np.arccos(c2) * (1.0 if elbow == "down" else -1.0)
    q1 = np.arctan2(y, x) - np.arctan2(ARM.l2 * np.sin(q2),
                                       ARM.l1 + ARM.l2 * np.cos(q2))
    return np.array([q1, q2])


def _payload_terms(q, dq):
    """``(M1, C1, G1)`` — the wrist-payload contribution to ``M``, ``C``, ``G``
    for a *unit* point mass at the link-2 tip (matches Experiment 23B)."""
    l1, l2 = NOM.l1, NOM.l2
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    M1 = np.array([[l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * c2, l2 ** 2 + l1 * l2 * c2],
                   [l2 ** 2 + l1 * l2 * c2, l2 ** 2]])
    h = l1 * l2 * s2
    C1 = np.array([[-h * dq[1], -h * (dq[0] + dq[1])], [h * dq[0], 0.0]])
    q12 = q[0] + q[1]
    G1 = NOM.g * np.array([l1 * np.cos(q[0]) + l2 * np.cos(q12), l2 * np.cos(q12)])
    return M1, C1, G1


class _Reg:
    """Drive the wrist to ``target`` — a live array the sandbox mutates in
    place (mouse / arrow keys).  The raw target can jump (a click), so it is
    passed through a critically-damped 2nd-order *reference model* that emits a
    smooth, self-consistent ``(qr, dqr, ddqr)`` — no finite differencing, and
    the model-based laws (and the adaptive estimator) stay well-posed."""

    wn = 3.0          # reference-model bandwidth [rad/s] (keeps ddqr within the
                      # arm's modest joint-torque budget)

    def __init__(self, target, dt: float = 0.02):
        self.target = target
        self.dt = dt

    def reset(self):
        self.qr = ik(self.target)
        self.dqr = np.zeros(2)

    def _ref(self, x):
        q, dq = np.asarray(x, float)[:2], np.asarray(x, float)[2:]
        if not hasattr(self, "qr"):
            self.reset()
        qd = ik(self.target)
        ddqr = self.wn ** 2 * (qd - self.qr) - 2 * self.wn * self.dqr
        self.dqr = self.dqr + ddqr * self.dt
        self.qr = self.qr + self.dqr * self.dt
        return q, dq, self.qr, self.dqr, ddqr


# gains are ~1/3 of Experiment 23B's — its joint spline is gentle, but a hand
# target circled in real time asks for more, and this teaching arm has only
# +-15 / +-10 N.m of joint torque.  Kept just under saturation on the default
# circle so the comparison is about *modelling*, not who clips first.
class PDGravComp(_Reg):
    name = "PD + gravity comp"
    Kp, Kd = np.diag([56.0, 49.0]), np.diag([9.0, 8.0])

    def update(self, x, dt):
        q, dq, qr, dqr, _ = self._ref(x)
        return self.Kp @ (qr - q) + self.Kd @ (dqr - dq) + NOM.G(q)


class ComputedTorque(_Reg):
    name = "Computed torque (0 kg model)"
    Kp, Kd = np.diag([56.0, 49.0]), np.diag([9.0, 8.0])

    def update(self, x, dt):
        q, dq, qr, dqr, ddqr = self._ref(x)
        a = ddqr + self.Kp @ (qr - q) + self.Kd @ (dqr - dq)
        return NOM.M(q) @ a + NOM.C(q, dq) @ dq + NOM.G(q) + NOM.b * dq


class AdaptiveComputedTorque(_Reg):
    """Slotine-Li: computed torque on the nominal model plus ``mhat*Y1`` with a
    one-parameter payload estimate driven by the sliding variable ``s`` — the
    same law as Experiment 23B, gains scaled to this arm's torque budget."""

    name = "Adaptive computed torque"
    lam, gamma = 7.0, 3.0
    Kd_s = np.diag([9.0, 8.0])
    sigma = 0.05           # leakage: pulls m̂ back toward 0 when s is small, so
                           # tracking *lag* is not misread as a payload
    MHAT_MAX = 0.6         # projection bound = the arm's rated max wrist payload

    def reset(self):
        super().reset()
        self.mhat = 0.0

    def update(self, x, dt):
        q, dq, qr, dqr, ddqr = self._ref(x)
        Lam = self.lam * np.eye(2)
        e, de = q - qr, dq - dqr
        s = de + Lam @ e
        dqr_ = dqr - Lam @ e
        ddqr_ = ddqr - Lam @ de
        tau0 = (NOM.M(q) @ ddqr_ + NOM.C(q, dq) @ dqr_ + NOM.G(q)
                + NOM.b * dqr_)
        M1, C1, G1 = _payload_terms(q, dq)
        Y1 = M1 @ ddqr_ + C1 @ dqr_ + G1
        tau = tau0 + self.mhat * Y1 - self.Kd_s @ s
        # sigma-modified gradient step + a projection bound (robust adaptive
        # control: identifies a real payload, ignores mere tracking lag)
        mdot = -self.gamma * (float(Y1 @ s) + self.sigma * self.mhat)
        self.mhat = float(np.clip(self.mhat + dt * mdot, 0.0, self.MHAT_MAX))
        return tau


def _clip_reach(x, y):
    r = np.hypot(x, y)
    scale = min(1.0, 0.98 * REACH / (r + 1e-9))
    return np.array([x * scale, y * scale])


DT = 0.02
CENTRE0 = np.array([0.42, 0.22])       # hold point (arrow keys / click move it)
RADIUS = 0.0                           # >0 makes the target circle; 0 = a fixed hold
RATE = 0.6                             # rad/s around the circle (if RADIUS > 0)


class _Path:
    """Where the wrist should be.  A fixed hold point by default; set ``RADIUS``
    for a slow circle.  Either way the payload step is what the demo turns on."""

    def __init__(self):
        self.centre = CENTRE0.copy()

    def point(self, t):
        if RADIUS <= 0.0:
            return self.centre.copy()
        return self.centre + RADIUS * np.array([np.cos(RATE * t), np.sin(RATE * t)])


def _clip_reach_centre(pth, x, y):
    r = np.hypot(x, y)
    scale = min(1.0, 0.80 * REACH / (r + 1e-9))
    pth.centre[:] = [x * scale, y * scale]


def build() -> tuple[Sandbox, _Path]:
    pth = _Path()
    target = pth.point(0.0).copy()                  # shared, mutated in place
    x0 = np.concatenate([ik(target), [0.0, 0.0]])
    ctrls = {c.name: c for c in (PDGravComp(target, DT), ComputedTorque(target, DT),
                                 AdaptiveComputedTorque(target, DT))}
    box = Sandbox(
        ARM, ctrls, x0=x0, target=target, dt=DT, substeps=8,
        title="Live 2-link arm - computed torque vs. an unknown payload",
        disturbance=Disturbance(
            sliders=[("wrist payload [kg]", 0.0, 0.6, 0.0)],
            hotkeys=[("p", "poke", lambda s: s.kick([0, 0, 4.0, -4.0])),
                     ("up", "centre +y", lambda s: pth.centre.__iadd__([0, 0.03])),
                     ("down", "centre -y", lambda s: pth.centre.__iadd__([0, -0.03])),
                     ("left", "centre -x", lambda s: pth.centre.__iadd__([-0.03, 0])),
                     ("right", "centre +x", lambda s: pth.centre.__iadd__([0.03, 0]))],
            on_slider=lambda s, name, v: setattr(ARM, "payload", float(v)),
            help_text="click / arrows: move circle   p: poke   slider: payload",
        ),
        on_step=lambda s: s.set_target(pth.point(s.t)),
        on_click=lambda s, x, y: _clip_reach_centre(pth, x, y),
    )
    return box, pth


def _run_scenario(name: str, payload: float, steps: int, step_at: int = 120):
    """Hold the target; add ``payload`` kg at ``step_at``.  Return the settled
    tip error (last 20 %) and the payload estimate (adaptive only)."""
    from aimct.viz import get_artist

    box, _ = build()
    box.set_controller(name)
    art = get_artist(ARM)
    errs, peak = [], 0.0
    for k in range(steps):
        if k == step_at:
            ARM.payload = payload
        box.step()
        e = np.linalg.norm(art.position(box.x) - box.target)
        errs.append(e)
        if k > step_at:
            peak = max(peak, e)
    settled = float(np.mean(errs[int(0.8 * len(errs)):]) * 1e3)
    return settled, peak * 1e3, getattr(box.controllers[name], "mhat", None)


def _headless() -> int:
    names = ("PD + gravity comp", "Computed torque (0 kg model)",
             "Adaptive computed torque")
    print("live_arm headless check - hold a target, then hang an unknown "
          "0.4 kg payload on the wrist at t = 2.4 s\n")
    for n in names:
        settled, peak, mhat = _run_scenario(n, 0.4, 900)
        tag = f"   mhat -> {mhat:.2f} kg (true 0.40)" if mhat is not None else ""
        print(f"   {n:<32} droop peak {peak:6.1f} mm   settled {settled:6.1f} mm{tag}")
    print("\nheadless check OK")
    return 0


def main() -> int:
    if "--headless" in sys.argv:
        return _headless()
    build()[0].run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
