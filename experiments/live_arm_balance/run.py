r"""Live double-inverted-pendulum sandbox — balancing the two-link arm upright.

`experiments/live_arm` manipulates a payload; this is the opposite control
problem on the *same* `aimct.systems.TwoLinkArm`: hold it standing straight
up (link 1 vertical, link 2 in line with it) against gravity — a genuine
double inverted pendulum, open-loop unstable (`A` has an eigenvalue with
positive real part at this equilibrium; `TwoLinkArm.linearize()` targets it by
default). The 3-way LQR story from the drone sandboxes reappears here:

* **LQR (stiff)** — regulates the linearised balance point; a *steady* torque
  disturbance ("wind") leaves a standing tilt (no integrator -> no way to
  cancel a constant input).
* **LQR + integral** — the same design plus an integral of the joint-angle
  error; drives the steady tilt to ~zero.
* **LQR (soft)** — much weaker gains: recovers a poke fine, but sags far more
  under a steady wind and is the one most likely to be pushed over.

Run:  python -m aimct live armbalance     (or: python experiments/live_arm_balance/run.py)
      python -m aimct live armbalance --headless
"""

from __future__ import annotations

import sys

import numpy as np

from aimct.controllers.lqr import solve_care
from aimct.systems import TwoLinkArm
from aimct.viz import Disturbance, Sandbox

ARM = TwoLinkArm()
XEQ = np.array([np.pi / 2.0, 0.0, 0.0, 0.0])     # standing straight up
UEQ = ARM.G(XEQ[:2])                             # ~0: a true equilibrium
DT, SUBSTEPS = 0.02, 8
FALL_LIMIT = 1.3                                 # rad from vertical -> "fallen"


def _lqr_gain(A, B, Q, R):
    P = solve_care(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


A, B = ARM.linearize()                           # about XEQ by default
# gains sized to this arm's real torque budget (+-15 / +-10 N.m): a poke or a
# steady wind should ask for feedback torque well inside that, not saturate
# and land in an unintended nonlinear sticking point (it did, at first - see
# the README's "how these gains were chosen" note).
Q = np.diag([8.0, 8.0, 1.0, 1.0])
R = np.diag([1.0, 1.0])
K_STIFF = _lqr_gain(A, B, Q, R)

# integral-augmented LQR on the two joint angles -> nulls a steady torque bias
_Ci = np.zeros((2, 4)); _Ci[0, 0] = _Ci[1, 1] = 1.0
_Aa = np.block([[A, np.zeros((4, 2))], [_Ci, np.zeros((2, 2))]])
_Ba = np.vstack([B, np.zeros((2, 2))])
_Qa = np.diag(np.concatenate([np.diag(Q), [3.0, 3.0]]))
_Ka = _lqr_gain(_Aa, _Ba, _Qa, R)
K_X, K_I = _Ka[:, :4], _Ka[:, 4:]

# "soft" is the *same* controller shape at 45% gain, not a re-tuned design -
# it is what happens when you turn this exact controller down. At this arm's
# coupling it is close to its stability margin (a linear eigenvalue is barely
# positive) - it settles a poke into a large, wrong offset instead of zero.
K_SOFT = 0.45 * K_STIFF


class LqrStiff:
    name = "LQR (stiff)"

    def reset(self):
        pass

    def update(self, x, dt):
        return UEQ - K_STIFF @ (np.asarray(x, float) - XEQ)


class LqrIntegral:
    name = "LQR + integral (wind-adaptive)"

    def reset(self):
        self.integ = np.zeros(2)

    def update(self, x, dt):
        x = np.asarray(x, float)
        self.integ = self.integ + (x[:2] - XEQ[:2]) * dt
        return UEQ - K_X @ (x - XEQ) - K_I @ self.integ


class LqrSoft:
    name = "LQR (soft)"

    def reset(self):
        pass

    def update(self, x, dt):
        return UEQ - K_SOFT @ (np.asarray(x, float) - XEQ)


def _wind(t, x, u, knobs):
    """A steady external torque ("wind") on each joint - independent of the
    motor's own torque limit, the way a real disturbance would be."""
    d = np.array([knobs.get("wind q1 [N.m]", 0.0), knobs.get("wind q2 [N.m]", 0.0)])
    ddq = np.linalg.solve(ARM.M(np.asarray(x, float)[:2]), d)
    return np.array([0.0, 0.0, ddq[0], ddq[1]])


_falls = {"count": 0}


def build() -> Sandbox:
    ctrls = {c.name: c for c in (LqrStiff(), LqrIntegral(), LqrSoft())}
    box = Sandbox(
        ARM, ctrls, x0=XEQ.copy(), dt=DT, substeps=SUBSTEPS,
        title="Live double-inverted-pendulum - balance under gravity",
        disturbance=Disturbance(
            sliders=[("wind q1 [N.m]", -2.0, 2.0, 0.0),
                     ("wind q2 [N.m]", -1.3, 1.3, 0.0)],
            hotkeys=[("p", "poke", lambda s: s.kick([0, 0, 1.5, -1.0]))],
            xdot_extra=_wind,
            help_text="p: poke   sliders: steady wind torque",
        ),
        aux_extra=lambda s: {"falls": _falls["count"]},
        on_step=_autoreset,
    )
    return box


def _tilt(x):
    return float(np.hypot(x[0] - XEQ[0], x[1] - XEQ[1]))


def _autoreset(box: Sandbox):
    x = box.x
    if not np.all(np.isfinite(x)) or abs(x[0] - XEQ[0]) > FALL_LIMIT:
        _falls["count"] += 1
        box.x[:] = XEQ + np.array([0.05, -0.03, 0.0, 0.0])
        box.controllers[box.active].reset()


def _headless() -> int:
    names = ("LQR (stiff)", "LQR + integral (wind-adaptive)", "LQR (soft)")

    print("live_arm_balance headless check\n")
    print("A) a poke (kick dq = [+1.5, -1.0] rad/s), no wind:")
    for name in names:
        box = build()
        box.set_controller(name)
        box.kick([0, 0, 1.5, -1.0])
        for _ in range(2000):
            box.step()
        tilt = np.degrees(_tilt(box.x))
        print(f"   {name:<32} settled {tilt:6.2f} deg from vertical")

    print("\nB) a steady wind torque [0.6, 0.4] N.m, no poke:")
    for name in names:
        box = build()
        box.set_controller(name)
        box.knobs["wind q1 [N.m]"] = 0.6
        box.knobs["wind q2 [N.m]"] = 0.4
        for _ in range(2500):
            box.step()
        tilt = np.degrees(_tilt(box.x))
        print(f"   {name:<32} standing tilt  {tilt:6.2f} deg   "
              f"fallen={'yes' if abs(box.x[0]-XEQ[0]) > FALL_LIMIT else 'no'}")
    print("\nheadless check OK")
    return 0


def main() -> int:
    if "--headless" in sys.argv:
        return _headless()
    build().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
