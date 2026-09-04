# Live double-inverted-pendulum sandbox — balancing the two-link arm upright

`experiments/live_arm` manipulates a payload; this is the opposite problem on
the *same* `aimct.systems.TwoLinkArm`: hold it standing straight up (link 1
vertical, link 2 in line with it) against gravity. That equilibrium is
open-loop **unstable** — `TwoLinkArm.linearize()` targets it by default, and
its `A` matrix has an eigenvalue with positive real part there — so this is a
genuine double inverted pendulum, not a manipulator holding a pose.

```bash
python -m aimct live armbalance            # or: python experiments/live_arm_balance/run.py
python -m aimct live armbalance3d          # the same sandbox as a real 3-D PyVista scene
python -m aimct live armbalance --headless
```

The 3-D view is the identical physics, gains and controllers, rendered by the
shared `aimct.viz.pv_arm.run_pyvista_arm` (same driver `live_arm` uses) —
orbit the camera edge-on and the "double pendulum" collapses back to the line
it actually is. Every control below, including `h`/`g`/`c` (see
[docs/VISUALIZATION.md](../../docs/VISUALIZATION.md)), works the same in both.

## Controls

| input | action |
| :--- | :--- |
| **p** | poke the arm (a `[+1.5, -1.0]` rad/s velocity kick) |
| **wind q1 / wind q2 sliders** | a steady external torque per joint — independent of the motor, the way a real disturbance would be |
| **1 / 2 / 3** | LQR (stiff) / LQR + integral (wind-adaptive) / LQR (soft) |
| **r** | reset |

A fall (either link tilts past ~75° from vertical) auto-resets near-upright
and increments the on-screen fall counter — the sandbox never needs a manual
reset just to keep going.

## The three controllers — same 3-way story as the drone sandboxes

| controller | a poke, no wind | a steady wind torque |
| :--- | :--- | :--- |
| **LQR (stiff)** | fully recovers | a **standing tilt** (no integrator can null a constant input) |
| **LQR + integral (wind-adaptive)** | fully recovers | tilt driven to **~0°** |
| **LQR (soft)** — the *same* gain matrix at 45 % | settles into a **large wrong offset**, not zero | a **dramatic droop**, close to falling |

Headless numbers:

```
A) poke (kick dq = [+1.5, -1.0] rad/s), no wind:
   LQR (stiff)                      settled   0.00 deg from vertical
   LQR + integral (wind-adaptive)   settled   0.00 deg from vertical
   LQR (soft)                       settled  36.98 deg from vertical

B) steady wind torque [0.6, 0.4] N.m, no poke:
   LQR (stiff)                      standing tilt    6.71 deg
   LQR + integral (wind-adaptive)   standing tilt    0.00 deg
   LQR (soft)                       standing tilt   54.91 deg
```

## How these gains were chosen (an honest note)

The first "stiff" design solved cleanly on the linearisation and looked fine
on paper — but simulating it on the real nonlinear arm, even a modest poke
made it **stick at a ~27° offset instead of returning to vertical**. The cause
wasn't a bug: the initial transient torque the optimal linear gain demanded
briefly exceeded the arm's real `tau_max = [15, 10] N.m`, the controller
clipped, and because the two joints are strongly coupled at this equilibrium,
saturating even briefly threw the *nonlinear* system onto a different (bad)
attractor instead of degrading gracefully. The fix was gains sized to the
actual torque budget (`Q = diag([8, 8, 1, 1])`, `R = diag([1, 1])`) so a
realistic poke or wind never asks for more torque than the arm has. "Soft" is
deliberately the *same* controller at 45 % gain — not a separately retuned
design — so what you are watching fail is specifically "not enough feedback,"
with everything else held constant.
