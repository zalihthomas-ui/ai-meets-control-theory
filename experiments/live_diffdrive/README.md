# Live differential-drive sandbox — path followers vs. a shove

The interactive counterpart of [Experiment 22](../22_diffdrive_path_following/).
A `aimct.systems.DifferentialDriveRobot` drives a fixed figure-8 loop under one
of three path-followers while **you** shove it off course and steal wheel
traction.

```bash
python -m aimct live diffdrive            # or:  python experiments/live_diffdrive/run.py
python -m aimct live diffdrive --headless # smoke check, no GUI
```

## Controls

| input | action |
| :--- | :--- |
| **arrow keys** | shove the robot ±x / ±y (position impulse) |
| **t** | kick the body speed (a lurch) |
| **wheel-slip slider** | 0 → 0.6 of commanded speed lost to slip — a steady drag on `v` and `ω` |
| **1 / 2 / 3** | Pure pursuit / Stanley / Path LQR |
| **r** | reset |

## The three followers

| follower | how it steers | character |
| :--- | :--- | :--- |
| **Pure pursuit** | aim at a look-ahead point on the path | re-acquires fast once it can simply steer back toward a point ahead |
| **Stanley** | null heading error + `atan` cross-track term at the front axle | snaps the heading in hard; can wag through high curvature |
| **Path LQR** | LQR on the path-error state `[y, θ, v, ω]` + a curvature feed-forward | smooth, model-based; slower to re-settle under sustained slip since it is also fighting a `v`/`ω` error the slip keeps re-injecting |

Headless numbers (a hard shove + 30 % wheel slip at t = 10 s):

```
Pure pursuit    peak cross-track  149 mm   back < 30 mm after  0.5 s
Stanley         peak cross-track  199 mm   back < 30 mm after  2.3 s
Path LQR        peak cross-track  287 mm   back < 30 mm after  9.3 s
```

Under sustained wheel slip none of them close to zero at first — that is a
modelling gap (every follower assumes the commanded twist is achieved), and
the honest read is in the HUD's live cross-track number. (The path's
look-ahead / nearest-point search uses progress hysteresis rather than a
fresh global search each frame — the figure-8 self-intersects, and a naive
per-frame nearest point flips between the two branches at the crossing,
which used to show up as pure pursuit's target visibly teleporting.)
