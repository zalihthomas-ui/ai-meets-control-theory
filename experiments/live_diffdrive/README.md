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
| **Pure pursuit** | aim at a look-ahead point on the path | smooth, cuts corners, forgives a big lateral error but drifts back slowly |
| **Stanley** | null heading error + `atan` cross-track term at the front axle | snaps back to the line hard; can wag through high curvature |
| **Path LQR** | LQR on the path-error state `[y, θ, v, ω]` + a curvature feed-forward | the balance — lowest peak excursion, steady recovery |

Headless numbers (a hard shove + 30 % wheel slip at t = 10 s):

```
Pure pursuit    peak cross-track  276 mm   back < 30 mm after  7.9 s
Stanley         peak cross-track  186 mm   back < 30 mm after  2.2 s
Path LQR        peak cross-track  174 mm   back < 30 mm after  8.3 s
```

Stanley's front-axle `atan` term makes it the quickest to re-acquire the line
after a displacement; pure pursuit is the gentlest but the slowest; Path LQR
keeps the *peak* excursion smallest. Under sustained wheel slip none of them
close to zero — that is a modelling gap (the followers assume the commanded
twist is achieved), and the honest read is in the HUD's live cross-track number.
