# Experiment 24 — gradient vs sampling for online nonlinear MPC

**Question.** Experiment 21 noted that the cross-entropy sampling MPC (CEM) used
throughout the capstone is "loose and slow". How much does a *gradient-based*
nonlinear MPC — iLQR run as a real-time iteration (RTI) — actually buy over CEM,
on the same plants, horizon and cost?

## Setup

Two receding-horizon planners, re-solving online from the measured state every
20 ms:

| planner | how it plans | derivatives | warm start |
| --- | --- | --- | --- |
| **Sampling MPC (CEM)** | refine a Gaussian over action sequences, keep the elite rollouts (`aimct.controllers.SamplingMPC`) | none | shift the elite mean, re-inflate σ |
| **iLQR / RTI-NMPC** | regularised backward Riccati sweep + line-searched forward rollout (`aimct.controllers.ILQR`) | finite-difference Jacobians of one RK4 step | full solve on step 1, then **one** iLQR iteration per step from the shifted plan; applies `u₀ + K₀(x − x₀)` |

Both get the **same** stage weights `Q, R`, the same horizon, and the same
terminal cost (the infinite-horizon LQR cost-to-go of the hover linearisation
for the quad; a heavy diagonal for the cart-pole). Input limits are hard for
both — CEM clips its samples, iLQR clamps its forward pass.

### Task 1 — cart-pole swing-up, solved online

From hanging (`θ = π`), drive the pole upright and hold it, `|F| ≤ 20 N`,
horizon 60 steps (1.2 s), 4 s run. A genuine nonlinear OCP with no pre-computed
plan — the planner has to *find* the swing-up itself.

### Task 2 — quadrotor figure-8 tracking

The Experiment-14 lemniscate on the Crazyflie 2.0 (planar 6-state), no obstacle,
horizon 20 steps, 12 s.

We score task / tracking error, control effort, and **per-step wall-clock
latency** — median and p95 over the run, with the cold first solve reported
separately (`lat_cold_ms`) because iLQR deliberately front-loads its work there.

## Results

See `table.md` / `table.csv` / `figure.png` (committed artifacts are the
`AIMCT_EXP_FULL=1` run).

Headline numbers (`AIMCT_EXP_FULL=1`, single machine — latencies are relative,
not absolute specs):

| | CEM | iLQR / RTI |
| --- | --- | --- |
| swing-up: time to upright | ~1.6 s | ~1.1 s |
| swing-up: residual angle | ~1.6° | ~0.4° |
| swing-up: median latency | ~100 ms | ~30 ms (cold solve ~0.8 s) |
| figure-8: RMS position error | ~200 mm | **~1.3 mm** |
| figure-8: control energy | ~0.030 | ~0.004 |
| figure-8: median latency | ~27 ms | ~15 ms |

## What it shows

1. **On smooth dynamics with a smooth cost, gradients win by a lot.** iLQR/RTI
   tracks the figure-8 to ~1.3 mm — two orders of magnitude tighter than CEM's
   ~200 mm — using ~7× less control energy, at *lower* per-step latency. CEM's
   ~200 mm here reproduces Experiment 20's sampling-MPC number (~234 mm) on the
   same course, so this is the method's real ceiling, not a tuning artefact: a
   few-hundred-sample Gaussian search cannot resolve a millimetre-scale
   correction on a 20-step horizon.

2. **iLQR moves its cost around.** The first call runs a full multi-iteration
   solve (~0.8 s for the swing-up) to *find* the trajectory; every later step is
   a single cheap iteration (~30 ms vs CEM's ~100 ms on the swing-up). CEM pays
   the same moderate cost every step and never gets cheaper. If you have a
   one-time budget at startup (or a decent initial guess), RTI is far cheaper in
   steady state; if every step must fit the same hard deadline with no
   exceptions, CEM's flat profile can be easier to certify.

3. **CEM is the more robust default when the problem is nasty.** It needs no
   derivatives, tolerates a discontinuous or non-convex running cost (the
   obstacle penalty in Exp 20/21), and cannot be trapped by a bad linearisation.
   iLQR here relies on finite-difference Jacobians and a line search; on a
   stiff or contact-rich model it needs the regularisation schedule and a
   sensible warm start or it stalls in a local minimum.

4. **Both are borderline real-time in pure Python at these horizons** (p95 ≈ the
   20 ms budget). The takeaway is not "iLQR is fast enough" — it is that for the
   same compute, iLQR returns a vastly more accurate plan plus a stabilising
   feedback gain `K₀`, which CEM does not produce at all.

**Rule of thumb.** Smooth model, smooth cost, need accuracy or a feedback gain →
iLQR / RTI. Non-smooth cost, black-box or contact dynamics, want robustness over
the last millimetre → CEM. The capstone's obstacle course is the second case;
clean trajectory tracking is emphatically the first.

## Run

```
python experiments/24_ilqr_vs_sampling_mpc/run.py
AIMCT_EXP_FULL=1 python experiments/24_ilqr_vs_sampling_mpc/run.py   # committed artifacts
```
