# Experiment 32 — direct collocation vs shooting vs sampling (offline)

**Question.** Experiments [24](../24_ilqr_vs_sampling_mpc/),
[25](../25_diffdrive_moving_obstacle/) and [26](../26_harder_reference_paths/)
raced **iLQR** (indirect / single shooting) against **CEM** (sampling) as
*online* receding-horizon controllers. Every one of those controllers rests on
solving a finite-horizon optimal-control problem. There is a third way to solve
it that the repo did not have: **direct transcription** — make the state *and*
the input at every knot decision variables of one big nonlinear program, and
enforce the discretised dynamics as equality constraints. This experiment puts
all three paradigms on the *same offline problem* and asks what each actually
delivers.

Companion: [`aimct.planning.DirectCollocation`](../../src/aimct/planning/collocation.py)
(new — Hermite-Simpson collocation, SLSQP),
[`aimct.controllers.iLQR`](../../src/aimct/controllers/ilqr.py),
[`aimct.controllers.SamplingMPC`](../../src/aimct/controllers/sampling_mpc.py).

## The shared optimal-control problem

Minimum-effort cart-pole swing-up:

```
min  ∫₀ᵀ u(t)² dt
s.t. ẋ = f(x, u)          (the CartPole model)
     x(0) = [0, 0, π, 0]  (hanging, at rest, cart at origin)
     x(T) = [0, 0, 0, 0]  (upright, at rest, cart at origin)
     |u| ≤ 20 N,   T = 2 s
```

| planner | how the terminal condition is handled | grid |
| :-- | :-- | :-- |
| **Direct collocation (HS)** | **hard equality constraint** `x(T) = goal` | 41 knots |
| **iLQR / single shooting** | terminal **penalty** `Qf = diag([400, 40, 800, 40])` — shooting has no hard-constraint mechanism | 100 steps @ dt 0.02 |
| **CEM / sampling** | same terminal penalty `Qf`, run open-loop over the whole horizon | 100 steps @ dt 0.02 |

`term_err_rolled` re-integrates the returned `u` through the **true** dynamics
with a fine RK4 — first-order hold for collocation, zero-order hold for iLQR /
CEM (the input model each planner assumes). That is the honest *"does the plan
actually fly"* check, separate from the planner's own knot values.

```bash
python experiments/32_direct_collocation_vs_ilqr/run.py
AIMCT_EXP_FULL=1 python experiments/32_direct_collocation_vs_ilqr/run.py   # committed numbers
```

## Results (`AIMCT_EXP_FULL=1`)

| planner | effort ∫u² | term err (planned) | term err (re-integrated) | max dyn. drift | peak \|u\| | box ok | solve |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Direct collocation (HS)** | 74.95 | **0** | **8.1e-4** | 8.1e-4 | 12.9 N | yes | **0.71 s** |
| iLQR / single shooting | 65.03 | 0.253 | 0.253 | 3.5e-5 | 12.7 N | yes | 1.41 s |
| CEM / sampling | 88.02 | 0.649 | 0.649 | 1.3e-5 | 13.1 N | yes | 8.55 s |

Collocation Hermite-Simpson defect norm at the solution: **2.1e-12**.

![offline swing-up: three planners](figure.png)

## Takeaways

1. **Direct collocation is the only one that meets the terminal condition —
   because for it the terminal condition is a *constraint*, not a wish.** Planned
   terminal error is 0 to solver tolerance; re-integrated it is 8e-4, and that
   residual is *pure inter-knot quadrature error* (see takeaway 4), not a
   modelling gap. iLQR and CEM leave the pole **0.25** and **0.65** short of
   upright: their terminal penalty `Qf` is balanced against the effort term, and
   the balance point is simply not at the goal. You would have to hand-crank
   `Qf` up and re-solve — and even then only approach it. We tried
   `Qf = diag([2000, 150, 4000, 150])`: iLQR's residual falls to 0.086 (effort
   rises to 71) and CEM's to 0.51 but with the effort ballooning to 155 and the
   input nearly saturating — a stiff penalty makes the derivative-free search
   thrash.

2. **Lower "effort" for iLQR is not a win — it is a different problem.** iLQR
   reports ∫u² = 65 against collocation's 75, but only because it stops 0.25
   short of the goal; effort is comparable *only at equal terminal error*.
   Collocation's 75 is the true minimum-effort value subject to actually
   arriving. This is the classic trap in reading trajectory-optimisation
   numbers: always check the constraint residual before the objective.

3. **Collocation was also the fastest to solve here** — one SLSQP solve of a
   ~250-variable NLP in 0.71 s, versus 1.4 s for iLQR's iteration to
   convergence and 8.5 s for CEM's 2000-sample population over 240 refinement
   sweeps. (This is offline single-solve wall-clock; it is *not* the Exp-24
   online story, where iLQR's one-RTI-iteration-per-step is the cheap option and
   CEM blows the real-time budget.)

4. **Collocation's weakness is between the knots.** The NLP equalities are
   satisfied to 2e-12, but Hermite-Simpson only makes the dynamics hold to
   *third order* across each interval — so a fine re-integration of the plan
   drifts. That drift is **8e-4 at 41 knots, and was 1.7e-2 at 21 knots**: it is
   mesh-dependent and converges away. iLQR and CEM have the opposite profile —
   they roll the *true* RK4 dynamics internally, so their plans are
   self-consistent to ~1e-5 — just self-consistent about reaching the wrong
   place. Practical reading: a collocation plan wants either a fine mesh or a
   tracking controller (LQR / MPC on the linearisation) to fly open-loop; do not
   trust the coarse-mesh knot values as a feedforward on their own.

5. **Which paradigm, when.**
   - **Hard terminal / path constraints, offline, smooth model** → direct
     collocation. It is the only one that treats "arrive exactly here" and
     "stay out of there" as constraints rather than tuning weights.
   - **Online receding horizon, smooth model** → iLQR / RTI (Exp 24): one cheap
     iteration per step, and it hands back a stabilising feedback gain.
   - **Non-smooth cost, contact, or black-box dynamics** → CEM (Exp 25): no
     gradients or linearisation to be defeated, at the price of the slowest
     solve and the loosest constraint satisfaction.
