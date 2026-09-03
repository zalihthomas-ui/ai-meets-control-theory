# Experiment 20 — obstacle-aware nonlinear MPC on the quadrotor

**Capstone step C9.2.** A figure-8 with a keep-out disk sitting on the path.
`LQR + flatness feedforward` tracks the trajectory but has no notion of the
obstacle. A cross-entropy **sampling MPC** — planning through either the true
dynamics or a **learned grey-box model** — bends the drone around the disk while
holding the trajectory, and we check what the learned model costs.

Uses [`aimct.controllers.SamplingMPC`](../../src/aimct/controllers/sampling_mpc.py)
(now with a 3-arg `running_cost(X, U, k)` for trajectory tracking) and
[`aimct.ml.LearnedDynamics`](../../src/aimct/ml/dynamics.py) with `base_step`.

## Setup

- Plant: `PlanarQuadrotor` (Crazyflie 2.0). Lemniscate `x_r = 0.55 sin(ωt)`,
  `z_r = 1.0 + 0.30 sin(2ωt)`, period 6 s.
- **Keep-out disk**: centre `(0.30, 1.16)`, radius `0.16 m`, 0.04 m soft buffer —
  it straddles the upper-right lobe of the figure-8.
- **Learned model**: residual MLP `[8,48,48,6]` over an **RK4 hover-linearisation**
  base step, trained on 3000 steps of `LQR+ff` figure-8 flight (no obstacle).
  20-step open-loop prediction error: **0.001**.
- **MPC**: horizon 20 (0.4 s), 400 samples, 3 CEM iterations, CARE terminal cost.
  Running cost = trajectory tracking + control effort + a quadratic penalty for
  entering the keep-out.
- `dt = 0.02`, `T = 12 s` (two laps), `|u| ∈ [0, 0.3] N`.

Run: `python experiments/20_quadrotor_obstacle_nmpc/run.py`
Outputs (committed): `table.md`, `table.csv`, `figure.png`.

## Results

| controller | RMS pos err | min clearance | steps in keep-out | ctrl energy |
| --- | --- | --- | --- | --- |
| LQR + flatness feedforward   | 86 mm  | **−86 mm** | **26** | 0.48 |
| SamplingMPC (true model)     | 234 mm | +11 mm | **0** | 0.033 |
| SamplingMPC (learned model)  | 255 mm | +26 mm | **0** | 0.034 |

## Takeaways

1. **The obstacle-blind controller flies straight through the disk.** LQR +
   flatness feedforward has the lowest tracking error (86 mm) — but only because
   it is doing the wrong thing: it penetrates the keep-out by 86 mm and spends
   26 steps inside it (panel a, the orange line cutting the red circle).
2. **Both sampling-MPC planners route around it — zero violations.** They pay for
   it in tracking error (peaks of 350–490 mm in panel b, one per lap, exactly
   when the path detours past the obstacle) — but that error *is* the avoidance
   manoeuvre. Away from the disk they hold ~200 mm, the CEM planner's looseness
   (same as Experiments 10 and 14).
3. **The learned grey-box model costs the planner nothing.** True-model and
   learned-model planners give the same behaviour (255 vs 234 mm RMS, +26 vs
   +11 mm clearance, both zero violations). A residual MLP over an RK4 hover
   linearisation predicts the quad to 0.001 over 20 steps — accurate enough that
   planning through it is indistinguishable from planning through the equations.
4. **This is the first capability no single earlier controller had:** a
   *constraint* the planner reasons about online. It sets up the grand bake-off
   (C9.4) where this MPC meets an RL policy and the safety-shielded hybrid on the
   same figure-8 + wind + obstacle course.

## Follow-ups (capstone)

- C9.3: a PPO policy trained on a quad trajectory-tracking `ControlEnv` task.
- C9.4: the five-way bake-off (LQR+flatness · linear-MPC-preview ·
  learned-model sampling-MPC · RL · RL-behind-shield) scored on tracking, effort,
  constraint violations, robustness, and compute.
