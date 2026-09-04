# Decision Guide — which control method, and why

The point of this project was never to show that a neural network can control a
system. It was to build **engineering judgment**: given a problem, which method
does the evidence support? This document is the synthesis of all 21 experiments
into a practical guide. Every claim links to the experiment that earned it.

> **One sentence:** classical and optimal control win whenever you have a usable
> model; learning earns its keep only where you *don't* — and even then, a
> classical fallback is usually what makes it safe to deploy.

---

## Quick decision table

| Your situation | Use | Evidence |
| --- | --- | --- |
| You have equations of motion (or can get them) | **LQR / pole placement**, then **MPC** if there are constraints | [04](../experiments/04_lqr_vs_pole_placement_cartpole/), [08](../experiments/08_mpc_vs_lqr_constrained_cartpole/) |
| Hard state/actuator constraints matter | **Linear MPC** (condensed QP) | [08](../experiments/08_mpc_vs_lqr_constrained_cartpole/), [20](../experiments/20_quadrotor_obstacle_nmpc/) |
| Nonlinear dynamics with a hard real-time deadline | **iLQR / RTI-NMPC** ($1.3\,\text{mm}$ error at $14.6\,\text{ms}$) | [24](../experiments/24_ilqr_vs_sampling_mpc/) |
| Multi-body robot arm trajectory tracking | **Computed torque** / **Joint LQR**; **Slotine--Li MRAC** under unknown payload | [23](../experiments/23_twolink_arm_tracking/) |
| Wheeled mobile robot path following | **Path LQR** (curvature feedforward) / **Stanley** | [22](../experiments/22_diffdrive_path_following/) |
| A smooth reference *trajectory* to track | **LQR + differential-flatness feedforward**, or **preview MPC** | [14](../experiments/14_quadrotor_figure8_tracking/) |
| Constant disturbance / steady-state error unacceptable | add **integral action** (LQI) or **MRAC** | [03](../experiments/03_pid_stabilizes_unstable/), [17](../experiments/17_adaptive_vs_fixed_changing_plant/), live sandbox |
| The plant *changes* over time (wear, payload) | **MRAC** if the uncertainty is matched; **gain scheduling** if the parameter is measured | [17](../experiments/17_adaptive_vs_fixed_changing_plant/) |
| Only noisy / partial measurements | **Kalman filter** (LQG); **EKF/UKF** if nonlinear | [06](../experiments/06_lqg_vs_lqr_measurement_noise/), [15](../experiments/15_quadrotor_ekf_output_feedback/), [16](../experiments/16_ekf_vs_ukf/) |
| No model, but you can collect data | **least-squares / DMDc system ID** → design on the fitted model | [09](../experiments/09_control_on_identified_model/) |
| No model, nonlinear, you can simulate a *learned* model | **sampling (CEM) MPC** over a learned residual model | [10](../experiments/10_planning_learned_vs_true_model/), [20](../experiments/20_quadrotor_obstacle_nmpc/) |
| Genuinely no model and no simulator — only interaction | **RL** (tabular if low-dim, DQN/PPO otherwise) — expect a large sample bill | [11](../experiments/11_qlearning_vs_classical/), [18](../experiments/18_rl_zoo_vs_lqr/) |
| You must deploy an unverified learned policy | wrap it in a **safety shield** with a classical fallback | [12](../experiments/12_shielded_qlearning/), [21](../experiments/21_grand_capstone_bakeoff/) |

---

## Method by method

### PID
- **Wins:** SISO loops, when a steady-state-error-free response to a step or
  constant disturbance is the whole job. Anti-windup is non-negotiable under
  saturation ([03](../experiments/03_pid_stabilizes_unstable/): overshoot 53 % → 39 %).
- **Fails:** multi-state, single-actuator plants — it cannot coordinate them.
  On the cart-pole a single-loop angle PID balances the pole but drifts the cart
  0.82 m ([04](../experiments/04_lqr_vs_pole_placement_cartpole/)).

### LQR / pole placement
- **Wins:** the default for a linear (or linearised) model. It *is* the optimal
  controller for a quadratic cost — nothing to beat, only to re-derive
  ([18](../experiments/18_rl_zoo_vs_lqr/): greedy return −0.3, 4 numbers, 0 samples).
  Pole placement at the LQR eigenvalues reproduces the LQR gain to 4 sig figs
  ([04](../experiments/04_lqr_vs_pole_placement_cartpole/)) — LQR just removes the
  guessing.
- **Fails:** no constraint handling (violates a rail limit by 23 %,
  [08](../experiments/08_mpc_vs_lqr_constrained_cartpole/)); no steady-state
  disturbance rejection without integral augmentation; the linearisation is
  strictly local ([02](../experiments/02_linearization_validity/): diverges past
  ~23°; [05](../experiments/05_cartpole_basin_of_attraction/) maps the ~57°
  cart-pole basin).
- **Watch out:** an ill-conditioned input matrix needs **Bryson-rule cost
  scaling** or you get a nonsensical gain (a ~2000 rad/s pole on the quadrotor,
  [14](../experiments/14_quadrotor_figure8_tracking/)).

### Linear MPC
- **Wins:** hard constraints, by construction — respects `|x| ≤ 0.5 m` where LQR
  does not ([08](../experiments/08_mpc_vs_lqr_constrained_cartpole/)). With a
  DARE terminal cost a short horizon is still stabilising; with a whole-horizon
  *reference preview* it matches flatness-feedforward LQR on trajectory tracking
  ([14](../experiments/14_quadrotor_figure8_tracking/): 142 mm → 48 mm).
- **Fails / costs:** compute. Even the condensed QP runs ~12–40 ms/step on these
  small problems — 6–20× a real 500 Hz flight-controller budget
  ([21](../experiments/21_grand_capstone_bakeoff/)). A single tiled setpoint
  cuts corners on curved paths.

### Estimation (Luenberger / Kalman / EKF / UKF)
- **Kalman over Luenberger** when there is noise: a fast pole-placed observer
  amplifies sensor noise into the actuator (11× control energy,
  [06](../experiments/06_lqg_vs_lqr_measurement_noise/)); the Kalman filter is
  the noise-optimal trade. LQG recovers ~95 % of full-state performance from
  half the (noisy) sensors ([15](../experiments/15_quadrotor_ekf_output_feedback/)).
- **UKF over EKF** only when the nonlinearity over your uncertainty is real: the
  EKF gets trapped in a wrong basin where the UKF's sigma points escape
  ([16](../experiments/16_ekf_vs_ukf/)) — but near a good operating point they
  are identical and the UKF just costs more evaluations.
- **Never** finite-difference a noisy position for velocity: 150× the control
  energy ([15](../experiments/15_quadrotor_ekf_output_feedback/)).

### System identification & learned models
- A controller designed on a **least-squares-identified** model works if you have
  enough clean data; 1 s of noisy data destabilises, 24 s is fine
  ([09](../experiments/09_control_on_identified_model/)). Naive closed-loop LS is
  **biased** — the estimate does not converge to truth even with abundant data.
- A **learned residual over an approximate physics model** (grey-box) beats both
  the physics alone and a pure black-box net ([`ml` tests];
  [10](../experiments/10_planning_learned_vs_true_model/),
  [20](../experiments/20_quadrotor_obstacle_nmpc/)). Planning through a 4.8k-param
  learned model is indistinguishable from planning through the equations.

### Adaptive control (MRAC / gain scheduling)
- **MRAC wins** when the plant drifts and the uncertainty is *matched*: ~1 mrad
  tracking error through a 5× stiffness change, where every fixed and
  gain-scheduled LQR droops 0.15 → 0.55 ([17](../experiments/17_adaptive_vs_fixed_changing_plant/)).
- **Costs:** ~2.5× the control energy and a real convergence transient — worth
  nothing on a plant that does not change. **Does not apply** when the
  uncertainty is unmatched (a horizontal wind on the planar quadrotor enters
  through pitch, not the input matrix — the live sandbox uses integral-LQR there
  instead).

### Reinforcement learning
- **Tabular Q** learns a swing-up but chatters near the goal (coarse grid →
  limit cycle) and needs a 50k-entry table vs a 2–3-number classical law
  ([11](../experiments/11_qlearning_vs_classical/)).
- **From-scratch DQN and PPO** re-discover the LQR on cart-pole — after 90k–240k
  environment steps and a ~4.6k-param net ([18](../experiments/18_rl_zoo_vs_lqr/)).
- **Pure from-scratch RL does not bootstrap** on a plant as pitch-sensitive as
  the Crazyflie (`I_yy = 1.4e-5`) — every early rollout collapses before the
  critic learns ([21](../experiments/21_grand_capstone_bakeoff/) uses a
  behaviour-cloned policy instead).
- **Library defaults are not magic:** Stable-Baselines3 PPO under-performs on a
  non-standard reward/action scale without re-tuning
  ([18](../experiments/18_rl_zoo_vs_lqr/)).

### Hybrids (safety shield)
- A **classical shield locks an unreliable RL policy to the goal** with *less*
  control effort than the raw policy ([12](../experiments/12_shielded_qlearning/):
  1.4 rad limit cycle → 0.00 rad, −35 % energy).
- But combining only helps **when each component covers a genuine weakness of the
  others**. In the grand bake-off the base policy's one edge (0.09 ms latency)
  was diluted away because it needed the shield so often that the obstacle-aware
  MPC was driving most of the time — the hybrid *tied* the MPC alone
  ([21](../experiments/21_grand_capstone_bakeoff/)).

---

## Lessons that recur across many experiments

1. **Feedforward does the heavy lifting.** Model-based feedforward (differential
   flatness, hover thrust, reference preview) means feedback only has to correct
   error and disturbance, not reconstruct the whole manoeuvre — [03], [14], [17].
2. **Integral action is the only cure for a constant disturbance.** No amount of
   proportional-state-feedback tuning nulls a steady wind — [03], [17], live sandbox.
3. **Cost/weight scaling is not optional** on any real, ill-conditioned plant —
   [14] (Bryson rule), [17], [18] (SB3).
4. **Model error is survivable; the amount matters.** LQR's `[½, ∞)` gain margin
   absorbs a ~50 % gain error ([09]); a tighter design would not.
5. **The real-time deadline is a first-class axis.** MPC's optimality is worth
   nothing if it misses the control step — [21] scores it explicitly.
6. **Report the failures.** Half of what is useful here is knowing where a method
   breaks: linearisation past 23° [02], EKF basin trapping [16], closed-loop
   SysID bias [09], RL sample cost [18], from-scratch RL failing outright on the
   quad [21].

---

## What this project did *not* settle

- Deep RL is shallow here (DQN, PPO from scratch; no SAC, no distributional
  methods, no real hyper-parameter search).
- One real system (the planar quadrotor). No ground vehicle, manipulator, or
  multi-agent case.
- No hardware-in-the-loop, no real sensor logs, no formal robustness certificate
  beyond LQR/LQG margins.
- Nonlinear MPC is only the sampling (CEM) planner — no SQP / real-time-iteration
  NMPC.

See [`docs/roadmap.md`](roadmap.md) and the report's roadmap section for where
these would go next.
