# Experiments Catalog (01–41)

Every experiment in `aimct` is self-contained with its own configuration (`config.yaml`), runner (`run.py`), benchmark table (`table.md`), and publication-ready figure (`figure.png` / `figure.svg`).

See the **[Master Results Table](../RESULTS.md)** for quantitative metrics across all experiments.

---

## 1. Classical Foundations & Limits
- **[Exp 01 — Integrator Accuracy](01_rk4_vs_euler.md):** 4th-order Runge-Kutta (RK4) vs. Forward Euler on Mass-Spring-Damper.
- **[Exp 02 — Linearization Validity](02_jacobian_linearization.md):** Linear state-space divergence envelope ($|\theta| > 23^\circ$) on Inverted Pendulum.
- **[Exp 03 — PID Clamping & Anti-Windup](03_pid_stabilizes_unstable.md):** Anti-windup clamping preventing unstable integrator growth under torque limits.

## 2. Modern State-Space, Observers & Kalman Filtering
- **[Exp 04 — LQR vs. Pole Placement](04_lqr_vs_pole_placement_cartpole.md):** Algebraic CARE optimal gain design vs. Ackermann pole guessing on Cart-Pole.
- **[Exp 05 — Cart-Pole Basin of Attraction](05_cartpole_basin_of_attraction.md):** Quantified $26.4^\circ$ recoverable envelope under saturation.
- **[Exp 06 — LQG vs. High-Gain Observers](06_lqg_vs_lqr_measurement_noise.md):** Optimal Kalman noise-bandwidth trade-off vs. high-frequency encoder thrashing.
- **[Exp 15 — Quadrotor EKF Output Feedback](15_quadrotor_ekf_output_feedback.md):** Full velocity reconstruction on Crazyflie 2.0 under sensor noise.
- **[Exp 16 — EKF vs. Unscented Kalman Filter (UKF)](16_ekf_vs_ukf.md):** Sigma-point nonlinear covariance propagation escaping false $\pm \pi$ basins.
- **[Exp 38 — Moving-Horizon Estimation (MHE)](38_mhe_vs_ekf.md):** Constrained MAP state estimation strictly enforcing non-negative physical bounds.
- **[Exp 41 — Bearings-Only Target Tracking (PF)](41_particle_filter_bearings_only.md):** Sequential Monte Carlo particle filter resolving unobservable range cone.

## 3. Underactuated Agility & Energy Shaping
- **[Exp 07 — Spong Energy Shaping Swing-Up](07_cartpole_swingup_hybrid.md):** Lyapunov homoclinic orbit pumping with 1-switch LQR catch.
- **[Exp 28 — Furuta Rotary Inverted Pendulum](28_furuta_pendulum_control.md):** Quanser QUBE-Servo 2 benchmark, LQR catch, constrained MPC, and Åström-Furuta swing-up.
- **[Exp 33 — Ball and Beam Rolling Balance](33_ball_and_beam_control.md):** Multivariable LQR/MPC coordinating relative-degree-4 tilt and roll dynamics.

## 4. Constrained & Nonlinear Model Predictive Control (MPC)
- **[Exp 08 — Constrained Linear MPC](08_mpc_vs_lqr_constrained_cartpole.md):** Active-set condensed QP enforcing track boundaries ($|x| \le 0.5\,\text{m}$).
- **[Exp 14 — Quadrotor 3D Lemniscate](14_quadrotor_figure8_tracking.md):** Nonlinear geometric tracking on SE(3) along high-speed 3D figure-8.
- **[Exp 24 — iLQR / RTI-NMPC vs. Sampling MPC (CEM)](24_ilqr_vs_sampling_mpc.md):** Real-time iteration quadratic convergence ($0.34\,\text{mm}$ in $0.6\,\text{ms}$) on Quadrotor.
- **[Exp 26 — iLQR vs. Sampling on Complex Geometries](26_harder_reference_paths.md):** Benchmark on Lissajous 3:2 and Archimedean Spiral paths.
- **[Exp 30 — Coupled Two-Tank Process Control](30_two_tank_level_control.md):** Multivariable MPC coordinating nonlinear Torricelli outflow with zero level violations.

## 5. Obstacle Avoidance & Non-Convex Trajectories
- **[Exp 20 — Quadrotor Obstacle Avoidance NMPC](20_quadrotor_obstacle_nmpc.md):** Derivative-free CEM navigating geometric keep-out barriers.
- **[Exp 25 — Differential-Drive Moving Obstacle Avoidance](25_diffdrive_moving_obstacle.md):** Receding-horizon avoidance of dynamic obstacle disks.

## 6. Trajectory Optimization & Transcription
- **[Exp 32 — Direct Collocation vs. Shooting vs. Sampling](32_direct_collocation_vs_ilqr.md):** Hermite–Simpson direct NLP transcription enforcing exact terminal equality constraints in $0.71\,\text{s}$.

## 7. Robust Control, $H_\infty$ & Disturbance Rejection
- **[Exp 17 — Model Reference Adaptive Control (MRAC)](17_adaptive_vs_fixed_changing_plant.md):** Dynamic compensation for $300\%$ plant parameter drift.
- **[Exp 23 — Two-Link Arm Computed Torque & Slotine–Li Adaptation](23_twolink_arm_tracking.md):** On-line wrist payload mass identification restoring millimeter precision.
- **[Exp 34 — Disturbance Observer (DOB) Wind Rejection](34_dob_wind_rejection.md):** 2-DOF Q-filter acceleration feedforward providing $6.4\times$ faster settling under aerodynamic wind.
- **[Exp 35 — $H_\infty$ Mixed-Sensitivity vs. LQG on Resonant Plant](35_hinf_vs_lqg.md):** Robust loop shaping with $S/KS/T$ weighting on flexible joint resonance ($1.8\,\text{Hz}$).
- **[Exp 40 — $\mu$ Analysis: Structured Robust Stability](40_mu_analysis_rs_rp.md):** Structured singular value quantifying simultaneous gain and resonant mode uncertainty.

## 8. Real Multi-Body, High-Speed Vehicles & Hardware Bridges
- **[Exp 22 — Differential-Drive Mobile Robot Path Following](22_diffdrive_path_following.md):** Pure Pursuit vs. Stanley vs. Path LQR with curvature feedforward on TurtleBot3.
- **[Exp 27 — Dynamic Bicycle Double Lane Change](27_bicycle_double_lane_change.md):** Linear vs. Pacejka tire dynamics under friction saturation ($\mu=0.6$).
- **[Exp 36 — Hardware-in-the-Loop Arm Balancing](36_hil_arm_balance.md):** Real-time loop execution, latency compensation, and 12-bit quantization watchdog validation.

## 9. Reinforcement Learning, Imitation & Safety Shields
- **[Exp 09 — Control on Identified Models (SysID)](09_control_on_identified_model.md):** Least-squares / DMDc identification under closed-loop data.
- **[Exp 10 — Planning on Learned Neural Models](10_planning_learned_vs_true_model.md):** Residual MLP grey-box dynamics planning.
- **[Exp 11 — Tabular Q-Learning vs. Energy Shaping](11_qlearning_vs_classical.md):** Sample inefficiency and chattering in model-free RL.
- **[Exp 12 — Shielded Reinforcement Learning](12_shielded_qlearning.md):** Formal safety shield guaranteeing zero keep-out violations.
- **[Exp 18 — RL Zoo vs. LQR Baseline](18_rl_zoo_vs_lqr.md):** Continuous PPO sample cost ($500\,\text{k}$ steps) vs. analytical LQR ($0$ steps).
- **[Exp 19 — Intelligent Control Challenge (ICC)](19_icc_leaderboard.md):** Blind black-box multi-plant cross-paradigm leaderboard.
- **[Exp 21 — Grand Capstone Bake-Off](21_grand_capstone_bakeoff.md):** Five-way grand course bake-off on 6-state Crazyflie quadrotor.
- **[Exp 29 — DAgger vs. Behavior Cloning Under High Slip](29_dagger_vs_bc_lane_change.md):** Interactive expert relabeling repairing distribution shift.
- **[Exp 31 — Soft Actor-Critic (SAC) vs. PPO Sample Efficiency](31_sac_vs_ppo_sample_efficiency.md):** Off-policy experience replay delivering $10\text{--}20\times$ sample savings.
