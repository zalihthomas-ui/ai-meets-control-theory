# Experiments Catalog (01–36)

Every experiment in imct is self-contained with its own configuration (config.yaml), runner (
un.py), benchmark table (	able.md), and publication-ready figure (igure.png / igure.svg).

See the **[Master Results Table](../RESULTS.md)** for quantitative metrics across all experiments.

---

## 1. Classical Foundations & Limits
- **Exp 01 — Integrator Accuracy:** 4th-order Runge-Kutta (RK4) vs. Forward Euler on Mass-Spring-Damper.
- **Exp 02 — Linearization Validity:** Linear state-space divergence envelope ($|\\theta| > 23^\\circ$) on Inverted Pendulum.
- **Exp 03 — PID Clamping & Anti-Windup:** Anti-windup clamping preventing unstable integrator growth under torque limits.

## 2. Modern State-Space, Observers & Kalman Filtering
- **Exp 04 — LQR vs. Pole Placement:** Algebraic CARE optimal gain design vs. Ackermann pole guessing on Cart-Pole.
- **Exp 05 — Cart-Pole Basin of Attraction:** Quantified ^\\circ$ recoverable envelope under saturation.
- **Exp 06 — LQG vs. High-Gain Observers:** Optimal Kalman noise-bandwidth trade-off vs. high-frequency encoder thrashing.
- **Exp 15 — Quadrotor EKF Output Feedback:** Full velocity reconstruction on Crazyflie 2.0 under sensor noise.
- **Exp 16 — EKF vs. Unscented Kalman Filter (UKF):** Sigma-point nonlinear covariance propagation escaping false \\pi$ basins.

## 3. Underactuated Agility & Energy Shaping
- **Exp 07 — Spong Energy Shaping Swing-Up:** Lyapunov homoclinic orbit pumping with 1-switch LQR catch.
- **Exp 28 — Furuta Rotary Inverted Pendulum:** Quanser QUBE-Servo 2 benchmark, LQR catch, constrained MPC, and Åström-Furuta swing-up.
- **Exp 33 — Ball and Beam Rolling Balance:** Multivariable LQR/MPC coordinating relative-degree-4 tilt and roll dynamics.

## 4. Constrained & Nonlinear Model Predictive Control (MPC)
- **Exp 08 — Constrained Linear MPC:** Active-set condensed QP enforcing track boundaries ($|x| \le 0.5\\,\\text{m}$).
- **Exp 24 — iLQR / RTI-NMPC vs. Sampling MPC (CEM):** Real-time iteration quadratic convergence (.34\\,\\text{mm}$ in .6\\,\\text{ms}$) on Quadrotor.
- **Exp 26 — iLQR vs. Sampling on Complex Geometries:** Benchmark on Lissajous 3:2 and Archimedean Spiral paths.
- **Exp 30 — Coupled Two-Tank Process Control:** Multivariable MPC coordinating nonlinear Torricelli outflow with zero level violations.

## 5. Obstacle Avoidance & Non-Convex Trajectories
- **Exp 20 — Quadrotor Obstacle Avoidance NMPC:** Derivative-free CEM navigating geometric keep-out barriers.
- **Exp 25 — Differential-Drive Moving Obstacle Avoidance:** Receding-horizon avoidance of dynamic obstacle disks.

## 6. Trajectory Optimization & Transcription
- **Exp 32 — Direct Collocation vs. Shooting vs. Sampling:** Hermite--Simpson direct NLP transcription enforcing exact terminal equality constraints in .71\\,\\text{s}$.

## 7. Robust Control, \\infty$ & Disturbance Rejection
- **Exp 17 — Model Reference Adaptive Control (MRAC):** Dynamic compensation for \\%$ plant parameter drift.
- **Exp 23 — Two-Link Arm Computed Torque & Slotine--Li Adaptation:** On-line wrist payload mass identification restoring millimeter precision.
- **Exp 34 — Disturbance Observer (DOB) Wind Rejection:** 2-DOF Q-filter acceleration feedforward providing \\times$ faster settling under aerodynamic wind.
- **Exp 35 — \\infty$ Mixed-Sensitivity vs. LQG on Resonant Plant:** Robust loop shaping with /KS/T$ weighting on flexible joint resonance (.8\\,\\text{Hz}$).

## 8. Real Multi-Body, High-Speed Vehicles & Hardware Bridges
- **Exp 22 — Differential-Drive Mobile Robot Path Following:** Pure Pursuit vs. Stanley vs. Path LQR with curvature feedforward on TurtleBot3.
- **Exp 27 — Dynamic Bicycle Double Lane Change:** Linear vs. Pacejka tire dynamics under friction saturation ($\\mu=0.6$).
- **Exp 36 — Hardware-in-the-Loop Arm Balancing:** Real-time loop execution, latency compensation, and 12-bit quantization watchdog validation.

## 9. Reinforcement Learning, Imitation & Safety Shields
- **Exp 09 — Control on Identified Models (SysID):** Least-squares / DMDc identification under closed-loop data.
- **Exp 10 — Planning on Learned Neural Models:** Residual MLP grey-box dynamics planning.
- **Exp 11 — Tabular Q-Learning vs. Energy Shaping:** Sample inefficiency and chattering in model-free RL.
- **Exp 12 — Shielded Reinforcement Learning:** Formal safety shield guaranteeing zero keep-out violations.
- **Exp 18 — RL Zoo vs. LQR Baseline:** Continuous PPO sample cost (\\,\\text{k}$ steps) vs. analytical LQR ($ steps).
- **Exp 19 — Intelligent Control Challenge (ICC):** Blind black-box multi-plant cross-paradigm leaderboard.
- **Exp 21 — Grand Capstone Bake-Off:** Five-way grand course bake-off on 6-state Crazyflie quadrotor.
- **Exp 29 — DAgger vs. Behavior Cloning Under High Slip:** Interactive expert relabeling repairing distribution shift.
- **Exp 31 — Soft Actor-Critic (SAC) vs. PPO Sample Efficiency:** Off-policy experience replay delivering \\text{--}20\\times$ sample savings.
