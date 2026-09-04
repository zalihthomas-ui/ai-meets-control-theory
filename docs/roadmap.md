# AI Meets Control Theory — Project Roadmap

> **Engineering Agreement:** Priority order for all contributions: **clarity → correctness → reproducibility → performance.**

---

## Phase 1 — Foundation, Curriculum, & Capstone (✅ COMPLETED)

The entire foundational curriculum, capstone robotics showcases, deep RL agents, adaptive controllers, and benchmark challenge suites are implemented from scratch, tested, and validated:

- [x] **Module 01: Mathematical Foundations** — Matrix exponential, Runge--Kutta integration (RK4), numerical optimization.
- [x] **Module 02: Dynamic System Modeling** — State-space representations, analytical/numerical Jacobian linearisation, discretization.
- [x] **Module 03: Classical Control** — Filtered derivative PID with conditional anti-windup clamping (Exp 03).
- [x] **Module 04: Modern Control & Estimation** — Controllability, observability, Ackermann pole placement, full-state Luenberger observers, Filter Algebraic Riccati Equation (FARE) duality, continuous/discrete Kalman filtering with Joseph covariance update, Extended Kalman Filter (EKF), and Unscented Kalman Filter (UKF) (Exp 04, 05, 06, 15, 16).
- [x] **Module 05: Optimal & Constrained Control** — Hamiltonian Schur CARE solver, continuous full-state LQR gain margins, constrained receding-horizon Model Predictive Control with active-set QP solver and reference preview (Exp 08, 14).
- [x] **Module 06: System ID & Machine Learning** — Closed-loop least-squares state-space ID, DMDc, matrix logarithm ZOH inversion, from-scratch MLP with Adam, residual/grey-box LearnedDynamics, and Sampling-based MPC (CEM) with CARE terminal cost (Exp 07, 09, 10, 20).
- [x] **Module 07: Reinforcement Learning** — Gymnasium `ControlEnv` adapter, state/action discretization, Tabular Q-Learning, Deep Q-Networks (DQN), REINFORCE, and Proximal Policy Optimization (PPO) (Exp 11, 18).
- [x] **Module 08: AI + Control (Hybrid Architectures)** — Supervisory safety shielding, action filtering, barrier predicates, auditable intervention logging (Exp 12).
- [x] **Module 09: Advanced State Estimation, Adaptive Control & Robotics** — Full 6-state Planar Quadrotor (Crazyflie 2.0), differential flatness feedforward trajectory generation, Bryson-rule scaling, EKF/UKF output feedback, Model Reference Adaptive Control (MRAC), and obstacle-aware Sampling NMPC (Exp 14, 15, 16, 17, 20).
- [x] **Module 10: The Intelligent Control Challenge & Grand Synthesis** — Standardized multi-track benchmarking engine, robustness sweeps, cross-paradigm leaderboard, and 5-way capstone grand bake-off (Exp 19, 21).
- [x] **Interactive Live Sandboxes** — 2D planar quadrotor sandbox (`python -m aimct live`) and 6-DOF WebGL Three.js 3D sandbox (`python -m aimct live3d --web`).
- [x] **Test Harness & Reports** — 323 passing unit tests, living technical report PDF, master results table (`docs/RESULTS.md`), and engineering decision guide (`docs/DECISION-GUIDE.md`).

---

## Phase 2 — Post-Capstone Depth, Real Systems, & Packaging (🚀 IN PROGRESS)

Phase 2 builds upon the completed curriculum by adding **breadth** (more real-world multi-body systems), **depth** (addressing the algorithmic gaps exposed during the capstone bake-off), and turning the repository into an installable, published PyPI package.

### Track A — More Real Dynamical Systems
Each system is an `aimct.systems` model with real datasheet parameters, analytical/numerical Jacobians, unit tests, and comparison experiments:
- [ ] **Differential-Drive Mobile Robot:** Unicycle kinematics with motor/wheel slip lag. Task: waypoint path tracking in cluttered map. Comparisons: Pure Pursuit vs. Stanley vs. Linearised Path LQR vs. Kinematic MPC.
- [ ] **2-Link Planar Robot Arm:** Euler--Lagrange multi-body dynamics with realistic link masses and inertias. Task: joint-space trajectory tracking with payload step. Comparisons: PD + Gravity Compensation vs. Computed Torque vs. LQR vs. MPC vs. Adaptive MRAC under unknown payload.
- [ ] **Bicycle-Model Ground Vehicle:** Dynamic lateral tire-force model (Pacejka / linear). Task: high-speed double lane-change maneuvers. Comparisons: Stanley vs. LQR vs. Kinematic MPC vs. RL policy.
- [ ] *(Stretch)* Furuta Pendulum, Ball-and-Beam, Coupled Two-Tank benchmarks.

### Track B — Method Depth & Advanced Synthesis
Addressing the key algorithmic trade-offs identified in Experiment 21:
- [ ] **Nonlinear MPC (iLQR / Real-Time Iteration):** Iterative LQR / sequential quadratic programming over rollout trajectory. Replaces loose stochastic CEM with deterministic, real-time optimal planning.
- [ ] **Soft Actor-Critic (SAC) & Continuous RL Search:** Off-policy maximum-entropy actor-critic for sample-efficient continuous control on complex multi-rotor tasks.
- [ ] **Direct Trajectory Optimization:** Collocation and multiple-shooting methods for offline optimal open-loop trajectory generation.
- [ ] **First-Class Behavior Cloning & DAgger:** Formalized `aimct.rl.imitation` module enabling structured expert demonstration harvesting and interactive policy fine-tuning.
- [ ] *(Stretch)* $\mathcal{H}_\infty$ / $\mu$-synthesis loop shaping and disturbance-observer control for unmatched wind rejection.

### Track C — Benchmark & Trajectory Infrastructure
- [ ] **Trajectory-Tracking Harness Suite:** Standardized reference generators and tracking error metrics (cross-track error, along-track error, heading error, path completion percentage).
- [ ] **`aimct.trajectories` Library:** Reusable trajectory generators: Lemniscate, minimum-jerk polynomials, Dubins paths, cubic splines.

### Track D — PyPI Packaging & Distribution
- [ ] **Distribution Configuration:** Package name `aimct` (fallback `ai-meets-control-theory`), full metadata in `pyproject.toml`, console entry point `aimct = "aimct.__main__:main"`.
- [ ] **Wheel & Source Distribution:** Clean packaging excluding heavy PDF artifacts and raw experiment figures.
- [ ] **Automated Release Pipeline:** GitHub Action `release.yml` with PyPI Trusted Publishing (OIDC).
- [ ] **Documentation & Release Runbook:** `CHANGELOG.md` (Keep-a-Changelog) and `docs/PACKAGING.md`.

---

## Phase 3 — Hardware & Physical Deployment (🔮 PLANNED)

- [ ] Hardware-in-the-loop (HIL) testing interfaces.
- [ ] Flight log ingestion and telemetry playback (Crazyflie CFclient / ROS2 rosbag).
- [ ] Hosted interactive documentation portal.
- [ ] Multi-agent collaborative control and partially-observable decentralized systems.
