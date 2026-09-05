# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-05

The Phase-2 release: real multi-body systems, the remaining planning and RL
paradigms, a unified visualization layer, and a design-time authoring tool.
34 experiments, 460+ passing unit tests.

### Added

**Systems (Track A).** `DifferentialDriveRobot` (unicycle + first-order
actuator lag), `TwoLinkArm` (planar Euler-Lagrange, settable wrist payload),
`BicycleVehicle` (dynamic single-track, linear + Pacejka tyre models),
`FurutaPendulum` (rotary inverted pendulum), `TwoTank` (coupled-tank process
control, Torricelli outflow), `BallAndBeam` (relative-degree-4 underactuated).
Every one is parameterised on a real hardware class (TurtleBot3, Quanser
2-DOF arm / QUBE-Servo 2 / Coupled Tanks / Ball & Beam) with a datasheet-grade
reference doc under `docs/references/`.

**Control & planning (Track B).**
- `aimct.controllers.ilqr` — iterative-LQR trajectory optimiser + a
  real-time-iteration nonlinear-MPC controller.
- `aimct.planning.DirectCollocation` — Hermite-Simpson direct transcription of
  a finite-horizon OCP (SLSQP / trust-constr), with input/state boxes and an
  optional path-inequality hook; the offline mirror of the online solvers.
- `aimct.controllers.DisturbanceObserver` + `QFilter` — a 2-DOF DOB wrapping
  any base controller, with matched cancellation and virtual-tilt reallocation
  for the unmatched (horizontal-wind) channel on the quadrotor.

**Reinforcement learning & imitation.**
- `aimct.rl.sac` — from-scratch Soft Actor-Critic (squashed-Gaussian actor,
  twin critics + Polyak targets, auto-tuned temperature).
- `aimct.rl.imitation` — `BehaviorCloning` + a `dagger` interactive
  data-aggregation loop.
- `aimct.ml.MLP.grad_input` — backprop-to-input (used by the SAC reparam actor).

**Benchmarks & trajectories.**
- `aimct.benchmarks.tracking` — `track_trajectory` path-following harness (RMS
  / cross-track error, completion %, energy) and `TrackingResult.animate()`.
- `aimct.trajectories` — `Lissajous`, `Rose`, `Spiral` (alongside the existing
  `Lemniscate` / `MinimumJerk` / `Spline` / `Dubins`).

**Visualization — `aimct.viz`.** A `SystemArtist` "draw one state" contract,
`animate()` (replay any simulated run as video/GIF with a telemetry HUD),
`Sandbox` + `Disturbance` (real-time interactive sandboxes with sliders /
hot-keys / a switchable controller), and `aimct.viz.pv_arm` (a shared 3-D
PyVista renderer). Every `Sandbox` gets a help overlay (`h`), a "surprise me"
randomiser (`g`), PNG snapshotting (`c`), and a session-best score for free.
Shipped sandboxes: `live_arm`, `live_arm_balance`, `live_diffdrive` (+ 3-D
views for the arm ones) — `python -m aimct live {arm,arm3d,diffdrive,
armbalance,armbalance3d}`.

**Design-time authoring — `aimct.dev`.** `python -m aimct preview MODULE:Class`
renders a live design dashboard for a system under development — pole map,
controllability / observability, analytic-vs-numeric Jacobian residual, and
free / step / impulse / sinusoid response traces — re-rendering on file save
with `--watch`.

**Experiments 22–34.** Differential-drive path following (22); two-link-arm
joint tracking + adaptive payload rejection (23); iLQR/RTI-NMPC vs. sampling
MPC (24); moving-obstacle avoidance (25); tracking robustness on harder
reference paths (26); dynamic-bicycle double lane change with a Pacejka tyre
swap (27); Furuta pendulum (28); DAgger recovery vs. plain BC (29); two-tank
level control (30); SAC vs. PPO sample efficiency (31); direct collocation vs.
iLQR vs. CEM offline planning (32); ball-and-beam (33); disturbance-observer
wind rejection (34).

**Docs & packaging.** `docs/GETTING-STARTED.md` (ties the decision guide /
report / library / reproduce lanes together), `docs/DECISION-GUIDE.md`
rewritten as a structured decision system (flowchart + master matrix + the
recurring engineering laws), an `examples/` gallery (7 runnable scripts), a
`py.typed` marker (the package ships inline type info), and CI now runs the
full suite with coverage (`--cov-fail-under=80`).

### Fixed
- `live_diffdrive`'s path follower could show its look-ahead point teleport
  across the figure-8's self-intersection (a global nearest-point search
  flipping branches); replaced with progress-hysteresis search.
- CI's install step was missing the `ml` extra, so `pytest` failed to even
  collect (`aimct.rl` imports `gymnasium` unconditionally) on every push.
- `live_drone_3d/pv3d.py`'s interactive path called a PyVista method
  (`add_callback`) absent from the installed PyVista version — switched to
  `add_timer_event`.
- Packaging: `aimct.__version__` and the built distribution's version had
  drifted (`0.0.1` vs. `0.1.0`); `pyproject.toml` now takes its version from
  `aimct.__version__` (single source of truth).

## [0.1.0] - 2026-09-04

### Added
- **Core State-Space & Classical Control (`aimct.controllers`)**:
  - From-scratch Continuous Algebraic Riccati Equation (CARE) and Discrete Algebraic Riccati Equation (DARE) solvers.
  - Linear Quadratic Regulator (`LQR`) with Bryson scaling and integral augmentation (`LQI`).
  - Proportional-Integral-Derivative (`PID`) controller with anti-windup clamping and low-pass derivative filtering.
  - State Feedback with setpoint tracking (`StateFeedback`).
  - Full-state and reduced-order Luenberger Observers (`LuenbergerObserver`).
  - Continuous and Discrete Kalman Filters (`KalmanFilter`), Extended Kalman Filter (`EKF`), and Unscented Kalman Filter (`UKF`).
- **Constrained & Optimal Control (`aimct.controllers`)**:
  - Active-set dense Quadratic Program solver (`solve_qp`) with warm-starting.
  - Receding-horizon Linear Model Predictive Control (`LinearMPC`) with hard input and soft state constraints.
  - Model Predictive Path Integral / Sampling MPC (`SamplingMPC`).
- **Nonlinear & Underactuated Hybrid Control (`aimct.controllers`)**:
  - Mark Spong Partial Feedback Linearization (`EnergyShapingSwingUp`).
  - Hysteresis mode-switching swing-up to balance handoff (`HybridSwingUpLQR`).
  - Model Reference Adaptive Control (`MRAC`) with Lyapuov weight adaptation.
- **Safe Control & Barrier Functions (`aimct.safety`, `aimct.shield`)**:
  - Real-time Control Barrier Function Quadratic Program safety filters (`CBFShield`).
  - Forward-invariance certificates around untrusted RL policies and manual inputs.
- **Data-Driven & Physics-Informed Dynamics (`aimct.sysid`, `aimct.ml`)**:
  - Sparse Identification of Nonlinear Dynamics (`SINDy`) with STLSQ regression.
  - Continuous-depth Neural Ordinary Differential Equations (`NeuralODE`) with adjoint backpropagation.
  - Proximal Policy Optimization (`PPO`) and Deep Deterministic Policy Gradients (`DDPG`).
- **Dynamical Systems Benchmark Library (`aimct.systems`)**:
  - Linear Mechanical Oscillator (`MassSpringDamper`, `InvertedMassSpringDamper`).
  - Armature-Controlled DC Motor (`DCMotor`, `DCMotor2`).
  - Inverted Pendulum on a Cart (`CartPole`).
  - Simple Nonlinear Pendulum (`Pendulum`).
  - Planar Quadrotor UAV (`PlanarQuadrotor`).
  - Differential-Drive Mobile Robot (`DifferentialDriveRobot`).
  - Two-Link Planar Manipulator (`TwoLinkArm`).
- **Benchmarking & Scoring Engine (`aimct.benchmarks`)**:
  - Automated multi-controller comparison harness (`compare`, `ComparisonResult`).
  - Intelligent Control Challenge (ICC) 4-track scoring engine (`score_run`, Track 3 & 4 wrappers).
  - Grand Capstone Five-Way Bake-Off rubric (`score_capstone`, `capstone_leaderboard_table`).
- **Interactive Tools & Notebooks**:
  - Guided interactive tour (`notebooks/01_tour.ipynb`).
  - CLI entry point (`python -m aimct`).
  - 3D real-time simulation visualization (`python -m aimct live3d`).
