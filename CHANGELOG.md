# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Real systems (Track A):** `DifferentialDriveRobot` (unicycle + first-order
  actuator lag) and `TwoLinkArm` (planar Euler-Lagrange manipulator with a
  settable wrist payload), both parameterised on real hardware classes
  (TurtleBot3-Burger-class mobile robot; Quanser 2-DOF-arm-class manipulator).
- **`aimct.controllers.ilqr`:** iterative-LQR trajectory optimiser + a
  real-time-iteration nonlinear-MPC controller.
- **`aimct.trajectories`:** `Lissajous`, `Rose`, `Spiral` reference paths
  (alongside the existing `Lemniscate`/`MinimumJerk`/`Spline`/`Dubins`).
- **`aimct.benchmarks.tracking`:** `track_trajectory` path-following harness
  (RMS/cross-track error, completion %, energy) and `TrackingResult.animate()`.
- **`aimct.viz`:** a unified visualization layer — `SystemArtist` (one draw
  contract per system), `animate()` (replay any simulated run), `Sandbox` +
  `Disturbance` (real-time interactive sandboxes with sliders/hot-keys), and
  `aimct.viz.pv_arm` (a shared 3-D PyVista renderer for the arm systems).
  Every `Sandbox` gets a help overlay, a "surprise me" randomiser, PNG
  snapshotting, and a session-best score for free.
- **Interactive sandboxes:** `live_arm` (unknown-payload identification),
  `live_arm_balance` (double-inverted-pendulum balance under gravity),
  `live_diffdrive` (path-follower recovery from a shove) — each with a 2-D
  (matplotlib) and, for the arm sandboxes, a 3-D (PyVista) view. Run via
  `python -m aimct live {arm,arm3d,diffdrive,armbalance,armbalance3d}`.
- **`aimct.dev`:** a design-time preview tool for a `DynamicalSystem` under
  development (pole map, controllability/observability, analytic-vs-numeric
  Jacobian check, animated replay) — `python -m aimct preview MODULE:Class`.
- **Experiments 22–26:** differential-drive path following, two-link-arm
  tracking + adaptive payload rejection, iLQR/RTI-NMPC vs. sampling MPC (CEM),
  moving-obstacle avoidance, and tracking-robustness on harder reference paths.

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
