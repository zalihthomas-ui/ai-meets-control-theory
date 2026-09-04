# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

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
