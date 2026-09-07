# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-09-07

The **v1.0.0 Release Candidate**: comprehensive end-to-end framework unification with 41 empirical experiments, formal API stability contract (`docs/STABILITY.md`), multi-agent consensus, particle filtering, moving-horizon estimation, structured $\mu$-analysis, hardware-in-the-loop bridges, and publication-ready documentation portal.

### Added
- **Formal API Stability Contract (`docs/STABILITY.md`)**:
  - Full SemVer 2.0.0 compliance rules for all Tier 1 public surfaces.
  - Strict 2-minor-cycle deprecation policy with `DeprecationWarning` enforcement.
  - Public surface declaration across all 12 core subpackages.
- **Concepts Documentation Portal (`docs/concepts/`)**:
  - 7 comprehensive mathematical narrative chapters covering the 6 conceptual pillars: State-Space Dynamics, Estimation, Optimal & Constrained Control, Robustness, Data-Driven & Safe RL, and Hardware Bridge.
- **Advanced State Estimation & Robustness**:
  - `aimct.estimation.MovingHorizonEstimator` (MHE): Constrained MAP sliding-window state estimation with arrival cost Riccati propagation.
  - `aimct.estimation.ParticleFilter`: Bootstrap Sequential Monte Carlo filter with systematic adaptive resampling and log-sum-exp numerical stabilization.
  - `aimct.robust.mu` & `aimct.robust.dk_iterate`: Structured Singular Value ($\mu$-analysis) and D-K iteration for mixed real/complex uncertainty blocks.
  - `aimct.systems.MultiAgentSystem` & `aimct.controllers.FormationController`: Distributed consensus formation control under dynamic graph switching ($K_5 \to C_5 \to P_5 \to \text{Disconnected} \to S_5$) with collision barrier avoidance.
- **Experiments 37–41**:
  - **Exp 37**: Robust $\mu$-synthesis under high-order parameter perturbation.
  - **Exp 38**: Moving-Horizon Estimation (MHE) vs. EKF on coupled two-tank process with non-negative liquid constraints.
  - **Exp 39**: Multi-agent formation control under dynamic communication graph switching.
  - **Exp 40**: $\mu$-analysis showing blind failure modes of classical single-loop margins.
  - **Exp 41**: Bearings-only target tracking benchmark (Particle Filter vs. EKF vs. UKF).
- **Community Governance & Contributor Tooling**:
  - GitHub issue forms (`bug_report.yml`, `feature_request.yml`) and PR checklist (`pull_request_template.md`).
  - Refreshed `CONTRIBUTING.md` developer guide.

---

## [0.3.0] - 2026-09-06

The **Phase 3 Release: Robustness, Hardware & Reach**:
- **$H_\infty$ Mixed-Sensitivity Loop Shaping (`aimct.controllers.hinf`)**:
  - Continuous state-space plant augmentation (`mixsyn`), $S/KS/T$ weighting filter design, and 2-Riccati $H_\infty$ optimal controller synthesis.
- **Hardware-in-the-Loop & Embedded Deployment (`aimct.hil`, `aimct.deploy`)**:
  - Real-time simulation harness with 12-bit ADC/DAC quantization, transport latency ($\tau_d$), and clock jitter injection.
  - Zero-allocation standalone C99 code generator (`aimct.deploy.emit_c`) and MicroPython exporter.
- **Experiments 35–36**:
  - **Exp 35**: $H_\infty$ mixed-sensitivity vs. LQG under unmodelled structural resonance.
  - **Exp 36**: Hardware-in-the-loop two-link arm balancing under transport delay and quantization limits.

---

## [0.2.0] - 2026-09-05

The **Phase-2 Release: Modern Multi-Body Systems, Trajectory Optimization & Safe RL**:
- **Systems**: `DifferentialDriveRobot`, `TwoLinkArm`, `BicycleVehicle`, `FurutaPendulum`, `TwoTank`, `BallAndBeam`.
- **Control & Planning**: Iterative LQR (`iLQR`), Nonlinear Model Predictive Control (RTI-NMPC), Direct Collocation (`aimct.planning.DirectCollocation`), 2-DOF Disturbance Observer (`DOB`).
- **Reinforcement Learning**: Soft Actor-Critic (`SAC`), DAgger interactive imitation learning, and Behavioral Cloning (`BC`).
- **Experiments 22–34**: Comprehensive multi-body path following, obstacle avoidance, and sample-efficiency benchmarks.

---

## [0.1.0] - 2026-09-04

The **Phase 1 Release: Foundations & Benchmarks**:
- Core state-space models, algebraic Riccati solvers (CARE/DARE), LQR, LQG, Pole Placement, and PID with anti-windup.
- Linear MPC with active-set QP solver.
- Nonlinear observers (EKF, UKF) and energy shaping swing-up controllers.
- Control Barrier Function safety shields (`CBFShield`).
- Initial benchmark suite (Experiments 01–21).
