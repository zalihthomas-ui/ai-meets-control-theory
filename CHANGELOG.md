# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Nothing yet.

---

## [1.0.0] - 2026-09-07

**The API freeze.** From this release the public API — enumerated in
[`docs/STABILITY.md`](docs/STABILITY.md) across all 16 subpackages — is stable
under [Semantic Versioning](https://semver.org): a breaking change to it
requires a major version, with a `DeprecationWarning` for at least one minor
cycle first. 1.0.0 adds no new feature areas; it is the stability commitment,
an API-consistency pass, and finalised documentation.

### Added
- **`docs/STABILITY.md`** — the versioning / public-API / deprecation policy,
  with the certified public surface for every subpackage and the 18
  field-stable return dataclasses.
- **Top-level `aimct.__all__`** + lazy subpackage loading: `import aimct;
  aimct.controllers.LQR` works, and a bare `import aimct` still does not pull
  `matplotlib` (viz) or `gymnasium` (rl).
- `ExtendedKalmanFilter.from_system` / `UnscentedKalmanFilter.from_system` —
  all four nonlinear estimators (EKF, UKF, MHE, PF) now share one
  construction story.
- `aimct.controllers.solve_qp` + `QPResult` are now public — the from-scratch
  active-set QP is a reusable primitive alongside `solve_care` / `dare` /
  `solve_lyapunov`.
- `aimct.robust` returns `MuBounds` (from `mu`) and `RobustMarginResult`
  (from `robust_stability_margin` / `robust_performance_margin` /
  `dk_iterate`) — typed results, matching every other subpackage.
- API-reference pages for `aimct.robust` and `aimct.simulate`; `aimct.hybrid`
  added to the top-level package.

### Changed
- **Breaking (`aimct.robust`):** `mu` and the margin functions return
  dataclasses instead of plain dicts. Code that indexed the dict keys must
  switch to attribute access (`r["peak_upper"]` → `r.peak_upper`). The module
  is one release old.
- CI coverage gate 80% → 85% (the full suite sits at ~90%).
- CI / Docs workflows cancel superseded runs (concurrency groups); the perf
  baseline only auto-refreshes on a >20% move.
- `TubeMPC.from_system` dropped a dead `dt=` parameter.

### Fixed
- Two nondeterministic test failures that flaked CI on Python 3.12: an
  unseeded RNG in `test_mhe.py`, and a too-tight forward-Euler drift bound in
  the disturbance-observer test.

---

## [0.3.0] - 2026-09-07

**Phase 3 — Robust control, the hardware bridge, estimation depth, and reach.**
34 → 41 experiments; 560+ unit tests; a hosted docs portal.

### Added

**Robust & certified control.**
- `aimct.controllers.hinf` — from-scratch H∞ mixed-sensitivity (S/KS/T)
  synthesis: `StateSpace` LTI toolkit with `hinf_norm`, `weight_S`/`weight_KS`/
  `weight_T` shaping filters, `augment_plant`, DGKF γ-bisection `hinf_syn`,
  `mixsyn`, `HinfController`. Cross-checked against `control.hinfsyn`.
- `aimct.robust` — structured-uncertainty **analysis**: `BlockStructure`,
  `mu` (D-scaling upper bound + power-iteration lower bound),
  `robust_stability_margin`, `robust_performance_margin`, `dk_iterate`.
- `aimct.controllers.TubeMPC` — tube / robust MPC: a box outer-approximation
  of the minimal robust positively-invariant set, nominal MPC on the
  tightened constraints, `u = u_nom + K(x - x_nom)`. `MRPISet` dataclass.

**Estimation.**
- `aimct.estimation.MovingHorizonEstimator` (`MHE`) — constrained sliding-window
  MAP estimation with an EKF arrival cost; state / disturbance / general
  inequality constraints.
- `aimct.estimation.ParticleFilter` (`PF`) — bootstrap SMC with systematic
  adaptive resampling and log-space weights.

**Multi-agent.**
- `aimct.systems.MultiAgentSystem` — N stacked agents with a communication
  graph, Laplacian, and algebraic connectivity (time-varying graph supported).
- `aimct.controllers.FormationController` — consensus + rigid-formation +
  leader-follower control, with collision-barrier avoidance.

**Planning.**
- `aimct.planning.DirectCollocation` gained decision-variable scaling
  (`x_scale`/`u_scale`, auto by default) and a sparse Hermite–Simpson
  constraint Jacobian — an *active nonconvex path constraint on a
  badly-scaled ≥4-state plant* (a keep-out disk on `PlanarQuadrotor`) now
  solves in ~1.5 s.

**Identification & deployment (the hardware bridge).**
- `aimct.sysid.identify_manipulator` — fit a two-link arm's base inertial
  parameters from a `(q, q̇, q̈, τ)` log via the linear-in-parameters
  regressor, with excitation / validation diagnostics and `to_twolink_arm()`.
  Plus `finite_difference_derivatives` (central / Savitzky–Golay).
- `aimct.deploy` — `export_controller` → `controller.json`,
  `PortableController` (runs unchanged in `simulate` and `RealTimeLoop`),
  `emit_c` / `emit_micropython` reference executors.
- `aimct.hil` — `RealTimeLoop` (deadline accounting), In-process / UDP /
  Serial transports, and `PlantEmulator` (encoder quantisation, torque
  saturation + slew, transport delay, jitter, sensor noise).

**Simulation & benchmarking.**
- `aimct.simulate.simulate_batch` + `BatchResult` — Monte-Carlo rollout over
  an array of initial states, with per-trial disturbances / parameter
  overrides and `.map_metric`.
- `benchmarks/perf/` + a `Perf` CI workflow — times the hot paths against a
  committed, drift-gated baseline.

**Visualization.**
- `aimct.viz.animate` gained an `aux_fn(t)` hook for per-frame context
  (an active disturbance, a mode label).

**Experiments 35–41.** H∞ vs LQG under an unmodelled resonance (35); HIL
two-link-arm balance under transport delay + quantisation (36); tube MPC vs
nominal MPC under a persistent bounded disturbance (37); MHE vs EKF/UKF with a
hard physical state bound (38); formation control on a switching communication
graph (39); μ-analysis catching a coupled instability that single-loop
gain/phase margins miss (40); bearings-only tracking, particle filter vs
EKF/UKF (41).

**Docs & packaging.**
- A hosted **mkdocs-material documentation portal** (GitHub Pages): API
  reference via `mkdocstrings`, a 7-chapter Concepts narrative, a page per
  experiment, the living technical report (4 parts + executive summary).
- `docs/STABILITY.md` — the semver / public-API / deprecation policy.
- A JOSS `paper.md` + `paper.bib` draft; issue / PR templates; a refreshed
  `CONTRIBUTING.md`.
- Top-level `aimct.__all__` + lazy subpackage loading — `import aimct;
  aimct.controllers.LQR` works, and a bare `import aimct` still doesn't pull
  `matplotlib` (viz) or `gymnasium` (rl).
- `examples/07` now shows the wind gust in its animation;
  `examples/08_multisystem_relay_handoff.py` — a crane → mobile-robot →
  slung-load-quad payload relay (three systems, three controllers, gated
  hand-offs).

### Fixed
- A nondeterministic `test_mhe.py` failure (unseeded `np.random` in an
  assertion loop) that flaked CI on Python 3.12.
- Six `SyntaxWarning`s from unescaped LaTeX in `multi_agent.py` docstrings;
  `src/` is now `SyntaxWarning`-free.
- PEP-701 f-strings in the Exp 34/36 `run.py` table writers that broke the
  smoke-run on Python 3.10/3.11.
- Two `mkdocs --strict`-breaking doc-tree-external links in the μ-analysis
  reference.

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
