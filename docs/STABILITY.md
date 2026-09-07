# API Stability & Versioning

From **v1.0.0** onward, `aimct` strictly follows [Semantic Versioning 2.0.0](https://semver.org):

- **PATCH** (`1.0.x`) — Bug fixes, documentation improvements, numerical tolerances, and performance optimizations. Zero API changes.
- **MINOR** (`1.x.0`) — New modules, classes, functions, or optional keyword arguments added in a backward-compatible manner. Existing code continues to work without modification.
- **MAJOR** (`2.0.0`) — Breaking changes to the public API as defined below.

---

## What Constitutes the Public API

A symbol is considered part of the `aimct` public API if and only if **all** of the following conditions hold:

1. It is exported from one of the **16 core subpackages'** `__init__.py` and listed in that module's `__all__` (or top-level `aimct.__all__`), **or** explicitly documented in the API reference portal.
2. Its import path does not contain a leading-underscore component (`aimct.controllers.LQR` is public; `aimct.controllers._qp` or `aimct._internal` is private).
3. It is not marked as *experimental* in its docstring or class documentation.

Everything else — helper functions, internal optimizers, private methods, and non-dataclass internal container states — may change in any release without triggering a major version bump.

---

## The Public Surface at v1.0

The following table enumerates the certified public API across all 16 core subpackages:

| Subpackage | Public API Symbols & Core Contracts | Guarantees & Consistency Notes |
| :--- | :--- | :--- |
| `aimct.systems` | `DynamicalSystem` (the ABC base class contract: `n_states`, `n_inputs`, `dynamics(x, u)`, `step(x, u, dt)`, optional `linearize(x0, u0)`, `output(x, u)`), `rotation_matrix`, and all concrete dynamical systems: `MassSpringDamper`, `Pendulum`, `CartPole`, `FurutaPendulum`, `BallAndBeam`, `TwoTank`, `TwoLinkArm`, `DCMotor`, `DCMotor2`, `LinearSystem`, `PlanarQuadrotor`, `Quadrotor3D`, `BicycleVehicle`, `DifferentialDriveRobot`, `MultiAgentSystem` (alias `MultiAgent`), graph topology generators (`complete_graph`, `cycle_graph`, `line_graph`, `star_graph`, `disconnected_graph`). | Concrete systems strictly follow continuous-time ODE and discrete `step()` conventions. |
| `aimct.controllers` | `Controller` (ABC), `PID`, `StateFeedback`, `LQR`, `ObserverFeedback`, `LinearMPC`, `TubeMPC`, `MRPISet`, `mrpi_box`, `SamplingMPC`, `ILQR` / `iLQR`, `iLQRResult`, `MRAC`, `DisturbanceObserver`, `QFilter`, `GainScheduledLQR`, `EnergyShapingSwingUp`, `HybridSwingUpLQR`, `ConsensusFormationController` (alias `FormationController`), formation geometry generators (`polygon_formation`, `line_formation`, `diamond_formation`, `wedge_formation`), $H_\infty$ robust loop shaping subpackage (`StateSpace`, `weight_S`, `weight_KS`, `weight_T`, `augment_plant`, `lft_lower`, `hinf_syn`, `mixsyn`, `HinfSynResult`, `HinfController`), active-set optimization (`solve_qp`, `QPResult`), matrix utilities (`dare`, `solve_care`, `solve_lyapunov`, `place_poles`, `controllability_matrix`, `is_controllable`, `wrap_angle`). | `solve_qp` and `QPResult` provide a reusable from-scratch active-set quadratic programming primitive, like `solve_care`/`dare`. `TubeMPC` computes minimal robust positively invariant (mRPI) outer bounding sets and Pontryagin difference tightening. |
| `aimct.planning` | `DirectCollocation`, `CollocationResult`. | Hermite–Simpson direct transcription NLP enforcing hard midpoint defects and terminal equality constraints. |
| `aimct.robust` | `BlockStructure`, `MuBounds`, `RobustMarginResult`, `mu`, `robust_stability_margin`, `robust_performance_margin`, `dk_iterate`. | Structured singular value analysis ($\mu$) for mixed real/complex parametric perturbations. `mu()` returns typed `MuBounds(lower, upper, rho, sigma_max, worst_delta)` with `.value`; `robust_stability_margin`, `robust_performance_margin`, and `dk_iterate` return typed `RobustMarginResult(omega, mu_lower, mu_upper, peak_upper, peak_lower, omega_peak, margin, robust)`. |
| `aimct.estimation` | `LuenbergerObserver`, `place_observer`, `solve_fare`, `KalmanFilter`, `DiscreteKalmanFilter`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, `MovingHorizonEstimator` (alias `MHE`), `ParticleFilter` (alias `PF`), `systematic_resample`, `finite_diff_jacobian`, `observability_matrix`, `observability_rank`, `is_observable`. | **Consistency Guarantee:** All four nonlinear estimators (`ExtendedKalmanFilter`, `UnscentedKalmanFilter`, `MovingHorizonEstimator`/`MHE`, `ParticleFilter`/`PF`) share a unified construction interface: `(f, h, Q, R, *, dt, ...)` and a `.from_system(system, Q, R, *, dt, h=None, **kwargs)` classmethod. |
| `aimct.simulate` | `simulate`, `simulate_batch`, `Trajectory`, `BatchResult`, `rk4_step`. | Deterministic numerical integration and trajectory logging. |
| `aimct.sysid` | `least_squares_id`, `dmdc`, `to_continuous`, `prediction_error`, `model_mismatch`, `identify_manipulator`, `manipulator_regressor`, `finite_difference_derivatives`, `ManipulatorID`. | System identification and Euler–Lagrange manipulator parameter regression. |
| `aimct.benchmarks` | Standardized metrics (`rise_time`, `settling_time`, `peak_overshoot`, `peak_time`, `steady_state_error`, `rmse`, `iae`, `itae`, `ise`, `control_energy`, `peak_control`, `slew_rate`, `saturation_duty_cycle`, `compute_all_metrics`), comparative harness (`compare`, `ComparisonResult`, `sweep`, `SweepResult`, `track_trajectory`, `TrackingResult`), challenge/capstone evaluation suites (`score_capstone_entry`, `score_capstone`, `capstone_leaderboard_table`, `BlackBoxPlant`, `BlackBoxEnvironment`, `SafetyEnvelope`, `ChallengeScoreResult`, `ParamPerturbed`, `perturbed_system`, `ActuatorLag`, `ImpulseDisturbance`, `ImpulseInjector`, `robust_degradation`, `evaluate_safety`, `score_run`, `WEIGHTS`, `NORMALISERS`, `ScoreWeights`, `BaselineCosts`, `CAPSTONE_WEIGHTS`, `CAPSTONE_BASELINES`). | Standardized comparative benchmarking across classical and AI paradigms. |
| `aimct.trajectories` | `Trajectory` (ABC reference trajectory), `Setpoint`, `Circle`, `Lemniscate`, `MinimumJerk`, `Spline`, `Dubins`, `Lissajous`, `Rose`, `Spiral`. | Parametric time- and arc-length-indexed reference trajectories. |
| `aimct.viz` | `animate`, `Replay`, `Sandbox`, `Disturbance`, `SystemArtist`, `register_artist`, `get_artist`, `has_artist`. | Publication-grade rendering and real-time interactive sandboxes. |
| `aimct.dev` | `DesignReport`, `build_report`, `render`, `load_system`, `preview_once`, `watch`. | Design-time system introspection and verification CLI (`python -m aimct preview MODULE:Class [--watch]`). |
| `aimct.deploy` | `export_controller`, `load_controller`, `PortableController`, `ControllerSpec`, `UnsupportedControllerError`, `emit_c`, `emit_micropython`. | Zero-dependency standalone C99 and MicroPython code generation. |
| `aimct.hil` | `RealTimeLoop`, `HILResult`, `DeadlineMissInfo`, `PlantEmulator`, `Transport`, `InProcessTransport`, `UDPTransport`, `SerialTransport`. | Real-time hardware-in-the-loop emulation and serial/UDP transport. |
| `aimct.hybrid` | `ShieldedController`, `box_predicate`, `barrier_predicate`. | Formal safety filtering and predicate-based control intervention. |
| `aimct.ml` | `MLP`, `LearnedDynamics`, `system_step`, `batched_rk4`. | Pure NumPy neural network forward/backward propagation and learned residual dynamics. |
| `aimct.rl` | `ControlEnv`, `TASKS`, `make`, `wrap_to_pi`, `figure8_reference`, `figure8_obs`, `FIGURE8_PERIOD`, `Discretizer`, `QLearning`, `GreedyPolicy`, `train`, `evaluate`, `GaussianPolicy`, `reinforce`, `evaluate_policy`, `BehaviorCloning`, `aggregate`, `dagger`, `DQN`, `QNetwork`, `ReplayBuffer`, `dqn`, `PPO`, `ppo`, `SAC`, `sac`, `SACResult`. | The RL surface exposes both agent classes and thin functional wrappers; both are first-class supported. |
| **CLI** | Subcommands for `python -m aimct` (`compare`, `preview`, `live`, `list`). | Command-line developer interface. |

---

## Important Conventions & Disambiguations

### Class Name Disambiguation
- **`Trajectory`**: Exists in **both** `aimct.simulate` (rollout result dataclass containing time vectors, state trajectories, input trajectories, and metadata) and `aimct.trajectories` (abstract base class defining reference trajectory generators). These are distinct classes designed for separate roles.
- **`TrackingResult`**: Exported from `aimct.benchmarks` for trajectory-tracking accuracy metrics (cross-track error, along-track error, completion percentage) and is distinct from simulation rollouts.

### Prediction Horizon Keyword Convention
Horizon length is spelled `N` on `LinearMPC` and `DirectCollocation` (knot/step count) and `horizon` on `ILQR` / `SamplingMPC` (receding-horizon length). This split is intentional and frozen for 1.0.

---

## Field-Stable Return Dataclasses

The following return **dataclasses** are guaranteed to be field-stable: fields may be added in minor releases, but will never be removed or renamed within major version `1.x`:

- `Trajectory` (`aimct.simulate`)
- `BatchResult` (`aimct.simulate`)
- `iLQRResult` (`aimct.controllers`)
- `CollocationResult` (`aimct.planning`)
- `TrackingResult` (`aimct.benchmarks`)
- `ComparisonResult` (`aimct.benchmarks`)
- `SweepResult` (`aimct.benchmarks`)
- `ChallengeScoreResult` (`aimct.benchmarks`)
- `DesignReport` (`aimct.dev`)
- `ManipulatorID` (`aimct.sysid`)
- `HinfSynResult` (`aimct.controllers`)
- `MRPISet` (`aimct.controllers`)
- `QPResult` (`aimct.controllers`)
- `MuBounds` (`aimct.robust`)
- `RobustMarginResult` (`aimct.robust`)
- `SACResult` (`aimct.rl`)
- `ControllerSpec` (`aimct.deploy`)
- `HILResult` (`aimct.hil`)
- `DeadlineMissInfo` (`aimct.hil`)

---

## Deprecation Policy

1. When an API symbol is slated for removal, it will be marked with a `DeprecationWarning` in minor release `1.n.0`, explicitly identifying its replacement. The deprecation will be documented in `CHANGELOG.md` and the symbol's docstring.
2. The deprecated symbol will remain functional throughout all subsequent `1.x` releases. It will be removed no earlier than **major version 2.0.0** and no sooner than **6 months** following the initial deprecation notice.
3. Experimental APIs (explicitly flagged in docstrings) are exempt and may evolve across minor releases.

---

## Supported Environments & Dependencies

- **Python Runtime:** Python 3.10, 3.11, 3.12, and 3.13 (verified across continuous integration runners). Dropping a Python version occurs only after official CPython end-of-life and constitutes a MINOR version change.
- **Core Dependencies:** Strict floors for NumPy and SciPy as specified in `pyproject.toml`.
- **Optional Extras:** Extras (`[ml]`, `[viz]`, `[xcheck]`, `[dev]`, `[docs]`) are required only for their specific optional submodules; the base library (`pip install aimct`) carries zero heavy dependencies.

---

## What Is Not Covered

- Exact numerical outputs of iterative algorithms (NLP solvers, gradient descent, Monte Carlo particles) across different BLAS/LAPACK implementations; unit test suites enforce tolerances rather than bit-for-bit equality.
- Generated figures and markdown tables in `experiments/` — these serve as the empirical evidentiary foundation rather than programmatic library exports.
- Minor visual layout parameters in matplotlib animation figures.
