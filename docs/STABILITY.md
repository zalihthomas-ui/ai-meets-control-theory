# API stability & versioning

> Draft for v1.0. Refined by the docs pass (D10); wired into the portal nav
> and linked from the README.

From **v1.0.0** onward, `aimct` follows [Semantic Versioning](https://semver.org):

- **PATCH** (`1.0.x`) — bug fixes, docs, performance. No API change.
- **MINOR** (`1.x.0`) — new modules, classes, functions, or keyword arguments,
  added in a backward-compatible way. Existing code keeps working.
- **MAJOR** (`2.0.0`) — a breaking change to something in the **public API**
  as defined below.

## What is public

A name is part of the public API if **all** of these hold:

1. It is exported from a subpackage's `__init__.py` and listed in that
   module's `__all__`, **or** documented in the API reference.
2. Its import path does not contain a leading-underscore component
   (`aimct.controllers.LQR` is public; `aimct.controllers._qp` is not).
3. It is not marked *experimental* in its docstring.

Everything else — helper functions, module internals, the exact contents of
non-dataclass return values, private attributes — may change in any release.

### The public surface at v1.0

| Subpackage | Public API |
| --- | --- |
| `aimct.systems` | `DynamicalSystem` (the ABC contract: `n_states`, `n_inputs`, `dynamics`, optional `linearize`/`output`) and every concrete system class |
| `aimct.controllers` | `PID`, `StateFeedback`, `LQR`, `LinearMPC`, `SamplingMPC`, `iLQR`/`ILQR`, `MRAC`, `DisturbanceObserver`/`QFilter`, `hinf` (`StateSpace`, `weight_*`, `augment_plant`, `hinf_syn`, `mixsyn`, `HinfController`), tube MPC, the swing-up/hybrid controllers, `solve_care`/`solve_qp`/`place_poles` |
| `aimct.planning` | `DirectCollocation`, `CollocationResult` |
| `aimct.robust` | `BlockStructure`, `mu`, `robust_stability_margin`, `robust_performance_margin`, `dk_iterate` |
| `aimct.estimation` | `LuenbergerObserver`, `KalmanFilter`/`DiscreteKalmanFilter`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, `MovingHorizonEstimator`/`MHE`, `ParticleFilter` |
| `aimct.sysid` | `least_squares_id`, `dmdc`, `to_continuous`, `identify_manipulator`, `finite_difference_derivatives`, `ManipulatorID` |
| `aimct.simulate` | `simulate`, `simulate_batch`, `Trajectory`, `BatchResult`, `rk4_step` |
| `aimct.benchmarks` | `compare`, `ComparisonResult`, `track_trajectory`, `TrackingResult`, the challenge/capstone scoring entry points |
| `aimct.trajectories` | `Setpoint`, `Circle`, `Lemniscate`, `MinimumJerk`, `Spline`, `Dubins`, `Lissajous`, `Rose`, `Spiral` |
| `aimct.viz` | `animate`, `Replay`, `Sandbox`, `Disturbance`, `SystemArtist`, `register_artist`/`get_artist`/`has_artist` |
| `aimct.dev` | `build_report`, `DesignReport`, `python -m aimct preview` |
| `aimct.deploy` | `export_controller`, `load_controller`, `PortableController`, `ControllerSpec`, `emit_c`, `emit_micropython` |
| `aimct.hil` | `RealTimeLoop`, `PlantEmulator`, the transport classes |
| `aimct.ml`, `aimct.rl` | learned-dynamics and agent classes as documented |
| CLI | `python -m aimct` subcommands (`compare`, `preview`, `live`, `list`) |

Return **dataclasses** (`Trajectory`, `BatchResult`, `iLQRResult`,
`CollocationResult`, `TrackingResult`, `ComparisonResult`, `DesignReport`,
`ManipulatorID`, `HinfSynResult`, …) are public: fields are only added, never
removed or renamed, within a major version.

## Deprecation process

1. In release `1.n.0`, the old name keeps working but emits a
   `DeprecationWarning` naming the replacement, and its docstring and the
   CHANGELOG record it.
2. The old name is removed no earlier than the **next major** release
   (`2.0.0`) and never less than **6 months** after the deprecating release,
   whichever is later.
3. Experimental APIs (docstring-marked) are exempt — they may change or be
   removed in a minor release, with a CHANGELOG note.

## Supported Python & dependencies

- Python: the versions in the CI matrix (currently 3.10–3.12). Dropping a
  Python version is a MINOR change once that version is end-of-life.
- NumPy / SciPy: the floors in `pyproject.toml`. Raising a floor is a MINOR
  change.
- Optional extras (`ml`, `viz`, `xcheck`, `docs`): required only for the
  features that import them; the core (`pip install aimct`) never depends on
  them.

## What is *not* covered

- Exact numeric outputs of iterative solvers (they may shift with a NumPy /
  SciPy / BLAS update); tests assert tolerances, not bit-for-bit values.
- The `experiments/` scripts and their generated tables/figures — these are
  the evidence base, versioned with the repo, not part of the package.
- Plot styling and animation frame details.
