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
- [x] **Interactive Live Sandboxes (v1)** — 2D planar quadrotor sandbox (`python -m aimct live`) and 6-DOF WebGL Three.js 3D sandbox (`python -m aimct live3d --web`). Superseded/extended by the `aimct.viz` sandbox family below.
- [x] **Test Harness & Reports** — living technical report PDF, master results table (`docs/RESULTS.md`), and engineering decision guide (`docs/DECISION-GUIDE.md`). Test count below reflects the current total, not the Phase-1 snapshot.

---

## Phase 2 — Post-Capstone Depth, Real Systems, & Packaging (✅ COMPLETE)

Phase 2 added **breadth** (more real-world multi-body systems), **depth** (the
algorithmic gaps the capstone bake-off exposed), a **visualisation layer**
(not originally scoped — added when the user asked for it), and turned the
repository into an installable, PyPI-ready package. All four original tracks
shipped; two items were consciously descoped as genuine stretch goals rather
than left silently undone (marked below).

### Track A — More Real Dynamical Systems

- [x] **Differential-Drive Mobile Robot** (`aimct.systems.DifferentialDriveRobot`,
  `d7f48bd`) — unicycle `[x, y, θ, v, ω]` with a
  first-order speed/yaw-rate actuator lag, analytic `linearize()` about a
  straight path, TurtleBot3-Burger-class parameters. Two experiments:
  - [Experiment 22](../experiments/22_diffdrive_path_following/) — pure
    pursuit vs. Stanley vs. path LQR vs. kinematic MPC on a waypoint spline.
    Verdict: each design trades along-track accuracy against cross-track
    tightness against energy; the MPC horizon buys nothing on a smooth,
    unconstrained path.
  - [Experiment 25](../experiments/25_diffdrive_moving_obstacle/) — the
    "cluttered map" task, done as analytic keep-out disks (two static, one
    **moving** across the path) rather than a full occupancy grid (a
    deliberate scope call — see the experiment's README). CEM and iLQR carry
    the same obstacle-penalty cost Experiments 20/21 used on the quad; CEM
    is the only entry that meaningfully avoids, iLQR's gradient step is
    limited by the barrier's local non-convexity near an obstacle's centre.
- [x] **2-Link Planar Robot Arm** (`aimct.systems.TwoLinkArm`,
  `d7f48bd`) — planar Euler–Lagrange dynamics with
  public `M(q)`/`C(q,q̇)`/`G(q)`, a settable wrist payload, real link
  masses/inertias. [Experiment 23](../experiments/23_twolink_arm_tracking/):
  PD+gravity-comp vs. computed torque vs. joint LQR vs. joint MPC on the
  nominal arm, then an unknown 0.5 kg payload with every controller still
  using the 0 kg model. Adaptive control is a **Slotine–Li one-parameter
  adaptive computed torque**, not the generic `aimct.controllers.MRAC` class
  the original roadmap named — `MRAC` assumes a linear matched uncertainty
  and does not fit a nonlinear manipulator's payload term; the adaptive law
  built for this experiment is the correct tool and ties directly to
  [Experiment 17](../experiments/17_adaptive_vs_fixed_changing_plant/)'s
  lesson (adapt the controller, don't identify the plant) on a real
  multi-body system.
- [ ] **Bicycle-Model Ground Vehicle** — dynamic lateral tire-force model
  (Pacejka / linear), high-speed double-lane-change, Stanley vs. LQR vs.
  kinematic MPC vs. an RL policy. **Not started** — open for Phase 3 or a
  future pass.
- [ ] *(Stretch, explicitly deferred)* Furuta pendulum, ball-and-beam,
  coupled two-tank classic lab benchmarks.

### Track B — Method Depth & Advanced Synthesis

- [x] **Nonlinear MPC — iLQR / Real-Time Iteration**
  (`aimct.controllers.ILQR` / `iLQR`, `c480efb`) —
  a regularised backward Riccati sweep (Tassa 2012 line search) with a
  Diehl real-time-iteration receding-horizon wrapper.
  [Experiment 24](../experiments/24_ilqr_vs_sampling_mpc/) closes the
  Experiment-21 remark directly: on the same cart-pole swing-up and quad
  figure-8, iLQR beats sampling MPC (CEM) by ~2 orders of magnitude in
  tracking error at comparable or lower compute.
  [Experiment 26](../experiments/26_harder_reference_paths/) confirms this
  is not a lemniscate artefact (iLQR wins 32×–840× on a sharper Lissajous
  and an easing-curvature spiral too), while
  [Experiment 25](../experiments/25_diffdrive_moving_obstacle/) shows the
  honest limit: on a **non-convex** obstacle-avoidance cost, gradient-based
  iLQR is outperformed by derivative-free CEM — smooth cost → iLQR wins big;
  non-smooth cost → CEM's population search still earns its keep. This is
  the fuller, evidence-backed version of Experiment 21's "CEM is loose and
  slow" remark, not a blanket win for either method.
- [ ] **Soft Actor-Critic (SAC)** and a proper PPO hyperparameter search —
  **not started**. Experiment 21's from-scratch on-policy PPO did not
  bootstrap on the quad (behaviour-cloned from the LQR+flatness expert
  instead — see the experiment's README); SAC as a genuinely sample-efficient
  off-policy alternative remains open.
- [ ] **Direct trajectory optimisation** (collocation / multiple shooting) —
  **not started**.
- [ ] **First-class behaviour cloning + DAgger** (`aimct.rl.imitation`) —
  **not started as a module.** Experiment 21 hand-rolled a one-off behaviour
  clone of the LQR+flatness expert (documented, not reusable); formalising
  it into a tested module is still open.
- [ ] *(Stretch, explicitly deferred)* H∞ / μ-synthesis loop shaping and
  disturbance-observer control for unmatched wind rejection.

### Track C — Benchmark & Trajectory Infrastructure

- [x] **Trajectory-tracking harness** (`aimct.benchmarks.tracking` —
  `track_trajectory`, `TrackingResult`, `70a22c8`,
  perf-fixed at `6846ff3`) — RMS/max position
  error, RMS cross-track error, path-completion %, control energy, a
  `space="joint"` mode for non-Cartesian tracking (Experiment 23), and
  `TrackingResult.animate()` (see `aimct.viz` below). Used by Experiments
  22, 23, 25, 26.
- [x] **`aimct.trajectories`** — `Setpoint`, `Circle`, `Lemniscate`,
  `MinimumJerk`, `Spline`, `Dubins` (`70a22c8`),
  plus `Lissajous`, `Rose`, `Spiral`
  (`168bcea`, added for Experiment 26's
  harder-path stress test).

### Track D — PyPI Packaging & Distribution

- [x] **Distribution configuration** — the `aimct` name is unreserved on
  PyPI (verified); full PEP 621 metadata in `pyproject.toml` (authors,
  description, `readme`, `license`, `keywords`, trove classifiers,
  `project.urls`); console entry point `aimct = "aimct.__main__:main"`
  (`2f05cfd`, version single-sourced at
  `d570324`).
- [x] **Wheel & source distribution** — `MANIFEST.in` excludes
  `experiments/`, `docs/papers/*.pdf`, and report build artifacts; a clean
  `aimct-0.1.0` wheel + sdist builds locally with `python -m build`; the
  installed wheel prints a clone hint instead of failing when an interactive
  sandbox script isn't present (`d570324`).
- [x] **Automated release pipeline** — `.github/workflows/release.yml`,
  dormant until the first tag, publishes via **PyPI Trusted Publishing
  (OIDC)** — no tokens in the repo.
- [x] **Documentation & release runbook** — `CHANGELOG.md`
  (Keep-a-Changelog) and `docs/PACKAGING.md`.
- [ ] **Actually publishing to PyPI** — deliberately not done; the runbook
  above is ready whenever that decision is made.

### Not originally scoped, shipped anyway — `aimct.viz` & `aimct.dev`

The user asked mid-Phase-2 for a unified visualisation layer and a
design-time system preview. Both landed as first-class subpackages:

- [x] **`aimct.viz`** (`0c8a45c` onward,
  `docs/VISUALIZATION.md`) — one `SystemArtist` contract powers two front
  ends: `animate(traj, system)` replays any `simulate()` /
  `track_trajectory()` run (reference + trail + telemetry HUD,
  `.save("run.mp4" | "run.gif")`), and `Sandbox` drives the same picture in
  real time with disturbances and hot-swappable controllers
  (`798b45e` added a help overlay, "surprise me",
  snapshot, and session-best tracking on top). Artists exist for `Pendulum`,
  `CartPole`, `TwoLinkArm`, `DifferentialDriveRobot`, `PlanarQuadrotor`;
  `register_artist()` extends it to a new system in one class.
- [x] **Seven interactive sandboxes** under `python -m aimct live
  [drone|drone3d|arm|arm3d|diffdrive|armbalance|armbalance3d]`
  (`0c8a45c` drone/arm/diffdrive,
  `6873322` armbalance,
  `6f1c19a` the PyVista 3-D arm variants) — the
  `drone3d` target auto-selects PyVista, falling back to matplotlib-3D or
  the experimental WebGL build.
- [x] **`aimct.dev`** (`4d146d9`,
  `docs/DEV_PREVIEW.md`, CLI wired at
  `e009282`) — a design-time preview for a new
  `DynamicalSystem`: pole map, controllability/observability, an
  analytic-vs-numeric `linearize()` Jacobian residual check, and four
  response traces (free/step/impulse/sinusoid), rebuilt on every file save
  via `python -m aimct preview mymodule.py:MyPlant --watch`. Deliberately
  separate from `aimct.viz` (authoring-time tool vs. the runtime replay/
  sandbox story), though it may borrow the replay panel from it later.

---

## Phase 3 — Hardware & Physical Deployment (🔮 PLANNED, not started)

- [ ] Hardware-in-the-loop (HIL) testing interfaces.
- [ ] Flight log ingestion and telemetry playback (Crazyflie CFclient / ROS2 rosbag).
- [ ] Hosted interactive documentation portal.
- [ ] Multi-agent collaborative control and partially-observable decentralized systems.
- [ ] The Track A/B items explicitly deferred above (bicycle-model vehicle,
  SAC, direct trajectory optimisation, a first-class imitation-learning
  module, H∞/μ-synthesis, and the classic-lab-benchmark stretch systems)
  remain open and are natural Phase 3 (or an earlier revisit) candidates
  ahead of anything in this list.

---

*Superseded: `docs/roadmap-phase2.md` was the Phase-2 planning document this
page now reports the outcome of; kept for history, not maintained further.*
