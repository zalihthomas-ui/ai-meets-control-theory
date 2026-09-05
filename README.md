# AI Meets Control Theory

### From Classical Control to Intelligent Autonomous Systems

[![CI](https://github.com/zalihthomas-ui/ai-meets-control-theory/actions/workflows/ci.yml/badge.svg)](https://github.com/zalihthomas-ui/ai-meets-control-theory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Report: Living PDF](https://img.shields.io/badge/Report-Living%20PDF-brightgreen)](docs/report/ai-meets-control-theory.pdf)

**AI Meets Control Theory** is a rigorous, from-scratch experimentation framework that systematically bridges classical control theory, modern state-space methods, constrained Model Predictive Control (MPC), Kalman filtering, adaptive control, and modern machine learning/reinforcement learning on physical dynamical systems. Under the core discipline **"derive it, build it from scratch, simulate it, visualise it, and compare it honestly"**, every controller—from PID and LQR to active-set MPC, EKF/UKF, PPO actor-critic, and safety shields—is evaluated on identical plants, sensor noise profiles, disturbances, and actuator limits.

📄 **[Read the Living Technical Report (PDF)](docs/report/ai-meets-control-theory.pdf)** &nbsp;|&nbsp; 📖 **[User Guide & API Recipes](docs/USAGE.md)** &nbsp;|&nbsp; 💡 **[Examples Gallery](examples/README.md)** &nbsp;|&nbsp; 🎨 **[Unified Visualization](docs/VISUALIZATION.md)** &nbsp;|&nbsp; 📊 **[Master Results & Verdicts Table](docs/RESULTS.md)** &nbsp;|&nbsp; 🧭 **[Engineering Decision Guide](docs/DECISION-GUIDE.md)** &nbsp;|&nbsp; 🚁 **[Live 3D WebGL Sandbox](https://claude.ai/code/artifact/69b12b78-d7b2-4732-a7af-2af14930139b)** &nbsp;|&nbsp; 🎯 **[Project Vision & Manifesto](docs/vision.md)**

---

## Quickstart & Installation

```bash
# 1. Clone and install in editable mode with development & ML extras
git clone https://github.com/zalihthomas-ui/ai-meets-control-theory.git
cd ai-meets-control-theory
pip install -e ".[dev,ml]"

# 2. Run the fast test suite (426 fast / 433 total passing unit tests from scratch)
pytest -m "not slow"

# 3. Run a canonical multi-controller benchmark comparison
python -m aimct compare --system quadrotor

# 4. Launch interactive physics sandboxes (2D drone, 2-link arm, diff-drive, or 3D 6-DOF WebGL)
python -m aimct live            # 2D quadrotor vs wind
python -m aimct live arm        # 2-link manipulator
python -m aimct live diffdrive  # mobile robot
python -m aimct live3d --web    # 6-DOF WebGL sandbox
```

📖 **Usage & Recipes:** See [`docs/USAGE.md`](docs/USAGE.md) for the 5-axis framework guide (*system × controller × trajectory × disturbance × parameters*) and copy-paste recipes.  
💡 **Examples Gallery:** See [`examples/README.md`](examples/README.md) for 6 concise, copy-pasteable runnable scripts showcasing every major framework feature.  
🎨 **Unified Visualization:** See [`docs/VISUALIZATION.md`](docs/VISUALIZATION.md) for replay animation (`aimct.viz.animate`) and real-time interactive sandboxes (`aimct.viz.Sandbox`).  
🛠️ **Design-Time Preview:** See [`docs/DEV_PREVIEW.md`](docs/DEV_PREVIEW.md) for model inspection and Jacobian validation (`python -m aimct preview <Plant> --watch`).  
📦 **Packaging & Releases:** See [`docs/PACKAGING.md`](docs/PACKAGING.md) for the PyPI distribution runbook and [`CHANGELOG.md`](CHANGELOG.md) for the version history.

---

## The Experiments (01–30)

Every experiment is self-contained with its own configuration, runner, Markdown/CSV benchmark table, and publication-ready 4-panel figure. See [`docs/RESULTS.md`](docs/RESULTS.md) for full metrics.

| Exp | Directory | Plant | Key Comparison | Empirical Finding & Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **01** | [`01_integrator_accuracy`](experiments/01_integrator_accuracy/) | Mass-Spring-Damper | RK4 vs Forward Euler | Euler adds false numerical energy; 4th-order RK4 is mandatory for stable physics. |
| **02** | [`02_linearization_validity`](experiments/02_linearization_validity/) | Inverted Pendulum | Linear vs Nonlinear ODE | Linear state-space diverges at $|\theta_0| > 23^\circ$; validity is strictly local. |
| **03** | [`03_pid_stabilizes_unstable`](experiments/03_pid_stabilizes_unstable/) | Inverted Pendulum | PID Clamping vs Raw PID | Conditional anti-windup clamping cuts overshoot from $53\%$ to $39\%$ and prevents windup instability. |
| **04** | [`04_lqr_vs_pole_placement_cartpole`](experiments/04_lqr_vs_pole_placement_cartpole/) | Cart-Pole (Balance) | LQR (CARE) vs Pole Placement | LQR finds optimal gain without pole guessing; single-loop PID drifts $0.82\,\text{m}$ off-rail. |
| **05** | [`05_cartpole_basin_of_attraction`](experiments/05_cartpole_basin_of_attraction/) | Cart-Pole (Nonlinear) | LQR Basin of Attraction | Quantified $57^\circ$ recoverable envelope; actuator saturation prevents divergence. |
| **06** | [`06_lqg_vs_lqr_measurement_noise`](experiments/06_lqg_vs_lqr_measurement_noise/) | Cart-Pole (Encoders) | Kalman Filter vs Luenberger | Fast observers amplify encoder noise ($16.2\,\text{N}^2\text{s}$); LQG gives smooth, optimal effort ($2.2\,\text{N}^2\text{s}$). |
| **07** | [`07_cartpole_swingup_hybrid`](experiments/07_cartpole_swingup_hybrid/) | Cart-Pole (Swing-Up) | Spong Energy Shaping + LQR | Energy pumping lifts from $\theta=\pi$ to orbit; hysteresis supervisor catches in 1 switch. |
| **08** | [`08_mpc_vs_lqr_constrained_cartpole`](experiments/08_mpc_vs_lqr_constrained_cartpole/) | Cart-Pole (Bounds) | Constrained MPC vs LQR | Active-set QP MPC strictly respects $|x| \le 0.5\,\text{m}$, while LQR violates rail by $23\%$. |
| **09** | [`09_control_on_identified_model`](experiments/09_control_on_identified_model/) | Cart-Pole (SysID) | Least-Squares / DMDc ID | LQR gain margins tolerate $20\%$ parameter residual on $24\,\text{s}$ data; $1\,\text{s}$ data destabilizes. |
| **10** | [`10_planning_learned_vs_true_model`](experiments/10_planning_learned_vs_true_model/) | Cart-Pole (Neural MLP) | Sampling MPC on Neural Model | 4,804-param residual MLP matches true physics planning ($3\%$ error); CARE terminal cost required. |
| **11** | [`11_qlearning_vs_classical`](experiments/11_qlearning_vs_classical/) | Inverted Pendulum | Tabular Q vs Energy Shaping | Model-free RL learns swing-up but chatters at $1.5\,\text{rad}$; classical needs zero training data. |
| **12** | [`12_shielded_qlearning`](experiments/12_shielded_qlearning/) | Inverted Pendulum | Shielded RL vs Raw RL | Classical safety shield locks RL swing-up to $0.00\,\text{rad}$ with $35\%$ less control effort. |
| **13** | [`13_robust_control_loop_shaping`](experiments/13_robust_control_loop_shaping/) | Stiff Dynamics | $\mathcal{H}_\infty$ Loop Shaping vs Nominal | Explicit sensitivity shaping guarantees stability under $\pm 40\%$ parameter uncertainty. |
| **14** | [`14_quadrotor_figure8_tracking`](experiments/14_quadrotor_figure8_tracking/) | Crazyflie 2.0 ($28\,\text{g}$) | Flatness Feedforward vs MPC | Differential flatness inversion cuts RMS error by $15\%$ to $43.5\,\text{mm}$; preview MPC matches at $47.9\,\text{mm}$. |
| **15** | [`15_quadrotor_ekf_output_feedback`](experiments/15_quadrotor_ekf_output_feedback/) | Crazyflie 2.0 (Noisy) | EKF Observer vs Differencing | EKF reconstructs unmeasured velocity to $8\,\text{mm/s}$; finite differencing explodes energy $150\times$. |
| **16** | [`16_ekf_vs_ukf`](experiments/16_ekf_vs_ukf/) | Inverted Pendulum | EKF vs Unscented UKF | UKF sigma points escape $\pi$-off false basin ($0.07\,\text{rad}$); EKF gets trapped at $6.28\,\text{rad}$. |
| **17** | [`17_adaptive_vs_fixed_changing_plant`](experiments/17_adaptive_vs_fixed_changing_plant/) | MSD (Drifting $k$) | Lyapunov MRAC vs Fixed LQR | MRAC holds $< 1\,\text{mm}$ error under $500\%$ spring constant drift, eliminating static LQR droop. |
| **18** | [`18_rl_zoo_vs_lqr`](experiments/18_rl_zoo_vs_lqr/) | Cart-Pole (Balance) | RL Zoo (DQN, PPO) vs LQR | Scratch continuous PPO matches LQR return $-0.3$ and $200/200$ hold, paying $240\text{k}$ sample cost. |
| **19** | [`19_icc_leaderboard`](experiments/19_icc_leaderboard/) | Multi-Plant Challenge | Blind Black-Box Leaderboard | MPC dominates precision (DC Motor $41.3$); Energy+LQR hybrid sweeps agility (Pendulum $23.8$, Track 3 $29.9$). |
| **20** | [`20_quadrotor_obstacle_nmpc`](experiments/20_quadrotor_obstacle_nmpc/) | Crazyflie 2.0 (Keep-Out) | Sampling NMPC vs Flatness LQR | NMPC bends trajectory around keep-out ($+11\,\text{mm}$ clearance); flatness LQR crashes straight through. |
| **21** | [`21_grand_capstone_bakeoff`](experiments/21_grand_capstone_bakeoff/) | Crazyflie 2.0 (Grand Course) | Five-Way Grand Bake-Off | Sampling NMPC scores 8.0 (0 violations); imitation tracks 41.4 mm but cuts keep-out 46 times; hybrid scores 7.9. |
| **22** | [`22_diffdrive_path_following`](experiments/22_diffdrive_path_following/) | TurtleBot3-Burger (Unicycle) | Pure Pursuit vs Stanley vs Path LQR | Path LQR curvature feedforward gives tightest cross-track error ($9.25\,\text{mm}$); pure pursuit cuts corners ($35\,\text{mm}$). |
| **23** | [`23_twolink_arm_tracking`](experiments/23_twolink_arm_tracking/) | 2-Link Planar Robot Arm | Computed Torque vs Slotine--Li MRAC | Nominal computed torque collapses under $+0.5\,\text{kg}$ load ($394\,\text{mm}$); Slotine--Li adapts to $4.93\,\text{mm}$ ($100\%$). |
| **24** | [`24_ilqr_vs_sampling_mpc`](experiments/24_ilqr_vs_sampling_mpc/) | Cart-Pole & Crazyflie 2.0 | iLQR / RTI-NMPC vs Sampling MPC | iLQR converges $150\times$ tighter on quad ($1.34\,\text{mm}$ error) and solves in $14.6\,\text{ms}$ (meets $20\,\text{ms}$ flight budget). |
| **25** | [`25_diffdrive_moving_obstacle`](experiments/25_diffdrive_moving_obstacle/) | TurtleBot3-Burger (Dynamic Disks) | Blind Trackers vs Obstacle-Aware Planners | CEM derivative-free sampling navigates around non-convex obstacle fields ($36$ collision steps) where iLQR gradient fails to clear ($69$ steps). |
| **26** | [`26_harder_reference_paths`](experiments/26_harder_reference_paths/) | Crazyflie 2.0 (Lissajous, Spiral) | iLQR vs Sampling MPC across Geometries | iLQR beats CEM by $32\times\text{--}840\times$ RMS error; CEM latency ($28\text{--}31\,\text{ms}$) violates $20\,\text{ms}$ flight budget on all paths. |
| **27** | [`27_bicycle_double_lane_change`](experiments/27_bicycle_double_lane_change/) | Dynamic Bicycle Sedan | Stanley vs LQR vs Kinematic MPC vs BC RL | Kinematic MPC wins nominal ($52.5\,\text{mm}$); Stanley wins Pacejka $\mu=0.6$ ($734\,\text{mm}$); BC RL fails off-road ($5.22\,\text{m}$ RMS). |
| **28** | [`28_furuta_pendulum_control`](experiments/28_furuta_pendulum_control/) | Furuta Rotary Pendulum (QUBE-2) | LQR vs Linear MPC vs Energy Swing-Up | Upright stabilization in $40\,\text{ms}$ ($e_{ss} < 6\times 10^{-7}\,\text{rad}$); MPC caps torque ($0.1343\,\text{N}\cdot\text{m}$); Swing-up in $6.0\,\text{s}$. |
| **29** | [`29_dagger_vs_bc_lane_change`](experiments/29_dagger_vs_bc_lane_change/) | Dynamic Bicycle (Pacejka $\mu=0.6$) | Plain BC vs DAgger (8 rounds) | Plain BC drifts off-road ($6.02\,\text{m}$ RMS); DAgger relabeling matches expert LQR ($768.8\,\text{mm}$ RMS) but inherits expert ceiling. |
| **30** | [`30_two_tank_level_control`](experiments/30_two_tank_level_control/) | Coupled Nonlinear Two-Tank | SISO PI vs Multivariable LQR vs Linear MPC | SISO PI eliminates nonlinear steady-state droop ($0.0\,\text{cm}$); LQR/MPC cuts pump energy by $22\%$ ($6659\,\text{V}^2\text{s}$) with $0\%$ level violation. |

---

## Framework Architecture & Package Layout

```
src/aimct/
  systems/        DynamicalSystem base + LinearSystem, MassSpringDamper,
                  Pendulum, CartPole, PlanarQuadrotor (Crazyflie 2.0), DCMotor,
                  DifferentialDriveRobot, TwoLinkArm, BicycleVehicle,
                  FurutaPendulum, TwoTank
  simulate.py     rk4_step(), simulate() -> Trajectory(t, x, u, y)
  controllers/    PID, StateFeedback, LQR, ObserverFeedback, MRAC, ComputedTorque,
                  EnergyShapingSwingUp, HybridSwingUpLQR,
                  LinearMPC (with preview), SamplingMPC (CEM + obstacles),
                  ILQR (trajectory optimiser + real-time-iteration NMPC)
  estimation/     LuenbergerObserver, KalmanFilter (LQE/FARE), DiscreteKalmanFilter,
                  ExtendedKalmanFilter, UnscentedKalmanFilter, observability_matrix
  trajectories/   Lemniscate, Spline, Minimum-Jerk Polynomials, Dubins, Lissajous, Spiral, Rose
  sysid/          least_squares_id, dmdc, to_continuous (block logm), prediction_error
  ml/             MLP (backprop + Adam), LearnedDynamics (grey-box / residual)
  rl/             ControlEnv (Gymnasium adapter), Discretizer, QLearning, DQN, REINFORCE, PPO,
                  imitation (BehaviorCloning, dagger)
  hybrid/         ShieldedController (switch/filter blends, predicate helpers)
  viz/            SystemArtist contract, animate() replay engine, Sandbox live GUI
  dev/            Design-time preview dashboard (poles, controllability, Jacobian residuals)
  benchmarks/     metrics.py (13 metrics), harness.py, sweep.py, challenge.py, tracking.py
  plot_style.py   Okabe-Ito color palette + publication-ready 4-panel comparison figures
```

---

## The Core Engineering Cycle

Every method follows the same rigorous pipeline:

```
THEORY → DERIVATION → IMPLEMENTATION → SIMULATION → VISUALISATION → VALIDATION → COMPARISON → EXPERIMENT
```

1. **Understand the mathematics:** First-principles ODEs, Riccati equations, Lyapunov stability, Hamiltonians, Hamilton-Jacobi-Bellman, GAE.
2. **Build from scratch:** No black-box library magic in core algorithms. Custom Hamiltonian Schur CARE solver, custom active-set QP solver, custom backpropagation + Adam, custom sigma-point UKF, custom PPO actor-critic.
3. **Validate against reality:** Hard actuator saturation, sensor noise, latency, parameter drift, unmeasured states, non-convex keep-out zones.
4. **Compare honestly:** Side-by-side Pareto tables under identical random seeds, step sizes, and initial conditions.

---

## Learning Curriculum

| Module | Topic | Description |
| :--- | :--- | :--- |
| **[01](modules/01-mathematical-foundations)** | Mathematical Foundations | Linear algebra, matrix exponential, RK4 integration, numerical optimization. |
| **[02](modules/02-dynamic-system-modeling)**  | Dynamic System Modeling | First-principles physics, state-space representations, Jacobian linearisation. |
| **[03](modules/03-classical-control)**        | Classical Control | Filtered derivative PID, conditional anti-windup clamping, frequency response. |
| **[04](modules/04-modern-control)**           | Modern Control & Estimation | Controllability, observability, Ackermann pole placement, Luenberger observers, Kalman filters (linear, EKF, UKF). |
| **[05](modules/05-optimal-control)**          | Optimal & Constrained Control | Algebraic Riccati equations (CARE/DARE), LQR robustness margins, constrained active-set Model Predictive Control. |
| **[06](modules/06-machine-learning)**         | ML for Dynamical Systems | Least-squares SysID, DMDc, neural MLP backprop + Adam, residual LearnedDynamics, sampling-based MPC (CEM). |
| **[07](modules/07-reinforcement-learning)**   | Reinforcement Learning | Gymnasium `ControlEnv`, Tabular Q-Learning, Deep Q-Networks (DQN), REINFORCE, and Proximal Policy Optimization (PPO). |
| **[08](modules/08-ai-plus-control)**          | AI + Control (Hybrid Safety) | Supervisory safety shielding, action filtering, control barrier functions, auditable intervention logging. |
| **[09](modules/09-robotics-capstones)**       | Robotics Capstones | Full 6-state Crazyflie 2.0 Quadrotor, differential flatness inversion, Bryson scaling, EKF output feedback, MRAC, obstacle NMPC. |
| **[10](modules/10-intelligent-control-challenge)** | Intelligent Control Challenge | Standardized black-box multi-track benchmark engine and cross-paradigm leaderboard. |

---

## Status: Phase 2 Complete (Phase 3 Planned) 🚀

The core curriculum (Modules 01–10), 30 empirical benchmark experiments, living technical report, unified visualization layer (`aimct.viz`), design-time preview dashboard (`aimct.dev`), and 7 interactive sandboxes are complete with **433 passing unit tests** across Python 3.10–3.13.

**Phase 2 Delivered:**
- **Track A (Real Systems):** Differential-drive mobile robots (Exp 22 path tracking, Exp 25 dynamic obstacle avoidance), 2-link planar manipulator arms (Exp 23 computed torque & Slotine–Li payload adaptation), dynamic bicycle vehicles (Exp 27 ISO-3888 double lane change with linear vs. Pacejka tire models), Furuta rotary inverted pendulums (Exp 28 Quanser QUBE-Servo 2 benchmark), and coupled nonlinear process tanks (Exp 30 Quanser Coupled Two-Tank level control).
- **Track B (Algorithmic Depth):** Real-time iteration Nonlinear MPC (Exp 24, 26, 25 iLQR / RTI-NMPC vs. Sampling MPC) and interactive imitation learning (Exp 29 DAgger vs. Behavior Cloning lane-change recovery).
- **Track C & D:** Reusable trajectory generation suite (`aimct.trajectories`), tracking benchmark harness (`aimct.benchmarks.tracking`), unified visualization (`aimct.viz`), design-time preview (`aimct.dev`), and PyPI distribution packaging (`aimct`).

**Phase 3 (Planned):**
Hardware-in-the-loop (HIL) physical deployment, flight log telemetry ingestion (CFclient/ROS2), dynamic bicycle vehicle model, Soft Actor-Critic (SAC), and direct collocation trajectory optimization.

See [`docs/roadmap.md`](docs/roadmap.md) for detailed deliverables and development history.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and engineering
agreement, and the [Code of Conduct](CODE_OF_CONDUCT.md). Found a security
issue? See [SECURITY.md](SECURITY.md) for how to report it privately.

## License

MIT — see [LICENSE](LICENSE).
