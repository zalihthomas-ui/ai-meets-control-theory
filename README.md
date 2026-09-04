# AI Meets Control Theory

### From Classical Control to Intelligent Autonomous Systems

[![CI](https://github.com/zalihthomas-ui/ai-meets-control-theory/actions/workflows/ci.yml/badge.svg)](https://github.com/zalihthomas-ui/ai-meets-control-theory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Report: Living PDF](https://img.shields.io/badge/Report-Living%20PDF-brightgreen)](docs/report/ai-meets-control-theory.pdf)

**AI Meets Control Theory** is a rigorous, from-scratch experimentation framework that systematically bridges classical control theory, modern state-space methods, constrained Model Predictive Control (MPC), Kalman filtering, adaptive control, and modern machine learning/reinforcement learning on physical dynamical systems. Under the core discipline **"derive it, build it from scratch, simulate it, visualise it, and compare it honestly"**, every controller—from PID and LQR to active-set MPC, EKF/UKF, PPO actor-critic, and safety shields—is evaluated on identical plants, sensor noise profiles, disturbances, and actuator limits.

📄 **[Read the Living Technical Report (PDF)](docs/report/ai-meets-control-theory.pdf)** &nbsp;|&nbsp; 📊 **[Master Results & Verdicts Table](docs/RESULTS.md)** &nbsp;|&nbsp; 🧭 **[Engineering Decision Guide](docs/DECISION-GUIDE.md)** &nbsp;|&nbsp; 🚁 **[Live 3D WebGL Sandbox](https://claude.ai/code/artifact/69b12b78-d7b2-4732-a7af-2af14930139b)** &nbsp;|&nbsp; 🎯 **[Project Vision & Manifesto](docs/vision.md)**

---

## Quickstart & Installation

```bash
# 1. Clone and install in editable mode with development & ML extras
git clone https://github.com/zalihthomas-ui/ai-meets-control-theory.git
cd ai-meets-control-theory
pip install -e ".[dev,ml]"

# 2. Run the unit test suite (348 passing tests from scratch)
pytest -m "not slow"

# 3. Run a canonical multi-controller benchmark comparison
python -m aimct compare --system quadrotor

# 4. Launch the interactive 3D WebGL physics sandbox (or 2D live sandbox)
python -m aimct live3d --web
# or standalone matplotlib 3D / 2D
python -m aimct live3d
python -m aimct live
```

📦 **Packaging & Releases:** See [`docs/PACKAGING.md`](docs/PACKAGING.md) for the PyPI distribution runbook and [`CHANGELOG.md`](CHANGELOG.md) for the version history.

---

## The Experiments (01–21)

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

---

## Framework Architecture & Package Layout

```
src/aimct/
  systems/        DynamicalSystem base + LinearSystem, MassSpringDamper,
                  Pendulum, CartPole, PlanarQuadrotor (Crazyflie 2.0), DCMotor
  simulate.py     rk4_step(), simulate() -> Trajectory(t, x, u, y)
  controllers/    PID, StateFeedback, LQR, ObserverFeedback, MRAC,
                  EnergyShapingSwingUp, HybridSwingUpLQR,
                  LinearMPC (with preview), SamplingMPC (CEM + obstacles), _qp.solve_qp
  estimation/     LuenbergerObserver, KalmanFilter (LQE/FARE), DiscreteKalmanFilter,
                  ExtendedKalmanFilter, UnscentedKalmanFilter, observability_matrix
  sysid/          least_squares_id, dmdc, to_continuous (block logm), prediction_error
  ml/             MLP (backprop + Adam), LearnedDynamics (grey-box / residual)
  rl/             ControlEnv (Gymnasium adapter), Discretizer, QLearning, DQN, REINFORCE, PPO
  hybrid/         ShieldedController (switch/filter blends, predicate helpers)
  benchmarks/     metrics.py (13 metrics), harness.py, sweep.py, challenge.py (ICC engine)
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

## Status: Phase 2 in Progress 🚀

The core curriculum (Modules 01–10), 21 empirical benchmark experiments, living technical report, and 6-DOF live sandboxes are complete with **348 passing unit tests**.

**Phase 2 is actively expanding the library with:**
- **Track A (Real Systems):** Differential-drive mobile robots, 2-link planar manipulator arms (computed torque vs. adaptive MRAC), and dynamic bicycle ground vehicles.
- **Track B (Algorithmic Depth):** Real-time iteration Nonlinear MPC (iLQR / SQP), Soft Actor-Critic (SAC) continuous RL, direct trajectory optimization, and formalized Behavior Cloning + DAgger.
- **Track C & D:** Reusable trajectory generation suite and PyPI distribution packaging (`aimct`).

See [`docs/roadmap.md`](docs/roadmap.md) for detailed deliverables and current priorities.

---

## License

MIT — see [LICENSE](LICENSE).
