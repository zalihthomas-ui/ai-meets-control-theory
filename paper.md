---
title: 'AIMCT: A Benchmark Suite and Unified Framework Bridging AI and Control Theory'
tags:
  - Python
  - control theory
  - reinforcement learning
  - dynamical systems
  - model predictive control
  - physics-informed neural networks
  - system identification
  - hardware-in-the-loop
authors:
  - name: Zalih Thomas
    orcid: 0009-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher / AI Meets Control Theory Initiative, Istanbul, Turkey
    index: 1
date: 6 September 2026
bibliography: paper.bib
---

# Summary

Modern autonomous systems increasingly operate at the intersection of classical control theory and modern data-driven artificial intelligence. While classical methods (such as Proportional-Integral-Derivative [PID], Linear Quadratic Regulators [LQR], Model Predictive Control [MPC], and Sliding Mode Control [SMC]) provide formal guarantees of stability, robustness margins, and constraint satisfaction, they often struggle when physical models are uncertain or highly nonlinear [@astrom2010feedback; @rawlings2017model; @slotine1991applied]. Conversely, modern machine learning paradigms (such as Reinforcement Learning [RL], Physics-Informed Neural Networks [PINNs], Neural Ordinary Differential Equations [Neural ODEs], and Deep Koopman operators) can learn complex behaviors from data but frequently lack formal stability certificates, sample efficiency, and reliable out-of-distribution generalization [@schulman2017proximal; @haarnoja2018soft; @raissi2019physics; @chen2018neural; @brunton2022koopman].

imct (*AI Meets Control Theory*) is an open-source Python library designed to bridge this divide. It provides a standardized, mathematically rigorous testbed comprising **34 empirical benchmark experiments** spanning 9 dynamical physical systems (from linear mass-spring-damper to underactuated inverted pendulums, quadrotors, flexible joints, and 2-link robotic manipulators). imct unifies classical state-space models, numerical optimal control, statistical state estimators, system identification algorithms, neural dynamics models, and deep RL agents under a cohesive, zero-friction object-oriented interface.

# Statement of Need

Despite substantial interest in combining machine learning with control theory, existing software ecosystems remain deeply fragmented:
1. **Classical Control Libraries:** Packages such as python-control [@fuller2016pythoncontrol] provide classical frequency-domain and linear state-space tools, but lack deep integration with neural network architectures, trajectory planning algorithms, and RL environments.
2. **Optimization and MPC Frameworks:** Packages such as do-mpc [@lucia2017dompc] and CasADi [@andersson2019casadi] excel at nonlinear optimal control and symbolic automatic differentiation, but require steep learning curves and do not directly offer plug-and-play RL benchmarks, disturbance observers, or standardized comparative metrics.
3. **Reinforcement Learning Suites:** Libraries like Stable-Baselines3 [@raffin2021stablebaselines3] and Gymnasium [@brockman2016openai] offer state-of-the-art RL algorithms, but treat environments as black-box Markov Decision Processes (MDPs), completely discarding known physics, energy conservation laws, controllability matrices, and Lyapunov stability proofs.

Researchers and practitioners seeking to benchmark whether a Deep RL policy out-performs a tuned LQR or Nonlinear MPC on a physical system are often forced to write ad-hoc wrappers, disparate simulation loops, and custom metric extractors.

imct addresses this gap by establishing:
- **Unified Abstractions:** A single DynamicalSystem API that supports ODE integration (Runge-Kutta 4th order, DOP853), Gym/Gymnasium environment exposure, automatic linearization (, B, C, D$ matrices), and physical energy tracking ( = T + V$).
- **Multi-System Benchmark Suite:** 34 systematically formulated experiments across 10 algorithmic categories (Linear, Underactuated, Multivariable, Constraints, Estimation, System Identification, Trajectory Optimization, Machine Learning, Reinforcement Learning, and Hybrid/Residual Control).
- **Rigorous Evaluation Metrics:** Standardized measurement of Settling Time ($), Overshoot ($), Integrated Absolute Error (IAE), Control Effort ($ norm of input $), Robustness Margins (gain/phase/delay margins), and Hardware Timing Jitter.
- **Hardware-in-the-Loop (HIL) Bridges:** Modular serial/UDP communication protocols enabling direct deployment of Python controllers to physical hardware platforms (such as the 2-Link Robotic Manipulator) with real-time safety watchdogs.

# The 34-Experiment Empirical Evidence Base

The core of imct is its comprehensive suite of 34 benchmark experiments, categorized into ten distinct research tracks:

| Track | Experiments | Key Focus Systems & Methods |
| :--- | :--- | :--- |
| **I. Baseline Linear** | EXP-01 to EXP-03 | Mass-Spring-Damper, Inverted Pendulum (PID, LQR, Pole Placement) |
| **II. Underactuated Systems** | EXP-04 to EXP-07 | Cart-Pole, Furuta Pendulum, Ball and Beam (Energy Swing-Up, LQR, SMC) |
| **III. Multivariable & Coupled** | EXP-08 to EXP-10 | Quadrotor UAV, Coupled 2-Tank, 2-Link Arm (MIMO Decoupling, State-Space) |
| **IV. Constraints & Optimal** | EXP-11 to EXP-14 | Linear & Nonlinear MPC with input/state saturation and obstacle avoidance |
| **V. Estimation & Observers** | EXP-15 to EXP-18 | Kalman Filter (KF), EKF, UKF, Extended State Observer (ESO), Disturbance Observers |
| **VI. System Identification** | EXP-19 to EXP-21 | Recursive Least Squares (RLS), Subspace SysId (N4SID), Sparse Identification (SINDy) |
| **VII. Trajectory Optimization** | EXP-22 to EXP-24 | Direct Collocation, Differential Dynamic Programming (iLQR), Minimum Snap Planning |
| **VIII. Machine Learning & PINNs** | EXP-25 to EXP-27 | Physics-Informed Neural Networks, Neural ODEs, Deep Koopman Linearization |
| **IX. Reinforcement Learning** | EXP-28 to EXP-31 | PPO, SAC, DDPG, Model-Based RL (MBPO) with reward-shaping ablation |
| **X. Hybrid & Advanced** | EXP-32 to EXP-34 | Residual RL (LQR + Policy), Active Disturbance Rejection Control (ADRC), Real-Time HIL |

Each experiment includes automated validation tests, convergence checks, phase portrait generators, and standardized Markdown reports comparing classical vs. AI approaches.

# Software Architecture and Interface Design

The architecture of imct is structured around modular, loosely coupled Python subpackages:

`
src/aimct/
├── systems/       # ODE definitions, energy equations, and Gym wrappers
├── controllers/   # PID, LQR, MPC, SMC, Adaptive, Feedback Linearization
├── planning/      # iLQR, Direct Collocation, RRT*, Spline generators
├── estimation/    # Linear KF, EKF, UKF, Particle Filter, ESO
├── sysid/         # Least Squares, N4SID, SINDy, Manipulator Regressor SysID
├── ml/            # PINNs, Neural ODEs, Deep Koopman operators (PyTorch)
├── rl/            # PPO, SAC implementations with Gymnasium integration
├── hybrid/        # Residual RL (Nominal + Neural), Neuro-Symbolic controllers
├── hil/           # Real-time hardware serial bridge, packet streamer, watchdogs
├── viz/           # Publication-grade Matplotlib/Plotly engine & live dashboard
├── dev/           # Developer tools, metric trackers, property testing contracts
└── benchmarks/    # Standardized experiment test runners and scorecards
`

A core principle of imct is mathematical transparency. For instance, simulating a closed-loop controller on an inverted pendulum requires fewer than 10 lines of clean, self-documenting code:

`python
import aimct
from aimct.systems import InvertedPendulum
from aimct.controllers import LQRController

system = InvertedPendulum(mass=0.2, length=0.3, damping=0.01)
controller = LQRController(system, Q=[10.0, 1.0], R=[0.1])
result = aimct.simulate(system, controller, x0=[0.2, 0.0], t_span=(0.0, 5.0))

print(f"Settling time: {result.settling_time:.3f} s, IAE: {result.iae:.3f}")
result.plot(save_path="pendulum_lqr.png")
`

# Comparison to Existing Software

| Feature | imct | python-control | do-mpc | CasADi | Stable-Baselines3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical State-Space (LQR/PID/SMC)** | Yes | Yes | Partial | No | No |
| **Model Predictive Control (MPC)** | Yes | No | Yes | Yes (Core) | No |
| **Nonlinear Estimation (EKF/UKF/ESO)** | Yes | Partial | Yes | No | No |
| **System Identification (RLS/SINDy)** | Yes | Partial | Yes | No | No |
| **Deep RL Algorithms (PPO/SAC)** | Yes | No | No | No | Yes |
| **Physics-Informed NNs (PINNs/NeuralODEs)** | Yes | No | No | No | No |
| **Hybrid / Residual Control (LQR+RL)** | Yes | No | No | No | No |
| **Pre-built 34-Experiment Benchmark Base** | Yes | No | No | No | No |
| **Hardware Bridge & Real-Time HIL** | Yes | No | No | No | No |

# Hardware-in-the-Loop Validation

In addition to pure simulations, imct includes an end-to-end Hardware-in-the-Loop (HIL) bridge and system identification pipeline for real physical systems, demonstrated on a 2-Link Direct-Drive Robotic Manipulator. The bridge supports sub-millisecond serial packet encoding, velocity numerical differentiation with Savitzky-Golay filtering, torque saturation clamping, and emergency heartbeat watchdogs.

# Documentation and Reproducibility

imct is accompanied by comprehensive, publication-grade hosted documentation featuring:
- A structured Decision Guide mapping control objectives to optimal algorithms.
- Full analytical equations rendered with MathJax.
- Auto-generated API reference using mkdocstrings covering all classes and methods.
- Interactive Jupyter tutorials (
otebooks/01_tour.ipynb).
- Complete experiment scorecards with downloadable artifact plots.

# Acknowledgements

The author thanks the open-source control and machine learning communities for developing foundational tools, including NumPy, SciPy, PyTorch, Matplotlib, and SymPy, upon which imct is built.

# References
