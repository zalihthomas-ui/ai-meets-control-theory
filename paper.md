---
title: 'AIMCT: A Benchmark Suite and Unified Framework Bridging AI and Control Theory'
tags:
  - Python
  - control theory
  - reinforcement learning
  - dynamical systems
  - model predictive control
  - physics-informed neural networks
  - state estimation
  - hardware-in-the-loop
authors:
  - name: Zalih Thomas
    orcid: 0009-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher / AI Meets Control Theory Initiative, Istanbul, Turkey
    index: 1
date: 7 September 2026
bibliography: paper.bib
---

# Summary

Modern autonomous cyber-physical systems increasingly operate at the intersection of classical feedback control theory and data-driven machine learning. While classical methods—such as Proportional-Integral-Derivative (PID) control, Linear Quadratic Regulators (LQR), $H_\infty$ loop shaping, and Model Predictive Control (MPC)—provide rigorous mathematical guarantees of stability, robustness margins, and constraint satisfaction, they rely heavily on accurate physical models and struggle with unmodelled complex kinematics [@astrom2010feedback; @rawlings2017model; @slotine1991applied]. Conversely, modern machine learning paradigms—such as deep Reinforcement Learning (RL), Physics-Informed Neural Networks (PINNs), and Neural Ordinary Differential Equations (Neural ODEs)—excel at extracting representations from trajectory data but lack formal stability certificates, fail catastrophically under distribution shift, and exhibit low sample efficiency [@schulman2017proximal; @haarnoja2018soft; @raissi2019physics; @chen2018neural; @brunton2022koopman].

`aimct` (*AI Meets Control Theory*) is an open-source Python library designed to bridge this foundational divide. It provides a standardized, mathematically rigorous testbed comprising **41 empirical benchmark experiments** spanning 12 physical systems (from harmonic oscillators and underactuated acrobats to dynamic multi-agent swarms and high-speed road vehicles). `aimct` unifies classical state-space models, numerical optimal control, statistical state estimators, system identification algorithms, neural dynamics models, and deep RL agents under a cohesive, zero-friction object-oriented interface.

# Statement of Need

Despite substantial interest in combining machine learning with control theory, existing software ecosystems remain deeply fragmented:
1. **Classical Control Libraries:** Packages such as `python-control` [@fuller2016pythoncontrol] provide classical frequency-domain and linear state-space tools, but lack native integration with neural network architectures, trajectory optimization algorithms, and RL environments.
2. **Optimization and MPC Frameworks:** Frameworks such as `do-mpc` [@lucia2017dompc] and `CasADi` [@andersson2019casadi] excel at nonlinear optimal control and symbolic automatic differentiation, but carry steep learning curves and do not directly offer plug-and-play RL benchmarks, disturbance observers, or standardized comparative metrics.
3. **Reinforcement Learning Suites:** Libraries like `Stable-Baselines3` [@raffin2021stablebaselines3] and `Gymnasium` [@brockman2016openai] offer state-of-the-art RL algorithms, but treat environments as black-box Markov Decision Processes (MDPs), completely discarding known physics, energy conservation laws, controllability matrices, and Lyapunov stability proofs.

Researchers and practitioners seeking to benchmark whether a Deep RL policy out-performs a tuned LQR or Nonlinear MPC on a physical system are often forced to write ad-hoc wrappers, disparate simulation loops, and custom metric extractors.

`aimct` addresses this gap by establishing:
- **Unified Abstractions:** A single `DynamicalSystem` API that supports ODE integration (Runge-Kutta 4th order), Gymnasium environment exposure, continuous Jacobian linearization ($A, B, C, D$), and Hamiltonian energy tracking ($E = T + V$).
- **Multi-System Benchmark Suite:** 41 systematically formulated experiments across 9 modules (Linear Foundations, State-Space & Observers, Underactuated Agility, Constrained MPC, Obstacle Avoidance, Trajectory Optimization, Robust & $H_\infty$ Control, Multi-Body Vehicles & Hardware Bridges, and Safe RL / Imitation).
- **Rigorous Evaluation Metrics:** Standardized measurement of Settling Time ($t_s$), Overshoot ($M_p$), Integrated Absolute Error (IAE), Control Effort ($\|u\|_2$), Robustness Margins (gain/phase/disk margins and structured singular value $\mu$), and Hardware Timing Jitter.
- **Hardware-in-the-Loop (HIL) & Code Generation:** Modular transport bridges and zero-allocation C99/MicroPython code generation for direct microcontroller deployment (STM32, ESP32, RP2040) with real-time safety watchdogs.

# The 41-Experiment Empirical Evidence Base

The core of `aimct` is its comprehensive suite of 41 benchmark experiments, categorized into nine distinct research modules:

| Module | Experiments | Key Focus Systems & Methods |
| :--- | :--- | :--- |
| **1. Classical Foundations** | Exp 01–03 | Mass-Spring-Damper, Inverted Pendulum (RK4 vs Euler, Jacobian validity, PID anti-windup) |
| **2. State-Space & Estimation** | Exp 04–06, 15, 16, 38, 41 | Cart-Pole, Quadrotor, Two-Tank (LQR vs Pole Placement, EKF, UKF, MHE, Particle Filter) |
| **3. Underactuated Systems** | Exp 07, 28, 33 | Cart-Pole, Furuta Rotary Pendulum, Ball and Beam (Spong energy swing-up, Åström-Furuta) |
| **4. Constrained & Nonlinear MPC** | Exp 08, 14, 24, 26, 30, 37 | Cart-Pole, Quadrotor, Two-Tank (Active-set QP, iLQR/RTI-NMPC, sampling MPPI, Tube MPC) |
| **5. Obstacle Avoidance** | Exp 20, 25 | Quadrotor, Differential-Drive (Non-convex geometric keep-out corridors) |
| **6. Trajectory Optimization** | Exp 32 | Double Pendulum (Direct Collocation Hermite-Simpson NLP vs Shooting) |
| **7. Robust & Frequency Domain** | Exp 17, 23, 34, 35, 40 | Two-Mass Resonator, Quadrotor (MRAC, DOB wind rejection, $H_\infty$ loop shaping, $\mu$-analysis) |
| **8. Multi-Body & Hardware** | Exp 22, 27, 36, 39 | TurtleBot3, Dynamic Bicycle, 2-Link Arm, Swarm (Pacejka slip, HIL latency, consensus) |
| **9. Data-Driven & Safe RL** | Exp 09–12, 18, 19, 21, 29, 31 | SysID (DMDc, SINDy), CBF Safety Shields, DAgger, PPO, Soft Actor-Critic (SAC) |

Each experiment includes automated validation tests, convergence checks, phase portrait generators, and standardized Markdown reports comparing classical vs. AI approaches.

# Software Architecture and Interface Design

The architecture of `aimct` is structured around modular, loosely coupled Python subpackages:

```
src/aimct/
├── systems/       # ODE definitions, energy equations, and Gym wrappers
├── controllers/   # PID, LQR, MPC, iLQR, MRAC, H-infinity, Formation Consensus
├── planning/      # Direct Collocation, Trajectory Optimizers
├── estimation/    # Kalman Filter, EKF, UKF, Particle Filter, MHE
├── sysid/         # Least Squares, DMDc, Manipulator Regressor SysID
├── ml/            # PINNs, Neural ODEs, Deep Koopman operators (PyTorch)
├── rl/            # PPO, SAC implementations with Gymnasium integration
├── hybrid/        # Control Barrier Function (CBF) real-time safety shields
├── hil/           # Real-time hardware emulation, quantization, transport delay
├── deploy/        # Standalone C99 and MicroPython zero-alloc code generators
├── robust/        # Structured singular value (mu-analysis), Disk Margins, D-K
└── benchmarks/    # Standardized experiment test runners and scorecards
```

A core principle of `aimct` is mathematical transparency. For instance, simulating a closed-loop controller on an inverted pendulum requires fewer than 10 lines of clean, self-documenting code:

```python
import aimct
from aimct.systems import InvertedPendulum
from aimct.controllers import LQR

system = InvertedPendulum(mass=0.2, length=0.3, damping=0.01)
A, B, C, D = system.linearize()
controller = LQR(A, B, Q=[[10.0, 0.0], [0.0, 1.0]], R=[[0.1]])
result = aimct.simulate(system, controller, x0=[0.2, 0.0], t_span=(0.0, 5.0))

print(f"Settling time: {result.settling_time:.3f} s, IAE: {result.iae:.3f}")
```

# Comparison to Existing Software

| Feature | `aimct` | `python-control` | `do-mpc` | `CasADi` | `Stable-Baselines3` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical State-Space (LQR/PID/CARE)** | **Yes** | Yes | Partial | No | No |
| **Constrained Linear & Nonlinear MPC** | **Yes** | No | Yes | Yes (Core) | No |
| **State Estimation (KF/EKF/UKF/PF/MHE)** | **Yes** | Partial | Yes | No | No |
| **Robust $H_\infty$ & $\mu$-Analysis** | **Yes** | Partial | No | No | No |
| **Deep RL Algorithms (PPO/SAC)** | **Yes** | No | No | No | Yes |
| **Real-Time CBF Safety Shields** | **Yes** | No | No | No | No |
| **Multi-Agent Consensus Swarms** | **Yes** | No | No | No | No |
| **41-Experiment Benchmark Suite** | **Yes** | No | No | No | No |
| **Zero-Alloc C99 Code Generation** | **Yes** | No | No | No | No |

# Documentation and Reproducibility

`aimct` is accompanied by comprehensive, publication-grade hosted documentation featuring:
- A structured Decision Guide mapping control objectives to optimal algorithms.
- 7 core mathematical concept chapters explaining fundamental feedback dynamics.
- Full analytical equations rendered with MathJax.
- Auto-generated API reference using `mkdocstrings` covering all classes and methods.
- Complete experiment scorecards with downloadable artifact plots and CSV logs.

# Acknowledgements

The author thanks the open-source control and machine learning communities for developing foundational tools, including NumPy, SciPy, PyTorch, Matplotlib, and SymPy, upon which `aimct` is built.

# References
