# Prior-Art Survey: Open-Source Control & Reinforcement Learning Frameworks

This survey analyzes the existing open-source ecosystem spanning classical control, optimal control, model predictive control (MPC), system identification, and reinforcement learning (RL). It articulates what **AI Meets Control Theory (AIMCT)** borrows, cross-checks against, wraps, and—crucially—builds from scratch.

---

## 1. Executive Summary & Ecosystem Gap

The computational control and autonomy landscape is deeply fragmented into two isolated paradigms:

```
+------------------------------------+         +------------------------------------+
|       CLASSICAL & MODERN           |         |         REINFORCEMENT LEARNING     |
|         CONTROL THEORY             |         |               & AI                 |
|                                    |         |                                    |
| - python-control, do-mpc, Drake    |         | - Gymnasium, Stable-Baselines3     |
| - Rigorous ODEs, transfer funcs    |  GAP    | - Black-box environments (step)    |
| - Stability proofs, margins (Bode) | <=====> | - Reward hacking, sample hungry    |
| - Exact mathematical guarantees    |         | - No Lyapunov/stability bounds     |
| - Rigid to unmodeled dynamics      |         | - Little classical cross-checking  |
+------------------------------------+         +------------------------------------+
```

### The AIMCT Mission
AIMCT bridges this divide. We provide an open-source, mathematically transparent virtual laboratory where classical controllers (PID, LQR, H-infinity), modern estimators (Kalman, Observers), optimal solvers (MPC), and machine learning (Neural ODEs, SINDy, PPO/SAC, Physics-Informed ML) operate on **identical dynamical systems, with identical integrators, evaluated under identical benchmark metrics.**

---

## 2. Framework-by-Framework Survey

### 2.1 `python-control` (Control Systems Library)
- **Primary Domain**: Classical and Modern Linear/LTI Control Theory.
- **Backend**: NumPy, SciPy, Slycot (FORTRAN SLICOT wrapper).
- **Core Strengths**:
  - Gold standard for transfer functions, state-space representations, Bode plots, Nyquist diagrams, Root Locus.
  - Robust continuous/discrete Algebraic Riccati Equation (ARE/DARE) solvers and Lyapunov solvers.
  - Linear state feedback and Kalman filter synthesis (`place`, `lqr`, `lqe`).
- **Limitations for AIMCT**:
  - Heavily centered on linear/LTI systems; nonlinear simulation and MPC support is minimal.
  - Slycot dependency can cause compilation friction on Windows/ARM platforms.
  - No native reinforcement learning interfaces or data-driven / neural dynamics hooks.
- **AIMCT Strategy**:
  - **Do NOT depend on Slycot in core**.
  - Write pure NumPy/SciPy solvers from scratch for Phase 0–1 (e.g., Hamiltonian/Schur Riccati solver for LQR).
  - Use `python-control` as an optional test-suite oracle to verify the numerical accuracy of our from-scratch algorithms in `tests/`.

---

### 2.2 `do-mpc` (Model Predictive Control in Python)
- **Primary Domain**: Nonlinear Model Predictive Control (NMPC), Moving Horizon Estimation (MHE), Robust Multi-Stage MPC.
- **Backend**: CasADi, Ipopt, Bonmin, NumPy.
- **Core Strengths**:
  - Industrial-grade formulation of constrained optimal control and trajectory generation.
  - Symbolic auto-differentiation via CasADi and C-code compilation.
  - Multi-stage robust formulation accounting for parametric uncertainty trees.
- **Limitations for AIMCT**:
  - High barrier to entry: heavy abstraction layer and steep CasADi symbolic DSL learning curve.
  - "Black box" solver experience: hides the underlying quadratic programming (QP) / active-set mechanics from students.
  - Unsuited for quick educational experiments or lightweight hybrid RL-PID controllers.
- **AIMCT Strategy**:
  - In Phase 0–2, implement transparent, from-scratch QP-based linear MPC and convex shooting methods.
  - Introduce `do-mpc` / `CasADi` in Phase 2 capstones as the advanced, high-performance reference implementation for complex constrained robotics.

---

### 2.3 `Gymnasium` (Farama Foundation / Classic Control)
- **Primary Domain**: Standardized RL Environment API (`step()`, `reset()`, `render()`).
- **Classic Control Environments**: `CartPole-v1`, `Pendulum-v1`, `MountainCarContinuous-v0`, `Acrobat-v1`.
- **Core Strengths**:
  - Universal standard interface across the machine learning community.
  - Huge ecosystem of compatible algorithms and benchmarking suites.
- **Limitations for AIMCT**:
  - Simplistic Euler integrators with low physical fidelity (e.g., `CartPole-v1` omits friction and pole inertia, uses coarse discrete step forward-Euler).
  - Truncated observations and ad-hoc reward functions that promote reward hacking rather than physical stability.
  - Lacks state-space matrices ($A, B$), linearization routines, controllability/observability checks, or frequency domain diagnostics.
- **AIMCT Strategy**:
  - Build physically rigorous dynamical systems in `src/aimct/systems/` with continuous ODEs and fixed-step RK4 integrators.
  - Provide a lightweight Gymnasium-compatible wrapper (`AIMCTEnv(gym.Env)`) so any external RL agent (Stable-Baselines3, CleanRL) can train directly on AIMCT systems.

---

### 2.4 `Stable-Baselines3` (SB3) & CleanRL
- **Primary Domain**: Deep Reinforcement Learning Algorithms (PyTorch).
- **Core Strengths**:
  - Reliable, battle-tested implementations of PPO, SAC, TD3, DQN, A2C.
  - Excellent documentation, logging, and evaluation callbacks.
- **Limitations for AIMCT**:
  - Pure RL focus: unaware of system poles, energy functions, Lyapunov criteria, or control effort penalties unless explicitly engineered into the reward function.
- **AIMCT Strategy**:
  - Use SB3 as the reference RL baseline engine for Phase 3 and Phase 4.
  - Compare SB3 agents directly against LQR and MPC on identical initial condition distributions and disturbance profiles.

---

### 2.5 Supplementary Frameworks

| Tool | Focus | Role in AIMCT |
| :--- | :--- | :--- |
| **Drake (MIT)** | Underactuated robotics, trajectory optimization, Lyapunov certificates | Conceptual reference for trajectory optimization and sums-of-squares (SOS) verification. |
| **PySINDy** | Sparse Identification of Nonlinear Dynamics | Reference library for Phase 3 data-driven system identification (discovering governing ODEs from time series). |
| **torchdiffeq** | Neural Ordinary Differential Equations | Backend tool for Phase 3 continuous-time neural ODEs and learned dynamics. |
| **ControlSystems.jl (Julia)** | Modern control toolbox in Julia | Benchmark for algorithm speed and clean mathematical design. |

---

## 3. Comprehensive Feature & Decision Matrix

| Dimension | python-control | do-mpc | Gymnasium | Stable-Baselines3 | **AIMCT (This Project)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | Classical/LTI Control | Nonlinear MPC/MHE | RL Environments | RL Algorithms | **Unified Classical + AI Control** |
| **Mathematical Transparency** | Medium (Slycot/C) | Low (CasADi C-bindings) | Low (Simplified physics) | Medium (PyTorch) | **High (Scratch math + verified libraries)** |
| **Integrator Fidelity** | Variable (`scipy.integrate`) | Collocation / IDAS | Low (Coarse Euler) | N/A | **High (RK4 fixed-step + SciPy adaptive)** |
| **Linearization & Pole Analysis** | Built-in | None | None | None | **Built-in (`system.linearize()`, Bode, Nyquist)** |
| **Controller Baselines Included** | PID, LQR, Pole Place | NMPC | Random / None | PPO, SAC, TD3 | **PID, Pole Place, LQR, MPC, RL, Hybrids** |
| **Standardized Benchmarking** | None | Ad-hoc | Reward curve only | Reward curve only | **Unified Control Effort, ITAE, Settling, Robustness** |
| **Dependencies** | NumPy, SciPy, Slycot | CasADi, Ipopt | NumPy | PyTorch | **NumPy, SciPy, Matplotlib (Zero C-compiler needed for core)** |

---

## 4. Architectural Decisions for AIMCT

### Decision 1: Pure-Python Mathematical Core
- **Choice**: All foundational controllers (`PID`, `StateFeedback`, `LQR`), estimators (`KalmanFilter`, `LuenbergerObserver`), and numerical integrators (`rk4_step`) will be written in pure NumPy/SciPy without mandatory C/Fortran extension dependencies.
- **Rationale**: Guarantees instant zero-friction installation on any platform and allows students/engineers to inspect every line of the mathematical implementation.

### Decision 2: Common `System` Interface
Every system in `src/aimct/systems/` must expose:
```python
class DynamicalSystem(ABC):
    def dynamics(self, state: np.ndarray, action: np.ndarray, t: float) -> np.ndarray:
        """Returns x_dot = f(x, u, t)."""
        ...
    
    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns analytical or numerical (A, B) matrices around operating point."""
        ...
        
    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """Advances state by dt using RK4 integration."""
        ...
```

### Decision 3: Common `Controller` Interface
Every controller (whether a 3-line PID, an LQR matrix multiply, an MPC QP solver, or a 100,000-parameter PyTorch neural policy) conforms to:
```python
class BaseController(ABC):
    def reset(self) -> None:
        """Resets internal integrator states, history buffers, or memory."""
        ...
        
    def compute_action(self, obs: np.ndarray, t: float) -> np.ndarray:
        """Computes control action u(t) given current observation."""
        ...
```

### Decision 4: Unified Comparison Harness (`aimct.benchmarks`)
- Rather than evaluating RL on "average episode reward" and classical control on "settling time", AIMCT's benchmark harness runs both through identical standardized test fixtures:
  - Setpoint tracking & regulation errors: ITAE, ISE, RMSE.
  - Transient response: Rise time ($t_r$), settling time ($t_s$), overshoot ($M_p\%$).
  - Actuator efficiency: Control effort $\int u(t)^2 dt$, slew rate / jerk $\int \dot{u}(t)^2 dt$, saturation duration.
  - Stability & safety: Maximum state excursion, constraint violation count.
  - Robustness: Monte Carlo sweep under parametric uncertainty ($\pm 30\%$) and noise injection.
