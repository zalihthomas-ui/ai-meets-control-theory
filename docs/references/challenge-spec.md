# Intelligent Control Challenge — Evaluation Specification & Benchmark Standard

This document establishes the official specification, evaluation metrics, tracks, and submission protocols for the **Intelligent Control Challenge (ICC)** within the **AI Meets Control Theory (AIMCT)** ecosystem.

---

## 1. Challenge Vision & Purpose

Modern autonomous systems increasingly demand control policies that are **accurate, robust to unmodeled dynamics, sample-efficient, computationally lightweight, and mathematically safe**. 

The Intelligent Control Challenge is a rigorous, open benchmark designed to answer:
> *Which control paradigm (Classical, Optimal/MPC, Reinforcement Learning, or Hybrid) provides the optimal trade-off across precision, robustness, computational overhead, and safety guarantees on complex dynamical systems?*

Competitors and researchers evaluate their controllers against standardized, reproducible dynamical benchmarks under identical environmental conditions and disturbance profiles.

---

## 2. Challenge Tracks

The challenge consists of four progressive tracks:

```
[ Track 1: Precision Tracking ] ===> [ Track 2: Nonlinear Underactuated ]
                                                  |
[ Track 4: Safe Black-Box Adaptation ] <=== [ Track 3: Parametric Uncertainty ]
```

### Track 1: Precision Regulation & Trajectory Tracking (L1)
- **Focus**: High-bandwidth tracking, zero steady-state error, minimal control effort, sensor noise rejection.
- **Plants**: Multi-degree-of-freedom Mass-Spring-Damper, DC Motor Position/Speed.
- **Conditions**: Known nominal plant dynamics, Gaussian measurement noise $\mathcal{N}(0, \sigma^2)$, continuous and step reference trajectories.

### Track 2: Nonlinear Underactuated Swing-Up & Balance (L2)
- **Focus**: Large-angle non-convex optimization, underactuation handling, hard actuator saturation.
- **Plants**: Inverted Pendulum, Cart-Pole System, Acrobot.
- **Conditions**: Starting from stable downward hanging rest ($x_0 = [0, 0, \pi, 0]^T$), swing up to vertical inverted equilibrium within bounded rail limits ($|x| \le 2.4\text{ m}$) with strictly bounded force ($|F| \le 20.0\text{ N}$).

### Track 3: Robustness & Parametric Uncertainty (L1 & L2)
- **Focus**: Controller resilience against unmodeled dynamics, parameter variations, and external impulse shocks.
- **Conditions**:
  - Model parameters (mass $m$, length $l$, friction $c$) sampled uniformly from nominal $\pm 30\%$.
  - Unmodeled actuator lag: first-order delay $\frac{1}{\tau_a s + 1}$ with $\tau_a = 0.05\text{ s}$.
  - Random impulse disturbance forces $F_{dist} \sim \text{Laplace}(0, b_d)$ injected at unpredictable time intervals.

### Track 4: Safe Black-Box Adaptive Control (L2+)
- **Focus**: Learning and adapting to an unknown dynamical plant online without violating critical safety envelopes.
- **Conditions**:
  - The controller receives only the state dimension, action dimension, and sensor observations (no explicit $A, B$ or differential equations provided).
  - Hard state constraints (e.g. cart position $|x| \le 2.4\text{ m}$, max angular velocity $|\dot{\theta}| \le 10\text{ rad/s}$). Any single violation results in **immediate disqualification / zero score for that episode**.
  - Strict interaction budget: maximum $10^4$ timesteps of online exploration.

---

## 3. Quantitative Evaluation Metrics

All candidate controllers are evaluated across five orthogonal dimensions:

```
                +-------------------------+
                |    1. Performance       |
                |    (ITAE, RMSE, ts, Mp) |
                +------------+------------+
                             |
+---------------------+      |      +---------------------+
| 5. Compute Cost     |<-----+----->|  2. Control Effort  |
| (Latency, RAM, Ops) |      |      |  (Energy, Slew rate)|
+---------------------+      |      +---------------------+
                             |
+---------------------+      |      +---------------------+
| 4. Robustness       |<-----+----->|  3. Safety & Bounds |
| (Degradation Ratio) |             |  (Violations, Margin)|
+---------------------+             +---------------------+
```

### 3.1 Tracking Performance Index ($J_{\text{perf}}$)
1. **ITAE (Integral of Time-weighted Absolute Error)**:
   $$J_{\text{ITAE}} = \int_0^T t \cdot \|y(t) - r(t)\|_1 \, dt$$
   *Penalizes sustained long-term tracking errors much more heavily than initial transient errors.*
2. **RMSE (Root Mean Square Tracking Error)**:
   $$J_{\text{RMSE}} = \sqrt{\frac{1}{T} \int_0^T \|y(t) - r(t)\|_2^2 \, dt}$$
3. **Transient Metrics**:
   - Rise Time ($t_r$): Time to reach $90\%$ of step setpoint.
   - Settling Time ($t_s$): Time to permanently remain within $\pm 2\%$ error band.
   - Peak Overshoot ($M_p\%$): $\frac{y_{\max} - r_{\text{step}}}{r_{\text{step}}} \times 100\%$.

### 3.2 Actuator & Effort Index ($J_{\text{effort}}$)
1. **Total Control Energy**:
   $$J_{\text{energy}} = \int_0^T \|u(t)\|_2^2 \, dt$$
2. **Actuator Slew Rate (Smoothness / Jerk Penalty)**:
   $$J_{\text{slew}} = \int_0^T \left\| \frac{du(t)}{dt} \right\|_2^2 \, dt$$
   *High-frequency chattering or bang-bang switching is heavily penalized to protect physical actuators.*
3. **Saturation Fraction**:
   $$\Phi_{\text{sat}} = \frac{1}{T} \int_0^T \mathbb{I}\left( \|u(t)\|_\infty \ge 0.99 \, u_{\max} \right) \, dt$$

### 3.3 Safety & Constraint Preservation ($J_{\text{safe}}$)
1. **State Constraint Violation Penalty**:
   $$C_{\text{viol}} = \sum_{k=1}^N \max\left(0, \|x_k\|_{\text{safe, normalized}} - 1.0\right)^2$$
2. **Hard Failure Condition**: If any state crosses the catastrophic collapse boundary (e.g. cart crashes into rail limits $x > x_{rail}$), the episode terminates with status `FAILED` and score $0$.

### 3.4 Robustness Degradation Score ($S_{\text{robust}}$)
Let $\bar{J}_{\text{nominal}}$ be the nominal cost and $\bar{J}_{\text{perturbed}}$ be the mean cost over a 50-seed Monte Carlo sweep of $\pm 30\%$ parametric shifts:
$$S_{\text{robust}} = \max\left( 0.0, \ 1.0 - \frac{\bar{J}_{\text{perturbed}} - \bar{J}_{\text{nominal}}}{\bar{J}_{\text{nominal}}} \right)$$
A controller whose performance degrades by $>100\%$ receives $S_{\text{robust}} = 0$.

### 3.5 Computational Efficiency Index ($J_{\text{comp}}$)
1. **Mean Step Latency ($\bar{t}_{\text{step}}$)**: Must not exceed the real-time deadline $t_{\text{deadline}} = 0.5 \cdot \Delta t$ (e.g., $\le 0.5\text{ ms}$ for a $1000\text{ Hz}$ control loop).
2. **Max Step Latency ($t_{\max}$)**: Maximum worst-case execution time (WCET) observed across all steps.
3. **Parameter Footprint**: Number of active parameters / weights in the policy.

---

## 4. Standardized Composite Scoring Formulation

For any evaluation run, the overall challenge score $S \in [0, 100]$ is computed as:

$$S = 100 \cdot \exp\left( - \left[ w_1 \frac{J_{\text{ITAE}}}{J_{\text{ITAE}}^{\text{base}}} + w_2 \frac{J_{\text{energy}}}{J_{\text{energy}}^{\text{base}}} + w_3 \frac{J_{\text{slew}}}{J_{\text{slew}}^{\text{base}}} \right] \right) \times S_{\text{robust}} \times \mathbb{I}(\text{No Safety Failures})$$

**Standard Default Weights**:
- $w_1 = 0.50$ (Tracking accuracy)
- $w_2 = 0.30$ (Energy efficiency)
- $w_3 = 0.20$ (Actuator smoothness)
- Baseline values $J^{\text{base}}$ are defined by the canonical reference LQR controller in `docs/references/benchmark-systems.md`.

---

## 5. Submission Protocol & Controller API

All challenge entries must inherit from `aimct.controllers.BaseController` and implement the standard Python interface:

```python
from abc import ABC, abstractmethod
import numpy as np

class ChallengeController(ABC):
    """Standardized controller interface for Intelligent Control Challenge."""
    
    @abstractmethod
    def __init__(self, spec: dict) -> None:
        """Initialize controller with state dimension, action bounds, and metadata."""
        self.state_dim: int = spec["state_dim"]
        self.action_dim: int = spec["action_dim"]
        self.action_limit: np.ndarray = np.array(spec["action_limit"])
        self.dt: float = spec["dt"]

    @abstractmethod
    def reset(self, target_state: np.ndarray) -> None:
        """Reset internal integrators, estimators, or recurrent policy states."""
        pass

    @abstractmethod
    def compute_action(self, observation: np.ndarray, t: float) -> np.ndarray:
        """
        Compute control action u(t).
        
        Args:
            observation: Current state estimate or sensor readings [state_dim].
            t: Current simulation time in seconds.
            
        Returns:
            action: Control input array [action_dim] within [-action_limit, +action_limit].
        """
        pass

    def observe_feedback(self, next_obs: np.ndarray, reward_or_cost: float, done: bool, info: dict) -> None:
        """Optional online learning/adaptation hook called after environment step."""
        pass

    def get_diagnostics(self) -> dict:
        """Optional telemetry hook (e.g. estimated parameters, Lyapunov value, QP iterations)."""
        return {}
```

---

## 6. Evaluation Harness Workflow

The automated evaluation runner executes the following pipeline:

```
+--------------------------------------------------------------------------------+
|                           ICC BENCHMARK RUNNER                                 |
|                                                                                |
| 1. Load Submission Model -> Verify API compliance                              |
| 2. Run Nominal Suite (10 Fixed Seeds) -> Record baseline metrics               |
| 3. Run Robustness Sweep (50 Perturbed Seeds) -> Record degradation             |
| 4. Run Disturbance Rejection Suite (Step Gusts + Noise)                         |
| 5. Measure Step Wall-Clock Latencies (Microsecond Resolution)                 |
| 6. Check Safety Envelopes -> Immediate zero score if breached                   |
| 7. Compute Composite Score (Eq. 1) -> Export JSON + Markdown + SVGs            |
+--------------------------------------------------------------------------------+
```

### Output Artifacts
Every challenge run emits:
1. `results.json`: Machine-readable raw metrics, per-seed scores, timing statistics.
2. `summary_table.md`: Formatted comparison table vs. standard baselines (PID, LQR, MPC, PPO).
3. `trajectory_plots.svg`: Time-series plots of state trajectories, tracking error, control effort, and phase portraits.
