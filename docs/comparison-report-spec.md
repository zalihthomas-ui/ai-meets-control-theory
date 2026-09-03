# Benchmark Comparison Report Specification

> **AI Meets Control Theory — Benchmark Standardization**  
> Version 1.0 | Maintainer: docs & design (`lava`)

---

## 1. Purpose & Scope

Every computational benchmark in this repository compares $N$ candidate controllers on a given dynamical system under identical initial conditions, setpoints, disturbances, and noise profiles.

To ensure consistency, scientific rigor, and direct comparability across classical, optimal, and learned control methods, **all benchmarks must emit reports conforming to this specification**.

A complete benchmark comparison consists of:
1. **Standard Metrics Table** (Markdown + committed `.csv`)
2. **Canonical 4-Panel Figure** (`.png` + `.svg`)
3. **Robustness / Parameter Sweep Plot** (if applicable)
4. **Structured Executive Summary & Engineering Verdict**

---

## 2. Standard Performance Metrics

All metrics must be evaluated over the closed-loop simulation trajectory $t \in [0, T]$ sampled at time step $\Delta t$, with reference $r(t)$, output $y(t)$, tracking error $e(t) = r(t) - y(t)$, and control input $u(t) \in \mathbb{R}^m$.

| Metric Symbol | Name | Exact Mathematical Definition | Units | Description / Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **$t_r$** | Rise Time | $t_{90\%} - t_{10\%}$ where $t_\alpha = \min \{t \mid y(t) \ge \alpha \cdot r_{target}\}$ | $\text{s}$ | Time required for output to rise from 10% to 90% of a step input. |
| **$t_s$** | Settling Time | $\min \{ t^* \mid \|y(t) - r(t)\| \le 0.02 \cdot \|r_{target}\| \quad \forall t \ge t^* \}$ | $\text{s}$ | Time after which output remains within $\pm 2\%$ error band. |
| **$M_p$** | Peak Overshoot | $\frac{\max_{t \ge 0} (y(t) - r_{target})}{r_{target}} \times 100\%$ | $\%$ | Maximum percentage excursion beyond setpoint. Zero if monotonic. |
| **$e_{ss}$** | Steady-State Error | $\lim_{t \to T} \|r(t) - y(t)\|$ (evaluated as mean over last 5% of $T$) | system units | Residual tracking error at the end of the simulation horizon. |
| **$\text{RMSE}$** | RMS Tracking Error | $\sqrt{\frac{1}{T} \int_0^T \|r(t) - y(t)\|_2^2 \, dt} \approx \sqrt{\frac{1}{N} \sum_{k=1}^N \|e_k\|_2^2}$ | system units | Global tracking precision across the full simulation horizon. |
| **$\text{IAE}$** | Integral Absolute Error | $\int_0^T \|r(t) - y(t)\|_1 \, dt \approx \sum_{k=1}^N \|e_k\|_1 \Delta t$ | $\text{unit}\cdot\text{s}$ | Cumulative transient error penalty (linear weighting). |
| **$\text{ITAE}$** | Integral Time-Abs Error | $\int_0^T t \cdot \|r(t) - y(t)\|_1 \, dt \approx \sum_{k=1}^N t_k \|e_k\|_1 \Delta t$ | $\text{unit}\cdot\text{s}^2$ | Heavily penalizes persistent or slow-decaying oscillations. |
| **$E_u$** | Control Effort (Energy) | $\int_0^T \|u(t)\|_2^2 \, dt \approx \sum_{k=1}^N \|u_k\|_2^2 \Delta t$ | $\text{u-unit}^2\cdot\text{s}$ | Total actuator energy consumed during maneuver. |
| **$u_{\max}$** | Peak Control Input | $\max_{t \in [0, T]} \|u(t)\|_\infty$ | $\text{u-unit}$ | Peak instantaneous actuator demand. |
| **$\text{Sat}\%$** | Saturation Duty Cycle | $\frac{1}{T} \int_0^T \mathbb{I}(\|u(t)\| \ge u_{\mathrm{limit}}) \, dt \times 100\%$ | $\%$ | Percentage of simulation time the actuator was clamped at its limit. |
| **$\Phi_m$** | Phase Margin | $\Phi_m = 180^\circ + \angle L(j\omega_{gc})$ where $\|L(j\omega_{gc})\| = 1$ | $\text{deg}$ | Robustness to unmodeled pure time delays: $\tau_{\max} \approx \Phi_m / \omega_{gc}$. |
| **$G_m$** | Gain Margin | $G_m = -20 \log_{10} \|L(j\omega_{pc})\|$ where $\angle L(j\omega_{pc}) = -180^\circ$ | $\text{dB}$ | Gain increase factor before closed-loop instability occurs. |

---

## 3. Standard Comparison Table Layout

Every benchmark suite must output both a Markdown table (in `README.md` / report) and an identical `metrics.csv` table.

### 3.1 Markdown Schema

```markdown
| Controller | Rise Time $t_r$ [s] | Settling Time $t_s$ [s] | Overshoot $M_p$ [%] | Steady Error $e_{ss}$ | RMSE | Energy $E_u$ [N²s] | Peak $u_{\max}$ [N] | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PID (Tuned)** | 0.42 | 1.15 | 8.4% | 0.001 | 0.142 | 34.2 | 18.5 | 0.0% | Stable |
| **State Feedback** | 0.38 | 0.85 | 4.1% | 0.000 | 0.098 | 48.1 | 24.0 | 2.1% | Stable |
| **LQR (Optimal)** | 0.45 | 0.92 | 1.8% | 0.000 | 0.104 | 28.6 | 14.8 | 0.0% | Stable |
| **MPC (Constrained)**| 0.46 | 0.94 | 0.0% | 0.000 | 0.108 | 27.4 | 12.0 | 4.8% | Stable (Constrained) |
| **RL (PPO)** | 0.51 | 1.48 | 12.6% | 0.014 | 0.198 | 62.8 | 29.5 | 8.4% | Stable |
```

---

## 4. Standard Figure Specification

### 4.1 Canonical 4-Panel Benchmark Figure

Every benchmark comparison script must save a figure named `comparison_benchmark.png` (and `.svg`) structured as follows:

```
+------------------------------------+------------------------------------+
|  Top-Left (a): State Tracking      |  Top-Right (b): Control Effort     |
|  - y(t) for all controllers        |  - u(t) for all controllers        |
|  - r(t) reference (grey dashed)    |  - Actuator bounds (red dashed)    |
|  - +/-2% settling band (grey fill) |  - Saturation events annotated     |
+------------------------------------+------------------------------------+
|  Bottom-Left (c): Error Trajectory |  Bottom-Right (d): Phase Portrait  |
|  - e(t) = r(t) - y(t)              |  - State x1 vs State x2            |
|  - Zero baseline (grey dashed)     |  - Equilibrium marked (star/dot)   |
|  - Transient decay visualizer      |  - Vector flow direction arrows    |
+------------------------------------+------------------------------------+
```

### 4.2 Disturbance Rejection & Robustness Sweep Panel

When evaluating robustness (e.g., parameter variation $\Delta m, \Delta k$ or step disturbance $d(t)$):
- **X-axis:** Parameter variation ratio (e.g., $m / m_0 \in [0.5, 2.0]$) or disturbance magnitude.
- **Y-axis (Left):** Settling time $t_s$ or RMSE.
- **Y-axis (Right):** Total control energy $E_u$.
- **Shaded Area:** Unstable / divergent operating region.

---

## 5. Standard Report Markdown Template

When producing a benchmark artifact or documentation report, follow this standard structure:

```markdown
# Benchmark Report: [System Name] Multi-Controller Evaluation

**Date:** YYYY-MM-DD  
**System:** `LinearSystem` / `MassSpringDamper` / `Pendulum` / `CartPole`  
**Simulation Horizon:** $T = 10.0\text{ s}$, $\Delta t = 0.001\text{ s}$, Integrator: `RK4`  
**Initial State:** $x_0 = [x_{0,1}, x_{0,2}, \dots]^T$  
**Target Reference:** $r(t) = \dots$  

---

## 1. Executive Summary

- **Top Performer (Precision):** [Controller Name] achieved lowest RMSE ([Value]).
- **Top Performer (Energy):** [Controller Name] consumed least energy ([Value]).
- **Robustness Winner:** [Controller Name] tolerated largest parameter variation ($\pm XX\%$).
- **Key Trade-off:** [Summary of engineering compromise].

---

## 2. Controller Formulations & Gains

| Controller | Formulation / Architecture | Gains / Hyperparameters |
| :--- | :--- | :--- |
| **PID** | $u(t) = K_p e + K_i \int e + K_d \dot{e}$ with anti-windup | $K_p = 100, K_i = 10, K_d = 20, \tau_f = 0.01$ |
| **LQR** | $u(t) = -K x$ | $Q = \text{diag}([10, 1]), R = 0.1 \implies K = [9.48, 3.12]$ |

---

## 3. Quantitative Comparison Table

[Embed Standard Metrics Table Here]

---

## 4. Visual Analysis

![Benchmark Comparison](figures/comparison_benchmark.png)

### Observations:
1. **Transient Response:** [Analysis of rise time, overshoot, damping].
2. **Actuator Demand:** [Analysis of peak control, chattering, saturation].
3. **Phase-Plane Trajectory:** [Analysis of convergence to equilibrium, limit cycles].

---

## 5. Robustness & Disturbance Analysis

![Robustness Sweep](figures/robustness_sweep.png)

- **Input Disturbance Rejection:** [Step force disturbance test results].
- **Parameter Sensitivity:** [Behavior when mass / stiffness varies by $\pm 50\%$].

---

## 6. Engineering Recommendation

> **When to use what for this system:**
> - Choose **PID** if state is not fully measurable and simplicity is paramount.
> - Choose **LQR** if full state is available and optimal energy-precision trade-off is required.
> - Choose **MPC** if actuator limits or state constraints are frequently active.
> - Choose **RL / Hybrid** if dynamics contain unknown nonlinearities.
```

---

## 6. Verification Checklist for Benchmark Authors

Before merging any benchmark results into the repository:
- [ ] All metrics computed with identical definitions via `aimct.benchmarks.metrics`.
- [ ] Same initial conditions $x_0$ and time vector $t$ used across all controllers.
- [ ] Actuator saturation limits enforced identically across all controllers.
- [ ] Both `metrics.csv` and `comparison_benchmark.png` committed under experiment folder.
- [ ] Report follows the standardized structure above with clear engineering takeaways.
