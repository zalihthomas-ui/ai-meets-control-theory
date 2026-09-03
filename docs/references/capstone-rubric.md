# Capstone Evaluation Rubric & Five-Way Controller Bake-Off Specification

This document defines the official scoring rubric, quantitative evaluation criteria, and baseline normalization protocols for the **Module 09 Capstone: Five-Way Quadrotor Obstacle & Trajectory Tracking Bake-Off** (C9.4).

---

## 1. Executive Summary & Benchmark Overview

The Capstone challenge tests 5 distinct control and learning paradigms on the **Planar Quadrotor** (`aimct.systems.PlanarQuadrotor`) tracking an aggressive dynamic figure-8 trajectory through an environment with static and moving spatial obstacles under crosswind disturbances:

```
                                [ TRAJECTORY TRACKING TASK ]
                              Dynamic Figure-8: (y_ref(t), z_ref(t))
                                              |
      +---------------------------------------+---------------------------------------+
      |                   |                   |                   |                   |
      v                   v                   v                   v                   v
[ CONTROLLER 1 ]   [ CONTROLLER 2 ]   [ CONTROLLER 3 ]   [ CONTROLLER 4 ]   [ CONTROLLER 5 ]
  LQR + Flatness     Linear MPC         Learned Sampling     Deep RL (PPO)      Shielded Hybrid
  Feedforward        Preview QP         NMPC (Diff-Model)    Policy (MLP)       RL + CBF Safety
```

---

## 2. Evaluation Dimensions & Mathematical Formulation

The Capstone rubric evaluates every submission across 5 weighted dimensions:

### 2.1 Trajectory Tracking Precision ($w_1 = 0.40$)
Measures the root-mean-square position tracking error in the vertical plane:
$$J_{\text{pos}} = \text{RMSE}_{\text{pos}} = \sqrt{\frac{1}{T}\int_0^T \left[ (y(t) - y_{\text{ref}}(t))^2 + (z(t) - z_{\text{ref}}(t))^2 \right] dt}$$
Ratio:
$$r_{\text{pos}} = \min\left(10.0, \ \frac{J_{\text{pos}}}{J_{\text{pos}}^{\text{base}}}\right)$$

### 2.2 Control Energy & Battery Efficiency ($w_2 = 0.20$)
Measures total thrust and torque effort exerted across both rotors:
$$J_{\text{energy}} = \int_0^T \left( u_1(t)^2 + u_2(t)^2 \right) dt$$
Ratio:
$$r_{\text{energy}} = \min\left(10.0, \ \frac{J_{\text{energy}}}{J_{\text{energy}}^{\text{base}}}\right)$$

### 2.3 Actuator Smoothness & Slew Rate ($w_3 = 0.10$)
Penalizes high-frequency motor chattering and jerk to prevent actuator wear:
$$J_{\text{slew}} = \int_0^T \left( \left(\frac{du_1}{dt}\right)^2 + \left(\frac{du_2}{dt}\right)^2 \right) dt$$
Ratio:
$$r_{\text{slew}} = \min\left(10.0, \ \frac{J_{\text{slew}}}{J_{\text{slew}}^{\text{base}}}\right)$$

### 2.4 Safety, Obstacle Clearance & Constraints ($w_4 = 0.15$)
Evaluates proximity to obstacles and actuator saturation:
$$J_{\text{safety}} = C_{\text{obs}} + 0.5 \cdot \text{SatDutyCycle}$$
where $C_{\text{obs}}$ is the integrated obstacle boundary violation penalty:
$$C_{\text{obs}} = \int_0^T \sum_{k=1}^K \max\left( 0.0, \ R_{\text{safe}, k} - \|p(t) - p_{\text{obs}, k}\|_2 \right)^2 dt$$
Ratio:
$$r_{\text{safety}} = \min\left(10.0, \ \frac{J_{\text{safety}}}{J_{\text{safety}}^{\text{base}}}\right)$$

### 2.5 Environmental Robustness ($w_5 = 0.15$)
Evaluated across a 20-seed Monte Carlo sweep under continuous horizontal crosswinds ($w_y \sim \mathcal{N}(0, 2.0\text{ m/s})$) and $\pm 20\%$ payload mass shifts ($m \in [0.8 m_0, 1.2 m_0]$):
$$S_{\text{robust}} = \max\left( 0.20, \ 1.0 - \frac{\bar{J}_{\text{wind}} - J_{\text{nominal}}}{J_{\text{nominal}}} \right)$$

---

## 3. Composite Capstone Score

The final composite score $S_{\text{capstone}} \in [0, 100]$ is computed as:

$$S_{\text{capstone}} = 100 \cdot \exp\left( - \left[ 0.40 r_{\text{pos}} + 0.20 r_{\text{energy}} + 0.10 r_{\text{slew}} + 0.15 r_{\text{safety}} \right] \right) \times S_{\text{robust}} \times \mathbb{I}(\text{No Hard Disqualifications})$$

### 3.1 Hard Disqualification Rules (Instant Score = 0.0)
A controller is immediately disqualified (`DISQUALIFIED`) if:
1. **Crash / Hard Obstacle Penetration**: Distance to obstacle center falls below physical obstacle radius: $\|p(t) - p_{\text{obs}, k}\|_2 < R_{\text{body}} + R_{\text{obs}, k}$.
2. **Computational Timeout**:
   - Mean step latency $\bar{t}_{\text{step}} > 2.0\text{ ms}$ ($500\text{ Hz}$ control loop deadline).
   - Maximum step latency (WCET) $t_{\max} > 10.0\text{ ms}$.
3. **State Divergence / Non-Finite Actions**: State explodes or control output contains NaN/Inf.

---

## 4. Five-Way Controller Comparison Matrix

| Controller Entry | Paradigm | Expected Tracking Precision | Obstacle Avoidance Mechanism | Robustness to Wind | Typical Step Latency |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **1. LQR + Flatness** | Classical Optimal + Analytical Feedforward | High ($\text{RMSE} < 0.05\text{ m}$) | None (Relies on collision-free reference) | Medium (Integral bias) | **$< 0.02\text{ ms}$** (Ultra-fast) |
| **2. Linear MPC Preview** | Receding-Horizon QP Optimization | Very High ($\text{RMSE} < 0.03\text{ m}$) | Soft hyperplane constraints | High (Anticipative) | **$\approx 0.35\text{ ms}$** (Real-time) |
| **3. Learned Sampling MPC** | Differentiable Model + MPPI Sampling | Medium ($\text{RMSE} \approx 0.08\text{ m}$) | Non-convex cost field exploration | Very High (Online adaptation) | **$\approx 1.80\text{ ms}$** (GPU/Parallel) |
| **4. Deep RL Policy (PPO)** | Model-Free Neural Policy | Moderate ($\text{RMSE} \approx 0.12\text{ m}$) | Reward penalties (Empirical) | Low-Medium (Sim-to-real gap) | **$\approx 0.08\text{ ms}$** (Neural forward) |
| **5. Shielded Hybrid RL** | RL Policy + Real-Time CBF Safety Filter | High ($\text{RMSE} \approx 0.06\text{ m}$) | **Guaranteed CBF-QP Invariance** | High (Certified bounds) | **$\approx 0.25\text{ ms}$** (Real-time QP) |

---

## 5. Canonical Baseline Values ($J^{\text{base}}$)

Baseline normalization costs derived from canonical LQR + Flatness tracking on nominal figure-8 trajectory ($T = 15.0\text{ s}$):
- $J_{\text{pos}}^{\text{base}} = 0.050\text{ m}$
- $J_{\text{energy}}^{\text{base}} = 150.0\text{ N}^2\cdot\text{s}$
- $J_{\text{slew}}^{\text{base}} = 2500.0\text{ N}^2/\text{s}$
- $J_{\text{safety}}^{\text{base}} = 0.10$
