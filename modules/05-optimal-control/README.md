# Module 05: Optimal Control

> **Curriculum Path:** [AI Meets Control Theory](../../README.md) $\rightarrow$ **Module 05: Optimal Control**

---

## Overview

Optimal control formalizes the trade-off between tracking precision and actuator energy consumption. By minimizing mathematically defined cost functionals subject to system dynamics and physical constraints, optimal control derives feedback policies with provable optimality and robustness properties.

```
+-----------------------------------------------------------------------------------+
|                                OPTIMAL CONTROL                                    |
+-------------------------+-------------------------+-------------------------------+
| Problem Formulations    | Calculus of Variations  | Linear Quadratic Regulator    |
| - Bolza / Lagrange Cost | - Pontryagin's Min (PMP)| - Cont. Riccati Eq. (CARE)    |
| - Energy vs Precision   | - Hamiltonian & Costates| - Disc. Riccati Eq. (DARE)    |
| - Constraints (u ∈ U)   | - Transversality        | - Optimal Feedback u = -Kx    |
+-------------------------+-------------------------+-------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|                 ROBUSTNESS GUARANTEES & MODEL PREDICTIVE CONTROL                  |
| - Kalman's Inequality: Guaranteed [1/2, ∞) Gain Margin & ≥60° Phase Margin        |
| - Model Predictive Control (MPC): Receding Horizon & Constrained QP Optimization  |
+-----------------------------------------------------------------------------------+
```

---

## Module Topics

1. **[01. Optimal Control Problem Formulation](01-optimal-control-formulation.md)**
   - General functional optimization: Bolza, Lagrange, and Mayer cost formulations.
   - Standard control objectives: minimum time, minimum energy, and quadratic regulation.
   - State and control constraint sets ($x \in \mathcal{X}, u \in \mathcal{U}$).

2. **[02. Calculus of Variations & Pontryagin's Minimum Principle](02-calculus-of-variations-and-pmp.md)**
   - Variational calculus on state trajectories.
   - The Hamiltonian function $\mathcal{H}(x, u, \lambda, t)$.
   - Pontryagin's Minimum Principle (PMP) necessary conditions: state ODE, costate ODE, Hamiltonian minimization, and transversality.
   - Bang-bang control for control-affine actuator limits.

3. **[03. The Linear Quadratic Regulator (LQR)](03-linear-quadratic-regulator.md)**
   - Infinite-horizon continuous-time LQR problem: $\min \int_0^\infty (x^T Q x + u^T R u) dt$.
   - Derivation of the Continuous Algebraic Riccati Equation (CARE): $A^T P + P A - P B R^{-1} B^T P + Q = 0$.
   - Solving CARE via Hamiltonian matrix eigenspaces / Schur decomposition.
   - Discrete-Time LQR and the Discrete Algebraic Riccati Equation (DARE).

4. **[04. LQR Tuning, Bryson's Rule & Robustness Guarantees](04-lqr-tuning-and-robustness.md)**
   - Bryson's rule for normalizing state and control weights.
   - Kalman's return difference inequality in the frequency domain.
   - Remarkable guaranteed stability margins of continuous LQR: gain margin $[1/2, \infty)$ ($-6\text{ dB}$ to $+\infty\text{ dB}$) and phase margin $\ge 60^\circ$.
   - Doyle's 1978 warning: loss of robustness in output-feedback LQG.

5. **[05. Introduction to Model Predictive Control (MPC)](05-model-predictive-control-intro.md)**
   - The Receding Horizon Control (RHC) principle.
   - Formulating finite-horizon optimal control as Quadratic Programming (QP).
   - Handling hard actuator saturation, slew rate limits, and state constraints.
   - Terminal costs and terminal invariant sets for recursive feasibility and stability.
   - Comparative evaluation: PID vs. State Feedback vs. LQR vs. MPC vs. RL.
