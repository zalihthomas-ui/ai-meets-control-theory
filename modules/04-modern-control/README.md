# Module 04: Modern Control

> **Curriculum Path:** [AI Meets Control Theory](../../README.md) $\rightarrow$ **Module 04: Modern Control**

---

## Overview

Modern control operates directly in the time-domain state space ($\mathbb{R}^n$). By exploiting the internal state vector, modern methods naturally handle multi-input multi-output (MIMO) systems, underactuated robotics, state estimation from noisy sensors, and optimal pole assignment.

```
+-----------------------------------------------------------------------------------+
|                                MODERN CONTROL                                     |
+-------------------------+-------------------------+-------------------------------+
| State-Space Foundations | Controllability & Obs.  | Feedback & Estimation         |
| - Coordinates z = Tx    | - Kalman Rank & PBH     | - State Feedback u = -Kx      |
| - Canonical Realizations| - Gramian Metrics       | - Ackermann & Bass-Gura       |
| - Modal Decompositions  | - Mathematical Duality  | - Luenberger Observer (A-LC)  |
+-------------------------+-------------------------+-------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|               SEPARATION PRINCIPLE & STOCHASTIC ESTIMATION                        |
| - Independent design of Controller σ(A-BK) and Observer σ(A-LC)                   |
| - Continuous & Discrete Kalman Filters (Algebraic Riccati Covariance Equation)    |
+-----------------------------------------------------------------------------------+
```

---

## Module Topics

1. **[01. State-Space Foundations & Canonical Forms](01-state-space-foundations.md)**
   - Matrix differential equations $\dot{x} = Ax + Bu, y = Cx + Du$.
   - Coordinate transformations $z = Tx$ and structural invariants.
   - Controllable Canonical Form (CCF) and Observable Canonical Form (OCF).

2. **[02. Controllability, Observability & Duality](02-controllability-and-observability.md)**
   - Controllability definition and the Kalman Controllability Matrix $\mathcal{C}$.
   - Popov-Belevitch-Hautus (PBH) rank tests for controllability and stabilizability.
   - Controllability Gramian $W_c$ and minimum energy steering.
   - Observability definition, Kalman matrix $\mathcal{O}$, and PBH detectability test.
   - Duality between actuation and sensing.

3. **[03. Full-State Feedback & Pole Placement](03-state-feedback-pole-placement.md)**
   - State feedback law $u = -Kx + k_r r$.
   - Closed-loop eigenvalue assignment: $\det(sI - (A - BK)) = \Delta_{\text{des}}(s)$.
   - Exact derivation of Ackermann's formula and Bass-Gura method.
   - Feedforward reference gain $k_r$ for zero steady-state tracking error.
   - Control effort vs. bandwidth trade-offs.

4. **[04. State Observers & The Separation Principle](04-state-observers-and-separation-principle.md)**
   - Full-state Luenberger observer structure $\dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})$.
   - Error dynamics $\dot{e} = (A - LC)e$ and dual observer pole assignment.
   - Rigorous mathematical proof of the Separation Principle.
   - Practical observer tuning rules (the $3\times$ to $5\times$ bandwidth rule).

5. **[05. Introduction to Kalman Filtering](05-kalman-filter-intro.md)**
   - State estimation under Gaussian process noise $w \sim \mathcal{N}(0, Q_w)$ and sensor noise $v \sim \mathcal{N}(0, R_v)$.
   - Continuous-time Kalman-Bucy filter and the Algebraic Riccati Covariance Equation.
   - Discrete-Time Kalman Filter predict-update recursions.
   - The Linear Quadratic Gaussian (LQG) framework.
