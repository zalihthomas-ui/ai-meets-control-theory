# Module 03: Classical Control

> **Curriculum Path:** [AI Meets Control Theory](../../README.md) $\rightarrow$ **Module 03: Classical Control**

---

## Overview

Classical control focuses on feedback loops in the frequency and complex Laplace domains ($s$-plane). This module covers fundamental feedback principles, rigorous mathematical design of Proportional-Integral-Derivative (PID) controllers (including real-world derivative filtering and anti-windup clamping), stability analysis via Root Locus, Bode, and Nyquist criteria, and a complete worked example stabilizing an open-loop unstable system.

```
+-----------------------------------------------------------------------------------+
|                              CLASSICAL CONTROL                                    |
+-------------------------+-------------------------+-------------------------------+
| Feedback Principles     | PID Controller Design   | Frequency & Stability         |
| - Loop Sensitivity S, T | - P, I, D Terms         | - Root Locus Evan's Rules     |
| - Algebraic S + T = 1   | - Filtered Derivative   | - Bode & Nyquist Criteria     |
| - Waterbed Effect       | - Anti-Windup Clamping  | - Gain & Phase Margins        |
+-------------------------+-------------------------+-------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|              WORKED EXAMPLE: STABILIZING AN UNSTABLE SYSTEM                       |
| - Analytical proof why P fails, PD stabilizes, and PID rejects steady disturbance |
| - Full simulation, anti-windup verification, and fundamental limits               |
+-----------------------------------------------------------------------------------+
```

---

## Module Topics

1. **[01. Feedback Principles & Fundamental Limitations](01-feedback-principles.md)**
   - The canonical single-loop feedback architecture.
   - Sensitivity function $S(s)$ and Complementary Sensitivity function $T(s)$.
   - The fundamental trade-off: $S(s) + T(s) = 1$.
   - The Bode Sensitivity Integral (Waterbed Effect) for unstable systems.

2. **[02. PID Controller Architecture & Implementation](02-pid-controller-design.md)**
   - Mathematical formulation: Proportional, Integral, and Derivative actions.
   - Practical filtered derivative: $D(s) = \frac{K_d s}{\tau_f s + 1}$ ($N$-parameter form).
   - Integrator windup mechanisms and anti-windup schemes (conditional clamping and back-calculation).
   - 2-DOF setpoint weighting (eliminating derivative kick).
   - Tuning heuristics: Ziegler-Nichols, Cohen-Coon, and pole placement.

3. **[03. Stability & Frequency Response Analysis](03-stability-and-frequency-response.md)**
   - Characteristic equation $1 + L(s) = 0$ and $s$-plane pole trajectories.
   - Routh-Hurwitz stability criterion.
   - Root Locus design rules.
   - Bode diagrams (magnitude/phase) and asymptotic approximations.
   - Nyquist stability criterion and encirclements ($N = Z - P$).
   - Quantitative robustness: Gain Margin $G_m$, Phase Margin $\Phi_m$, and Delay Margin $\tau_{\max}$.

4. **[04. Worked Example: Stabilizing an Unstable System](04-worked-example-unstable-system.md)**
   - Complete analytical and numerical demonstration on $G(s) = \frac{1}{s^2 - \omega_0^2}$.
   - Mathematical proof: why P-only fails (pure oscillatory center) and why derivative action is strictly necessary.
   - Gain derivation, derivative filter tuning, and anti-windup clamping under torque saturation.
   - Full simulation results, step response, load disturbance rejection, and phase portraits.
   - Critical appraisal: Where PID reaches its fundamental limitations.
