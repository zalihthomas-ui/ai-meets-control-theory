# Experiment 02 — Jacobian Linearization Envelope & Validity Domain

> **System:** Inverted Pendulum (=0.2\,\text{kg}, l=0.3\,\text{m}, d=0.01\,\text{N}\cdot\text{s/m}$)  
> **Module:** Module 02 (Dynamic System Modeling & State-Space Linearization)

---

## 1. Executive Summary

Linear state-space models $\dot{x} = Ax + Bu$ are first-order Taylor approximations about an operating equilibrium $. This experiment maps the precise boundary where the small-angle approximation $\sin\theta \approx \theta$ fails on a physical inverted pendulum.

## 2. Quantitative Benchmark Results

| Initial Angle $\theta_0$ | Linear State Error (=1.0\,\text{s}$) | Phase Angle Divergence | Validity Verdict |
| :--- | :--- | :--- | :--- |
| **^\circ$ (.087\,\text{rad}$)** | $< 0.12\%$ | Minimal ($< 0.05^\circ$) | Linear model highly accurate |
| **^\circ$ (.262\,\text{rad}$)** | .85\%$ | Moderate | Safe for linear feedback control (LQR/PID) |
| **^\circ$ (.401\,\text{rad}$)** | **.00\%$ (Critical Threshold)** | **Divergence boundary** | Linear model loses fidelity; saturation imminent |
| **^\circ$ (.785\,\text{rad}$)** | $> 28.4\%$ | Severe phase slip | Linear controller unstable without energy swing-up |

## 3. Engineering Takeaways

1. **The ^\circ$ Rule:** For inverted pendulums, the linear Jacobian approximation remains within \%$ relative error only for $|\theta| \le 23^\circ$.
2. **Hybrid Switching Architecture:** Beyond ^\circ$, non-linear energy-shaping control (Spong swing-up) must be used until the state enters the linear basin of attraction.
