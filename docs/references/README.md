# AIMCT Reference Library & Specifications

This directory contains research foundations, canonical parameter sets, framework surveys, and challenge specifications for the **AI Meets Control Theory (AIMCT)** project.

---

## Documents

1. **[Canonical Benchmark Systems (`benchmark-systems.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/benchmark-systems.md)**
   - Analytical differential equations and state-space formulations ($A, B, C, D$).
   - Literature standard parameters for Level 1 (Mass-Spring-Damper, Inverted MSD, DC Motor) and Level 2 (Nonlinear Pendulum, Cart-Pole, Duffing & Van der Pol Oscillators).
   - Numerical reference controller gains (PID, Pole Placement, LQR), Riccati solutions, closed-loop poles, and step response performance metrics.
   - Ground-truth verification protocols for tests in `tests/`.

2. **[Prior-Art & Framework Survey (`prior-art.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/prior-art.md)**
   - Analysis of existing open tools: `python-control`, `do-mpc`, `Gymnasium`, `Stable-Baselines3`, `Drake`, `PySINDy`, `torchdiffeq`.
   - Clear delineation of what AIMCT implements from scratch vs. what is wrapped or verified against external libraries.
   - Core architectural patterns for dynamical systems, controllers, and unified benchmark evaluation.

3. **[Intelligent Control Challenge Spec (`challenge-spec.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/challenge-spec.md)**
   - Official 4-track challenge specification (Precision Tracking, Nonlinear Underactuated, Parametric Robustness, Safe Black-Box Adaptation).
   - Quantitative evaluation formulas: ITAE, RMSE, control energy $\int u^2 dt$, slew rate / jerk $\int \dot{u}^2 dt$, and safety barrier penalties.
   - Standardized `ChallengeController` Python interface and automated scoring harness.
