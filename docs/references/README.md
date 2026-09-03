# AIMCT Reference Library & Specifications

This directory contains research foundations, canonical parameter sets, framework surveys, and challenge specifications for the **AI Meets Control Theory (AIMCT)** project.

---

## Documents

1. **[Canonical Benchmark Systems (`benchmark-systems.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/benchmark-systems.md)**
   - Analytical differential equations and state-space formulations ($A, B, C, D$).
   - Literature standard parameters for Level 1 (Mass-Spring-Damper, Inverted MSD, DC Motor) and Level 2 (Nonlinear Pendulum, Cart-Pole, Duffing & Van der Pol Oscillators).
   - Numerical reference controller gains (PID, Pole Placement, LQR), Riccati solutions, closed-loop poles, and step response performance metrics.
   - Ground-truth verification protocols for tests in `tests/`.

2. **[Cart-Pole LQR Reference Specification (`cartpole-lqr-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/cartpole-lqr-reference.md)**
   - Canonical physical parameters matching `CartPole` ($M=1.0, m=0.1, l=0.5$).
   - Analytical and numerical $(A, B)$ linearized matrices matching `CartPole.linearize()`.
   - 3 canonical LQR tuning configurations (Standard Balanced, Aggressive Angle, Soft Energy-Saving) with exact continuous Riccati solutions $P$, optimal feedback gains $K$, and closed-loop eigenvalues.

3. **[Swing-Up & Basin of Attraction Reference (`swingup-and-basin.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/swingup-and-basin.md)**
   - Energy-shaping swing-up formulations (Åström-Furuta for Simple Pendulum, Mark Spong Partial Feedback Linearization for Cart-Pole).
   - Hybrid switching state machine and hysteresis boundary conditions.
   - Quantitative Lyapunov basin-of-attraction envelopes and recovery bounds across LQR tunings.

4. **[State Estimation, Observers & Kalman Filtering (`observers-kalman-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/observers-kalman-reference.md)**
   - Observability analysis on Cart-Pole across 4 sensor configurations (full-state, cart position only, pole angle only, dual encoders).
   - Deterministic Luenberger observer design and Separation Principle proof.
   - Continuous Filter Algebraic Riccati Equation (FARE) and Discrete Kalman Filter (DKF) predict-update algorithm.
   - Golden numerical test fixtures for `src/aimct/estimation/` (Kalman gain matrix $L$, error poles, covariance $\Sigma$).

5. **[Prior-Art & Framework Survey (`prior-art.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/prior-art.md)**
   - Analysis of existing open tools: `python-control`, `do-mpc`, `Gymnasium`, `Stable-Baselines3`, `Drake`, `PySINDy`, `torchdiffeq`.
   - Clear delineation of what AIMCT implements from scratch vs. what is wrapped or verified against external libraries.
   - Core architectural patterns for dynamical systems, controllers, and unified benchmark evaluation.

6. **[Intelligent Control Challenge Spec (`challenge-spec.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/challenge-spec.md)**
   - Official 4-track challenge specification (Precision Tracking, Nonlinear Underactuated, Parametric Robustness, Safe Black-Box Adaptation).
   - Quantitative evaluation formulas: ITAE, RMSE, control energy $\int u^2 dt$, slew rate / jerk $\int \dot{u}^2 dt$, and safety barrier penalties.
   - Standardized `ChallengeController` Python interface and automated scoring harness.
