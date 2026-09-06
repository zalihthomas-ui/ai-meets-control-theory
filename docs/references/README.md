# AIMCT Reference Library & Specifications

This directory contains research foundations, canonical parameter sets, framework surveys, and challenge specifications for the **AI Meets Control Theory (AIMCT)** project.

---

## Documents

1. **[Canonical Benchmark Systems (`benchmark-systems.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/benchmark-systems.md)**
   - Analytical differential equations and state-space formulations ($A, B, C, D$).
   - Literature standard parameters for Level 1 (Mass-Spring-Damper, Inverted MSD, DC Motor) and Level 2 (Nonlinear Pendulum, Cart-Pole, Duffing & Van der Pol Oscillators).
   - Numerical reference controller gains (PID, Pole Placement, LQR), Riccati solutions, closed-loop poles, and step response performance metrics.
   - Ground-truth verification protocols for tests in `tests/`.

2. **[Robotics Benchmark Systems Reference (`robotics-systems-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/robotics-systems-reference.md)**
   - Differential-Drive Mobile Robot (`DifferentialDriveRobot`): unicycle kinematics, motor velocity lag ($\tau_m=0.05\text{ s}$), wheelbase geometry ($r=0.033\text{ m}, W=0.160\text{ m}$), velocity bounds ($v_{\max}=0.22\text{ m/s}, \omega_{\max}=2.84\text{ rad/s}$), and Brockett nonholonomic analysis.
   - Two-Link Planar Robotic Manipulator (`TwoLinkArm`): complete Euler-Lagrange equations $M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) + F_v \dot{q} = \tau$, Coriolis Christoffel symbols, passivity skew-symmetry, forward kinematics, and geometric Jacobian.

3. **[Furuta Pendulum Reference Specification (`furuta-pendulum-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/furuta-pendulum-reference.md)**
   - Rotary Inverted Pendulum (Quanser QUBE-Servo 2 standard): 2-DOF underactuated rotary mechanics.
   - Analytical Euler-Lagrange equations $M(\alpha)\ddot{q} + C(\alpha, \dot{q})\dot{q} + G(\alpha) + D\dot{q} = \tau$, analytical $(A, B)$ linearization about upright equilibrium, and Åström-Furuta energy shaping swing-up law.

4. **[Coupled Two-Tank Liquid Level Reference (`two-tank-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/two-tank-reference.md)**
   - Process control fluid hydraulics (Quanser Coupled Tanks standard): 2-state coupled nonlinear capacity lag.
   - Torricelli square-root discharge law, operating equilibrium calculations, fluid flow resistance $R_{12}, R_2$, Jacobian state-space linearization $(A, B)$, and interactive capacity analysis.

5. **[Cart-Pole LQR Reference Specification (`cartpole-lqr-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/cartpole-lqr-reference.md)**
   - Canonical physical parameters matching `CartPole` ($M=1.0, m=0.1, l=0.5$).
   - Analytical and numerical $(A, B)$ linearized matrices matching `CartPole.linearize()`.
   - 3 canonical LQR tuning configurations (Standard Balanced, Aggressive Angle, Soft Energy-Saving) with exact continuous Riccati solutions $P$, optimal feedback gains $K$, and closed-loop eigenvalues.

6. **[Swing-Up & Basin of Attraction Reference (`swingup-and-basin.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/swingup-and-basin.md)**
   - Energy-shaping swing-up formulations (Åström-Furuta for Simple Pendulum, Mark Spong Partial Feedback Linearization for Cart-Pole).
   - Hybrid switching state machine and hysteresis boundary conditions.
   - Quantitative Lyapunov basin-of-attraction envelopes and recovery bounds across LQR tunings.

7. **[State Estimation, Observers & Kalman Filtering (`observers-kalman-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/observers-kalman-reference.md)**
   - Observability analysis on Cart-Pole across 4 sensor configurations (full-state, cart position only, pole angle only, dual encoders).
   - Deterministic Luenberger observer design and Separation Principle proof.
   - Continuous Filter Algebraic Riccati Equation (FARE) and Discrete Kalman Filter (DKF) predict-update algorithm.
   - Golden numerical test fixtures for `src/aimct/estimation/` (Kalman gain matrix $L$, error poles, covariance $\Sigma$).

8. **[Prior-Art & Framework Survey (`prior-art.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/prior-art.md)**
   - Analysis of existing open tools: `python-control`, `do-mpc`, `Gymnasium`, `Stable-Baselines3`, `Drake`, `PySINDy`, `torchdiffeq`.
   - Clear delineation of what AIMCT implements from scratch vs. what is wrapped or verified against external libraries.
   - Core architectural patterns for dynamical systems, controllers, and unified benchmark evaluation.

9. **[Intelligent Control Challenge Spec (`challenge-spec.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/challenge-spec.md)**
   - Official 4-track challenge specification (Precision Tracking, Nonlinear Underactuated, Parametric Robustness, Safe Black-Box Adaptation).
   - Quantitative evaluation formulas: ITAE, RMSE, control energy $\int u^2 dt$, slew rate / jerk $\int \dot{u}^2 dt$, and safety barrier penalties.
   - Ratio capping ($10.0$) and $S_{\text{robust}}$ floor ($0.20$) calibrations.

10. **[Capstone Evaluation Rubric (`capstone-rubric.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/capstone-rubric.md)**
    - Capstone 21 Five-Way Quadrotor Trajectory & Obstacle Avoidance Bake-Off specification.
    - 5-dimensional evaluation: Tracking RMSE ($w_1=0.40$), Control Energy ($w_2=0.20$), Slew Rate ($w_3=0.10$), Obstacle & Safety Violations ($w_4=0.15$), and Crosswind Robustness ($w_5=0.15$).

11. **[Research Papers & Foundations Survey (`research-papers-survey.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/research-papers-survey.md)**
    - Comprehensive literature synthesis of 10 foundational open-access papers across RL continuous control foundations, parsimonious data-driven dynamics (SINDy, Neural ODEs, Koopman), and safe control (CBFs, Differentiable MPC, MPPI).

12. **[Ball and Beam Reference Specification (`ball-and-beam-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/ball-and-beam-reference.md)**
    - Canonical Quanser Ball and Beam apparatus standard: underactuated, relative degree 4 nonlinear rolling mechanics ($m = 0.064\text{ kg}, M = 0.20\text{ kg}, L = 0.425\text{ m}, J = 3.010 \times 10^{-3}\text{ kg}\cdot\text{m}^2, \tau_{\max} = 1.50\text{ N}\cdot\text{m}$).
    - Complete Euler-Lagrange equations of motion, analytical $(A, B)$ state-space linearization about horizontal origin, open-loop instability ($\lambda = +4.123\text{ rad/s}$), and Cascade PID vs. PFL vs. LQR vs. Constrained MPC benchmark formulations.

13. **[Disturbance-Observer (DOB) Control Reference Specification (`disturbance-observer-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/disturbance-observer-reference.md)**
    - Disturbance Observer (DOB) two-degree-of-freedom architecture, causal rational Q-filter design, relative degree matching, and Small-Gain robust stability conditions ($\|W_m(s)Q(s)\|_\infty < 1$).
    - Matched vs. unmatched disturbance mechanics on underactuated quadrotors, direct feedforward cancellation vs. virtual tilt reallocation ($\theta_{\text{virtual}} = \hat{d}_x / g$), and proof of DOB superiority over classical integral action lag.

14. **[H-infinity Mixed-Sensitivity Reference Specification (`hinf-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/hinf-reference.md)**
    - The $H_\infty$ norm, the S/KS/T mixed-sensitivity objective $\|[W_S S;\ W_{KS} KS;\ W_T T]\|_\infty$, and first-order weight selection ($W_S$, $W_{KS}$, $W_T$ Skogestad-Postlethwaite forms).
    - Generalised-plant construction $P = \mathcal{F}(G, W_S, W_{KS}, W_T)$, the two DGKF Hamiltonian Riccati equations with the $\gamma^{-2}$ term, the sub-optimality conditions ($X_\infty \succeq 0$, $Y_\infty \succeq 0$, $\rho(X_\infty Y_\infty) < \gamma^2$), $\gamma$-bisection, and the central-controller formula.
    - $H_\infty$ vs. LQG on an unmodelled lightly-damped high-frequency mode: why a $W_T$ that rolls off past crossover preserves gain / phase / disk margins that LQG loses (Experiment 35).

15. **[Moving-Horizon Estimation (MHE) Reference Specification (`mhe-reference.md`)](file:///C:/Users/salih/Desktop/ai-meets-control-theory/docs/references/mhe-reference.md)**
    - Optimization-based state estimation on sliding finite window $N$, handling hard state & disturbance inequality constraints ($x_{\text{lo}} \le x \le x_{\text{hi}}$).
    - Arrival cost approximation via EKF covariance propagation, stability under uniform observability, and analysis of why MHE outperforms Gaussian filters (EKF, UKF) on non-Lipschitz / hard-boundary physical systems (e.g. liquid level floor $h \ge 0$).



