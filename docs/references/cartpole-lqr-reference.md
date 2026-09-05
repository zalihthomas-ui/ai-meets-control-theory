# Cart-Pole LQR Reference Specification & Tuning Guide

This document contains canonical parameters, continuous and discrete linearized models, and golden LQR tuning sets for the **Cart-Pole (Inverted Pendulum on a Cart)** dynamical system (`aimct.systems.CartPole`).

---

## 1. System Physical Parameters (`CartPole`)

| Parameter | Symbol | Canonical Value | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| Cart Mass | $M$ (`mc`) | $1.0$ | $\text{kg}$ | Mass of the linear carriage |
| Pole Mass | $m$ (`mp`) | $0.1$ | $\text{kg}$ | Mass of the inverted pendulum rod |
| Distance to CoM | $l$ (`length`) | $0.5$ | $\text{m}$ | Half-length of rod (uniform rod length $L = 1.0\text{ m}$) |
| Gravity | $g$ | $9.81$ | $\text{m/s}^2$ | Gravitational acceleration |
| Moment of Inertia | $I$ | $\frac{1}{3} m l^2 = 0.008333$ | $\text{kg}\cdot\text{m}^2$ | Uniform rod inertia around center of mass |
| Rail Boundary | $x_{\max}$ | $\pm 2.4$ | $\text{m}$ | Physical track travel limit |
| Max Force | $F_{\max}$ | $\pm 20.0$ | $\text{N}$ | Saturated actuator force limit |

---

## 2. Linearized State-Space Formulation Around Upright Equilibrium

Operating point: $x_{\text{eq}} = [0, 0, 0, 0]^T$ ($x = 0, \dot{x} = 0, \theta = 0, \dot{\theta} = 0$), $u_{\text{eq}} = 0\text{ N}$.

$$\Delta_0 = l \cdot \left( \frac{4}{3} - \frac{m}{M + m} \right) = 0.5 \cdot \left( \frac{4}{3} - \frac{0.1}{1.1} \right) = 0.621212\text{ m}$$

$$\dot{x}(t) = A x(t) + B u(t)$$

$$A = \begin{bmatrix} 
0.0 & 1.0 & 0.0 & 0.0 \\ 
0.0 & 0.0 & -0.71780 & 0.0 \\ 
0.0 & 0.0 & 0.0 & 1.0 \\ 
0.0 & 0.0 & 15.79171 & 0.0 
\end{bmatrix}, \quad 
B = \begin{bmatrix} 
0.0 \\ 
0.97561 \\ 
0.0 \\ 
-1.46341 
\end{bmatrix}$$

### Open-Loop Pole Locations
- $\lambda_1, \lambda_2 = 0.0, 0.0$ (Double integrator along horizontal track)
- $\lambda_3 = +3.9739\text{ rad/s}$ (**Unstable Right-Half Plane Pole** — inverted pendulum saddle)
- $\lambda_4 = -3.9739\text{ rad/s}$ (Stable Left-Half Plane Pole)
- Controllability Matrix $\mathcal{C} = [B, AB, A^2B, A^3B]$: $\text{rank}(\mathcal{C}) = 4$ (Full Rank).

---

## 3. Canonical LQR Weighting Configurations & Golden Gains

### Tuning Set 1: Standard Balanced Regulation (Recommended Baseline)
- **Design Intent**: Balanced cart position tracking and fast pole stabilization without excessive control authority demand.
- **State Cost**: $Q = \text{diag}([10.0, 1.0, 100.0, 10.0])$
- **Input Cost**: $R = [0.1]$
- **CARE Solution $P$**:
  $$P = \begin{bmatrix}
  13.0776 & 8.0507 & 24.9743 & 6.0505 \\
  8.0507 & 7.9716 & 26.1007 & 6.2149 \\
  24.9743 & 26.1007 & 129.0132 & 23.5347 \\
  6.0505 & 6.2149 & 23.5347 & 5.7248
  \end{bmatrix}$$
- **Optimal Feedback Gain $K = R^{-1} B^T P$**:
  $$K_1 = [-10.0000, \ -12.8705, \ -87.1377, \ -23.2235]$$
- **Closed-Loop Eigenvalues $\lambda(A - BK)$**:
  $$\lambda_{cl} = \{-15.6206, \ -3.1855, \ -1.3114 \pm 1.0795 j\}$$
- **Settling Time ($2\%$ band from $\theta_0 = 0.1\text{ rad}$)**: $t_s \approx 2.45\text{ s}$
- **Peak Cart Excursion**: $x_{\max} \approx 0.082\text{ m}$

---

### Tuning Set 2: Aggressive Angle Penalization (Stiff Balance)
- **Design Intent**: Tight pendulum angle regulation for high-disturbance environments, accepting higher actuator force.
- **State Cost**: $Q = \text{diag}([1.0, 0.1, 1000.0, 10.0])$
- **Input Cost**: $R = [0.01]$
- **Optimal Feedback Gain $K$**:
  $$K_2 = [-10.0000, \ -27.0519, \ -364.8368, \ -56.6862]$$
- **Closed-Loop Eigenvalues $\lambda(A - BK)$**:
  $$\lambda_{cl} = \{-45.6004, \ -10.1621, \ -0.4004 \pm 0.3866 j\}$$
- **Transient Response from $\theta_0 = 0.1\text{ rad}$**:
  - Angle settles in $< 0.8\text{ s}$.
  - Peak force demand: $F_{\text{peak}} \approx 36.5\text{ N}$ (requires $\ge 40\text{ N}$ actuator limit).

---

### Tuning Set 3: Energy-Conserving / Soft Actuation
- **Design Intent**: Minimal energy expenditure on low-power motor drivers ($|F| \le 10\text{ N}$).
- **State Cost**: $Q = \text{diag}([1.0, 0.1, 10.0, 1.0])$
- **Input Cost**: $R = [1.0]$
- **Optimal Feedback Gain $K$**:
  $$K_3 = [-1.0000, \ -2.0466, \ -30.8464, \ -7.8675]$$
- **Closed-Loop Eigenvalues $\lambda(A - BK)$**:
  $$\lambda_{cl} = \{-4.5337, \ -3.3134, \ -0.6762 \pm 0.6441 j\}$$
- **Transient Response from $\theta_0 = 0.1\text{ rad}$**:
  - Peak force demand: $F_{\text{peak}} \approx 3.08\text{ N}$ (well within $10\text{ N}$ limit).

---

## 4. Nonlinear Basin of Attraction & Stabilization Envelope

When applied to the true nonlinear `CartPole` dynamics without swing-up logic, the linear LQR control law $u = -K x$ stabilizes the system within the initial condition envelope below.

We distinguish between two bounds:
1. **Lyapunov Invariant Ellipsoid (Guaranteed Inner Certificate)**: Certified via $x_0^T P x_0 \le c^*$ where $\dot{V}(x) < 0$.
2. **Empirical Recovery Boundary (Measured Basin with $\pm 20\text{ N}$ Actuator)**: Measured in Experiment 05.

| Metric | Tuning Set 1 (Standard, $R=0.1$) | Tuning Set 2 (Aggressive, $R=0.01$) | Tuning Set 3 (Soft Energy-Saving, $R=1.0$) |
| :--- | :--- | :--- | :--- |
| **Lyapunov Inner Bound ($x_0^T P x_0 \le c^*$)** | $|\theta_0| \le 0.38\text{ rad}$ ($21.8^\circ$) | $|\theta_0| \le 0.45\text{ rad}$ ($25.8^\circ$) | $|\theta_0| \le 0.22\text{ rad}$ ($12.6^\circ$) |
| **Measured Max Angle ($\theta_0$ at $\dot{\theta}_0=0$)** | **$0.83\text{ rad}$ ($48^\circ$)** | **$0.92\text{ rad}$ ($53^\circ$)** | **$1.00\text{ rad}$ ($57^\circ$)** |
| **Measured Max Ang. Velocity ($\dot{\theta}_0$ at $\theta_0=0$)** | **$4.4\text{ rad/s}$** | **$5.3\text{ rad/s}$** | **$5.3\text{ rad/s}$** |
| **Control Energy $E_u = \int u^2 dt$** | $\approx 4.82\text{ N}^2\cdot\text{s}$ | $\approx 28.4\text{ N}^2\cdot\text{s}$ | $\approx 0.94\text{ N}^2\cdot\text{s}$ |
| **Phase Margin $\Phi_m$** | $> 60^\circ$ | $> 75^\circ$ | $> 50^\circ$ |
| **Gain Margin $G_m$** | $[0.5, \infty)$ (Guaranteed by LQR) | $[0.5, \infty)$ | $[0.5, \infty)$ |

> **Note on Actuator Limits:** Actuator saturation ($\pm 20\text{ N}$) actually *enlarges* the usable basin on large angles by preventing high-gain controllers from issuing destructive torque spikes that blow up early rollouts. The Lyapunov certificate is strictly conservative.
