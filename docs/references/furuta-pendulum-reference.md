# Furuta Pendulum (Rotary Inverted Pendulum) Reference Specification

This document provides analytical equations of motion, Euler-Lagrange derivations, state-space linearizations, canonical Quanser QUBE-Servo 2 parameters, and baseline LQR / Energy Swing-Up control laws for the **Furuta Pendulum** (`aimct.systems.FurutaPendulum`).

---

## 1. Physical Architecture & Canonical Parameters

The Furuta Pendulum (Rotary Inverted Pendulum, RIP) consists of a horizontal rotary arm driven by a DC motor and a pendulum link attached to the arm tip via a perpendicular unactuated revolute joint.

### 1.1 Canonical Parameters (Quanser QUBE-Servo 2 RIP Standard)

| Component | Parameter | Symbol | Nominal Value | Unit | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Rotary Arm** | Arm Mass | $m_r$ | `0.095` | $\text{kg}$ | Mass of rotary base arm |
| | Arm Length | $L_r$ | `0.085` | $\text{m}$ | Kinematic length of rotary arm ($85\text{ mm}$) |
| | Arm Moment of Inertia | $J_r$ | `2.288e-4` | $\text{kg}\cdot\text{m}^2$ | Moment of inertia about vertical pivot ($\frac{1}{3} m_r L_r^2$) |
| | Arm Viscous Damping | $D_r$ | `5.0e-4` | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | Damping at base rotary bearing |
| **Pendulum Link** | Pendulum Mass | $m_p$ | `0.024` | $\text{kg}$ | Mass of pendulum rod |
| | Pendulum Full Length | $L_p$ | `0.129` | $\text{m}$ | Full rod length ($129\text{ mm}$) |
| | Pendulum COM Distance | $l_p$ | `0.0645` | $\text{m}$ | Distance from hinge to COM ($L_p / 2$) |
| | Pendulum COM Inertia | $J_p$ | `3.328e-5` | $\text{kg}\cdot\text{m}^2$ | Inertia about COM ($\frac{1}{12} m_p L_p^2$) |
| | Effective Pendulum Inertia | $J_p^{\text{eff}}$ | `1.332e-4` | $\text{kg}\cdot\text{m}^2$ | Inertia about hinge ($J_p + m_p l_p^2 = \frac{4}{3} m_p l_p^2$) |
| | Pendulum Viscous Damping | $D_p$ | `1.0e-4` | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | Damping at pendulum pivot |
| **Environment** | Gravitational Acceleration | $g$ | `9.81` | $\text{m/s}^2$ | Downward gravitational field |
| **Actuator** | Max Motor Torque | $\tau_{\max}$ | `0.15` | $\text{N}\cdot\text{m}$ | Maximum peak motor torque |

---

## 2. Generalized Coordinates & Nonlinear Dynamics

### 2.1 State Vector & Coordinate Conventions
- State vector $x = [\theta, \alpha, \dot{\theta}, \dot{\alpha}]^T \in \mathbb{R}^4$:
  - $\theta$: Rotary arm angle [rad] ($\theta = 0$ is the nominal reference orientation).
  - $\alpha$: Pendulum angle [rad] ($\alpha = 0$ is **upright equilibrium**, $\alpha = \pm \pi$ is downward hanging).
  - $\dot{\theta}$: Rotary arm angular velocity [rad/s].
  - $\dot{\alpha}$: Pendulum angular velocity [rad/s].
- Control input $u = \tau \in [-\tau_{\max}, +\tau_{\max}]$: Torque applied by base motor.

### 2.2 Euler-Lagrange Equations of Motion
The nonlinear dynamics satisfy:
$$M(\alpha) \begin{bmatrix} \ddot{\theta} \\ \ddot{\alpha} \end{bmatrix} + C(\alpha, \dot{\theta}, \dot{\alpha}) \begin{bmatrix} \dot{\theta} \\ \dot{\alpha} \end{bmatrix} + G(\alpha) + D \begin{bmatrix} \dot{\theta} \\ \dot{\alpha} \end{bmatrix} = \begin{bmatrix} \tau \\ 0 \end{bmatrix}$$

#### 1. Inertia Matrix $M(\alpha) = M(\alpha)^T > 0$:
$$M(\alpha) = \begin{bmatrix} J_t + m_p l_p^2 \sin^2\alpha & m_p L_r l_p \cos\alpha \\ m_p L_r l_p \cos\alpha & J_p^{\text{eff}} \end{bmatrix}$$
where $J_t = J_r + m_p L_r^2$ is the total base inertia about the rotary axis.

#### 2. Coriolis & Centrifugal Coupling:
$$C(\alpha, \dot{\theta}, \dot{\alpha}) \begin{bmatrix} \dot{\theta} \\ \dot{\alpha} \end{bmatrix} = \begin{bmatrix} m_p l_p^2 \sin(2\alpha) \dot{\theta}\dot{\alpha} - m_p L_r l_p \sin\alpha \dot{\alpha}^2 \\ -\frac{1}{2} m_p l_p^2 \sin(2\alpha) \dot{\theta}^2 \end{bmatrix}$$

#### 3. Gravity Vector (Upright Reference $\alpha = 0$):
$$G(\alpha) = \begin{bmatrix} 0 \\ -m_p g l_p \sin\alpha \end{bmatrix}$$

#### 4. Viscous Damping Matrix:
$$D = \begin{bmatrix} D_r & 0 \\ 0 & D_p \end{bmatrix}$$

---

## 3. State-Space Linearization About Upright Equilibrium

Linearizing about the unstable upright equilibrium $x_{\text{eq}} = [0, 0, 0, 0]^T, u_{\text{eq}} = 0$:

$$M_0 = \begin{bmatrix} J_t & m_p L_r l_p \\ m_p L_r l_p & J_p^{\text{eff}} \end{bmatrix}, \quad \det M_0 = J_t J_p^{\text{eff}} - (m_p L_r l_p)^2$$

The continuous state-space matrices $\dot{x} = A x + B u$ are:
$$A = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & -\frac{m_p^2 L_r l_p^2 g}{\det M_0} & -\frac{J_p^{\text{eff}} D_r}{\det M_0} & \frac{m_p L_r l_p D_p}{\det M_0} \\ 0 & \frac{J_t m_p g l_p}{\det M_0} & \frac{m_p L_r l_p D_r}{\det M_0} & -\frac{J_t D_p}{\det M_0} \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ \frac{J_p^{\text{eff}}}{\det M_0} \\ -\frac{m_p L_r l_p}{\det M_0} \end{bmatrix}$$

### 3.1 Numerical Golden Values (QUBE-Servo 2 RIP)
$$\det M_0 = 3.6230 \times 10^{-8}\text{ kg}^2\cdot\text{m}^4$$
$$A = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & -55.1525 & -1.8373 & 0.3632 \\ 0 & 168.5810 & 1.8159 & -1.1101 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ 3674.56 \\ -3631.83 \end{bmatrix}$$

- **Open-Loop Poles**: $\lambda(A) = \{0.0, \ +12.7244, \ -13.2385, \ -2.4332\}$
- **Controllability**: $\operatorname{rank}(\mathcal{C}) = 4$ (Full rank).

---

## 4. Control Strategies

### 4.1 Upright LQR Balance
With state weighting $Q = \operatorname{diag}([5.0, 10.0, 0.5, 0.5])$ and control weighting $R = [1.0]$:
$$K_{\text{LQR}} = \begin{bmatrix} -2.2361 & -29.8967 & -1.0452 & -2.0699 \end{bmatrix}$$
Closed-loop poles: $\lambda(A - BK) = \{-3653.3, \ -11.70 \pm 4.64j, \ -3.17\}$.

### 4.2 Åström-Furuta Energy Swing-Up Law
The pendulum's mechanical energy relative to the upright separatrix ($E = 0$ upright, $E = -2 m_p g l_p$ hanging):
$$E(\alpha, \dot{\alpha}) = \frac{1}{2} J_p^{\text{eff}} \dot{\alpha}^2 + m_p g l_p (\cos\alpha - 1)$$
Choosing desired arm acceleration:
$$\ddot{\theta}_{\text{des}} = k_E \cdot E \cdot \operatorname{sign}(\dot{\alpha}\cos\alpha) - k_p \theta - k_d \dot{\theta}$$
Torque command:
$$\tau = J_t \ddot{\theta}_{\text{des}} + D_r \dot{\theta}$$
When $|\operatorname{wrap}(\alpha)| \le 0.35\text{ rad}$ and $|\dot{\alpha}| \le 3.0\text{ rad/s}$, hand off seamlessly to $K_{\text{LQR}}$.
