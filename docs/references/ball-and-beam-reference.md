# Ball and Beam Benchmark System Reference Specification

This document provides analytical equations of motion, Euler-Lagrange derivations, state-space linearizations, canonical Quanser Ball and Beam apparatus parameters, and classical / modern control formulations for the **Ball and Beam System** (`aimct.systems.BallAndBeam`).

---

## 1. Executive Summary & Physical Architecture

The Ball and Beam is an iconic underactuated nonlinear control benchmark exhibiting **relative degree 4** (input torque $\tau \to$ ball position $r$), open-loop instability, and state-dependent inertial coupling:
- A solid steel sphere of mass $m$ and radius $R_b$ rolls without slipping along a grooved rigid beam of mass $M$ and length $L$.
- A DC motor applies torque $\tau$ to tilt the beam about its central pivot.
- As the beam tilts, gravitational acceleration rolls the ball along the beam. The rolling ball simultaneously exerts a state-dependent torque $m g r \cos\theta$ on the beam.

```
                     Ball (Mass m, Radius Rb, Position r)
                             (o)
        ======================|======================  Beam (Length L, Mass M)
                              ^ Pivot (Angle theta)
                              |
                     [ DC Motor Torque tau ]
```

---

## 2. Canonical Hardware Parameters (Quanser Standard Class)

| Component | Parameter | Symbol | Nominal Value | Unit | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ball** | Ball Mass | $m$ | `0.064` | $\text{kg}$ | Solid steel ball mass ($64.0\text{ g}$) |
| | Ball Radius | $R_b$ | `0.0127` | $\text{m}$ | Ball radius ($12.7\text{ mm} = 0.5\text{ in}$) |
| | Ball Inertia | $J_b$ | `4.129e-6` | $\text{kg}\cdot\text{m}^2$ | Moment of inertia about center ($\frac{2}{5} m R_b^2$) |
| | Effective Inertial Mass | $m_{\text{eff}}$ | `0.0896` | $\text{kg}$ | Effective rolling mass ($m + \frac{J_b}{R_b^2} = \frac{7}{5} m$) |
| | Rolling Friction | $c_r$ | `0.002` | $\text{N}\cdot\text{s/m}$ | Ball rolling viscous resistance |
| | Travel Limits | $[r_{\min}, r_{\max}]$ | `[-0.20, +0.20]` | $\text{m}$ | Usable rolling track length ($40.0\text{ cm}$) |
| **Beam** | Beam Mass | $M$ | `0.20` | $\text{kg}$ | Mass of aluminum grooved beam |
| | Beam Length | $L$ | `0.425` | $\text{m}$ | Total beam length ($42.5\text{ cm}$) |
| | Beam Inertia | $J$ | `3.010e-3` | $\text{kg}\cdot\text{m}^2$ | Moment of inertia about central pivot ($\frac{1}{12} M L^2$) |
| | Pivot Damping | $b$ | `0.05` | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | Bearing viscous friction at pivot |
| | Tilt Angle Limits | $[\theta_{\min}, \theta_{\max}]$ | `[-0.45, +0.45]` | $\text{rad}$ | Mechanical hard stops ($\pm 25.8^\circ$) |
| **Actuator**| Peak Motor Torque | $\tau_{\max}$ | `1.50` | $\text{N}\cdot\text{m}$ | Actuator saturation limit |
| **Environment**| Gravity | $g$ | `9.81` | $\text{m/s}^2$ | Gravitational acceleration |

---

## 3. Euler-Lagrange Nonlinear Equations of Motion

### 3.1 Generalized Coordinates & Energy Formulation
- Generalized coordinates: $q = [r, \theta]^T \in \mathbb{R}^2$.
  - $r$: Position of the ball along the beam [m] ($r = 0$ at center pivot).
  - $\theta$: Tilt angle of the beam from horizontal [rad] ($\theta = 0$ is horizontal).
- **Kinetic Energy**:
  $$T = \frac{1}{2} \left(m + \frac{J_b}{R_b^2}\right) \dot{r}^2 + \frac{1}{2} \left(J + m r^2\right) \dot{\theta}^2 = \frac{7}{10} m \dot{r}^2 + \frac{1}{2} (J + m r^2) \dot{\theta}^2$$
- **Potential Energy**:
  $$V = m g r \sin\theta$$
- **Lagrangian**: $\mathcal{L} = T - V$.

### 3.2 Dynamic Equations
Applying Euler-Lagrange equations $\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{q}}\right) - \frac{\partial \mathcal{L}}{\partial q} = Q$:

1. **Ball Rolling Dynamics**:
   $$\left(\frac{7}{5} m\right) \ddot{r} - m r \dot{\theta}^2 + m g \sin\theta + c_r \dot{r} = 0$$
   $$\ddot{r} = \frac{5}{7} \left( r \dot{\theta}^2 - g \sin\theta \right) - \frac{c_r}{m_{\text{eff}}} \dot{r}$$

2. **Beam Rotational Dynamics**:
   $$(J + m r^2) \ddot{\theta} + 2 m r \dot{r} \dot{\theta} + m g r \cos\theta + b \dot{\theta} = \tau$$
   $$\ddot{\theta} = \frac{\tau - 2 m r \dot{r} \dot{\theta} - m g r \cos\theta - b \dot{\theta}}{J + m r^2}$$

---

## 4. State-Space Linearization About Center Horizontal Rest

Linearizing about equilibrium $x_0 = [0, 0, 0, 0]^T, u_0 = 0$ with state $x = [r, \dot{r}, \theta, \dot{\theta}]^T$:

$$\dot{x} = A x + B u$$

$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & -\frac{c_r}{m_{\text{eff}}} & -\frac{5}{7} g & 0 \\ 0 & 0 & 0 & 1 \\ -\frac{m g}{J} & 0 & 0 & -\frac{b}{J} \end{bmatrix} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & -0.02232 & -7.0071 & 0 \\ 0 & 0 & 0 & 1 \\ -208.56 & 0 & 0 & -16.61 \end{bmatrix}$$

$$B = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \frac{1}{J} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 332.18 \end{bmatrix}$$

### 4.1 System Properties
- **Relative Degree**: $\mu = 4$ from $\tau \to r$.
- **Open-Loop Poles**: $\lambda(A) = \{-16.91, \ -1.92 \pm 4.16j, \ \mathbf{+4.123}\}\text{ rad/s}$.
- **Unstable Pole**: $\lambda_4 = +4.123\text{ rad/s}$ reflects the open-loop unstable gravity-driven roll.
- **Controllability**: $\operatorname{rank}(\mathcal{C}) = 4$ (Full rank).

---

## 5. Control Benchmark Paradigms

### 5.1 Cascade PID (Inner-Outer Loop)
- **Outer Loop**: Ball position error $e_r = r^* - r \implies \theta_{\text{cmd}} = \operatorname{PID}_r(e_r)$ (commands beam tilt).
- **Inner Loop**: Beam angle error $e_\theta = \theta_{\text{cmd}} - \theta \implies \tau = \operatorname{PD}_\theta(e_\theta)$.
- Simple to implement, but outer bandwidth must be significantly slower than inner bandwidth to maintain loop stability.

### 5.2 Multivariable LQR
State feedback $u = -K (x - x_{\text{ref}})$ optimal under Bryson quadratic penalties:
$$Q = \operatorname{diag}([50.0, 5.0, 10.0, 1.0]), \ R = [0.1] \implies K = \begin{bmatrix} -23.00 & -15.00 & 27.74 & 3.14 \end{bmatrix}$$
Places closed-loop poles at $\{-1050.6, \ -2.80 \pm 2.80j, \ -3.16\}$.

### 5.3 Constrained Linear MPC
Receding-horizon quadratic programming enforcing hard track limits ($|r| \le 0.20\text{ m}$), beam tilt limits ($|\theta| \le 0.45\text{ rad}$), and torque bounds ($|\tau| \le 1.5\text{ N}\cdot\text{m}$).
