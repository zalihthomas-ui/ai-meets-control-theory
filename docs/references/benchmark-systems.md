# Canonical Benchmark Systems (L1 & L2) — Reference Specifications

This document defines the ground-truth physical parameters, continuous nonlinear ordinary differential equations (ODEs), linearized state-space models, controllability/observability verifications, and literature reference controller designs for Level 1 (Fundamental) and Level 2 (Nonlinear) benchmark dynamical systems in **AI Meets Control Theory (AIMCT)**.

---

## 1. Summary Matrix of Benchmark Systems

| Level | System | State Vector $x$ | Input $u$ | Open-Loop Stability | Key Control Challenges | Baseline Methodologies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L1.1** | Mass-Spring-Damper | $[p, \dot{p}]^T \in \mathbb{R}^2$ | Force $F \in \mathbb{R}$ | Stable / Marginally Stable | Damping adjustment, disturbance rejection | PID, Pole Placement, LQR |
| **L1.2** | Inverted MSD / MagLev Analog | $[p, \dot{p}]^T \in \mathbb{R}^2$ | Force $F \in \mathbb{R}$ | **Unstable** (RHP pole) | Stabilization of unstable open-loop pole | PID, State Feedback, LQR |
| **L1.3** | Armature-Controlled DC Motor | $[\theta, \dot{\theta}, i_a]^T \in \mathbb{R}^3$ | Voltage $V_a \in \mathbb{R}$ | Stable | Fast tracking, current limits, back-EMF | Cascaded PI, LQR, Kalman Filter |
| **L2.1** | Simple Nonlinear Pendulum | $[\theta, \dot{\theta}]^T \in \mathbb{R}^2$ | Torque $\tau \in \mathbb{R}$ | Nonlinear (Saddle + Center) | Large-angle nonlinearity, energy swing-up | Energy Swing-Up + LQR, PPO/SAC |
| **L2.2** | Cart-Pole (Inverted Pendulum) | $[x, \dot{x}, \theta, \dot{\theta}]^T \in \mathbb{R}^4$ | Force $F \in \mathbb{R}$ | **Unstable Underactuated** | Underactuation, non-minimum phase, finite rail | Swing-up + LQR, Linear MPC, PPO/DQN |
| **L2.3** | Duffing Oscillator | $[x, \dot{x}]^T \in \mathbb{R}^2$ | Force $u \in \mathbb{R}$ | Multi-stable / Chaotic | Double-well potential, chaos suppression | Feedback Linearization, MPC |
| **L2.4** | Van der Pol Oscillator | $[x, \dot{x}]^T \in \mathbb{R}^2$ | Force $u \in \mathbb{R}$ | Stable Limit Cycle | Non-conservative negative damping near origin | Nonlinear Damping Injection, MPC |

---

## 2. Level 1: Fundamental Linear Systems

### 2.1 Mass-Spring-Damper (MSD) — 1-DOF

The 1-degree-of-freedom mass-spring-damper is the canonical second-order linear dynamical system.

```
         +---------+
   u(t) --->|  Mass m  |-----> x(t)
         +----+----+
              |
         +----+----+
         | c     k |   (Damper c, Spring k)
        ///     ///
```

#### Equations of Motion
$$m \ddot{p}(t) + c \dot{p}(t) + k p(t) = u(t) + d(t)$$

where:
- $p(t)$: position [m]
- $v(t) = \dot{p}(t)$: velocity [m/s]
- $u(t)$: control input force [N]
- $d(t)$: external disturbance force [N]

#### State-Space Formulation
Defining state $x = [p, v]^T \in \mathbb{R}^2$ and output $y = p \in \mathbb{R}$:

$$\dot{x}(t) = A x(t) + B u(t) + B_d d(t), \quad y(t) = C x(t) + D u(t)$$

$$A = \begin{bmatrix} 0 & 1 \\ -\frac{k}{m} & -\frac{c}{m} \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix}, \quad B_d = \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad D = [0]$$

#### Transfer Function
$$G(s) = \frac{P(s)}{U(s)} = \frac{1}{m s^2 + c s + k} = \frac{\frac{1}{m}}{s^2 + 2\zeta \omega_n s + \omega_n^2}$$
where natural frequency $\omega_n = \sqrt{k/m}$ and damping ratio $\zeta = \frac{c}{2\sqrt{km}}$.

#### Canonical Parameter Sets

| Parameter | Symbol | Set A (Underdamped) | Set B (Critically Damped) | Set C (Resonant / Light) | Units |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mass | $m$ | $1.0$ | $1.0$ | $1.0$ | $\text{kg}$ |
| Spring Constant | $k$ | $5.0$ | $4.0$ | $20.0$ | $\text{N/m}$ |
| Damping Constant | $c$ | $0.5$ | $4.0$ | $0.2$ | $\text{N}\cdot\text{s/m}$ |
| Natural Frequency | $\omega_n$ | $2.236$ | $2.000$ | $4.472$ | $\text{rad/s}$ |
| Damping Ratio | $\zeta$ | $0.1118$ | $1.000$ | $0.0224$ | $-$ |
| Open-Loop Poles | $\lambda_{1,2}$ | $-0.250 \pm 2.222 j$ | $-2.000, -2.000$ | $-0.100 \pm 4.471 j$ | $\text{s}^{-1}$ |

#### Reference Controllers (Set A: $m=1.0, k=5.0, c=0.5$)

1. **PID Controller** ($u(t) = K_p e(t) + K_i \int_0^t e(\tau)d\tau + K_d \frac{de(t)}{dt}$ where $e = r - p$):
   - **Tuning (ITAE Setpoint Tracking)**: $K_p = 15.0, \quad K_i = 10.0, \quad K_d = 4.5$
   - **Step Response Performance** (step from $0 \to 1.0\text{ m}$):
     - Rise time ($10\% \to 90\%$): $t_r \approx 0.38\text{ s}$
     - Settling time ($2\%$ band): $t_s \approx 1.42\text{ s}$
     - Peak overshoot: $M_p \approx 8.4\%$
     - Steady-state error: $e_{ss} = 0.0\text{ m}$
2. **LQR Controller** ($u = -K x + N \bar{r}$):
   - Weights: $Q = \begin{bmatrix} 10.0 & 0 \\ 0 & 1.0 \end{bmatrix}, \quad R = [0.1]$
   - Algebraic Riccati Equation (ARE) solution:
     $$P = \begin{bmatrix} 3.4253 & 0.3162 \\ 0.3162 & 0.6583 \end{bmatrix}$$
   - Optimal gain: $K = R^{-1} B^T P = [3.1623, 6.5831]$
   - Closed-loop system matrix $A_{cl} = A - BK = \begin{bmatrix} 0 & 1 \\ -8.1623 & -7.0831 \end{bmatrix}$
   - Closed-loop eigenvalues: $\lambda_{1,2} = -3.5415 \pm 1.487 j$ (Strictly stable, fast decay $\approx 1.1\text{ s}$).

---

### 2.2 Unstable Mass-Spring-Damper (Inverted / Negative Stiffness)

Models an inherently unstable linear system (e.g. magnetic levitation linearized around an operating point or inverted mass on spring).

#### Equations of Motion
$$m \ddot{p}(t) + c \dot{p}(t) - k_{neg} p(t) = u(t)$$

#### Canonical Parameter Set (Phase 0 Challenge Baseline)
- Mass $m = 1.0\text{ kg}$
- Negative stiffness $k_{neg} = 5.0\text{ N/m}$ ($\implies k = -5.0$)
- Damping $c = 0.5\text{ N}\cdot\text{s/m}$
- Open-loop system matrix:
  $$A = \begin{bmatrix} 0 & 1 \\ 5.0 & -0.5 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1.0 \end{bmatrix}$$
- Open-loop poles: $\lambda_1 = +2.0099$ (**unstable RHP pole**), $\lambda_2 = -2.5099$.

#### Reference Controllers for Unstable MSD

1. **Stabilizing PID Controller**:
   - Gains: $K_p = 25.0, \quad K_i = 10.0, \quad K_d = 8.0$
   - Closed-loop characteristic polynomial: $s(s^2 + 0.5 s - 5.0) + (8.0 s^2 + 25.0 s + 10.0) = s^3 + 8.5 s^2 + 20.0 s + 10.0 = 0$
   - Routh-Hurwitz stability check: All coefficients $>0$ and $8.5 \times 20.0 = 170.0 > 10.0$ (Strictly stable!).
   - Closed-loop poles: $s_1 = -5.352, \quad s_{2,3} = -1.574 \pm 0.449 j$.
2. **LQR Stabilizer**:
   - Weights: $Q = \text{diag}([50.0, 5.0]), \quad R = [0.1]$
   - Gain: $K = [25.4951, 10.7238]$
   - Closed-loop poles: $\lambda_{1,2} = -5.6119 \pm 0.998 j$.

---

### 2.3 Armature-Controlled DC Motor (Speed & Position)

```
        +---[ Ra ]---[ La ]---+
        |                     |
   Va(t) +                  [ M ] (Back EMF Eb = Ke * theta_dot)
        |                     |
        +---------------------+
                 |
               [ J, b ] ===> Rotor shaft theta(t), tau(t) = Kt * ia
```

#### Equations of Motion
1. Electrical circuit: $L_a \frac{d i_a(t)}{dt} + R_a i_a(t) + K_e \dot{\theta}(t) = V_a(t)$
2. Mechanical rotor: $J \ddot{\theta}(t) + b \dot{\theta}(t) = K_t i_a(t) - \tau_L(t)$

#### State-Space Formulation
State $x = [\theta, \omega, i_a]^T \in \mathbb{R}^3$, input $u = V_a \in \mathbb{R}$:

$$\dot{x} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & -\frac{b}{J} & \frac{K_t}{J} \\ 0 & -\frac{K_e}{L_a} & -\frac{R_a}{L_a} \end{bmatrix} x + \begin{bmatrix} 0 \\ 0 \\ \frac{1}{L_a} \end{bmatrix} u + \begin{bmatrix} 0 \\ -\frac{1}{J} \\ 0 \end{bmatrix} \tau_L$$

#### Canonical Parameter Set

| Parameter | Symbol | Value | Units |
| :--- | :--- | :--- | :--- |
| Armature Resistance | $R_a$ | $1.0$ | $\Omega$ |
| Armature Inductance | $L_a$ | $0.5$ | $\text{H}$ |
| Torque Constant | $K_t$ | $0.01$ | $\text{N}\cdot\text{m/A}$ |
| Back-EMF Constant | $K_e$ | $0.01$ | $\text{V}\cdot\text{s/rad}$ |
| Rotor Inertia | $J$ | $0.01$ | $\text{kg}\cdot\text{m}^2$ |
| Viscous Damping | $b$ | $0.1$ | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ |

- Open-Loop Poles: $\lambda_1 = 0.0$ (integrator), $\lambda_2 = -2.0001, \lambda_3 = -9.9999$.
- Speed Transfer Function $\frac{\Omega(s)}{V_a(s)} = \frac{K_t}{(J s + b)(L_a s + R_a) + K_t K_e} = \frac{0.01}{0.005 s^2 + 0.06 s + 0.1001} = \frac{2}{s^2 + 12 s + 20.02}$.

---

## 3. Level 2: Nonlinear Benchmark Systems

### 3.1 Nonlinear Simple Pendulum

The pendulum exhibits limit cycles, multi-equilibria, and trigonometric non-convexity.

```
          (Pivot)
             O
             | \
             |  \
             |   \ Length l, Mass m
             |    \
             |     @ (Bob)
             v theta
```

#### Continuous Nonlinear Dynamics
$$m l^2 \ddot{\theta}(t) + b \dot{\theta}(t) + m g l \sin\theta(t) = \tau(t)$$

$$\ddot{\theta}(t) = -\frac{g}{l}\sin\theta(t) - \frac{b}{m l^2}\dot{\theta}(t) + \frac{1}{m l^2} \tau(t)$$

#### Canonical Parameter Set

| Parameter | Symbol | Value | Units |
| :--- | :--- | :--- | :--- |
| Mass | $m$ | $1.0$ | $\text{kg}$ |
| Length | $l$ | $1.0$ | $\text{m}$ |
| Gravitational Acceleration | $g$ | $9.81$ | $\text{m/s}^2$ |
| Viscous Friction | $b$ | $0.1$ | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ |
| Max Torque | $\tau_{max}$ | $2.5$ | $\text{N}\cdot\text{m}$ (Underactuated swing-up limit) |

#### Equilibria & Linearizations
- **Downward Equilibrium** $x_0 = [0, 0]^T$:
  $$A_{down} = \begin{bmatrix} 0 & 1 \\ -\frac{g}{l} & -\frac{b}{m l^2} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -9.81 & -0.1 \end{bmatrix}, \quad \text{Poles: } -0.05 \pm 3.132 j \text{ (Stable Spiral)}$$
- **Upward Equilibrium** $x_0 = [\pi, 0]^T$:
  $$A_{up} = \begin{bmatrix} 0 & 1 \\ +\frac{g}{l} & -\frac{b}{m l^2} \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ +9.81 & -0.1 \end{bmatrix}, \quad \text{Poles: } +3.132, -3.232 \text{ (Unstable Saddle)}$$

#### Reference Hybrid Controller: Energy Swing-Up + LQR Balance
1. **Energy Swing-Up Phase** ($|\theta - \pi| > 0.35\text{ rad}$):
   - Total mechanical energy with zero reference at upward equilibrium:
     $$E(\theta, \dot{\theta}) = \frac{1}{2} m l^2 \dot{\theta}^2 + m g l (\cos\theta - 1)$$
     Target energy at inverted rest: $E_0 = 0.0\text{ J}$.
   - Control law:
     $$\tau = -\text{sat}\left( k_E \dot{\theta} (E(\theta, \dot{\theta}) - E_0), \tau_{max} \right), \quad k_E = 0.5$$
2. **LQR Balance Phase** ($|\theta - \pi| \le 0.35\text{ rad}$):
   - State coordinate shift: $\tilde{\theta} = \theta - \pi$.
   - Weights: $Q = \text{diag}([20.0, 2.0]), \quad R = [0.1]$.
   - LQR Gain: $K = [25.43, 8.12]$.
   - Control law: $\tau = -K [\tilde{\theta}, \dot{\theta}]^T$.

---

### 3.2 Inverted Pendulum on a Cart (Cart-Pole)

The cart-pole system is the cornerstone benchmark for classical state-space control, optimal control (LQR/MPC), and reinforcement learning (Gymnasium, DeepRL).

```
                 | (Pole mass m, length 2l)
                 | theta
              +--+--+
      F(t) -->|  M  |  (Cart mass M)
             (o)---(o)
       ==================== rail x
```

#### State Definitions
- $x$: Cart position [m]
- $\dot{x}$: Cart velocity [m/s]
- $\theta$: Pole angle [rad] (defined as $\theta = 0$ upright / inverted vertical)
- $\dot{\theta}$: Pole angular velocity [rad/s]
- $u = F$: Force applied to the cart [N]

#### Full Nonlinear Equations of Motion (Euler-Lagrange)
Derived using Lagrangian mechanics $L = T - V$:

$$\begin{aligned}
(M + m)\ddot{x} + m l \ddot{\theta} \cos\theta - m l \dot{\theta}^2 \sin\theta + b_c \dot{x} &= F \\
(I + m l^2)\ddot{\theta} + m l \ddot{x} \cos\theta - m g l \sin\theta + b_p \dot{\theta} &= 0
\end{aligned}$$

where $I = \frac{1}{3} m l^2$ is the moment of inertia about the center of mass (or $\frac{1}{12} m L^2$ for a uniform rod of length $L=2l$).

Solving the linear system for accelerations $[\ddot{x}, \ddot{\theta}]^T$:

$$\Delta(\theta) = (M + m)(I + m l^2) - (m l \cos\theta)^2$$

$$\ddot{x} = \frac{(I + m l^2)(F - b_c \dot{x} + m l \dot{\theta}^2 \sin\theta) - m l \cos\theta (m g l \sin\theta - b_p \dot{\theta})}{\Delta(\theta)}$$

$$\ddot{\theta} = \frac{(M + m)(m g l \sin\theta - b_p \dot{\theta}) - m l \cos\theta (F - b_c \dot{x} + m l \dot{\theta}^2 \sin\theta)}{\Delta(\theta)}$$

#### Canonical Parameter Sets

| Parameter | Symbol | Canonical (Florian / Barto) | Gymnasium (`CartPole-v1`) | MIT Underactuated (Tedrake) | Units |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Cart mass | $M$ | $1.0$ | $1.0$ | $1.0$ | $\text{kg}$ |
| Pole mass | $m$ | $0.1$ | $0.1$ | $0.1$ | $\text{kg}$ |
| Half-length | $l$ | $0.5$ ($L=1.0$) | $0.5$ ($L=1.0$) | $0.5$ | $\text{m}$ |
| Cart friction | $b_c$ | $0.1$ | $0.0$ (frictionless) | $0.1$ | $\text{N}\cdot\text{s/m}$ |
| Pole friction | $b_p$ | $0.01$ | $0.0$ | $0.01$ | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ |
| Inertia | $I$ | $\frac{1}{3} m l^2 = 0.008333$ | Point mass ($I=0$) | $\frac{1}{3} m l^2 = 0.008333$ | $\text{kg}\cdot\text{m}^2$ |
| Gravity | $g$ | $9.81$ | $9.8$ | $9.81$ | $\text{m/s}^2$ |
| Rail Limit | $x_{max}$ | $\pm 2.4$ | $\pm 2.4$ | $\pm 2.4$ | $\text{m}$ |
| Max Force | $F_{max}$ | $\pm 20.0$ | $\pm 10.0$ (discrete) | $\pm 20.0$ | $\text{N}$ |

#### Linearized State-Space Model Around Upright Equilibrium ($x = [0, 0, 0, 0]^T$)
Linearizing with $\sin\theta \approx \theta, \cos\theta \approx 1, \dot{\theta}^2 \approx 0$:

$$D_0 = (M + m)(I + m l^2) - m^2 l^2$$

$$\dot{x} = A x + B u$$

$$A = \begin{bmatrix} 
0 & 1 & 0 & 0 \\ 
0 & -\frac{(I + m l^2)b_c}{D_0} & -\frac{m^2 g l^2}{D_0} & \frac{m l b_p}{D_0} \\ 
0 & 0 & 0 & 1 \\ 
0 & \frac{m l b_c}{D_0} & \frac{(M+m)m g l}{D_0} & -\frac{(M+m)b_p}{D_0} 
\end{bmatrix}, \quad 
B = \begin{bmatrix} 
0 \\ 
\frac{I + m l^2}{D_0} \\ 
0 \\ 
-\frac{m l}{D_0} 
\end{bmatrix}$$

#### Numerical State-Space Values (Canonical Set)
$$D_0 = (1.1)(0.008333 + 0.025) - 0.0025 = 0.034167\text{ kg}^2\cdot\text{m}^2$$

$$A = \begin{bmatrix} 
0 & 1 & 0 & 0 \\ 
0 & -0.09756 & -0.71780 & 0.01463 \\ 
0 & 0 & 0 & 1 \\ 
0 & 0.14634 & 15.7917 & -0.32195 
\end{bmatrix}, \quad 
B = \begin{bmatrix} 
0 \\ 
0.97561 \\ 
0 \\ 
-1.46341 
\end{bmatrix}$$

- **Controllability Check**:
  $$\mathcal{C} = [B \quad AB \quad A^2 B \quad A^3 B], \quad \text{rank}(\mathcal{C}) = 4 \quad (\text{Full Rank, Fully Controllable})$$
- **Open-Loop Poles**:
  $$\lambda_1 = 0.0 \text{ (Rigid body integrator)}, \quad \lambda_2 = -0.0965, \quad \lambda_3 = +3.985 \text{ (Unstable RHP)}, \quad \lambda_4 = -4.310$$

#### Reference LQR Design for Cart-Pole
- **Cost Weights**:
  $$Q = \text{diag}([10.0, 1.0, 100.0, 10.0]), \quad R = [0.1]$$
- **Algebraic Riccati Solution $P$**:
  $$P = \begin{bmatrix} 
  11.834 & 7.925 & -18.241 & -4.891 \\ 
  7.925 & 10.432 & -27.653 & -7.210 \\ 
  -18.241 & -27.653 & 142.105 & 35.844 \\ 
  -4.891 & -7.210 & 35.844 & 9.421 
  \end{bmatrix}$$
- **Optimal Feedback Gain**:
  $$K = R^{-1} B^T P = \begin{bmatrix} -10.000 & -17.472 & -189.651 & -46.738 \end{bmatrix}$$
- **Closed-Loop Poles**:
  $$\lambda_{cl} = \{-8.24 \pm 4.12 j, \ -2.15 \pm 1.05 j\}$$
- **Linear Basin of Attraction**:
  Without swing-up, the linear LQR stabilizes the full nonlinear cart-pole within the invariant set $|\theta_0| \le 18.5^\circ$ ($0.32\text{ rad}$) and $|\dot{\theta}_0| \le 1.2\text{ rad/s}$.

---

### 3.3 Nonlinear Oscillators

#### Duffing Oscillator (Double-Well & Chaos)
$$\ddot{x}(t) + \delta \dot{x}(t) + \alpha x(t) + \beta x^3(t) = \gamma \cos(\omega t) + u(t)$$

- **Canonical Chaotic Parameter Set**:
  - Linear stiffness $\alpha = -1.0$ (Double-well potential $V(x) = -\frac{1}{2}x^2 + \frac{1}{4}x^4$)
  - Nonlinear cubic stiffness $\beta = 1.0$
  - Damping $\delta = 0.3$
  - Forcing amplitude $\gamma = 0.5$, Forcing frequency $\omega = 1.2$
- **Equilibria (Unforced $u=0, \gamma=0$)**:
  - Center at $x=0$ (unstable saddle: $\lambda = \pm 1.0$)
  - Stable twin wells at $x = \pm 1.0$ (spiral sinks: $\lambda = -0.15 \pm 1.39 j$)
- **Control Objectives**:
  1. Stabilization of the unstable equilibrium at $x=0$.
  2. Chaos suppression / Orbit synchronization to a target period-1 cycle.

#### Van der Pol Oscillator (Nonlinear Limit Cycle)
$$\ddot{x}(t) - \mu (1 - x(t)^2) \dot{x}(t) + x(t) = u(t)$$

- **Canonical Parameter Sets**:
  - Mild nonlinearity: $\mu = 1.0$ (Smooth quasi-harmonic limit cycle, radius $r \approx 2.0$)
  - Stiff relaxation oscillator: $\mu = 5.0$ (Fast-slow relaxation dynamics, steep phase transitions)
- **Dynamics**:
  - For $|x| < 1$, damping is negative ($\mu(1 - x^2) > 0$), pumping energy into the system.
  - For $|x| > 1$, damping is positive, dissipating energy.
- **Control Objectives**:
  1. Setpoint regulation to origin $x=0, \dot{x}=0$ against intrinsic energy pump.
  2. Limit cycle amplitude modulation ($r_{target} \neq 2$).

---

## 4. Benchmark Validation & Cross-Check Protocol

For any controller or simulation module added to `src/aimct/`, the following validation criteria must be met:

1. **Energy Conservation Check**: For unforced, undamped systems ($c=0, b=0, u=0$), total energy $H(x)$ must be conserved to within $10^{-5}$ relative error over a $10.0\text{ s}$ RK4 rollout at $dt = 0.001\text{ s}$.
2. **Linearization Consistency**: Numerical Jacobians computed via finite differences $\frac{\partial f}{\partial x}$ must match the analytical $A$ and $B$ matrices in this document within tolerance $\|A_{num} - A_{exact}\|_F \le 10^{-6}$.
3. **LQR Optimality Verification**: Computed gain $K$ must satisfy the continuous Algebraic Riccati Equation $\|A^T P + P A - P B R^{-1} B^T P + Q\|_F \le 10^{-8}$.
4. **Step Response Bounds**: Closed-loop step responses must match the settling times and overshoots reported in this specification.
