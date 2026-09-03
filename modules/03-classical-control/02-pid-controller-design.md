# PID Controller Architecture & Implementation

> **Module 03: Classical Control** | Theory Note 02  
> Focus: Mathematical formulation, practical derivative filtering, anti-windup clamping and back-calculation, 2-DOF setpoint weighting, and tuning rules.

---

## 1. Mathematical Formulation of PID Control

The Proportional-Integral-Derivative (PID) controller generates control action based on present, past, and predicted future errors:

$$u(t) = \underbrace{K_p e(t)}_{\text{Present Error}} + \underbrace{K_i \int_0^t e(\tau) \, d\tau}_{\text{Accumulated Past Error}} + \underbrace{K_d \frac{de(t)}{dt}}_{\text{Predicted Error Slope}}$$

where $e(t) \triangleq r(t) - y(t)$ is the tracking error.

### 1.1 Standard (ISA) Parameterization
Control engineers frequently express gains in terms of integral time $T_i$ and derivative time $T_d$:

$$u(t) = K_p \left( e(t) + \frac{1}{T_i} \int_0^t e(\tau) \, d\tau + T_d \frac{de(t)}{dt} \right)$$

where $K_i = \frac{K_p}{T_i}$ and $K_d = K_p T_d$.

In the Laplace domain:
$$C(s) = K_p + \frac{K_i}{s} + K_d s = \frac{K_d s^2 + K_p s + K_i}{s}$$

- **Proportional Action ($K_p$):** Adjusts open-loop gain and speeds up transient rise time. High $K_p$ reduces steady-state error but degrades damping margin.
- **Integral Action ($K_i$):** Adds a pole at $s = 0$ (Type 1 system), guaranteeing zero steady-state error ($e_{ss} = 0$) to constant step references and step disturbances. Adds $-90^\circ$ phase lag.
- **Derivative Action ($K_d$):** Adds phase lead $+90^\circ$ around the crossover frequency, providing artificial damping and reducing peak overshoot.

---

## 2. Practical Real-World Enhancements

An "ideal" textbook PID controller cannot be deployed directly on physical hardware due to three critical engineering realities:

### 2.1 Filtered Derivative Action
An ideal derivative $K_d s$ has infinite high-frequency gain ($\lim_{\omega \to \infty} |j\omega K_d| = \infty$). High-frequency sensor noise (e.g., encoder quantization, ADC noise) gets amplified uncontrollably, causing actuator chattering and overheating.

**Solution:** Implement derivative action with a first-order low-pass filter:

$$D(s) = \frac{K_d s}{\tau_f s + 1} = \frac{K_d s}{\frac{T_d}{N} s + 1} = \frac{K_d s}{\frac{K_d}{N K_p} s + 1}$$

where $N \in [8, 20]$ (typically $N = 10$). At low frequencies ($\omega \ll N/\tau_f$), $D(s) \approx K_d s$; at high frequencies ($\omega \gg N/\tau_f$), gain is strictly bounded by $N K_p$.

### 2.2 Integrator Windup & Anti-Windup Protection
All real actuators have physical saturation boundaries $u(t) \in [u_{\min}, u_{\max}]$.
- **The Windup Mechanism:** During a large step change or physical constraint, the actuator saturates at $u_{\max}$. However, tracking error $e(t) > 0$ continues to integrate, causing the internal integrator state $x_i = \int e \, d\tau$ to grow to huge values. When the output finally reaches the target, the error changes sign, but the controller cannot reduce $u(t)$ until the integrator discharges ("unwinds"). This causes massive overshoot and prolonged settling time.

**Anti-Windup Method 1: Conditional Integration (Clamping)**  
Freeze integration whenever the actuator is saturated and the error would drive it deeper into saturation:

$$\dot{x}_i(t) = \begin{cases} 0, & \text{if } u(t) \ge u_{\max} \text{ and } e(t) > 0 \\ 0, & \text{if } u(t) \le u_{\min} \text{ and } e(t) < 0 \\ e(t), & \text{otherwise} \end{cases}$$

**Anti-Windup Method 2: Back-Calculation**  
Feed back the saturation error $\Delta u = u_{\text{sat}} - u_{\text{unsat}}$ into the integrator with tracking time constant $T_t = \sqrt{T_i T_d}$:

$$\dot{x}_i(t) = e(t) + \frac{1}{T_t} \left( \text{clip}(u(t), u_{\min}, u_{\max}) - u(t) \right)$$

### 2.3 Setpoint Weighting & Derivative on Measurement (2-DOF PID)
A sudden step change in setpoint $r(t)$ creates a Dirac delta impulse in $\frac{de}{dt} = \frac{dr}{dt} - \frac{dy}{dt}$ ("derivative kick").

**Solution:** Compute derivative action solely on the negative measured output $-y(t)$, and apply reference weighting $\beta \in [0, 1]$ to the proportional term:

$$u(t) = K_p (\beta r(t) - y(t)) + K_i \int_0^t (r(\tau) - y(\tau)) \, d\tau - \frac{K_d s}{\tau_f s + 1} y(t)$$

Setting $\beta = 1, \gamma = 0$ provides fast disturbance rejection without derivative kicking on setpoint changes.

---

## 3. Discrete-Time Implementation

On a microcontroller with sample time $\Delta t$, the discrete equations using Forward Euler for integration and Backward Difference for filtered derivative are:

$$\begin{aligned}
e_k &= r_k - y_k \\
d_k &= \frac{\tau_f}{\tau_f + \Delta t} d_{k-1} + \frac{K_d}{\tau_f + \Delta t} (y_k - y_{k-1}) \quad (\text{derivative on measurement}) \\
u_{\text{unsat}, k} &= K_p (\beta r_k - y_k) + x_{i, k} - d_k \\
u_k &= \text{clip}(u_{\text{unsat}, k}, u_{\min}, u_{\max}) \\
x_{i, k+1} &= x_{i, k} + K_i \Delta t \, e_k + \frac{\Delta t}{T_t} (u_k - u_{\text{unsat}, k}) \quad (\text{back-calculation update})
\end{aligned}$$

---

## 4. Tuning Rules of Thumb

| Method | Tuning Basis | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **Ziegler-Nichols (Closed-Loop)** | Ultimate Gain $K_u$ & Period $T_u$ at boundary of stability | Fast setup; requires no prior mathematical model | Aggressive ($M_p \approx 25\%$), poor gain margin ($< 2$). |
| **Pole Placement / Root Locus** | Exact $s$-plane closed-loop pole locations | Explicitly shapes damping ratio $\zeta$ and natural frequency $\omega_n$ | Requires accurate transfer function model $G(s)$. |
| **Direct Synthesis (Lambda Tuning)** | Closed-loop target $T(s) = \frac{1}{\lambda s + 1}$ | Guaranteed non-overshooting, robust first-order response | Conservative for disturbance rejection. |
