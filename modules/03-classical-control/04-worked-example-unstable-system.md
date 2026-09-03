# Worked Example: Stabilizing an Unstable System with PID

> **Module 03: Classical Control** | Worked Example 01  
> Focus: Analytical derivation, root locus analysis, steady-state DC offsets, anti-windup under saturation, and committed benchmark results.

---

## 1. Problem Statement & Physical System

Consider an open-loop unstable second-order dynamical system representing an inverted pendulum or magnetic levitation system linearized about its unstable equilibrium:

$$\ddot{y}(t) - \omega_0^2 y(t) = u(t) + d(t)$$

where:
- $y(t)$: System output (e.g., angular displacement $\theta(t)\text{ [rad]}$).
- $u(t)$: Control input (e.g., control torque $\tau(t)\text{ [N}\cdot\text{m]}$), subject to actuator saturation $|u(t)| \le u_{\max} = 10.0\text{ N}\cdot\text{m}$.
- $d(t)$: Unmeasured external disturbance torque ($d = 0.5\text{ N}\cdot\text{m}$ step applied at $t = 5.0\text{ s}$).
- $\omega_0 = 2.0\text{ rad/s}$: Unstable natural frequency ($\omega_0^2 = 4.0\text{ s}^{-2}$).

The open-loop plant transfer function is:

$$G(s) = \frac{1}{s^2 - \omega_0^2} = \frac{1}{(s - 2)(s + 2)}$$

The open-loop plant has poles at $s = +2.0$ (strictly unstable RHP pole) and $s = -2.0$ (stable LHP pole).

---

## 2. Analytical Control Design

### 2.1 Why Proportional-Only (P) Control Fails

Applying proportional control $u(t) = K_p e(t)$ with $C(s) = K_p$ and error $e(t) = r(t) - y(t)$:

The closed-loop characteristic equation is:

$$1 + K_p G(s) = 0 \implies s^2 - \omega_0^2 + K_p = 0 \implies s^2 = \omega_0^2 - K_p$$

- **Case 1 ($K_p < \omega_0^2 = 4.0$):**  
  $$s_{1,2} = \pm \sqrt{\omega_0^2 - K_p}$$  
  One pole remains strictly in the Right-Half Plane ($s_1 > 0$). The closed-loop system **diverges exponentially**.
- **Case 2 ($K_p > \omega_0^2 = 4.0$):**  
  $$s_{1,2} = \pm j \sqrt{K_p - \omega_0^2}$$  
  Both poles lie strictly on the imaginary axis ($j\omega$). The system is undamped ($\zeta = 0$), oscillating perpetually. Under finite actuator saturation ($|u| \le 10\text{ N}\cdot\text{m}$), the actuator clamps and the trajectory quickly diverges into extreme instability.

> **Mathematical Conclusion:** Proportional feedback alone **can never asymptotically stabilize** an unstable second-order system.

---

### 2.2 Adding Derivative Action: PD Control & The DC Gain Offset

Applying Proportional-Derivative control $u(t) = K_p e(t) + K_d \dot{e}(t)$ with $C(s) = K_p + K_d s$:

The closed-loop characteristic equation becomes:

$$s^2 - \omega_0^2 + (K_p + K_d s) = 0 \iff s^2 + K_d s + (K_p - \omega_0^2) = 0$$

Matching this with the standard second-order desired polynomial $s^2 + 2\zeta\omega_n s + \omega_n^2 = 0$:

$$K_p = \omega_n^2 + \omega_0^2, \qquad K_d = 2\zeta\omega_n$$

**Design Targets:**
- Desired natural frequency: $\omega_n = 4.0\text{ rad/s}$
- Desired damping ratio: $\zeta = 0.7071$ (Butterworth optimal damping)

Computing controller gains:
$$K_p = 4.0^2 + 2.0^2 = 20.0\text{ N}\cdot\text{m/rad}$$
$$K_d = 2 \times 0.7071 \times 4.0 \approx 5.657\text{ N}\cdot\text{m}\cdot\text{s/rad}$$

Both closed-loop poles are placed in the left-half plane at $s = -2.828 \pm j2.828$.

#### The Inherent Steady-State DC Offset of PD on an Unstable Plant
The closed-loop transfer function from reference $r$ to output $y$ is:

$$\frac{Y(s)}{R(s)} = \frac{G(s)C(s)}{1 + G(s)C(s)} = \frac{K_d s + K_p}{s^2 + K_d s + (K_p - \omega_0^2)}$$

Evaluating the steady-state DC gain ($s \to 0$):

$$\text{DC Gain} = \lim_{s \to 0} \frac{Y(s)}{R(s)} = \frac{K_p}{K_p - \omega_0^2} = \frac{20.0}{20.0 - 4.0} = \frac{20}{16} = 1.25$$

For a unit step reference $r = 1.0\text{ rad}$, the output converges to $y_{ss} = 1.25\text{ rad}$, resulting in a **permanent 25% tracking offset** ($e_{ss} = |1.0 - 1.25| = 0.25\text{ rad}$) even without external disturbances.

When an input disturbance torque $d = 0.5\text{ N}\cdot\text{m}$ is applied at $t = 5.0\text{ s}$, the disturbance transfer function:

$$\frac{Y(s)}{D(s)} = \frac{G(s)}{1 + G(s)C(s)} = \frac{1}{s^2 + K_d s + (K_p - \omega_0^2)}$$

induces an additional steady-state displacement:

$$\Delta y_{ss} = \frac{d}{K_p - \omega_0^2} = \frac{0.5}{16.0} = 0.03125\text{ rad}$$

Total steady-state output under disturbance is $y_{ss} = 1.25 + 0.03125 = 1.28125\text{ rad}$, yielding a total steady-state error of **$e_{ss} \approx 0.2812\text{ rad}$**.

---

### 2.3 Adding Integral Action: PID Control for Perfect Tracking & Disturbance Rejection

To eliminate both the $25\%$ reference offset and the disturbance offset, we add integral action:

$$C(s) = K_p + \frac{K_i}{s} + \frac{K_d s}{\tau_d s + 1} \approx \frac{K_d s^2 + K_p s + K_i}{s}$$

The closed-loop characteristic polynomial is:

$$s(s^2 - \omega_0^2) + (K_d s^2 + K_p s + K_i) = s^3 + K_d s^2 + (K_p - \omega_0^2)s + K_i = 0$$

By the Routh-Hurwitz criterion, closed-loop stability requires:
1. $K_d > 0$
2. $K_p > \omega_0^2 = 4.0$
3. $K_i > 0$
4. $K_d (K_p - \omega_0^2) > K_i \implies K_i < 5.657 \times 16.0 \approx 90.51$

Setting $K_i = 15.0$ places the integral gain safely within the stable band $[0, 90.51]$, providing rapid integral error elimination while maintaining strong damping.

---

## 3. Practical Implementation: Derivative Filter & Anti-Windup

1. **Filtered Derivative:** Realized with filter coefficient $N = 10$:
   $$\tau_d = \frac{K_d}{N K_p} = \frac{5.657}{10 \times 20.0} \approx 0.0283\text{ s}$$
2. **Actuator Saturation & Anti-Windup Clamping:**  
   Actuators are limited to $|u(t)| \le u_{\max} = 10.0\text{ N}\cdot\text{m}$.  
   Under **Conditional Integration (Clamping)**, the integrator state is frozen whenever the commanded torque $u_{\text{unsat}}$ exceeds $10.0\text{ N}\cdot\text{m}$ and the error $e(t)$ has the same sign as $u(t)$.

---

## 4. Benchmark Results & Quantitative Comparison

The experiment is implemented in [`experiments/03_pid_stabilizes_unstable/run.py`](../../experiments/03_pid_stabilizes_unstable/run.py) and simulated for $T = 10.0\text{ s}$ with $\Delta t = 0.001\text{ s}$ (RK4 integration) under a unit step reference $r = 1.0\text{ rad}$ and a step disturbance $d = 0.5\text{ N}\cdot\text{m}$ at $t = 5.0\text{ s}$.

### 4.1 Quantitative Performance Table

| Controller | Rise Time $t_r$ [s] | Settling Time $t_s$ [s] | Overshoot $M_p$ [%] | Steady Error $e_{ss}$ | IAE [rad·s] | ITAE [rad·s²] | Control Energy $E_u$ [N²s] | Peak $u_{\max}$ [N] | Saturation [%] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P-only** | 0.275 | 10.0 | $9.54 \times 10^9$ | $6.04 \times 10^7$ | $4.77 \times 10^7$ | $4.53 \times 10^8$ | 984.3 | 10.0 | 97.62% |
| **PD** | 0.389 | 10.0 | 28.19% | 0.2812 | 2.811 | 13.63 | 316.3 | 10.0 | 1.60% |
| **PID (no AW)** | 0.334 | 6.162 | 53.08% | $3.90 \times 10^{-5}$ | 0.9274 | 1.043 | 262.2 | 10.0 | 5.55% |
| **PID + AW** | 0.362 | 6.161 | **39.43%** | **$3.90 \times 10^{-5}$** | **0.7979** | **0.876** | **245.1** | 10.0 | **1.61%** |

*(Committed source: [`experiments/03_pid_stabilizes_unstable/table.md`](../../experiments/03_pid_stabilizes_unstable/table.md))*

---

## 5. Visual Analysis & Narrative

![Experiment 03 Comparison Figure](../../experiments/03_pid_stabilizes_unstable/figure.png)

### Key Observations from the 4-Panel Benchmark:

1. **P-Only Catastrophic Failure (Panel a & d):**  
   Because proportional control cannot add damping to the unstable plant, the output initially rises rapidly ($t_r = 0.275\text{ s}$) but immediately blows through the setpoint. The actuator saturates at $+10\text{ N}\cdot\text{m}$ for $97.6\%$ of the run, leaving the system in an unrecoverable diverging orbit in phase space $(y, \dot{y})$.
2. **PD Steady-State Droop (Panel a & c):**  
   The PD controller successfully stabilizes the plant, preventing divergence and maintaining well-damped trajectories. However, because the plant lacks a pure integrator, PD settles to $y = 1.25\text{ rad}$ ($25\%$ reference error). When the disturbance hits at $t = 5\text{ s}$, the output shifts further to $y = 1.2812\text{ rad}$, showing that PD is fundamentally incapable of eliminating steady-state offset on non-integrating unstable plants.
3. **The Danger of Integrator Windup (PID no AW):**  
   Adding integral action eliminates the steady error ($e_{ss} \approx 0$). However, because the initial step request demands more torque than the $10\text{ N}\cdot\text{m}$ limit, the integrator accumulates error during saturation ($5.55\%$ duty cycle). When the output crosses $r = 1.0\text{ rad}$, the overcharged integrator continues pushing the plant forward, causing a severe **$53.08\%$ peak overshoot**.
4. **The Power of Anti-Windup Clamping (PID + AW):**  
   Conditional integration halts the integrator during saturation, keeping the actuator duty cycle at just $1.61\%$. Peak overshoot drops from **$53.08\% \to 39.43\%$** (a $26\%$ relative reduction), control energy drops by $6.5\%$, and ITAE improves by $16\%$.

---

## 6. Engineering Summary & Limits of PID

1. **Derivative action is mathematically mandatory** to provide phase lead and stabilize open-loop RHP poles.
2. **Integral action is strictly necessary** to cancel the plant's inherent open-loop pole DC gain bias ($\frac{K_p}{K_p - \omega_0^2} \ne 1$) and reject load disturbances.
3. **Anti-windup is essential under actuator limits** to prevent destructive transient overshoots on unstable systems.
4. **Limits of PID:** While PID effectively controls this single-input single-output plant, it cannot directly manage multi-state coupled systems (like Cart-Pole $[x, \dot{x}, \theta, \dot{\theta}]$ with a single force input) or enforce optimal energy-precision trade-offs, motivating **State Feedback and LQR** in Modules 04 and 05.
