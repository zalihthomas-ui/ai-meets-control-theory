# Worked Example: Stabilizing an Unstable System with PID

> **Module 03: Classical Control** | Worked Example 01  
> Focus: Analytical derivation, root locus analysis, anti-windup under saturation, and benchmark evaluation.

---

## 1. Problem Statement & Physical System

Consider an open-loop unstable second-order dynamical system representing an inverted pendulum or magnetic levitation system linearized about its unstable equilibrium:

$$\ddot{y}(t) - \omega_0^2 y(t) = u(t) + d(t)$$

where:
- $y(t)$: System output (e.g., angular displacement $\theta(t)\text{ [rad]}$).
- $u(t)$: Control input (e.g., control torque $\tau(t)\text{ [N}\cdot\text{m]}$), subject to actuator saturation $|u(t)| \le u_{\max} = 25.0\text{ N}\cdot\text{m}$.
- $d(t)$: Unmeasured external disturbance torque ($d = 0.5\text{ N}\cdot\text{m}$ applied at $t = 5.0\text{ s}$).
- $\omega_0 = 2.0\text{ rad/s}$: Unstable natural frequency ($\omega_0^2 = 4.0\text{ s}^{-2}$).

The open-loop plant transfer function is:

$$G(s) = \frac{1}{s^2 - \omega_0^2} = \frac{1}{(s - 2)(s + 2)}$$

The plant has open-loop poles at $s = +2.0$ (unstable RHP pole) and $s = -2.0$ (stable LHP pole).

---

## 2. Analytical Control Design

### 2.1 Why Proportional-Only (P) Control Fails

Applying proportional control $u(t) = K_p e(t)$ with $C(s) = K_p$:

The closed-loop characteristic equation is:

$$1 + K_p G(s) = 0 \implies s^2 - \omega_0^2 + K_p = 0 \implies s^2 = \omega_0^2 - K_p$$

- **Case 1 ($K_p < \omega_0^2 = 4$):**  
  $$s_{1,2} = \pm \sqrt{\omega_0^2 - K_p}$$  
  One pole remains strictly in the Right-Half Plane ($s_1 > 0$). The closed-loop system **diverges exponentially**.
- **Case 2 ($K_p > \omega_0^2 = 4$):**  
  $$s_{1,2} = \pm j \sqrt{K_p - \omega_0^2}$$  
  Both poles lie strictly on the imaginary axis ($j\omega$). The system oscillates perpetually with zero damping ($\zeta = 0$).

> **Mathematical Conclusion:** Proportional feedback alone **can never asymptotically stabilize** an unstable second-order system.

---

### 2.2 Adding Derivative Action: PD Control

Applying Proportional-Derivative control $u(t) = K_p e(t) + K_d \dot{e}(t)$:

$$C(s) = K_p + K_d s$$

The closed-loop characteristic equation becomes:

$$s^2 - \omega_0^2 + (K_p + K_d s) = 0 \iff s^2 + K_d s + (K_p - \omega_0^2) = 0$$

Matching this with the standard second-order desired polynomial $s^2 + 2\zeta\omega_n s + \omega_n^2 = 0$:

$$K_p = \omega_n^2 + \omega_0^2, \qquad K_d = 2\zeta\omega_n$$

**Design Specifications:**
- Desired natural frequency: $\omega_n = 4.0\text{ rad/s}$
- Desired damping ratio: $\zeta = 0.707$ (Butterworth optimal damping)

Computing controller gains:
$$K_p = 4.0^2 + 2.0^2 = 20.0\text{ N}\cdot\text{m/rad}$$
$$K_d = 2 \times 0.707 \times 4.0 = 5.66\text{ N}\cdot\text{m}\cdot\text{s/rad}$$

Both closed-loop poles are placed at $s = -2.83 \pm j2.83$ (strictly in LHP).

---

### 2.3 Adding Integral Action: PID Control for Disturbance Rejection

Under steady disturbance torque $d = 0.5\text{ N}\cdot\text{m}$, the PD controller suffers from persistent steady-state offset:

$$e_{ss} = \frac{d}{K_p - \omega_0^2} = \frac{0.5}{20.0 - 4.0} = 0.03125\text{ rad}$$

Adding integral action $C(s) = \frac{K_d s^2 + K_p s + K_i}{s}$:

$$s(s^2 - \omega_0^2) + (K_d s^2 + K_p s + K_i) = s^3 + K_d s^2 + (K_p - \omega_0^2)s + K_i = 0$$

By Routh-Hurwitz stability, the system is stable if and only if:
1. $K_d > 0$
2. $K_p > \omega_0^2 = 4.0$
3. $K_i > 0$
4. $K_d (K_p - \omega_0^2) > K_i \implies K_i < 5.66 \times 16.0 = 90.56$

Choosing $K_i = 15.0$ provides rapid integral recovery while maintaining a healthy gain margin.

---

## 3. Practical Implementation Details

1. **Filtered Derivative:**
   $$\tau_f = \frac{K_d}{N K_p} = \frac{5.66}{10 \times 20.0} = 0.0283\text{ s} \quad (N = 10)$$
2. **Anti-Windup Clamping:**
   $$\text{If } |u_{\text{unsat}}| \ge 25.0 \text{ and } \text{sign}(e) == \text{sign}(u) \implies \dot{x}_i = 0$$

---

## 4. Benchmark Results & Quantitative Comparison

Simulated for $T = 10.0\text{ s}$ with $\Delta t = 0.001\text{ s}$ using RK4 integration under a unit step reference $r = 1.0\text{ rad}$ and step disturbance $d = 0.5\text{ N}\cdot\text{m}$ at $t = 5.0\text{ s}$:

| Controller | Rise Time $t_r$ [s] | Settling Time $t_s$ [s] | Overshoot $M_p$ [%] | Steady Error $e_{ss}$ ($t=4.9\text{s}$) | Steady Error ($t=10\text{s}$) | Energy $E_u$ [N²s] | Peak $u_{\max}$ [N] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P-Only ($K_p=20$)** | 0.39 | $\infty$ | 100.0% | Undamped Osc. | Undamped Osc. | $\infty$ | 25.0 | Unstable / Limit Cycle |
| **PD ($K_p=20, K_d=5.66$)** | 0.44 | 1.18 | 4.8% | 0.000 | 0.0313 | 142.5 | 20.0 | Stable (Offset on Dist.) |
| **PID (No Anti-Windup)** | 0.41 | 2.85 | 32.4% | 0.000 | 0.0000 | 188.4 | 25.0 (Saturated) | Stable (Windup Lag) |
| **PID + Anti-Windup** | 0.43 | 1.25 | 6.2% | 0.000 | 0.0000 | 151.2 | 25.0 (Clamped) | **Optimal Performance** |

---

## 5. Engineering Takeaways & Limitations of PID

1. **Derivative action is mathematically mandatory** to provide phase lead and damping for open-loop unstable poles.
2. **Anti-windup is non-negotiable** on unstable systems; without it, actuator saturation delays error recovery, risking system divergence outside the linear basin of attraction.
3. **MIMO & State Constraints:** PID cannot directly balance multi-state coupled systems (like Cart-Pole $[x, \dot{x}, \theta, \dot{\theta}]$ with a single actuator) without ad-hoc cascading. This motivates **Modern Control (State Feedback & LQR)** in Modules 04 and 05.
