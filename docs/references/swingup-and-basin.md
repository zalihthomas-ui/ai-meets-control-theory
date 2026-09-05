# Nonlinear Swing-Up & Basin of Attraction Reference Specification

This document provides analytical formulations, Lyapunov energy-shaping laws, switching conditions, and quantitative basin-of-attraction envelopes for the **Simple Pendulum** (`aimct.systems.Pendulum`) and **Cart-Pole System** (`aimct.systems.CartPole`).

---

## 1. Energy-Based Swing-Up Control

Nonlinear swing-up control steers an underactuated pendulum from the stable downward resting state to the vicinity of the unstable upright equilibrium, where a local linear controller (such as LQR or State Feedback) captures and stabilizes the state.

```
       [ Downward Rest: theta = pi ]
                    |
                    v
       [ Energy-Shaping Swing-Up ]  (Pumps mechanical energy into orbit)
                    |
                    v
       [ Homoclinic Capture Region ] (|theta| <= 20 deg, |theta_dot| <= 1.5 rad/s)
                    |
                    v
       [ Linear LQR Balance Mode ]  (Asymptotic regulation to origin)
```

---

### 1.1 Simple Pendulum Swing-Up (Åström-Furuta Law)

#### System Dynamics
$$J \ddot{\theta} + b \dot{\theta} + m g l \sin\theta = u$$

where $J = m l^2$, $\theta = 0$ is the upward inverted position, and $\theta = \pm \pi$ is the downward rest.

#### Energy Formulation
Total mechanical energy (with potential zero set at the upright equilibrium $\theta = 0$):
$$E(\theta, \dot{\theta}) = \frac{1}{2} J \dot{\theta}^2 + m g l (\cos\theta - 1)$$

- At upright equilibrium ($\theta = 0, \dot{\theta} = 0$): $E_0 = 0.0\text{ J}$.
- At downward rest ($\theta = \pi, \dot{\theta} = 0$): $E_{\text{bottom}} = -2 m g l\text{ J}$.

#### Lyapunov Energy-Shaping Control Law
Taking the time derivative of energy along system trajectories:
$$\dot{E} = J \dot{\theta} \ddot{\theta} - m g l \sin\theta \dot{\theta} = \dot{\theta} (u - b \dot{\theta})$$

Ignoring friction during swing-up, setting $u = -k_E \dot{\theta} (E - E_0)$ yields:
$$\dot{E} = -k_E \dot{\theta}^2 (E - E_0)$$
which guarantees $\dot{E} > 0$ whenever $E < E_0$, driving the system monotonically toward the target homoclinic orbit $E = E_0$.

#### Saturated Control Law with Sign-Velocity Modulation
$$u(t) = \text{clip}\left( k_E \cdot \text{sign}(\dot{\theta} \cos\theta) \cdot (E_0 - E(\theta, \dot{\theta})), \ -u_{\max}, \ +u_{\max} \right)$$

- **Canonical Tuning**: $k_E = 1.5 \cdot \frac{1}{m g l} \approx 0.153\text{ J}^{-1}$, $u_{\max} = 2.5\text{ N}\cdot\text{m}$.

---

### 1.2 Cart-Pole Swing-Up (Spong Energy-Shaping Law)

For the inverted pendulum on a cart, actuation acts on the cart position $x$, coupling into the pendulum through inertial reaction force.

#### Pendulum Energy
$$E_p(\theta, \dot{\theta}) = \frac{1}{2} (I + m l^2) \dot{\theta}^2 + m g l (\cos\theta - 1)$$
Target homoclinic energy: $E_0 = 0.0\text{ J}$.

#### Desired Cart Acceleration ($\ddot{x}_{\text{des}}$)
Following Mark Spong (1995), the cart acceleration is commanded to pump energy into the pendulum while maintaining the cart near the center of the track:

$$\ddot{x}_{\text{des}} = k_E \cdot (E_p - E_0) \cdot \text{sign}(\dot{\theta} \cos\theta) - k_{p,x} x - k_{d,x} \dot{x}$$

where:
- $k_E = 0.8\text{ m}/(\text{s}^2\cdot\text{J})$: Energy pumping gain.
- $k_{p,x} = 2.0\text{ s}^{-2}$: Cart centering proportional gain (prevents rail collisions).
- $k_{d,x} = 1.5\text{ s}^{-1}$: Cart centering derivative damping.

#### Force Mapping via Partial Feedback Linearization
Given desired cart acceleration $\ddot{x}_{\text{des}}$, the commanded motor force $F$ is computed from the coupled equations:

$$F = \left( (M + m) - \frac{m^2 l^2 \cos^2\theta}{I + m l^2} \right) \ddot{x}_{\text{des}} - m l \dot{\theta}^2 \sin\theta - \frac{m g l \sin\theta \cos\theta}{I + m l^2}$$

---

## 2. Hybrid Switching Architecture (Swing-Up to LQR)

To guarantee smooth transition and avoid high-frequency switching chattering, a **hysteresis state-machine** is employed:

```
                  +-----------------------------------+
                  |        MODE 1: SWING-UP           |
                  |     (Spong Energy Shaping)        |
                  +-----------------+-----------------+
                                    |
          Condition: |theta| <= 0.35 rad (20 deg) AND
                     |theta_dot| <= 1.5 rad/s
                                    |
                                    v
                  +-----------------------------------+
                  |        MODE 2: LQR BALANCE        |
                  |          u = -K (x - x_ref)       |
                  +-----------------+-----------------+
                                    |
          Exit Condition (Escape): |theta| > 0.60 rad (34 deg)
                                    |
                                    +---> (Fall back to Swing-Up)
```

---

## 3. Quantitative Basin of Attraction for Cart-Pole LQR

When initialized near upright without swing-up assistance, the linear LQR controller $u = -K x$ stabilizes the full nonlinear `CartPole` system within the basin of attraction $\mathcal{B}$.

We distinguish between two bounds:
1. **Guaranteed Lyapunov Invariant Sub-Level Set (Inner Certificate)**: $\Omega_{c^*} = \{x \in \mathbb{R}^4 \mid x^T P x \le c^*\}$ where $\dot{V}(x) < 0$ along unconstrained linear flow.
2. **Measured Recovery Basin (Empirical Boundary with $\pm 20\text{ N}$ Saturation)**: Empirically measured via high-resolution parameter sweeps in Experiment 05.

### 3.1 Initial Condition Recovery Boundaries ($x_0 = [0, 0, \theta_0, \dot{\theta}_0]^T$)

Simulated on `CartPole` ($M=1.0\text{ kg}, m=0.1\text{ kg}, l=0.5\text{ m}$) under RK4 integration ($dt = 0.001\text{ s}$, $T = 6.0\text{ s}$–$8.0\text{ s}$, $|F| \le 20\text{ N}$):

| LQR Configuration | Weights $(Q, R)$ | Lyapunov Inner Bound $|\theta_0|_{\text{lyap}}$ | Measured Max Angle $|\theta_0|_{\max}$ ($\dot{\theta}_0=0$) | Measured Max Ang. Vel $|\dot{\theta}_0|_{\max}$ ($\theta_0=0$) | Settling Time $t_s$ ($2\%$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Set 1: Standard Balanced** | $Q=\text{diag}([10, 1, 100, 10]), R=0.1$ | $0.38\text{ rad}$ ($21.8^\circ$) | **$0.83\text{ rad}$ ($47.6^\circ$)** | **$4.4\text{ rad/s}$** | $2.45\text{ s}$ |
| **Set 2: Aggressive Angle** | $Q=\text{diag}([1, 0.1, 1000, 10]), R=0.01$ | $0.45\text{ rad}$ ($25.8^\circ$) | **$0.92\text{ rad}$ ($52.7^\circ$)** | **$5.3\text{ rad/s}$** | $0.82\text{ s}$ |
| **Set 3: Soft Energy-Saving** | $Q=\text{diag}([1, 0.1, 10, 1]), R=1.0$ | $0.22\text{ rad}$ ($12.6^\circ$) | **$1.00\text{ rad}$ ($57.3^\circ$)** | **$5.3\text{ rad/s}$** | $4.10\text{ s}$ |

---

### 3.2 Ellipsoidal Lyapunov Invariant Sets vs. True Basin

The quadratic Lyapunov function $V(x) = x^T P x$ provides a conservative analytical certificate $\Omega_{c^*} = \{x \in \mathbb{R}^4 \mid x^T P x \le c^*\}$.

- **Why Lyapunov Certificates are Conservative:**  
  The sub-level set $\Omega_{c^*}$ requires $\dot{V}(x) < 0$ for *all* points in the ellipsoid simultaneously. However, system trajectories that start outside $\Omega_{c^*}$ can temporarily increase $V(x)$ while still remaining bounded and eventually entering $\Omega_{c^*}$ asymptotically.
- **Why Saturation Enlarges the Recoverable Basin:**  
  On large initial angles ($> 45^\circ$), an unconstrained linear law $u = -Kx$ commands massive forces ($> 200\text{ N}$), driving numerical instability in the coupled non-minimum phase dynamics. Clamping to $\pm 20\text{ N}$ naturally tempers cart acceleration, allowing the pole's momentum to be smoothly arrested.
