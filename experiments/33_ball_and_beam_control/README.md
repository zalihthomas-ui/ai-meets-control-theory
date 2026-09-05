# Experiment 33: Ball and Beam Underactuated Balance & Position Control Benchmark

## 1. Executive Summary

This experiment evaluates classical cascaded loop control (**Cascade PID**), nonlinear geometric decoupling (**Partial Feedback Linearization / PFL**), state-space optimal control (**Multivariable LQR**), and receding-horizon constrained quadratic programming (**Linear MPC**) on the **Ball and Beam** underactuated control benchmark using canonical hardware parameters from the **Quanser Ball and Beam** laboratory apparatus.

```
                  +-----------------------------------+
                  |      Direct-Drive DC Motor        |
                  |     Control Torque tau(t)         |
                  +-----------------+-----------------+
                                    |
                                    v
            r = -L/2 = -0.2125 m          r = +L/2 = +0.2125 m
            [-----------------------o------------------------]  Beam length L = 0.425 m
                     <-- r(t) -->  ( ) Ball (m = 0.064 kg, r_b = 1.27 cm)
                                  /
                                 / theta(t) (Beam angle)
                                v
```

---

## 2. Physical Principles & Governing Equations

The analytical equations of motion are derived via Euler-Lagrange mechanics in [`docs/references/ball-and-beam-reference.md`](../../docs/references/ball-and-beam-reference.md):

$$\begin{aligned}
\ddot{r} &= \frac{5}{7}\left( r \dot{\theta}^2 - g \sin\theta \right) - \frac{c_r}{m_{\text{eff}}} \dot{r} \\
\ddot{\theta} &= \frac{\tau - 2 m r \dot{r} \dot{\theta} - m g r \cos\theta - b \dot{\theta}}{J + m r^2}
\end{aligned}$$

where:
- Ball effective inertia factor: $m_{\text{eff}} = \frac{7}{5} m$ (due to solid sphere rolling constraint $J_b = \frac{2}{5} m r_b^2$).
- Total beam + base moment of inertia: $J = J_{\text{beam}} + J_{\text{motor}} = \frac{1}{12} M L^2 + J_{\text{motor}} = 3.010 \times 10^{-3}\text{ kg}\cdot\text{m}^2$.

### 2.1 Canonical Hardware Parameters (Quanser Standard)
- **Ball mass**: $m = 0.064\text{ kg}$, **Ball radius**: $r_b = 0.0127\text{ m}$ ($1.27\text{ cm}$).
- **Beam mass**: $M = 0.20\text{ kg}$, **Beam length**: $L = 0.425\text{ m}$ ($42.5\text{ cm}$).
- **Motor torque limit**: $\tau_{\max} = 1.50\text{ N}\cdot\text{m}$.
- **Beam travel limit**: $|\theta| \le 30^\circ \approx 0.5236\text{ rad}$.
- **Ball travel limit**: $|r| \le L/2 = 0.2125\text{ m}$.
- **Friction**: Viscous beam damping $b = 0.05\text{ N}\cdot\text{m}\cdot\text{s/rad}$, rolling friction $c_r = 0.002\text{ N}\cdot\text{s/m}$.

### 2.2 Linearization & Open-Loop Instability
Linearizing the equations of motion around the horizontal origin $x_0 = [0, 0, 0, 0]^T, u_0 = 0$:

$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & -\frac{c_r}{m_{\text{eff}}} & -\frac{5}{7}g & 0 \\ 0 & 0 & 0 & 1 \\ -\frac{mg}{J} & 0 & 0 & -\frac{b}{J} \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \frac{1}{J} \end{bmatrix}$$

For the canonical parameters:
- Open-loop eigenvalues: $\lambda = \{-16.91, -1.92 \pm 4.16j, \mathbf{+4.123}\}\text{ rad/s}$.
- The system possesses an unstable open-loop pole at $+4.123\text{ rad/s}$ and relative degree 4 from torque input $\tau$ to ball position $r$.

---

## 3. Benchmark Quantitative Results

The benchmark evaluates a large setpoint transition from initial position $r_0 = -10.0\text{ cm}$ to target setpoint $r^* = +10.0\text{ cm}$ over a $5.0\text{ s}$ horizon ($dt = 0.002\text{ s}$).

| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ [cm] | RMSE [cm] | Energy $E_u$ [$\text{N}^2\cdot\text{m}^2\cdot\text{s}$] | Peak Torque $|\tau|_{\max}$ [N$\cdot$m] | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cascade PID** | 0.284 | 5.000 | 10.5 | 5.51e-01 | 5.81 | 0.0243 | 1.500 | 0.0 | **Stable** |
| **PFL (Nonlinear)** | 0.302 | 2.404 | 18.3 | 2.51e-03 | 5.88 | 0.0249 | 1.500 | 0.1 | **Stable** |
| **Multivariable LQR** | 0.576 | 1.492 | 1.3 | 1.85e-05 | 6.46 | 0.0187 | 0.966 | 0.0 | **Stable** |
| **Linear MPC** | 0.576 | 1.494 | 1.3 | 1.83e-05 | 6.47 | 0.0184 | 0.784 | 0.0 | **Stable** |

---

## 4. Key Controller Insights & Dynamics

1. **Cascade PID vs. Multivariable State-Space**:
   - The classical cascade architecture uses an outer loop ($r \to \theta_{\text{cmd}}$) and inner loop ($\theta \to \tau$). While intuitive, inner-loop phase lag causes slight transient oscillation ($M_p = 10.5\%$) and a longer settling tail ($t_s = 5.0\text{ s}$).
   - In contrast, multivariable state-feedback (**LQR** & **Linear MPC**) coordinates beam rotation and ball translation simultaneously, achieving near-zero overshoot ($M_p = 1.3\%$) and rapid monotonic settling ($t_s = 1.49\text{ s}$).

2. **Partial Feedback Linearization (PFL)**:
   - PFL computes the exact nonlinear gravity torque cancellation $\tau_{\text{grav}} = m g r \cos\theta + 2 m r \dot{r} \dot{\theta}$ and inverts the inertial coupling.
   - It converges precisely ($e_{ss} = 2.51 \times 10^{-3}\text{ cm}$) with settling time $t_s = 2.40\text{ s}$, but displays higher overshoot ($18.3\%$) during aggressive acceleration phases.

3. **Linear MPC Constraint Management**:
   - Linear MPC explicitly constrains the control authority to $\tau \in [-1.5, +1.5]\text{ N}\cdot\text{m}$ and beam angle to $\theta \in [-30^\circ, +30^\circ]$ over a prediction horizon $N = 25$ ($T_h = 0.5\text{ s}$).
   - MPC achieves the lowest peak control torque ($0.784\text{ N}\cdot\text{m}$) and lowest control energy ($0.0184\text{ N}^2\cdot\text{m}^2\cdot\text{s}$) while matching LQR tracking accuracy.

---

## 5. Visualizations

The generated phase portraits, state trajectories, and control torque histories are shown in `ball_and_beam_benchmark.png`.

---

## 6. How to Run

To execute the full simulation benchmark and regenerate all artifacts:

```bash
python experiments/33_ball_and_beam_control/run.py
```
