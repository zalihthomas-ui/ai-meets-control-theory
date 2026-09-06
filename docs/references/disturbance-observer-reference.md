# Disturbance-Observer (DOB) Control & Q-Filter Reference Specification

## 1. Executive Summary & Foundational Principles

A **Disturbance Observer (DOB)** (pioneered by Ohnishi in 1987 and formalized by Umeno & Hori in 1991) is a two-degree-of-freedom control architecture that reconstructs lumped internal and external disturbances from measurable plant inputs and outputs, feeding an estimated cancellation signal back into the control channel.

```
                      d(t) [Disturbance]
                       |
                       v
   r(t)   +     u_base +       u(t)   +-----------------+  y(t)
   ----->( + )-------->( + )--------->|   Plant P(s)    |--------+-------->
          ^ -           ^ -           +-----------------+        |
          |             |                                        |
          |             +----------------[ Q(s) ]<---------------+
          |                                 ^                    |
          |                                 | -                  |
          |                                ( + )<--[ P_n^{-1}(s) ]
          |                                 |
          |                                u(t)
          +------------------------------------------------------+
```

### Key Architectural Advantages
1. **Nominal Loop Recovery**: Inside the filter bandwidth ($\omega < \omega_Q$), the closed-loop system behaves identically to the nominal model $P_n(s)$, rendering outer-loop controllers (LQR, MPC, pole placement) insensitive to parameter variations and external forces.
2. **Phase-Lead Rejection vs. Integrator Lag**: Unlike classical integral action $\int e(t) dt$, which introduces a $-90^\circ$ phase lag and requires substantial tracking error accumulation before developing corrective authority, DOB reconstructs disturbance forces directly from acceleration/momentum mismatch, cancelling disturbances before tracking errors accumulate.
3. **Bandwidth Separation**: By tuning the low-pass Q-filter cutoff frequency $\omega_Q$, high-frequency sensor noise and unmodeled high-frequency structural resonances are blocked from the feedback path.

---

## 2. Frequency-Domain Formulation & Q-Filter Design

### 2.1 Closed-Loop Sensitivity Analysis
Let the true physical plant be $P(s)$, subject to input disturbance $d(s)$ and measurement noise $\eta(s)$:

$$Y(s) = P(s)[U(s) + D(s)] + N(s)$$

With nominal model $P_n(s)$ and low-pass filter $Q(s)$, the disturbance estimate is:

$$\hat{D}(s) = Q(s) \left[ P_n^{-1}(s) Y(s) - U(s) 
ight]$$

Applying the inner-loop cancellation $U(s) = U_{    ext{base}}(s) - \hat{D}(s)$, the actual control applied to the plant is:

$$U(s) = rac{1}{1 - Q(s) + Q(s) P_n^{-1}(s) P(s)} U_{    ext{base}}(s) - rac{Q(s) P_n^{-1}(s)}{1 - Q(s) + Q(s) P_n^{-1}(s) P(s)} [P(s) D(s) + N(s)]$$

When the nominal model matches the plant ($P(s) = P_n(s)$), the transfer functions simplify to:

$$egin{aligned}
Y(s) &= G_{ry}(s) U_{    ext{base}}(s) + G_{dy}(s) D(s) + G_{ny}(s) N(s) \
G_{ry}(s) &= P_n(s) \
G_{dy}(s) &= P_n(s) \left[ 1 - Q(s) 
ight] \
G_{ny}(s) &= -Q(s)
\end{aligned}$$

### 2.2 Rejection & Robustness Duality
- **Disturbance Sensitivity**: $S_{    ext{dob}}(s) = 1 - Q(s)$.
  - For low frequencies $\omega \ll \omega_Q$: $Q(j\omega) pprox 1 \implies S_{    ext{dob}}(j\omega) pprox 0$ ($\implies$ perfect disturbance rejection).
- **Complementary Sensitivity / Noise Transmission**: $T_{    ext{dob}}(s) = Q(s)$.
  - For high frequencies $\omega \gg \omega_Q$: $Q(j\omega)     o 0 \implies T_{    ext{dob}}(j\omega)     o 0$ ($\implies$ complete sensor noise attenuation).

---

## 3. Q-Filter Topologies & Relative Degree Matching

For the inverse nominal model $P_n^{-1}(s)$ to be causal and physically realizable in $Q(s) P_n^{-1}(s)$, the filter $Q(s)$ must satisfy the **relative degree condition**:

$$    ext{deg}_{    ext{rel}}(Q(s)) \ge     ext{deg}_{    ext{rel}}(P_n(s)) = n - m$$

where $n = \deg(    ext{denominator})$ and $m = \deg(    ext{numerator})$.

### 3.1 Standard Q-Filter Topologies

1. **Relative Degree 1 (First-Order Velocity Systems)**:
   $$Q_1(s) = rac{1}{    au_Q s + 1}, \quad \omega_Q = rac{1}{    au_Q}$$

2. **Relative Degree 2 (Double-Integrator Mechanical / Second-Order Systems)**:
   - *Binomial (Critically Damped)*:
     $$Q_{2,    ext{bin}}(s) = rac{1}{(    au_Q s + 1)^2} = rac{1}{    au_Q^2 s^2 + 2    au_Q s + 1}$$
   - *Butterworth Standard (Maximally Flat Passband, $\zeta = 1/\sqrt{2}$)*:
     $$Q_{2,    ext{bw}}(s) = rac{1 + \sqrt{2}    au_Q s}{(    au_Q s + 1)^2} \quad (    ext{deg}_{    ext{rel}} = 1) \quad     ext{or} \quad Q_{2,    ext{bw}}(s) = rac{\omega_Q^2}{s^2 + \sqrt{2}\omega_Q s + \omega_Q^2} \quad (    ext{deg}_{    ext{rel}} = 2)$$

3. **Relative Degree 3 (Third-Order Jitter / Torque-Driven Flexible Systems)**:
   $$Q_3(s) = rac{1 + 3    au_Q s}{(    au_Q s + 1)^3} \quad (    ext{deg}_{    ext{rel}} = 2) \quad     ext{or} \quad Q_3(s) = rac{1}{(    au_Q s + 1)^3} \quad (    ext{deg}_{    ext{rel}} = 3)$$

### 3.2 Discrete-Time Realization via Tustin (Bilinear) Transform
Using Tustin bilinear substitution $s = rac{2}{dt} rac{1 - z^{-1}}{1 + z^{-1}}$ with pre-warping $\omega_w = rac{2}{dt}    an\left(rac{\omega_Q dt}{2}
ight)$:

For the second-order filter $Q(s) = rac{\omega_Q^2}{s^2 + 2\zeta \omega_Q s + \omega_Q^2}$:
$$Q(z) = rac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}$$

where:
$$egin{aligned}
a_0 &= 4 + 4\zeta \omega_Q dt + (\omega_Q dt)^2 \
a_1 &= -8 + 2(\omega_Q dt)^2 \
a_2 &= 4 - 4\zeta \omega_Q dt + (\omega_Q dt)^2 \
b_0 &= (\omega_Q dt)^2, \quad b_1 = 2(\omega_Q dt)^2, \quad b_2 = (\omega_Q dt)^2
\end{aligned}$$

---

## 4. State-Space & Nonlinear Generalized Momentum DOB

### 4.1 State-Space Disturbance Observer (Extended State Formulation)
Consider the linear multivariable plant with unmatched and matched disturbances:

$$egin{aligned}
\dot{x} &= A x + B u + B_d d \
y &= C x
\end{aligned}$$

Assuming constant/slowly varying disturbance dynamics $\dot{d} = 0 + w_d$, the augmented extended system is:

$$egin{bmatrix} \dot{x} \ \dot{d} \end{bmatrix} = egin{bmatrix} A & B_d \ 0 & 0 \end{bmatrix} egin{bmatrix} x \ d \end{bmatrix} + egin{bmatrix} B \ 0 \end{bmatrix} u$$

With observer gain $L = egin{bmatrix} L_x \ L_d \end{bmatrix}$, the observer equations are:

$$egin{aligned}
\dot{\hat{x}} &= A \hat{x} + B u + B_d \hat{d} + L_x(y - C \hat{x}) \
\dot{\hat{d}} &= L_d(y - C \hat{x})
\end{aligned}$$

### 4.2 Nonlinear State-Space DOB with Auxiliary State (Chen et al. Formulation)
For nonlinear dynamics $\dot{x} = f(x) + g(x)u + B_d d$, direct numerical differentiation of $x$ is avoided by introducing an auxiliary state $z = \hat{d} - p(x)$:

$$egin{aligned}
\dot{z} &= -L(x) B_d z - L(x) \left[ f(x) + g(x) u + B_d p(x) 
ight] \
\hat{d} &= z + p(x)
\end{aligned}$$

where the observer gain matrix is $L(x) = rac{\partial p(x)}{\partial x}$.
Defining the disturbance estimation error $e_d = d - \hat{d}$:

$$\dot{e}_d = \dot{d} - \dot{\hat{d}} = -\dot{z} - L(x)\dot{x} = L(x)B_d(z + p(x) - d) = -L(x)B_d e_d$$

Choosing constant gain $p(x) = K_{    ext{dob}} x \implies L(x) = K_{    ext{dob}}$, the estimation error decays exponentially:

$$\dot{e}_d = -K_{    ext{dob}} B_d e_d \implies e_d(t) = e^{-K_{    ext{dob}} B_d t} e_d(0)$$

---

## 5. Matched vs. Unmatched Disturbance Rejection on Planar Quadrotor

### 5.1 Planar Quadrotor Mechanics & Wind Disturbance Model
The planar quadrotor state is $x = [p_x, p_z,     heta, v_x, v_z, \omega]^T \in \mathbb{R}^6$ with control inputs $u = [T_1, T_2]^T \in \mathbb{R}^2$.

Total thrust: $T = T_1 + T_2$; Differential torque: $    au = (T_1 - T_2)\ell$.

Under external wind force $\mathbf{F}_{    ext{wind}} = [F_{w, x}, F_{w, z}]^T$ and aerodynamic moment $    au_w$:

$$egin{aligned}
\ddot{x} &= -rac{T}{m}\sin    heta - rac{c_d}{m} v_x + rac{F_{w, x}}{m} \
\ddot{z} &= rac{T}{m}\cos    heta - g - rac{c_d}{m} v_z + rac{F_{w, z}}{m} \
\ddot{    heta} &= rac{    au}{I_{yy}} + rac{    au_w}{I_{yy}}
\end{aligned}$$

### 5.2 Matched vs. Unmatched Channel Classification

| Channel | Physical Variable | Control Input | Disturbance | Coupling Classification | Rejection Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Altitude ($z$)** | $\ddot{z}$ | Total Thrust $T$ | $F_{w, z} / m$ | **MATCHED** ($    ext{span}(B)$) | Direct cancellation: $\Delta T = -\hat{F}_{w, z} / \cos    heta$ |
| **Attitude ($    heta$)** | $\ddot{    heta}$ | Differential Torque $    au$ | $    au_w / I_{yy}$ | **MATCHED** ($    ext{span}(B)$) | Direct cancellation: $\Delta     au = -\hat{    au}_w$ |
| **Horizontal ($x$)** | $\ddot{x}$ | *None* ($B_x = 0$) | $F_{w, x} / m$ | **UNMATCHED** ($
otin     ext{span}(B)$) | Virtual control reallocation via pitch tilt: $    heta_{    ext{cmd}} =     heta_{    ext{base}} + rac{\hat{F}_{w, x}}{m g}$ |

### 5.3 Why DOB Outperforms Integral Action on Unmatched Disturbance

1. **Integral Action on Position Error ($e_x = x - x_{    ext{ref}}$)**:
   - Control law: $    heta_{    ext{cmd}} = -K_{p,x} e_x - K_{d,x} \dot{e}_x - K_{i,x} \int e_x dt$.
   - **Mechanism**: The drone *must drift away downwind* by a substantial distance $\Delta x$ over multiple seconds for $\int e_x dt$ to accumulate sufficient tilt angle $    heta_{    ext{trim}} = rctan(F_{w,x} / (mg))$.
   - **Phase Lag & Overshoot**: The integrator adds a pole at $s = 0$ ($-90^\circ$ phase lag), eroding phase margin, causing sluggish recovery, and creating substantial transient overshoot when gusts change suddenly.

2. **Disturbance Observer (DOB) on Acceleration Mismatch**:
   - Reconstructed translational acceleration mismatch:
     $$\hat{d}_x = Q(s) \left[ \ddot{x}_{    ext{meas}} + rac{T}{m}\sin    heta + rac{c_d}{m} v_x 
ight] pprox rac{F_{w, x}}{m}$$
   - **Mechanism**: DOB detects the wind force *instantaneously* through dynamic inertial mismatch without requiring positional displacement.
   - **Feedforward Tilt Correction**:
     $$    heta_{    ext{cmd}} =     heta_{    ext{base}} + rcsin\left(rac{m \hat{d}_x}{T}
ight) pprox     heta_{    ext{base}} + rac{\hat{d}_x}{g}$$
   - **Result**: The drone immediately pitches its nose into the oncoming wind gust, maintaining positional hold with near-zero drift ($e_{ss} pprox 0$) and zero integrator-induced phase lag.

---

## 6. Robust Stability & Small-Gain Theorem

### 6.1 Multiplicative Uncertainty Bound
Let the true plant model have multiplicative unmodeled dynamics:

$$P(s) = P_n(s) \left[ 1 + \Delta_m(s) 
ight], \quad |\Delta_m(j\omega)| \le W_m(\omega)$$

### 6.2 Small-Gain Condition
By the Small-Gain Theorem, the closed-loop system with DOB is robustly stable if and only if:

$$\| W_m(s) Q(s) \|_\infty < 1 \iff |Q(j\omega)| < rac{1}{W_m(\omega)}, \quad orall \omega \in \mathbb{R}$$

### 6.3 Tuning Guidelines for Cutoff Frequency $\omega_Q$
1. **Low-Frequency Rejection**: $\omega_Q \ge 3 \cdot \omega_{    ext{wind}}$ (where $\omega_{    ext{wind}}$ is the dominant frequency of atmospheric turbulence gusts, typically $0.5 - 2    ext{ rad/s}$).
2. **High-Frequency Attenuation**: $\omega_Q \le rac{1}{3} \cdot \omega_{    ext{resonance}}$ (where $\omega_{    ext{resonance}}$ is the lowest unmodeled structural resonance or motor lag pole, typically $20 - 50    ext{ rad/s}$).
3. **Canonical Quadrotor Tuning**: $\omega_Q = 10.0    ext{ rad/s}$ ($    au_Q = 0.10    ext{ s}$), damping $\zeta_Q = 1.0$ (critically damped binomial).
