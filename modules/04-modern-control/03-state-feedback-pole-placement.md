# Full-State Feedback & Pole Placement

> **Module 04: Modern Control** | Theory Note 03  
> Focus: State feedback control law, closed-loop pole placement, Ackermann's formula, reference tracking feedforward, and actuator trade-offs.

---

## 1. Full-State Feedback Control Architecture

If all state variables $x(t) \in \mathbb{R}^n$ are measured or estimated, the control input is formed as a linear combination of states:

$$u(t) = -K x(t) + k_r r(t)$$

where $K \in \mathbb{R}^{m \times n}$ is the state feedback gain matrix, $r(t) \in \mathbb{R}^p$ is the reference command, and $k_r$ is the reference feedforward scaling factor.

```
r(t) ──►[ k_r ]──(+)──► u(t) ──►[ Plant: ẋ = Ax + Bu ]──► x(t) ──►[ C ]──► y(t)
                  ▲ -                                       │
                  │                                         │
                  └──────────────[ Gain Matrix K ]──────────┘
```

Substituting the feedback law into the open-loop dynamics $\dot{x} = Ax + Bu$:

$$\dot{x}(t) = A x(t) + B (-K x(t) + k_r r(t)) = (A - BK) x(t) + B k_r r(t)$$

The closed-loop dynamics are governed by the **closed-loop system matrix** $A_{\text{cl}} \triangleq A - BK$.

---

## 2. Eigenvalue Assignment & Ackermann's Formula

**Theorem (Wonham, 1967):** If the pair $(A, B)$ is controllable, the closed-loop eigenvalues $\sigma(A - BK) = \{\mu_1, \mu_2, \dots, \mu_n\}$ can be placed **arbitrarily** in the complex plane (with complex conjugate pairing).

### 2.1 Desired Characteristic Polynomial
Let the target closed-loop pole locations define the desired polynomial:

$$\Delta_{\text{des}}(s) \triangleq \prod_{i=1}^n (s - \mu_i) = s^n + d_{n-1} s^{n-1} + \dots + d_1 s + d_0$$

### 2.2 Derivation of Ackermann's Formula (SISO)
By the Cayley-Hamilton theorem, the matrix $A_{\text{cl}} = A - BK$ satisfies its own characteristic equation:

$$\Delta_{\text{des}}(A - BK) = 0$$

Expanding $\Delta_{\text{des}}(A)$ in terms of $(A - BK)$:

$$\Delta_{\text{des}}(A) = \mathcal{C} \begin{bmatrix} * \\ \vdots \\ * \\ K \end{bmatrix}$$

Premultiplying by the last row of the inverse controllability matrix $\mathcal{C}^{-1}$:

$$K = \begin{bmatrix} 0 & 0 & \dots & 0 & 1 \end{bmatrix} \mathcal{C}(A, B)^{-1} \Delta_{\text{des}}(A)$$

where $\Delta_{\text{des}}(A) = A^n + d_{n-1}A^{n-1} + \dots + d_1 A + d_0 I$.

---

## 3. Reference Tracking & Feedforward Scaling $k_r$

With pure regulator feedback $u = -Kx$, the steady-state output for a step reference $r$ is:

$$0 = (A - BK) x_{ss} + B k_r r \implies x_{ss} = -(A - BK)^{-1} B k_r r$$
$$y_{ss} = C x_{ss} = -C (A - BK)^{-1} B k_r r$$

To achieve unity steady-state gain ($y_{ss} \equiv r$):

$$k_r = -\frac{1}{C (A - BK)^{-1} B}$$

---

## 4. Engineering Trade-Offs in Pole Selection

While controllability permits placing poles arbitrarily far into the left-half plane ($\text{Re}(\mu) \ll 0$), real-world physics imposes strict limits:

1. **Actuator Saturation:** Faster closed-loop bandwidth demands exponentially larger instantaneous control efforts $u(t) \approx \|K\|_2 \|x_0\|$.
2. **Noise Sensitivity:** High-gain feedback amplifies sensor measurement noise into violent actuator chattering.
3. **Bandwidth Rule:** Never push closed-loop bandwidth beyond $1/5$-th of the highest unmodeled structural resonance or digital sampling frequency $f_s$.
