# Discretization & Sampled-Data Systems

> **Module 02: Dynamic System Modeling** | Theory Note 04  
> Focus: Sampled-data systems, Zero-Order Hold (ZOH) discretization, Van Loan's method, and numerical comparison.

---

## 1. Digital Control & Sampled-Data Architecture

Modern control laws are executed on digital processors (microcontrollers, DSPs, embedded computers) running at discrete sampling period $\Delta t$:

```
        Discrete Controller               D/A (ZOH)            Continuous Plant
      ┌─────────────────────┐          ┌─────────────┐       ┌──────────────────┐
r_k ─►│  u_k = -K x_k       │──► u_k ─►│ u(t) = u_k  │─►u(t)─┤ ẋ(t) = Ax(t)+Bu  │──► x(t)
      │  (Digital Algorithm)│          │ t ∈ [tk, tk+1)      │ y(t) = Cx(t)     │      │
      └─────────────────────┘          └─────────────┘       └──────────────────┘      │
                 ▲                                                                     │
                 │                     A/D Sampler                                     │
                 └─────────────────────────────────────────────────────────────────────┘
                                       x_k = x(kΔt)
```

---

## 2. Derivation of Exact Zero-Order Hold (ZOH) Discretization

Given continuous LTI dynamics $\dot{x}(t) = Ax(t) + Bu(t)$ and constant input $u(\tau) = u_k$ over $\tau \in [t_k, t_k + \Delta t]$:

Applying Duhamel's formula across one sampling interval $[t_k, t_{k+1}]$:

$$x(t_{k+1}) = e^{A\Delta t} x(t_k) + \int_{t_k}^{t_k + \Delta t} e^{A(t_k + \Delta t - \tau)} B u(\tau) \, d\tau$$

Substituting $\sigma = t_k + \Delta t - \tau$ (so $d\tau = -d\sigma$):

$$x_{k+1} = e^{A\Delta t} x_k + \left( \int_0^{\Delta t} e^{A\sigma} d\sigma \right) B u_k$$

Thus, the exact discrete-time state-space representation is:

$$x_{k+1} = A_d x_k + B_d u_k, \qquad y_k = C_d x_k + D_d u_k$$

where:
$$A_d \triangleq e^{A\Delta t}$$
$$B_d \triangleq \left( \int_0^{\Delta t} e^{A\sigma} d\sigma \right) B$$
$$C_d = C, \qquad D_d = D$$

When matrix $A$ is invertible ($\det(A) \ne 0$):

$$B_d = A^{-1}\left( e^{A\Delta t} - I \right) B$$

---

## 3. Van Loan's Method for Exact Numerical Computation

When matrix $A$ is singular (e.g., systems containing pure rigid-body integrators), directly inverting $A$ fails. **Van Loan's Method** computes $A_d$ and $B_d$ simultaneously via a single augmented matrix exponential:

$$\mathcal{M} = \begin{bmatrix} A & B \\ 0_{m \times n} & 0_{m \times m} \end{bmatrix} \in \mathbb{R}^{(n+m) \times (n+m)}$$

Computing the matrix exponential of $\mathcal{M} \Delta t$:

$$e^{\mathcal{M}\Delta t} = \exp \left( \begin{bmatrix} A \Delta t & B \Delta t \\ 0 & 0 \end{bmatrix} \right) = \begin{bmatrix} A_d & B_d \\ 0_{m \times n} & I_{m \times m} \end{bmatrix}$$

This method is numerically stable, exact, and requires no matrix inverses.

---

## 4. Comparison of Discretization Methods

| Method | Formulation ($A_d, B_d$) | Order of Accuracy | Stability & Phase Fidelity |
| :--- | :--- | :--- | :--- |
| **Exact ZOH** | $A_d = e^{A\Delta t}, \; B_d = \int_0^{\Delta t} e^{A\tau} B d\tau$ | Exact for piecewise-constant $u$ | Preserves exact state transitions at sample times. |
| **Forward Euler** | $A_d = I + A\Delta t, \; B_d = B\Delta t$ | $\mathcal{O}(\Delta t)$ | Shifts poles rightward; introduces artificial instability. |
| **Backward Euler** | $A_d = (I - A\Delta t)^{-1}, \; B_d = (I - A\Delta t)^{-1} B\Delta t$ | $\mathcal{O}(\Delta t)$ | Highly damping; maps left-half plane into stable unit circle. |
| **Tustin (Bilinear)** | $s \leftarrow \frac{2}{\Delta t} \frac{z-1}{z+1}$ | $\mathcal{O}(\Delta t^2)$ | Maps entire LHP exactly inside unit disk; warps frequencies. |

---

## 5. Python Implementation (`scipy.signal.cont2discrete`)

```python
import numpy as np
from scipy import linalg, signal

# Continuous Mass-Spring-Damper
m, c, k = 1.0, 0.5, 4.0
A = np.array([[0.0, 1.0], [-k/m, -c/m]])
B = np.array([[0.0], [1.0/m]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

dt = 0.05 # Sampling period 50 ms

# 1. Exact ZOH via Van Loan's Method
M = np.block([
    [A * dt, B * dt],
    [np.zeros((1, 2)), np.zeros((1, 1))]
])
exp_M = linalg.expm(M)
A_d_vanloan = exp_M[:2, :2]
B_d_vanloan = exp_M[:2, 2:]

# 2. Exact ZOH via Scipy
d_sys = signal.cont2discrete((A, B, C, D), dt=dt, method="zoh")
A_d_scipy, B_d_scipy = d_sys[0], d_sys[1]

# Verify exact match
assert np.allclose(A_d_vanloan, A_d_scipy, atol=1e-12)
print("Discrete Matrix A_d:\n", A_d_scipy)
print("Discrete Matrix B_d:\n", B_d_scipy)
```
