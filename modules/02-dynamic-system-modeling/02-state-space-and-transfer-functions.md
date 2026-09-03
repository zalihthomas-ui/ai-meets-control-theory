# State-Space Realizations & Transfer Functions

> **Module 02: Dynamic System Modeling** | Theory Note 02  
> Focus: State-space representations, Laplace domain transfer functions, coordinate changes, and canonical forms.

---

## 1. The Continuous-Time LTI State-Space Framework

Any linear time-invariant physical system can be formulated in standard state-space form:

$$\begin{aligned}
\dot{x}(t) &= A x(t) + B u(t) \qquad &(\text{State Dynamics Equation}) \\
y(t) &= C x(t) + D u(t) \qquad &(\text{Measurement Output Equation})
\end{aligned}$$

where:
- $x(t) \in \mathbb{R}^n$: State vector (minimal set of dynamic variables storing system energy/history).
- $u(t) \in \mathbb{R}^m$: Control input vector.
- $y(t) \in \mathbb{R}^p$: Measured sensor output vector.
- $A \in \mathbb{R}^{n \times n}$: System matrix.
- $B \in \mathbb{R}^{n \times m}$: Input matrix.
- $C \in \mathbb{R}^{p \times n}$: Output / measurement matrix.
- $D \in \mathbb{R}^{p \times m}$: Direct feedthrough / transmission matrix ($D = 0$ for strictly proper systems).

---

## 2. Derivation of the Transfer Function Matrix $G(s)$

Taking the bilateral Laplace transform $\mathcal{L}\{\cdot\}$ assuming zero initial conditions ($x(0) = 0$):

$$s X(s) = A X(s) + B U(s) \implies (sI - A) X(s) = B U(s)$$

Since $(sI - A)$ is invertible for all complex frequencies $s \notin \sigma(A)$:

$$X(s) = (sI - A)^{-1} B U(s)$$

Substituting $X(s)$ into the Laplace-transformed output equation $Y(s) = C X(s) + D U(s)$:

$$Y(s) = \left[ C (sI - A)^{-1} B + D \right] U(s)$$

The matrix **Transfer Function** $G(s) \in \mathbb{C}^{p \times m}$ is:

$$G(s) = C (sI - A)^{-1} B + D = \frac{C \text{adj}(sI - A) B}{\det(sI - A)} + D$$

---

## 3. Coordinate Transformations & System Realizations

State variables are not unique. Let $T \in \mathbb{R}^{n \times n}$ be an invertible change of coordinates:

$$z(t) = T x(t) \iff x(t) = T^{-1} z(t)$$

Differentiating with respect to time:

$$\dot{z}(t) = T \dot{x}(t) = T (A x(t) + B u(t)) = (T A T^{-1}) z(t) + (T B) u(t)$$
$$y(t) = C x(t) + D u(t) = (C T^{-1}) z(t) + D u(t)$$

The transformed realization $(\tilde{A}, \tilde{B}, \tilde{C}, \tilde{D})$ is:

$$\tilde{A} = T A T^{-1}, \quad \tilde{B} = T B, \quad \tilde{C} = C T^{-1}, \quad \tilde{D} = D$$

### 3.1 Invariance Properties
- **Transfer Function Invariance:** $\tilde{G}(s) = \tilde{C}(sI - \tilde{A})^{-1}\tilde{B} + \tilde{D} \equiv G(s)$.
- **Eigenvalue Invariance:** $\det(sI - \tilde{A}) = \det(T(sI - A)T^{-1}) = \det(sI - A)$.

---

## 4. Canonical Forms

For an $n$-th order SISO transfer function $G(s) = \frac{b_{n-1} s^{n-1} + \dots + b_1 s + b_0}{s^n + a_{n-1} s^{n-1} + \dots + a_1 s + a_0}$:

### 4.1 Controllable Canonical Form (CCF)
$$A_{ccf} = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & 1 \\ -a_0 & -a_1 & -a_2 & \dots & -a_{n-1} \end{bmatrix}, \quad B_{ccf} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$
$$C_{ccf} = \begin{bmatrix} b_0 & b_1 & \dots & b_{n-1} \end{bmatrix}, \quad D = 0$$

### 4.2 Observable Canonical Form (OCF)
$$A_{ocf} = A_{ccf}^T, \quad B_{ocf} = C_{ccf}^T, \quad C_{ocf} = B_{ccf}^T, \quad D = 0$$

---

## 5. Poles, Zeros, and Minimal Realizations

- **Poles:** The roots of the characteristic polynomial $\det(sI - A) = 0$. Poles dictate the unforced natural dynamics (damping and frequency).
- **Zeros:** The complex frequencies $s_0$ where the transmission gain $G(s_0) = 0$ (or loss of rank for MIMO).
- **Minimal Realization:** A state-space realization $(A, B, C, D)$ is **minimal** if and only if it has the smallest possible state dimension $n$ to represent $G(s)$. A realization is minimal if and only if it is simultaneously **controllable** and **observable** (no hidden pole-zero cancellations).

---

## 6. Python Implementation (`scipy.signal`)

```python
import numpy as np
from scipy import signal

# Define 2nd-order Mass-Spring-Damper state space
m, c, k = 1.0, 0.5, 4.0
A = np.array([[0.0, 1.0], [-k/m, -c/m]])
B = np.array([[0.0], [1.0/m]])
C = np.array([[1.0, 0.0]]) # Measure position
D = np.array([[0.0]])

# Convert State-Space to Transfer Function
ss_sys = signal.StateSpace(A, B, C, D)
tf_sys = signal.ss2tf(A, B, C, D)

num, den = tf_sys
print(f"Numerator coefficients:   {num}")
print(f"Denominator coefficients: {den}")
# G(s) = 1 / (s^2 + 0.5s + 4)
```
