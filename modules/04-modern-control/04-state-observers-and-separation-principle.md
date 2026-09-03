# State Observers & The Separation Principle

> **Module 04: Modern Control** | Theory Note 04  
> Focus: Luenberger observer, error dynamics, dual pole placement, observer-based feedback, and mathematical proof of the Separation Principle.

---

## 1. The Full-State Luenberger Observer

In physical applications, measuring the complete state vector $x(t) \in \mathbb{R}^n$ is rarely feasible (due to sensor cost, weight, or accessibility). A **Luenberger Observer** reconstructs an estimate $\hat{x}(t)$ using known inputs $u(t)$ and measured outputs $y(t)$:

$$\dot{\hat{x}}(t) = \underbrace{A \hat{x}(t) + B u(t)}_{\text{Plant Model Simulation}} + \underbrace{L (y(t) - C \hat{x}(t))}_{\text{Output Error Correction}}$$

where:
- $\hat{x}(t) \in \mathbb{R}^n$: Estimated state vector.
- $\hat{y}(t) = C \hat{x}(t)$: Predicted measurement.
- $e_y(t) = y(t) - \hat{y}(t) = C(x - \hat{x})$: Innovation / measurement residual.
- $L \in \mathbb{R}^{n \times p}$: Observer gain matrix (innovation weighting).

---

## 2. Observer Error Dynamics

Define the state estimation error:

$$e(t) \triangleq x(t) - \hat{x}(t)$$

Differentiating with respect to time:

$$\begin{aligned}
\dot{e}(t) &= \dot{x}(t) - \dot{\hat{x}}(t) \\
&= (A x + B u) - (A \hat{x} + B u + L(C x - C \hat{x})) \\
&= A (x - \hat{x}) - L C (x - \hat{x}) \\
&= (A - LC) e(t)
\end{aligned}$$

The error dynamics $\dot{e}(t) = (A - LC)e(t)$ are:
1. **Autonomous:** Driven solely by initial estimation error $e(0) = x(0) - \hat{x}(0)$, completely independent of control input $u(t)$ or reference $r(t)$.
2. **Asymptotically Stable:** If and only if $(A - LC)$ is Hurwitz ($\text{Re}(\lambda_i(A - LC)) < 0$).

### 2.1 Dual Ackermann's Formula for Observer Gain $L$
Using the mathematical duality $(A, C) \leftrightarrow (A^T, C^T)$:

$$L = \Delta_{\text{des}}^{\text{obs}}(A) \, \mathcal{O}(A, C)^{-1} \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$

---

## 3. Observer-Based Output Feedback & The Separation Principle

When true state $x(t)$ is replaced by estimated state $\hat{x}(t)$, the control law becomes:

$$u(t) = -K \hat{x}(t) = -K (x(t) - e(t))$$

Substituting $u(t)$ into the plant dynamics:

$$\dot{x}(t) = A x(t) + B(-K x(t) + K e(t)) = (A - BK)x(t) + BK e(t)$$

Augmenting the plant state and estimation error into a combined system of dimension $2n$:

$$\begin{bmatrix} \dot{x}(t) \\ \dot{e}(t) \end{bmatrix} = \begin{bmatrix} A - BK & BK \\ 0 & A - LC \end{bmatrix} \begin{bmatrix} x(t) \\ e(t) \end{bmatrix}$$

### 3.1 Proof of the Separation Principle

The characteristic polynomial of the combined $2n$-dimensional closed-loop system is:

$$\det \left( s I_{2n} - \begin{bmatrix} A - BK & BK \\ 0 & A - LC \end{bmatrix} \right) = \det \begin{bmatrix} sI - (A - BK) & -BK \\ 0 & sI - (A - LC) \end{bmatrix}$$

Because the block matrix is upper block-triangular, the determinant is the product of diagonal block determinants:

$$\det(sI_{2n} - A_{\text{total}}) = \det(sI - (A - BK)) \cdot \det(sI - (A - LC))$$

> **The Separation Principle Theorem:**  
> The eigenvalues of the combined controller-observer system split cleanly into:
> $$\sigma(A_{\text{total}}) = \sigma(A - BK) \cup \sigma(A - LC)$$
> The state feedback gain $K$ and observer gain $L$ can be designed **completely independently** without altering each other's closed-loop poles.

---

## 4. Observer Tuning Guidelines

- **The $3\times$ to $5\times$ Bandwidth Rule:** Place observer eigenvalues 3 to 5 times faster (more negative) than controller eigenvalues:
  $$\text{Re}(\lambda_{\text{obs}}) \approx (3 \sim 5) \times \text{Re}(\lambda_{\text{ctrl}})$$
- *Reasoning:* Fast estimation ensures $\hat{x}(t) \to x(t)$ rapidly before control actions peak, minimizing transient lag.
- *Upper Limit:* Never place observer poles faster than sensor noise bandwidth or ADC sampling frequency to avoid injecting high-frequency measurement noise.
