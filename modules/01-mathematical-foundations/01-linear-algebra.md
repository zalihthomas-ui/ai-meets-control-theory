# Linear Algebra for Dynamical Systems

> **Module 01: Mathematical Foundations** | Theory Note 01  
> Focus: Vector spaces, spectral decomposition, matrix exponential, quadratic forms, and SVD.

---

## 1. State Vectors & Linear Transformations

In continuous-time control theory, the state of a physical system with $n$ degrees of freedom is represented as a state vector $x(t) \in \mathbb{R}^n$. The system's evolution is governed by linear transformations on this state space:

$$x(t) = \begin{bmatrix} x_1(t) \\ x_2(t) \\ \vdots \\ x_n(t) \end{bmatrix}, \quad \dot{x}(t) = A x(t)$$

where $A \in \mathbb{R}^{n \times n}$ is the system state matrix.

### 1.1 Vector and Matrix Norms

Quantifying system performance, tracking error, and stability requires vector and induced matrix norms:

- **Euclidean Norm ($\ell_2$):** $\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^T x}$. Represents geometric distance / energy in state space.
- **Maximum / Infinity Norm ($\ell_\infty$):** $\|x\|_\infty = \max_{1 \le i \le n} |x_i|$. Represents peak state deviation.
- **Induced 2-Norm (Spectral Norm of a Matrix):**
  $$\|A\|_2 = \sup_{x \ne 0} \frac{\|Ax\|_2}{\|x\|_2} = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$$

---

## 2. Spectral Decomposition & Modal Analysis

### 2.1 Eigenvalues and Eigenvectors

The scalar $\lambda \in \mathbb{C}$ and non-zero vector $v \in \mathbb{C}^n$ satisfy the eigenvalue problem:

$$A v = \lambda v \iff (\lambda I - A)v = 0$$

Non-trivial solutions exist if and only if $\lambda$ satisfies the **characteristic equation**:

$$p(\lambda) = \det(\lambda I - A) = \lambda^n + a_{n-1}\lambda^{n-1} + \dots + a_1 \lambda + a_0 = 0$$

### 2.2 Diagonalization & Modal Decoupling

If matrix $A$ has $n$ linearly independent eigenvectors $V = [v_1, v_2, \dots, v_n]$, then $A$ is diagonalizable:

$$A = V \Lambda V^{-1}, \quad \Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$$

Under the modal coordinate transformation $z(t) = V^{-1} x(t)$, the coupled system $\dot{x} = Ax$ decouples into $n$ independent scalar differential equations:

$$\dot{z}(t) = V^{-1}\dot{x}(t) = V^{-1} A V z(t) = \Lambda z(t) \implies \dot{z}_i(t) = \lambda_i z_i(t)$$

Each mode evolves independently as $z_i(t) = z_i(0) e^{\lambda_i t}$. The eigenvalues $\lambda_i = \sigma_i + j\omega_i$ dictate the dynamic behavior:
- $\text{Re}(\lambda_i) < 0$: Mode decays exponentially to zero (stable).
- $\text{Re}(\lambda_i) > 0$: Mode grows exponentially to infinity (unstable).
- $\text{Im}(\lambda_i) \ne 0$: Mode oscillates at natural frequency $\omega_i\text{ [rad/s]}$.

### 2.3 Defective Matrices & Jordan Canonical Form

When an eigenvalue $\lambda_k$ has algebraic multiplicity $m_a >$ geometric multiplicity $m_g$, matrix $A$ cannot be diagonalized. It is transformed into its **Jordan Canonical Form**:

$$A = V J V^{-1}, \quad J = \begin{bmatrix} J_1 & & 0 \\ & \ddots & \\ 0 & & J_p \end{bmatrix}, \quad J_k = \begin{bmatrix} \lambda_k & 1 & & 0 \\ & \lambda_k & \ddots & \\ & & \ddots & 1 \\ 0 & & & \lambda_k \end{bmatrix}$$

The corresponding state response contains polynomial-exponential terms $t^k e^{\lambda t}$, which govern resonant growth in critical systems.

---

## 3. Symmetric Matrices & Quadratic Forms

Quadratic forms represent energy, Lyapunov candidate functions, and cost functionals in optimal control:

$$V(x) = x^T P x = \sum_{i=1}^n \sum_{j=1}^n P_{ij} x_i x_j$$

Without loss of generality, $P \in \mathbb{R}^{n \times n}$ is assumed symmetric ($P = P^T$).

### 3.1 Positive Definiteness

A symmetric matrix $P = P^T$ is:
- **Positive Definite ($P \succ 0$):** $x^T P x > 0$ for all $x \ne 0 \iff \lambda_i(P) > 0 \quad \forall i$.
- **Positive Semi-Definite ($P \succeq 0$):** $x^T P x \ge 0$ for all $x \iff \lambda_i(P) \ge 0 \quad \forall i$.

### 3.2 Sylvester's Criterion & Cholesky Factorization

- **Sylvester's Criterion:** $P \succ 0$ if and only if all leading principal minors $\Delta_k = \det(P_{1:k, 1:k}) > 0$ for $k = 1, \dots, n$.
- **Cholesky Factorization:** If $P \succ 0$, there exists a unique lower-triangular matrix $L$ with strictly positive diagonal entries such that:
  $$P = L L^T$$

In numerical control routines (e.g., Riccati solvers), Cholesky factorization is used for numerically stable square-root filtering and state transformations.

---

## 4. The Matrix Exponential $e^{At}$

The matrix exponential is the fundamental solution operator for continuous linear systems $\dot{x}(t) = A x(t)$.

### 4.1 Formal Definition

$$e^{At} \triangleq \sum_{k=0}^\infty \frac{(At)^k}{k!} = I + At + \frac{1}{2!}A^2 t^2 + \frac{1}{3!}A^3 t^3 + \dots$$

### 4.2 Key Analytical Properties

1. **Identity at $t = 0$:** $e^{A \cdot 0} = I$.
2. **Derivative:** $\frac{d}{dt} e^{At} = A e^{At} = e^{At} A$.
3. **Inverse:** $(e^{At})^{-1} = e^{-At}$.
4. **Group Property:** $e^{A(t_1 + t_2)} = e^{A t_1} e^{A t_2}$ (holds for all $t_1, t_2$).
5. **Commuting Matrices:** $e^{(A+B)t} = e^{At} e^{Bt}$ if and only if $AB = BA$.

### 4.3 Computational Methods

1. **Diagonalization:** If $A = V \Lambda V^{-1}$, then:
   $$e^{At} = V \text{diag}\left(e^{\lambda_1 t}, e^{\lambda_2 t}, \dots, e^{\lambda_n t}\right) V^{-1}$$
2. **Laplace Inversion:** $e^{At} = \mathcal{L}^{-1}\left\{ (sI - A)^{-1} \right\}$.
3. **Padé Approximation with Scaling and Squaring:** The standard numerical algorithm implemented in `scipy.linalg.expm`. Matrix $A$ is scaled by $2^{-m}$ such that $\|2^{-m} A\| \le 0.5$, a rational Padé polynomial approximates $e^{2^{-m} A}$, and the result is squared $m$ times.

---

## 5. Singular Value Decomposition (SVD) & Directional Gain

For any matrix $M \in \mathbb{R}^{p \times m}$, the SVD decomposes $M$ into:

$$M = U \Sigma V^T$$

where $U \in \mathbb{R}^{p \times p}$ and $V \in \mathbb{R}^{m \times m}$ are orthogonal matrices, and $\Sigma \in \mathbb{R}^{p \times m}$ contains non-negative singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min} \ge 0$ on its diagonal.

### 5.1 Physical Interpretation in Multi-Variable Control

For a MIMO transfer function matrix $G(j\omega)$, the singular values quantify amplification as a function of input direction:

$$\bar{\sigma}(G(j\omega)) = \max_{u \ne 0} \frac{\|G(j\omega) u\|_2}{\|u\|_2}, \quad \underline{\sigma}(G(j\omega)) = \min_{u \ne 0} \frac{\|G(j\omega) u\|_2}{\|u\|_2}$$

- $\bar{\sigma}(G(j\omega))$: Maximum gain across all spatial input directions at frequency $\omega$ ($H_\infty$ norm peak).
- $\underline{\sigma}(G(j\omega))$: Minimum gain (worst-case attenuation direction).

---

## 6. Computational Verification (Python / NumPy)

```python
import numpy as np
from scipy import linalg

# Define a 2nd order system matrix (e.g. damped oscillator)
# x_dot = [0, 1; -k/m, -c/m] x
m, k, c = 1.0, 4.0, 0.5
A = np.array([[0.0, 1.0], [-k / m, -c / m]])

# 1. Eigenvalues and Eigenvectors
eigenvals, V = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvals}")

# 2. Matrix Exponential at t = 0.5s via Padé scaling & squaring
t = 0.5
Phi_pade = linalg.expm(A * t)

# 3. Matrix Exponential via Modal Decomposition
Lambda = np.diag(eigenvals)
Phi_modal = np.real(V @ np.diag(np.exp(eigenvals * t)) @ np.linalg.inv(V))

# Verify numerical agreement
assert np.allclose(Phi_pade, Phi_modal, atol=1e-12)
print(f"State transition matrix Phi(0.5):\n{Phi_pade}")
```
