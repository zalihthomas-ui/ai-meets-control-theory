# Differential Equations & Dynamical Systems

> **Module 01: Mathematical Foundations** | Theory Note 02  
> Focus: State solutions, Duhamel's principle, phase portraits, and Lyapunov stability.

---

## 1. Continuous-Time Linear State Equations

A continuous-time linear time-invariant (LTI) dynamical system is described by the first-order vector differential equation:

$$\dot{x}(t) = A x(t) + B u(t), \quad x(0) = x_0$$

where $x(t) \in \mathbb{R}^n$ is the state vector, $u(t) \in \mathbb{R}^m$ is the control input vector, $A \in \mathbb{R}^{n \times n}$ is the state dynamics matrix, and $B \in \mathbb{R}^{n \times m}$ is the input distribution matrix.

---

## 2. Derivation of the General Solution (Duhamel's Formula)

### 2.1 Integrating Factor Method

To solve the non-homogeneous vector ODE, premultiply both sides by the matrix integrating factor $e^{-At}$:

$$e^{-At} \dot{x}(t) - e^{-At} A x(t) = e^{-At} B u(t)$$

Using the product rule for matrix differentiation:

$$\frac{d}{dt}\left( e^{-At} x(t) \right) = -A e^{-At} x(t) + e^{-At} \dot{x}(t) = e^{-At} \dot{x}(t) - e^{-At} A x(t)$$

Thus, the differential equation simplifies to:

$$\frac{d}{dt}\left( e^{-At} x(t) \right) = e^{-At} B u(t)$$

Integrating both sides from $\tau = 0$ to $\tau = t$:

$$e^{-At} x(t) - e^{0} x(0) = \int_0^t e^{-A\tau} B u(\tau) \, d\tau$$

Premultiplying by $e^{At}$ yields the complete solution:

$$x(t) = \underbrace{e^{At} x_0}_{\text{Zero-Input Response (ZIR)}} + \underbrace{\int_0^t e^{A(t-\tau)} B u(\tau) \, d\tau}_{\text{Zero-State Response (ZSR)}}$$

### 2.2 The State Transition Matrix

The operator $\Phi(t, \tau) \triangleq e^{A(t-\tau)}$ satisfies:
1. $\Phi(t, t) = I$
2. $\Phi(t_2, t_0) = \Phi(t_2, t_1) \Phi(t_1, t_0)$
3. $\Phi(t, \tau)^{-1} = \Phi(\tau, t)$
4. $\frac{\partial}{\partial t} \Phi(t, \tau) = A \Phi(t, \tau)$

---

## 3. Phase Plane Analysis & 2D Equilibrium Classification

For an autonomous 2D system $\dot{x} = A x$ with eigenvalues $\lambda_{1, 2} = \sigma \pm j\omega$:

```
                        Im(λ)
                          │
                   Unstable Focus │
                          │  • (λ = +σ + jω)
                          │
  Stable Node             │             Unstable Node
  •───•───────────────────┼───────────────────•───•─── Re(λ)
(λ₂ < λ₁ < 0)             │                 (0 < λ₁ < λ₂)
                          │
                     Center (λ = ±jω)
                          │
                    Stable Focus  │
                          │  • (λ = -σ - jω)
                          │
```

| Eigenvalues $\lambda_{1, 2}$ | Phase Portrait Geometric Topology | Stability Classification |
| :--- | :--- | :--- |
| $\lambda_2 < \lambda_1 < 0$ | **Stable Node:** Trajectories converge tangentially to dominant eigenvector. | Asymptotically Stable |
| $0 < \lambda_1 < \lambda_2$ | **Unstable Node:** Trajectories diverge exponentially outward. | Unstable |
| $\lambda_1 < 0 < \lambda_2$ | **Saddle Point:** Converges along stable manifold, diverges along unstable manifold. | Unstable |
| $\sigma \pm j\omega$ ($\sigma < 0$) | **Stable Spiral / Focus:** Trajectories spiral inward toward origin. | Asymptotically Stable |
| $\sigma \pm j\omega$ ($\sigma > 0$) | **Unstable Spiral / Focus:** Trajectories spiral outward with growing amplitude. | Unstable |
| $\pm j\omega$ ($\sigma = 0$) | **Center:** Concentric closed periodic orbits with constant energy. | Marginally Stable |

---

## 4. Stability Theory & The Continuous Lyapunov Equation

### 4.1 Internal Stability (Hurwitz Condition)

The autonomous system $\dot{x} = Ax$ is **asymptotically stable** if and only if matrix $A$ is **Hurwitz**:

$$\text{Re}(\lambda_i(A)) < 0 \quad \forall i \in \{1, 2, \dots, n\}$$

### 4.2 Lyapunov Stability Theorem

Consider a quadratic energy function (Lyapunov candidate):

$$V(x) = x^T P x, \quad P = P^T \succ 0$$

Taking the time derivative along system trajectories $\dot{x} = Ax$:

$$\dot{V}(x) = \dot{x}^T P x + x^T P \dot{x} = (Ax)^T P x + x^T P (Ax) = x^T (A^T P + P A) x$$

For asymptotic stability, we require $\dot{V}(x) < 0$ for all $x \ne 0$. Setting:

$$A^T P + P A = -Q$$

for any symmetric positive definite matrix $Q \succ 0$ (e.g., $Q = I$):

$$\dot{V}(x) = -x^T Q x \le -\lambda_{\min}(Q) \|x\|_2^2 < 0 \quad \forall x \ne 0$$

**Theorem:** Matrix $A$ is Hurwitz if and only if for any given symmetric $Q \succ 0$, there exists a unique symmetric $P \succ 0$ satisfying the Continuous Lyapunov Equation $A^T P + P A = -Q$.

---

## 5. Computational Implementation (Python / NumPy)

```python
import numpy as np
from scipy import linalg

def solve_lyapunov_stability(A: np.ndarray, Q: np.ndarray = None) -> tuple[bool, np.ndarray]:
    """Solve A^T P + P A = -Q and verify asymptotic stability."""
    n = A.shape[0]
    if Q is None:
        Q = np.eye(n)
        
    # Solve continuous Lyapunov equation: A^T P + P A + Q = 0
    # scipy.linalg.solve_continuous_lyapunov solves A X + X A^H = Q -> pass A.T and -Q
    P = linalg.solve_continuous_lyapunov(A.T, -Q)
    
    # Check if P is positive definite (all eigenvalues > 0)
    p_eigvals = np.linalg.eigvalsh(P)
    is_stable = np.all(p_eigvals > 1e-9)
    return is_stable, P

# Example: Stable second-order oscillator with damping
A_stable = np.array([[0.0, 1.0], [-5.0, -2.0]])
stable, P_mat = solve_lyapunov_stability(A_stable)
print(f"System is Hurwitz: {stable}")
print(f"Lyapunov Matrix P:\n{P_mat}")
```
