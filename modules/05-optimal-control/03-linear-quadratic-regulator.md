# The Linear Quadratic Regulator (LQR)

> **Module 05: Optimal Control** | Theory Note 03  
> Focus: Infinite-horizon continuous and discrete LQR, Riccati equation derivations, Hamiltonian Schur method, and Python solvers.

---

## 1. Continuous-Time Infinite-Horizon LQR Problem

Given a continuous LTI system $\dot{x}(t) = Ax(t) + Bu(t)$ with initial condition $x(0) = x_0$, find the state feedback law $u^*(t)$ that minimizes the infinite-horizon quadratic performance index:

$$J(u) = \int_0^\infty \left( x(t)^T Q x(t) + u(t)^T R u(t) \right) \, dt$$

where:
- $Q = Q^T \succeq 0$: State weighting matrix (penalizes state deviations).
- $R = R^T \succ 0$: Control weighting matrix (penalizes actuator energy).

### 1.1 Solvability Conditions
A unique positive-definite stabilizing solution exists if and only if:
1. The pair $(A, B)$ is **stabilizable**.
2. The pair $(A, Q^{1/2})$ is **detectable** (or observable).

---

## 2. Derivation of the Continuous Algebraic Riccati Equation (CARE)

Assume a quadratic optimal value function (cost-to-go from state $x$):

$$V(x) = x^T P x, \qquad P = P^T \succ 0$$

By the **Hamilton-Jacobi-Bellman (HJB)** equation of dynamic programming:

$$\min_{u} \left[ x^T Q x + u^T R u + \nabla_x V(x)^T (A x + B u) \right] = 0$$

Since $\nabla_x V(x) = 2 P x$:

$$\min_{u} \left[ x^T Q x + u^T R u + 2 x^T P A x + 2 x^T P B u \right] = 0$$

Taking the partial derivative with respect to $u$ and setting to zero:

$$\frac{\partial}{\partial u}[\dots] = 2 R u + 2 B^T P x = 0 \implies u^*(t) = -R^{-1} B^T P x(t)$$

Defining the **Optimal LQR Gain Matrix**:
$$K \triangleq R^{-1} B^T P \implies u^*(t) = -K x(t)$$

Substituting $u^*$ back into the HJB equation:

$$x^T Q x + (-Kx)^T R (-Kx) + 2 x^T P A x + 2 x^T P B (-Kx) = 0$$
$$x^T \left( Q + P B R^{-1} B^T P + P A + A^T P - 2 P B R^{-1} B^T P \right) x = 0$$

Since this must hold for all non-zero states $x(t) \in \mathbb{R}^n$, we obtain the **Continuous Algebraic Riccati Equation (CARE)**:

$$A^T P + P A - P B R^{-1} B^T P + Q = 0$$

The closed-loop dynamics $\dot{x} = (A - BK)x$ are guaranteed strictly asymptotically stable (Hurwitz), and the minimum cost is:

$$J^* = x_0^T P x_0$$

---

## 3. Numerical Solution via the Hamiltonian Matrix (Schur Method)

Construct the $2n \times 2n$ **Hamiltonian Matrix**:

$$\mathcal{H}_{\text{ham}} \triangleq \begin{bmatrix} A & -B R^{-1} B^T \\ -Q & -A^T \end{bmatrix}$$

### 3.1 Spectral Symmetry
The eigenvalues of $\mathcal{H}_{\text{ham}}$ are symmetric about the imaginary axis: if $\lambda \in \sigma(\mathcal{H}_{\text{ham}})$, then $-\lambda \in \sigma(\mathcal{H}_{\text{ham}})$. There are exactly $n$ eigenvalues in the open left-half plane ($\text{Re}(\lambda) < 0$) and $n$ in the open right-half plane.

### 3.2 Stable Invariant Subspace
Let the eigenvectors spanning the $n$ stable eigenvalues be partitioned as:

$$\mathcal{H}_{\text{ham}} \begin{bmatrix} X_1 \\ X_2 \end{bmatrix} = \begin{bmatrix} X_1 \\ X_2 \end{bmatrix} \Lambda_-, \qquad X_1, X_2 \in \mathbb{R}^{n \times n}$$

The unique positive-definite stabilizing solution to the Riccati equation is:

$$P = X_2 X_1^{-1}$$

This real Schur decomposition method is implemented in `scipy.linalg.solve_continuous_are`.

---

## 4. Discrete-Time LQR (DARE)

For sampled-data systems $x_{k+1} = A_d x_k + B_d u_k$ minimizing $J = \sum_{k=0}^\infty (x_k^T Q x_k + u_k^T R u_k)$:

The **Discrete Algebraic Riccati Equation (DARE)** is:

$$P = A_d^T P A_d - A_d^T P B_d (R + B_d^T P B_d)^{-1} B_d^T P A_d + Q$$

The optimal discrete state feedback gain is:

$$K_d = (R + B_d^T P B_d)^{-1} B_d^T P A_d \implies u_k = -K_d x_k$$

---

## 5. Python Implementation (`scipy.linalg`)

```python
import numpy as np
from scipy import linalg

def compute_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Compute continuous LQR gain K and Riccati solution P."""
    # Solve CARE: A^T P + P A - P B R^-1 B^T P + Q = 0
    P = linalg.solve_continuous_are(A, B, Q, R)
    # Compute gain K = R^-1 B^T P
    K = np.linalg.inv(R) @ (B.T @ P)
    # Closed loop eigenvalues
    A_cl = A - B @ K
    cl_poles = np.linalg.eigvals(A_cl)
    return K, P, cl_poles

# Cart-Pole Linearized Model
A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -0.71, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 15.76, 0.0]
])
B = np.array([[0.0], [1.0], [0.0], [-1.0]])

# Weights: Bryson-style
Q = np.diag([10.0, 1.0, 100.0, 10.0])
R = np.array([[0.1]])

K, P, poles = compute_lqr(A, B, Q, R)
print(f"Optimal LQR Gain K: {K}")
print(f"Closed-loop poles: {poles}")
```
