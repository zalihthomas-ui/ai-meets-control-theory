# Optimization Foundations

> **Module 01: Mathematical Foundations** | Theory Note 04  
> Focus: Convex optimization, Lagrange multipliers, Karush-Kuhn-Tucker (KKT) conditions, and Quadratic Programming (QP) in control.

---

## 1. Unconstrained Optimization

Optimal control policies minimize scalar performance criteria (e.g., tracking error, energy consumption, execution time). An unconstrained optimization problem has the form:

$$\min_{x \in \mathbb{R}^n} f(x)$$

where $f: \mathbb{R}^n \to \mathbb{R}$ is twice continuously differentiable ($C^2$).

### 1.1 Optimality Conditions

- **First-Order Necessary Condition (FONC):** If $x^*$ is a local minimizer, the gradient vanishes:
  $$\nabla f(x^*) = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \dots & \frac{\partial f}{\partial x_n} \end{bmatrix}^T = 0$$
- **Second-Order Sufficient Condition (SOSC):** If $\nabla f(x^*) = 0$ and the Hessian matrix is strictly positive definite:
  $$\nabla^2 f(x^*) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_i \partial x_j} \end{bmatrix} \succ 0$$
  then $x^*$ is an isolated strict local minimizer.

### 1.2 Convexity

A set $\mathcal{C} \subseteq \mathbb{R}^n$ is convex if $\alpha x + (1-\alpha)y \in \mathcal{C}$ for all $x, y \in \mathcal{C}$ and $\alpha \in [0, 1]$.  
A function $f: \mathcal{C} \to \mathbb{R}$ is **convex** if:

$$f(\alpha x + (1-\alpha)y) \le \alpha f(x) + (1-\alpha) f(y) \quad \forall x, y \in \mathcal{C}, \; \alpha \in [0, 1]$$

- **Fundamental Theorem of Convex Optimization:** For a convex objective over a convex feasible set, **every local minimum is a global minimum**. If $f$ is strictly convex ($\nabla^2 f(x) \succ 0$), the global minimizer is unique.

---

## 2. Constrained Optimization & Lagrange Multipliers

Consider equality-constrained minimization:

$$\min_{x \in \mathbb{R}^n} f(x) \quad \text{subject to} \quad h(x) = 0$$

where $h: \mathbb{R}^n \to \mathbb{R}^p$ ($p < n$) represents physical system dynamics or terminal state targets.

### 2.1 The Lagrangian Function

We define the Lagrangian $\mathcal{L}: \mathbb{R}^n \times \mathbb{R}^p \to \mathbb{R}$:

$$\mathcal{L}(x, \lambda) \triangleq f(x) + \lambda^T h(x) = f(x) + \sum_{i=1}^p \lambda_i h_i(x)$$

where $\lambda \in \mathbb{R}^p$ is the vector of **Lagrange multipliers** (also called costates or shadow prices).

### 2.2 First-Order Stationarity

At a regular point $x^*$ (where gradients $\{\nabla h_i(x^*)\}$ are linearly independent), the necessary condition for optimality is:

$$\nabla_x \mathcal{L}(x^*, \lambda^*) = \nabla f(x^*) + \sum_{i=1}^p \lambda_i^* \nabla h_i(x^*) = 0$$
$$\nabla_\lambda \mathcal{L}(x^*, \lambda^*) = h(x^*) = 0$$

---

## 3. Karush-Kuhn-Tucker (KKT) Conditions

When the optimization problem includes inequality constraints (e.g., actuator torque limits, state boundaries):

$$\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{subject to} \quad & h_i(x) = 0, \quad i = 1, \dots, p \\
& g_j(x) \le 0, \quad j = 1, \dots, m
\end{aligned}$$

The augmented Lagrangian is:

$$\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i=1}^p \lambda_i h_i(x) + \sum_{j=1}^m \mu_j g_j(x)$$

### 3.1 The Four KKT Conditions

If $x^*$ is a local minimum satisfying constraint qualifications (e.g., Linear Independence Constraint Qualification — LICQ), there exist multipliers $\lambda^* \in \mathbb{R}^p$ and $\mu^* \in \mathbb{R}^m$ such that:

1. **Stationarity:**
   $$\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = \nabla f(x^*) + \sum_{i=1}^p \lambda_i^* \nabla h_i(x^*) + \sum_{j=1}^m \mu_j^* \nabla g_j(x^*) = 0$$
2. **Primal Feasibility:**
   $$h_i(x^*) = 0 \quad (i = 1, \dots, p), \qquad g_j(x^*) \le 0 \quad (j = 1, \dots, m)$$
3. **Dual Feasibility:**
   $$\mu_j^* \ge 0 \quad (j = 1, \dots, m)$$
4. **Complementary Slackness:**
   $$\mu_j^* g_j(x^*) = 0 \quad (j = 1, \dots, m)$$

*Intuition for Complementary Slackness:*  
- If constraint $j$ is inactive ($g_j(x^*) < 0$), then $\mu_j^* = 0$ (the constraint exerts zero force on the optimal solution).
- If constraint $j$ is active ($g_j(x^*) = 0$), then $\mu_j^* \ge 0$ (the multiplier acts as the normal reaction force holding the state at the boundary).

---

## 4. Quadratic Programming (QP) in Control

A **Quadratic Program (QP)** has a quadratic objective and linear constraints:

$$\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & \frac{1}{2} x^T H x + c^T x \\
\text{subject to} \quad & A_{eq} x = b_{eq} \\
& A_{in} x \le b_{in}
\end{aligned}$$

with $H = H^T \succ 0$. QPs form the computational core of Model Predictive Control (MPC) and optimal control trajectory generation.

### 4.1 Equality-Constrained QP: Closed-Form KKT System

When only equality constraints $A_{eq} x = b_{eq}$ are present:

$$\begin{bmatrix} H & A_{eq}^T \\ A_{eq} & 0 \end{bmatrix} \begin{bmatrix} x^* \\ \lambda^* \end{bmatrix} = \begin{bmatrix} -c \\ b_{eq} \end{bmatrix}$$

Because $H \succ 0$ and $A_{eq}$ has full row rank, this **KKT Matrix** is non-singular and solvable via block elimination (Schur complement):

$$\begin{aligned}
\lambda^* &= (A_{eq} H^{-1} A_{eq}^T)^{-1} (A_{eq} H^{-1} c + b_{eq}) \\
x^* &= -H^{-1}(c + A_{eq}^T \lambda^*)
\end{aligned}$$

---

## 5. Computational Implementation (Python / NumPy / SciPy)

```python
import numpy as np
from scipy import optimize

# Solve an Equality-Constrained QP:
# min 0.5 * x^T H x + c^T x  s.t.  A_eq x = b_eq
n = 3
H = np.array([[4.0, 1.0, 0.0],
              [1.0, 2.0, 0.5],
              [0.0, 0.5, 3.0]])
c = np.array([-1.0, -2.0, 0.0])

A_eq = np.array([[1.0, 1.0, 1.0]])
b_eq = np.array([1.0])

# Form and solve the exact KKT system
KKT_mat = np.block([
    [H, A_eq.T],
    [A_eq, np.zeros((1, 1))]
])
rhs = np.concatenate([-c, b_eq])
sol = np.linalg.solve(KKT_mat, rhs)

x_opt = sol[:n]
lambda_opt = sol[n:]
print(f"Optimal state x*: {x_opt}")
print(f"Lagrange multiplier lambda*: {lambda_opt}")
assert np.isclose(np.sum(x_opt), 1.0)
```
