# Controllability, Observability & Duality

> **Module 04: Modern Control** | Theory Note 02  
> Focus: Kalman rank condition, PBH tests, Gramians, and duality.

---

## 1. Controllability

A continuous-time system $\dot{x}(t) = Ax(t) + Bu(t)$ is **controllable** if for any initial state $x(0) = x_0$ and any target final state $x_1$, there exists a piecewise continuous control trajectory $u(t)$ that transfers the state from $x_0$ to $x_1$ in finite time $T < \infty$.

### 1.1 The Kalman Controllability Matrix

By applying the Cayley-Hamilton theorem ($A^n = \sum_{k=0}^{n-1} \alpha_k A^k$) to the matrix exponential $e^{At}$, any state reachable from the origin lies in the range of the **Controllability Matrix**:

$$\mathcal{C} \triangleq \begin{bmatrix} B & AB & A^2 B & \dots & A^{n-1} B \end{bmatrix} \in \mathbb{R}^{n \times (n \cdot m)}$$

**Kalman Rank Condition:** The pair $(A, B)$ is controllable if and only if:

$$\text{rank}(\mathcal{C}) = n$$

---

## 2. Popov-Belevitch-Hautus (PBH) Tests

The Kalman rank test can suffer from numerical conditioning issues on large matrices. The **PBH Test** provides an eigenvalue-by-eigenvalue algebraic check:

### 2.1 PBH Controllability Test
The pair $(A, B)$ is controllable if and only if:

$$\text{rank}\left( \begin{bmatrix} sI - A & B \end{bmatrix} \right) = n \quad \forall s \in \mathbb{C}$$

Equivalently, no non-zero left eigenvector $w^T$ of $A$ ($w^T A = \lambda w^T$) is orthogonal to $B$:

$$w^T B \ne 0 \quad \forall w \ne 0 \text{ such that } w^T A = \lambda w^T$$

### 2.2 PBH Stabilizability Test
A system is **stabilizable** if all its uncontrollable modes are naturally stable ($\text{Re}(\lambda) < 0$):

$$\text{rank}\left( \begin{bmatrix} sI - A & B \end{bmatrix} \right) = n \quad \forall s \in \mathbb{C} \text{ with } \text{Re}(s) \ge 0$$

---

## 3. The Controllability Gramian

For a stable system ($A$ is Hurwitz), the infinite-horizon **Controllability Gramian** is:

$$W_c \triangleq \int_0^\infty e^{At} B B^T e^{A^T t} \, dt$$

$W_c$ is the unique symmetric positive-definite solution to the continuous Lyapunov equation:

$$A W_c + W_c A^T + B B^T = 0$$

### 3.1 Minimum Energy Control
The minimum control energy required to steer the state from $x(0) = 0$ to $x(T) = x_1$ is:

$$\min_{u} \int_0^T \|u(t)\|_2^2 \, dt = x_1^T W_c(T)^{-1} x_1$$

- Eigenvectors of $W_c$ associated with large eigenvalues represent directions in state space that require minimal control energy.
- Eigenvectors associated with tiny eigenvalues represent difficult, energy-expensive directions to control.

---

## 4. Observability

A system is **observable** if the initial state $x(0) = x_0$ can be uniquely determined from knowledge of the input trajectory $u(t)$ and output trajectory $y(t)$ over a finite observation window $t \in [0, T]$.

### 4.1 The Kalman Observability Matrix

$$\mathcal{O} \triangleq \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix} \in \mathbb{R}^{(n \cdot p) \times n}$$

**Kalman Rank Condition:** The pair $(A, C)$ is observable if and only if:

$$\text{rank}(\mathcal{O}) = n$$

### 4.2 PBH Observability & Detectability
- **Observability:** $\text{rank}\left( \begin{bmatrix} sI - A \\ C \end{bmatrix} \right) = n \quad \forall s \in \mathbb{C}$.
- **Detectability:** $\text{rank}\left( \begin{bmatrix} sI - A \\ C \end{bmatrix} \right) = n \quad \forall s \in \mathbb{C} \text{ with } \text{Re}(s) \ge 0$ (all unobservable modes decay exponentially).

---

## 5. Mathematical Duality

Control and estimation are exact mathematical duals:

| Primal Problem (Control) | Dual Problem (Estimation) |
| :--- | :--- |
| Dynamics Matrix $A$ | Transpose Dynamics Matrix $A^T$ |
| Actuator Matrix $B$ | Sensor Matrix $C^T$ |
| Controllability Matrix $\mathcal{C}(A, B)$ | Observability Matrix $\mathcal{O}(A^T, B^T)^T$ |
| State Feedback Gain $K$ | Observer Injection Gain $L^T$ |
| Closed-loop dynamics $A - BK$ | Observer error dynamics $(A - LC)^T = A^T - C^T L^T$ |

**Theorem of Duality:** The pair $(A, B)$ is controllable if and only if the dual pair $(A^T, B^T)$ is observable.
