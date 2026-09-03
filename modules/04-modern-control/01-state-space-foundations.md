# State-Space Foundations & Canonical Forms

> **Module 04: Modern Control** | Theory Note 01  
> Focus: State-space representations, state coordinate transformations, modal decomposition, and canonical forms.

---

## 1. The Multi-Input Multi-Output (MIMO) State Space

Modern control represents physical systems as systems of coupled first-order differential equations:

$$\begin{aligned}
\dot{x}(t) &= A x(t) + B u(t) \\
y(t) &= C x(t) + D u(t)
\end{aligned}$$

where state vector $x(t) \in \mathbb{R}^n$, control input $u(t) \in \mathbb{R}^m$, and sensor output $y(t) \in \mathbb{R}^p$.

---

## 2. Coordinate Transformations & Similarity Invariants

Let $T \in \mathbb{R}^{n \times n}$ be an invertible state transformation matrix:

$$z(t) = T x(t) \iff x(t) = T^{-1} z(t)$$

Under this coordinate change, the state equations transform to:

$$\begin{aligned}
\dot{z}(t) &= \tilde{A} z(t) + \tilde{B} u(t) \\
y(t) &= \tilde{C} z(t) + \tilde{D} u(t)
\end{aligned}$$

where:
$$\tilde{A} = T A T^{-1}, \qquad \tilde{B} = T B, \qquad \tilde{C} = C T^{-1}, \qquad \tilde{D} = D$$

### 2.1 Fundamental Similarity Invariants
Coordinate changes alter the internal algebraic representation but leave all physical input-output properties invariant:
1. **Eigenvalues / Spectrum:** $\sigma(\tilde{A}) \equiv \sigma(A)$ since $\det(sI - T A T^{-1}) = \det(T(sI - A)T^{-1}) = \det(sI - A)$.
2. **Transfer Function Matrix:** $\tilde{G}(s) = \tilde{C}(sI - \tilde{A})^{-1}\tilde{B} + \tilde{D} \equiv G(s)$.
3. **Controllability & Observability Subspaces:** Dimensions and ranks are preserved.

---

## 3. Canonical Realization Forms (SISO)

For a single-input single-output (SISO) transfer function with characteristic polynomial $a(s) = s^n + a_{n-1}s^{n-1} + \dots + a_1 s + a_0$ and numerator $b(s) = b_{n-1}s^{n-1} + \dots + b_1 s + b_0$:

### 3.1 Controllable Canonical Form (CCF)
In CCF, the control input directly excites the highest derivative:

$$A_{\text{ccf}} = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & 1 \\ -a_0 & -a_1 & -a_2 & \dots & -a_{n-1} \end{bmatrix}, \quad B_{\text{ccf}} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$
$$C_{\text{ccf}} = \begin{bmatrix} b_0 & b_1 & b_2 & \dots & b_{n-1} \end{bmatrix}, \quad D_{\text{ccf}} = 0$$

*Property:* The Kalman controllability matrix $\mathcal{C}(A_{\text{ccf}}, B_{\text{ccf}})$ has an upper-triangular structure, making pole-placement gain design trivial: $K_{\text{ccf}} = [d_0 - a_0, \; d_1 - a_1, \; \dots, \; d_{n-1} - a_{n-1}]$.

### 3.2 Observable Canonical Form (OCF)
In OCF, sensor output is formed directly from the top state:

$$A_{\text{ocf}} = A_{\text{ccf}}^T, \qquad B_{\text{ocf}} = C_{\text{ccf}}^T, \qquad C_{\text{ocf}} = B_{\text{ccf}}^T, \qquad D_{\text{ocf}} = 0$$

*Property:* The Kalman observability matrix $\mathcal{O}(A_{\text{ocf}}, C_{\text{ocf}})$ is easily inverted, making Luenberger observer design straightforward.

---

## 4. Modal (Diagonal / Jordan) Form

If matrix $A$ has distinct eigenvalues $\lambda_1, \dots, \lambda_n$ with modal eigenvector matrix $V = [v_1, \dots, v_n]$, choosing $T = V^{-1}$ yields:

$$\tilde{A} = \Lambda = \begin{bmatrix} \lambda_1 & & 0 \\ & \ddots & \\ 0 & & \lambda_n \end{bmatrix}, \quad \tilde{B} = V^{-1} B, \quad \tilde{C} = C V$$

- The $i$-th row of $\tilde{B}$ dictates the **controllability** of mode $\lambda_i$. If the $i$-th row is identically zero, mode $\lambda_i$ cannot be affected by any control input.
- The $i$-th column of $\tilde{C}$ dictates the **observability** of mode $\lambda_i$. If the $i$-th column is identically zero, mode $\lambda_i$ produces zero effect on the sensor output.
