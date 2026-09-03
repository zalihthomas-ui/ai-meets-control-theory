# Introduction to Model Predictive Control (MPC)

> **Module 05: Optimal Control** | Theory Note 05  
> Focus: Receding horizon principle, constrained quadratic programming, terminal ingredients for stability, and comparison with LQR/PID/RL.

---

## 1. Why Model Predictive Control?

While LQR produces optimal linear state feedback $u = -Kx$, it suffers from a fundamental engineering limitation: **it cannot enforce hard inequality constraints during trajectory planning**.

In real engineering systems:
- Actuators have hard physical saturation ($u \in [u_{\min}, u_{\max}]$) and rate limits ($|\Delta u| \le \Delta u_{\max}$).
- States have safety boundaries (e.g., maximum motor temperature, obstacle avoidance distances, voltage limits).

Simply clamping LQR control inputs ($u = \text{clip}(-Kx)$) destroys optimality, causes severe performance degradation, and can lead to closed-loop instability. **Model Predictive Control (MPC)** directly incorporates constraints into an online optimization problem.

---

## 2. The Receding Horizon Control (RHC) Principle

At every discrete sampling instant $t_k = k \Delta t$:

```
Past (Committed)      Current      Future Prediction Horizon (Np steps)
───────────────────────┼───────────────────────────────────────────────► Time
... u_{k-2}, u_{k-1}   │  [ u_0*,  u_1*,  u_2*,  ...,  u_{Np-1}* ] (Optimized Trajectory)
                       │    │
                       │    ▼
                       │  Apply u_k = u_0* to physical plant
                       │
                       └─► At step k+1: Measure x_{k+1}, shift horizon, and repeat
```

### 2.1 The Finite-Horizon Optimal Control Problem (FHOCP)
Given current state measurement $x_k$, solve:

$$\min_{U = \{u_0, u_1, \dots, u_{N-1}\}} \sum_{i=0}^{N-1} \left( x_i^T Q x_i + u_i^T R u_i \right) + x_N^T P_f x_N$$

subject to:
$$\begin{aligned}
x_{i+1} &= A_d x_i + B_d u_i, \qquad x_0 = x_k \\
u_{\min} &\le u_i \le u_{\max}, \qquad i = 0, 1, \dots, N-1 \\
x_{\min} &\le x_i \le x_{\max}, \qquad i = 0, 1, \dots, N \\
x_N &\in \mathcal{X}_f \quad (\text{Terminal Invariant Set})
\end{aligned}$$

---

## 3. Transformation to Quadratic Programming (QP)

The FHOCP is converted into a standard Quadratic Program:

$$\min_{z} \frac{1}{2} z^T H z + c^T z \quad \text{subject to} \quad A_{\text{in}} z \le b_{\text{in}}$$

### 3.1 Condensed (Dense) vs. Non-Condensed (Sparse) Formulations
- **Condensed (Dense):** State variables $x_i$ are substituted out algebraically:
  $$x_i = A_d^i x_0 + \sum_{j=0}^{i-1} A_d^{i-1-j} B_d u_j$$
  Decision variable is $z = U \in \mathbb{R}^{m N}$. Fast for short horizons ($N \le 15$).
- **Non-Condensed (Sparse):** Both states and inputs are decision variables $z = [x_0, u_0, x_1, u_1, \dots, x_N]^T$. System dynamics form equality constraints. The resulting KKT matrix is block-tridiagonal (sparse), which specialized solvers (e.g., OSQP, qpOASES) solve in $\mathcal{O}(N)$ time.

---

## 4. Stability & Recursive Feasibility

Two critical mathematical properties must be guaranteed in production MPC:
1. **Recursive Feasibility:** If a feasible solution exists at step $k$, a feasible solution is guaranteed to exist at step $k+1$.
2. **Asymptotic Stability:** The closed-loop origin is Lyapunov stable.

### The Standard Stability Recipe (Mayne et al., 2000):
- **Terminal Weight $P_f$:** Set $P_f$ equal to the unconstrained discrete Riccati solution (DARE), representing the infinite-horizon cost-to-go.
- **Terminal Controller $\kappa_f(x) = -K_d x$:** The discrete LQR gain.
- **Terminal Constraint Set $\mathcal{X}_f$:** A positively invariant set under $u = -K_d x$ where state and input constraints are strictly satisfied.

---

## 5. Architectural Comparison: Classical vs. Modern vs. Optimal vs. AI

| Attribute | PID (Classical) | State Feedback (Modern) | LQR (Optimal) | MPC (Constrained) | RL / Neural (AI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Needed?** | Transfer Function or None | Linear $(A, B)$ | Linear $(A, B)$ | Linear / Nonlinear Model | None (Model-Free) or Learned |
| **State Dimension** | SISO (Single Output) | Arbitrary $\mathbb{R}^n$ | Arbitrary $\mathbb{R}^n$ | Arbitrary $\mathbb{R}^n$ | Arbitrary $\mathbb{R}^n$ |
| **Hard Constraints?** | Ad-hoc (Anti-Windup) | No | No | **Strict & Optimal** | Soft penalties (Reward) |
| **Online Compute** | $\mathcal{O}(1)$ (Microseconds) | $\mathcal{O}(n)$ (Matrix mult) | $\mathcal{O}(n)$ (Matrix mult) | $\mathcal{O}(N^3)$ (QP Solver) | $\mathcal{O}(\text{NN FLOPs})$ |
| **Stability Proofs** | Routh / Nyquist | Exact Pole Assignment | Guaranteed Margins | Lyapunov via Cost | Empirical / Verification tools |
| **Nonlinear Systems**| Poor / Gain Scheduling | Linearized only | Linearized only | Nonlinear MPC (NMPC) | **High Capacity** |
