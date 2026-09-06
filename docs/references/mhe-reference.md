# Moving-Horizon Estimation (MHE) Reference Specification

## 1. Executive Summary & Optimization Problem Definition

**Moving-Horizon Estimation (MHE)** is an optimization-based state estimation framework that solves a constrained Maximum A Posteriori (MAP) trajectory estimation problem over a sliding finite window of the $N$ most recent measurements.

Whereas classical Kalman filtering (KF, EKF, UKF) makes an unconstrained Gaussian assumption—inherently allowing state estimates to violate physical laws (e.g. negative liquid levels, negative absolute temperatures, or penetrations into obstacles)—MHE incorporates **hard state and process disturbance inequality constraints** directly into the optimization problem.

```
 Past History (t < k - N)                 Sliding Estimation Window (N steps)              Current Time k
 [-----------------------------|---------------------------------------------------------|---->
       Summarized by           k - N                                                     k
        Arrival Cost           Decision Variables: x_{k-N}, w_{k-N}, ..., w_{k-1}
      ||x - x_bar||^2_{P^-1}   Measurements:       y_{k-N}, ............, y_k
                               Constraints:        x_min <= x_i <= x_max,  w_min <= w_i <= w_max
```

---

## 2. Mathematical Formulation

Consider the discrete-time nonlinear dynamical system with process disturbance $w_k$ and measurement noise $v_k$:

$$\begin{aligned}
x_{k+1} &= f(x_k, u_k) + w_k, \quad w_k \sim \mathcal{N}(0, Q) \\
y_k &= h(x_k) + v_k, \quad v_k \sim \mathcal{N}(0, R)
\end{aligned}$$

subject to physical state constraints $x_k \in \mathcal{X} = \{x \in \mathbb{R}^n \mid x_{\text{lo}} \le x \le x_{\text{hi}}\}$ and disturbance bounds $w_k \in \mathcal{W}$.

### 2.1 Finite-Horizon Objective Function
At current time step $k$, given an estimation horizon $N$ and sliding window start index $k_0 = \max(0, k - N)$, MHE solves the constrained nonlinear program:

$$\min_{x_{k_0}, \{w_i\}_{i=k_0}^{k-1}} \; \underbrace{\frac{1}{2} \|x_{k_0} - \bar{x}_{k_0}\|_{P_{k_0}^{-1}}^2}_{\mathcal{J}_{\text{arrival}}(x_{k_0})} + \frac{1}{2} \sum_{i=k_0}^{k-1} \|w_i\|_{Q^{-1}}^2 + \frac{1}{2} \sum_{i=k_0}^k \|y_i - h(x_i)\|_{R^{-1}}^2$$

subject to:

$$\begin{aligned}
x_{i+1} &= f(x_i, u_i) + w_i, \quad i = k_0, \dots, k-1 \\
x_{\text{lo}} &\le x_i \le x_{\text{hi}}, \quad i = k_0, \dots, k \\
w_{\text{lo}} &\le w_i \le w_{\text{hi}}, \quad i = k_0, \dots, k-1 \\
g(x_i) &\le 0
\end{aligned}$$

where $\|v\|_M^2 = v^T M v$.

---

## 3. Arrival Cost Approximation

The arrival cost $\mathcal{J}_{\text{arrival}}(x_{k_0})$ encapsulates all historical information from $t = 0$ to $t = k_0 - 1$. While the exact arrival cost requires solving a Hamilton-Jacobi-Bellman equation, practical MHE implementations approximate it with a quadratic prior $(\bar{x}_{k_0}, P_{k_0})$:

### 3.1 Extended Kalman Filter (EKF) Arrival Cost Update
When the estimation window shifts by one step ($k_0 - 1 \to k_0$):
1. **Measurement Update** at $k_0 - 1$ using the previous optimal smoothed estimate $\hat{x}_{k_0-1}^*$:
   $$H = \left.\frac{\partial h}{\partial x}\right|_{\hat{x}_{k_0-1}^*}, \quad K_k = P^- H^T (H P^- H^T + R)^{-1}$$
   $$P^+ = (I - K_k H) P^- (I - K_k H)^T + K_k R K_k^T$$
2. **Time Propagation** to $k_0$:
   $$F = \left.\frac{\partial f}{\partial x}\right|_{\hat{x}_{k_0-1}^*, u_{k_0-1}}, \quad \bar{x}_{k_0} = f(\hat{x}_{k_0-1}^*, u_{k_0-1})$$
   $$P_{k_0} = F P^+ F^T + Q$$

This EKF-propagated covariance provides a mathematically rigorous Bayesian prior that guarantees bounded covariance growth across long simulation horizons.

---

## 4. Why MHE Beats EKF and UKF under Hard Constraints

| Property | Extended Kalman Filter (EKF) | Unscented Kalman Filter (UKF) | Moving-Horizon Estimation (MHE) |
| :--- | :---: | :---: | :---: |
| **Distribution Assumption** | Unconstrained Gaussian | Unconstrained Gaussian (Sigma points) | Constrained MAP / Non-Gaussian |
| **State Inequality Constraints** | None (Violates bounds) | None (Sigma points violate bounds) | **Hard constraints strictly enforced** |
| **Non-Lipschitz / Root Boundaries** | Diverges / NaN on $h < 0$ | Diverges on negative sigma points | **Guaranteed feasible domain** |
| **Disturbance Modeling** | White Gaussian | White Gaussian | Bounded / Non-Gaussian / Slew limits |
| **Information Horizon** | 1 step (Markovian recursive) | 1 step (Markovian recursive) | **$N$ steps multi-point smoothing** |
| **Computational Complexity** | $\mathcal{O}(n^3)$ | $\mathcal{O}(n^3)$ | $\mathcal{O}(N \cdot n^3)$ QP / SQP |

### 4.1 The Square-Root Singularity Problem (e.g. Coupled Tanks)
In fluid flow and chemical processes (such as the Torricelli outflow $\dot{h} = -\frac{a}{A}\sqrt{2g h}$ in [`aimct.systems.TwoTank`](file:///C:/Users/salih/Desktop/ai-meets-control-theory/src/aimct/systems/two_tank.py)):
- When tank level approaches empty ($h \to 0$), sensor noise causes measurements $y_k = h + v_k$ to dip negative ($y_k < 0$).
- EKF updates $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K(y_k - \hat{y}_k)$, frequently driving the estimate into the unphysical negative domain $\hat{h} < 0$.
- Evaluating $\sqrt{2g \hat{h}}$ yields complex numbers / `NaN`, corrupting filter covariance matrices and causing complete catastrophic divergence.
- Simple heuristic clipping ($\hat{x} \leftarrow \max(0, \hat{x})$) breaks the Bayesian optimality of Kalman filtering and introduces persistent bias.
- MHE restricts optimization over $h \ge 0$, maintaining numerical regularity and providing strictly feasible, minimum-variance state trajectory reconstruction.

---

## 5. Stability & Convergence Properties

Under the assumption of **Uniform Observability** over the horizon $N$ (i.e. the observability Gramian $\mathcal{O}_N(x) \succeq \alpha I > 0$) and bounded disturbance sequences $\|w_k\| \le \bar{w}, \|v_k\| \le \bar{v}$, the estimation error $e_k = x_k - \hat{x}_k$ satisfies:

$$\|e_k\| \le c_1 \rho^k \|e_0\| + c_2 \bar{w} + c_3 \bar{v}$$

where $\rho \in (0, 1)$ is the exponential convergence rate (Rao, Rawlings & Mayne, 2003; Alessandri et al., 2008).
