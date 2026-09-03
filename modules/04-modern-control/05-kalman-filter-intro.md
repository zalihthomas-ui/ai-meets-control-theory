# Introduction to Kalman Filtering

> **Module 04: Modern Control** | Theory Note 05  
> Focus: Stochastic state estimation, Continuous Kalman-Bucy filter, Discrete Kalman Filter predict-update recursions, and LQG control.

---

## 1. Stochastic State-Space Systems

In real physical environments, systems are corrupted by random external forces (process disturbances) and imperfect sensor readings (measurement noise):

$$\begin{aligned}
\dot{x}(t) &= A x(t) + B u(t) + w(t) \\
y(t) &= C x(t) + v(t)
\end{aligned}$$

where:
- $w(t) \sim \mathcal{N}(0, Q_w)$: Zero-mean Gaussian white process noise with spectral density $Q_w \succeq 0$.
- $v(t) \sim \mathcal{N}(0, R_v)$: Zero-mean Gaussian white measurement noise with spectral density $R_v \succ 0$.
- $\mathbb{E}[w(t) v(\tau)^T] = 0$: Process and measurement noises are mutually uncorrelated.

---

## 2. The Continuous-Time Kalman-Bucy Filter

The optimal linear unbiased state estimator minimizes the steady-state mean-squared error covariance $P \triangleq \mathbb{E}[(x - \hat{x})(x - \hat{x})^T]$:

$$\dot{\hat{x}}(t) = A \hat{x}(t) + B u(t) + L_K (y(t) - C \hat{x}(t))$$

### 2.1 Optimal Kalman Gain Matrix $L_K$

$$L_K = P C^T R_v^{-1}$$

where the steady-state error covariance $P = P^T \succ 0$ is the unique positive-definite solution to the **Filter Algebraic Riccati Equation (FARE)**:

$$A P + P A^T - P C^T R_v^{-1} C P + Q_w = 0$$

*Physical Trade-off:*
- Large sensor noise ($R_v \to \infty$): $L_K \to 0$, observer trusts model simulation $\dot{\hat{x}} \approx A\hat{x} + Bu$.
- Large process noise ($Q_w \to \infty$): $L_K$ grows, observer heavily trusts sensor measurements $y(t)$.

---

## 3. The Discrete-Time Kalman Filter (DKF)

For digital sampled-data implementations ($x_{k+1} = A_d x_k + B_d u_k + w_k, \; y_k = C_d x_k + v_k$ with $w_k \sim \mathcal{N}(0, Q_d), \; v_k \sim \mathcal{N}(0, R_d)$), the filter executes two recursive steps per sample cycle:

```
          ┌──────────────────────────────────────────────────────────┐
          │                   1. PREDICT (Time Update)               │
          │  x̂_{k|k-1} = A_d x̂_{k-1|k-1} + B_d u_{k-1}               │
          │  P_{k|k-1} = A_d P_{k-1|k-1} A_d^T + Q_d                 │
          └────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼  New measurement y_k arrives
          ┌──────────────────────────────────────────────────────────┐
          │                  2. UPDATE (Measurement Correction)      │
          │  K_k = P_{k|k-1} C_d^T (C_d P_{k|k-1} C_d^T + R_d)⁻¹    │
          │  x̂_{k|k} = x̂_{k|k-1} + K_k (y_k - C_d x̂_{k|k-1})         │
          │  P_{k|k} = (I - K_k C_d) P_{k|k-1}                       │
          └──────────────────────────────────────────────────────────┘
```

---

## 4. The Linear Quadratic Gaussian (LQG) Framework

The **LQG Controller** combines an optimal LQR regulator with an optimal Kalman Filter:

$$u(t) = -K_{\text{LQR}} \, \hat{x}(t)$$

### Stochastic Separation Theorem
Even in the presence of stochastic noise:
1. The optimal control gain $K_{\text{LQR}}$ is computed strictly from $(A, B, Q, R)$ ignoring noise covariances $(Q_w, R_v)$ (Certainty Equivalence).
2. The optimal estimator gain $L_K$ is computed strictly from $(A, C, Q_w, R_v)$ ignoring performance weights $(Q, R)$.
3. The combination $(K_{\text{LQR}}, L_K)$ is the mathematically unique optimal output-feedback controller for linear systems with Gaussian noise.
