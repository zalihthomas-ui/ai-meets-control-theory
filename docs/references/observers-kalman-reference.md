# State Estimation, Observers & Kalman Filtering — Reference Specification

This document provides mathematical formulations, observability analyses, continuous/discrete algorithms, and canonical reference configurations for **Luenberger Observers**, **Linear Quadratic Estimators (LQE / Continuous Kalman Filter)**, and **Discrete Kalman Filters (DKF)** to guide `src/aimct/estimation/` and Module 04.

---

## 1. Problem Formulation & System Model

Consider a continuous-time linear dynamical plant subject to stochastic process disturbances and sensor measurement noise:

$$\dot{x}(t) = A x(t) + B u(t) + w(t)$$
$$y(t) = C x(t) + D u(t) + v(t)$$

where:
- $x(t) \in \mathbb{R}^n$: Internal state vector (often unmeasured or partially measured).
- $u(t) \in \mathbb{R}^m$: Control input.
- $y(t) \in \mathbb{R}^p$: Sensor measurement output ($p \le n$).
- $w(t) \sim \mathcal{N}(0, W)$: Zero-mean Gaussian process noise ($W = W^T \succeq 0$).
- $v(t) \sim \mathcal{N}(0, V)$: Zero-mean Gaussian measurement noise ($V = V^T \succ 0$).
- $\mathbb{E}[w(t) v(\tau)^T] = 0$: Uncorrelated noise assumption.

---

## 2. Observability Analysis (Cart-Pole Case Study)

The observability matrix determines whether the full internal state $x(t)$ can be reconstructed from the history of sensor measurements $y(t)$:

$$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix} \in \mathbb{R}^{pn \times n}$$

A system is **observable** if and only if $\text{rank}(\mathcal{O}) = n$.

### Observability Under Different Sensor Configurations (Cart-Pole $n=4$)

| Sensor Suite | Output Matrix $C$ | Observability Rank $\text{rank}(\mathcal{O})$ | Observable Subspace | Physical Insight |
| :--- | :--- | :---: | :--- | :--- |
| **Config A: Full State** | $C = I_4$ | **$4$ (Full)** | $\text{span}\{x, \dot{x}, \theta, \dot{\theta}\}$ | Trivial; all states directly measured. |
| **Config B: Cart Position Only** | $C = [1, 0, 0, 0]$ | **$4$ (Full)** | $\text{span}\{x, \dot{x}, \theta, \dot{\theta}\}$ | **Fully Observable!** Cart motion dynamically couples into pole acceleration $\ddot{\theta}$, allowing full reconstruction. |
| **Config C: Pole Angle Only** | $C = [0, 0, 1, 0]$ | **$2$ (Deficient)** | $\text{span}\{\theta, \dot{\theta}\}$ | **Unobservable!** Cart position $x$ and velocity $\dot{x}$ produce zero torque on the pivot. Absolute cart location cannot be recovered. |
| **Config D: Dual Encoders** | $C = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$ | **$4$ (Full)** | $\text{span}\{x, \dot{x}, \theta, \dot{\theta}\}$ | **Standard Industrial Setup.** Measures both degrees of freedom $(x, \theta)$ to estimate velocities $(\dot{x}, \dot{\theta})$ with low noise amplification. |

---

## 3. Deterministic Luenberger Observer

For deterministic systems without stochastic noise characterization, the Luenberger observer reconstructs the state using output error feedback:

$$\dot{\hat{x}}(t) = A \hat{x}(t) + B u(t) + L \left( y(t) - C \hat{x}(t) \right) = (A - LC)\hat{x}(t) + B u(t) + L y(t)$$

### Error Dynamics
Defining estimation error $e(t) = x(t) - \hat{x}(t)$:
$$\dot{e}(t) = \dot{x}(t) - \dot{\hat{x}}(t) = (A - LC) e(t)$$

The error converges asymptotically to zero ($e(t) \to 0$) if and only if all eigenvalues of $(A - LC)$ lie strictly in the open left-half plane ($\text{Re}(\lambda_i) < 0$).

### Duality to State Feedback Pole Placement
By matrix transpose duality:
$$\det(sI - (A - LC)) = \det(sI - (A^T - C^T L^T))$$
The observer gain $L$ is computed by applying standard pole placement (Ackermann's formula or `scipy.signal.place_poles`) to the dual pair $(A^T, C^T)$:
$$L = \text{place}(A^T, C^T, \text{poles}_{\text{obs}})^T$$

### Separation Principle (Certainty Equivalence)
When coupling a state feedback controller $u(t) = -K \hat{x}(t)$ with a Luenberger observer, the full augmented closed-loop system is:

$$\begin{bmatrix} \dot{x} \\ \dot{e} \end{bmatrix} = \begin{bmatrix} A - BK & BK \\ 0 & A - LC \end{bmatrix} \begin{bmatrix} x \\ e \end{bmatrix}$$

Because the block matrix is block upper-triangular:
$$\det\left( sI - \begin{bmatrix} A - BK & BK \\ 0 & A - LC \end{bmatrix} \right) = \det(sI - (A - BK)) \cdot \det(sI - (A - LC))$$

**Takeaway**: Controller poles $\lambda(A - BK)$ and observer poles $\lambda(A - LC)$ are designed completely independently.

---

## 4. Continuous-Time Kalman Filter (LQE)

When process noise $w(t)$ and sensor noise $v(t)$ are present, arbitrary pole placement amplifies high-frequency measurement noise. The continuous Kalman Filter computes the optimal observer gain $L$ that minimizes the steady-state estimation error covariance:

$$\Sigma = \lim_{t \to \infty} \mathbb{E}[(x(t) - \hat{x}(t))(x(t) - \hat{x}(t))^T]$$

### Filter Algebraic Riccati Equation (FARE)
$$\Sigma A^T + A \Sigma - \Sigma C^T V^{-1} C \Sigma + W = 0$$

Solved via the Hamiltonian eigenspace method (identical duality to LQR CARE with $A \to A^T, B \to C^T, Q \to W, R \to V$).

### Optimal Kalman Gain
$$L = \Sigma C^T V^{-1} \in \mathbb{R}^{n \times p}$$

---

## 5. Discrete-Time Kalman Filter (DKF Algorithm)

For discrete-time microcontrollers with sampling period $\Delta t$:

### Discrete Propagation Model
$$x_{k+1} = F x_k + G_u u_k + w_k, \quad w_k \sim \mathcal{N}(0, Q_d)$$
$$y_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R_d)$$

$$F \approx I_n + A \Delta t + \frac{1}{2} A^2 \Delta t^2, \quad G_u \approx B \Delta t, \quad H = C$$
$$Q_d \approx W \Delta t, \quad R_d \approx \frac{V}{\Delta t}$$

### Predict Step (Time Update)
1. Predicted State:
   $$\hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1} + G_u u_{k-1}$$
2. Predicted Covariance:
   $$P_{k|k-1} = F P_{k-1|k-1} F^T + Q_d$$

### Update Step (Measurement Correction)
1. Innovation / Measurement Residual:
   $$\tilde{y}_k = y_k - H \hat{x}_{k|k-1}$$
2. Innovation Covariance:
   $$S_k = H P_{k|k-1} H^T + R_d$$
3. Optimal Kalman Gain:
   $$K_k = P_{k|k-1} H^T S_k^{-1}$$
4. Corrected State Estimate:
   $$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \tilde{y}_k$$
5. Updated Covariance (Joseph Form for numerical symmetry):
   $$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R_d K_k^T$$

---

## 6. Canonical Golden Test Fixture: Cart-Pole State Estimation

### Plant & Sensor Setup
- Plant: `CartPole` ($M=1.0\text{ kg}, m=0.1\text{ kg}, l=0.5\text{ m}$) linearized at upright equilibrium.
- Dual Encoders ($x, \theta$ measured):
  $$C = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$
- Noise Covariances:
  $$W = \text{diag}([10^{-4}, 10^{-2}, 10^{-4}, 10^{-2}]), \quad V = \text{diag}([10^{-4}, 10^{-4}])$$

### Golden Reference Values
1. **Steady-State Error Covariance $\Sigma$**:
   $$\Sigma = \begin{bmatrix} 
   4.5852 \times 10^{-4} & 1.0019 \times 10^{-3} & -1.1827 \times 10^{-5} & -4.4635 \times 10^{-5} \\ 
   1.0019 \times 10^{-3} & 4.5983 \times 10^{-3} & -1.0851 \times 10^{-4} & -4.2567 \times 10^{-4} \\ 
   -1.1827 \times 10^{-5} & -1.0851 \times 10^{-4} & 8.3631 \times 10^{-4} & 3.4478 \times 10^{-3} \\ 
   -4.4635 \times 10^{-5} & -4.2567 \times 10^{-4} & 3.4478 \times 10^{-3} & 1.5633 \times 10^{-2} 
   \end{bmatrix}$$
2. **Optimal Continuous Kalman Gain Matrix $L$**:
   $$L = \begin{bmatrix} 
   4.5852 & -0.1183 \\ 
   10.0190 & -1.0851 \\ 
   -0.1183 & 8.3631 \\ 
   -0.4464 & 34.4780 
   \end{bmatrix}$$
3. **Estimator Error Eigenvalues $\lambda(A - LC)$**:
   $$\lambda_{\text{obs}} = \{-2.2912 \pm 2.1805 j, \ -4.1830 \pm 1.0957 j\}$$
   *(All poles strictly in LHP; decay rate $> 2.29\text{ s}^{-1}$, settling time of estimation error $< 1.75\text{ s}$)*.
