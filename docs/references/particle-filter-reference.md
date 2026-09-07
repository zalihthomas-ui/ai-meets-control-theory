# Particle Filter (Sequential Monte Carlo) Reference Specification

## 1. Executive Summary & Non-Gaussian State Estimation

The **Particle Filter (PF)**, also known as the **Sequential Importance Resampling (SIR)** or **Bootstrap Particle Filter**, is a non-parametric Bayesian estimation technique that represents arbitrary, non-Gaussian, multimodal probability density functions using a finite collection of weighted random samples called **particles**:

$$p(x_k \mid y_{1:k}) \approx \sum_{i=1}^{N_p} w_k^{(i)} \delta(x_k - x_k^{(i)}), \quad \sum_{i=1}^{N_p} w_k^{(i)} = 1$$

While Kalman filters (KF, EKF, UKF) are constrained to unimodal Gaussian distributions $\mathcal{N}(\hat{x}_k, P_k)$, the Particle Filter represents arbitrary nonlinearities, non-Gaussian disturbances, and non-convex multimodal posteriors (e.g. range-ambiguous bearings-only tracking, periodic angular wrap-arounds, and obstacle avoidance).

```
   Gaussian Assumption (EKF / UKF)             Non-Gaussian / Multimodal (Particle Filter)
           |                                                |
          / \                                             /   \       / \
         /   \                                           /     \     /   \
   -----/-----\-----                               -----/-------\---/-----\-----
   Single mean & cov                                Multiple distinct particle clusters
```

---

## 2. Mathematical Formulation & Algorithm

Consider the discrete-time nonlinear state-space system:

$$\begin{aligned}
x_{k+1} &= f(x_k, u_k) + w_k, \quad w_k \sim p_w(w) \\
y_k     &= h(x_k) + v_k,       \quad v_k \sim \mathcal{N}(0, R)
\end{aligned}$$

where $f$ may be continuous (integrated via RK4) or discrete, and $v_k$ is zero-mean measurement noise with covariance $R$.

### 2.1 Sequential Importance Sampling Step (Prediction)
For each particle $i = 1, \dots, N_p$:
1. Sample process disturbance $w_k^{(i)} \sim \mathcal{N}(0, Q)$.
2. Propagate state forward:
   $$x_k^{(i)} = f(x_{k-1}^{(i)}, u_{k-1}) + w_k^{(i)}$$

### 2.2 Measurement Likelihood & Weight Update with Log-Sum-Exp
The likelihood of receiving measurement $y_k$ given particle $x_k^{(i)}$ is:

$$p(y_k \mid x_k^{(i)}) = \frac{1}{(2\pi)^{p/2} |R|^{1/2}} \exp\left( -\frac{1}{2} (y_k - h(x_k^{(i)}))^T R^{-1} (y_k - h(x_k^{(i)})) \right)$$

To avoid numerical floating-point underflow when residuals are large, weights are evaluated and normalized in the log domain:

$$\ell_i = \log w_{k-1}^{(i)} - \frac{1}{2} (y_k - h(x_k^{(i)}))^T R^{-1} (y_k - h(x_k^{(i)}))$$

Let $\ell_{\max} = \max_{j} \ell_j$. Normalized weights are computed via **Log-Sum-Exp**:

$$w_k^{(i)} = \frac{\exp(\ell_i - \ell_{\max})}{\sum_{j=1}^{N_p} \exp(\ell_j - \ell_{\max})}$$

### 2.3 State & Covariance Point Estimates
The minimum mean square error (MMSE) state estimate and posterior empirical covariance are:

$$\hat{x}_k = \sum_{i=1}^{N_p} w_k^{(i)} x_k^{(i)}$$

$$P_k = \sum_{i=1}^{N_p} w_k^{(i)} (x_k^{(i)} - \hat{x}_k)(x_k^{(i)} - \hat{x}_k)^T$$

---

## 3. Particle Degeneracy & Systematic Adaptive Resampling

Over successive time steps, the variance of the particle weights strictly increases, leading to **degeneracy** where all but one particle have near-zero weight.

### 3.1 Effective Sample Size (ESS)
Degeneracy is measured by the **Effective Sample Size**:

$$N_{\text{eff}} = \frac{1}{\sum_{i=1}^{N_p} (w_k^{(i)})^2}, \quad 1 \le N_{\text{eff}} \le N_p$$

### 3.2 Adaptive Systematic Resampling
Resampling is triggered whenever $N_{\text{eff}} < N_{\text{thresh}} = \gamma \cdot N_p$ (typically $\gamma = 0.5$):
1. Construct cumulative sum of weights $C_j = \sum_{m=1}^j w_k^{(m)}$ with $C_0 = 0$.
2. Draw a single uniform offset $u_1 \sim \mathcal{U}[0, 1/N_p]$.
3. For $i = 1, \dots, N_p$:
   - Generate sample pointer: $u_i = u_1 + \frac{i - 1}{N_p}$.
   - Find index $j$ such that $C_{j-1} \le u_i < C_j$.
   - Select resampled particle: $x_{\text{new}}^{(i)} = x_k^{(j)}$.
4. Reset all weights to uniform: $w_k^{(i)} = \frac{1}{N_p}$.

Systematic resampling achieves minimum variance among all standard resampling schemes (multinomial, residual, stratified) and runs in $\mathcal{O}(N_p)$ linear time.

---

## 4. Benchmark Application: Bearings-Only Target Tracking

In **Bearings-Only Target Tracking (BOTT)**, an observer measures only the azimuth line-of-sight angle to a moving target:

$$\theta_k = \text{atan2}(y_{t,k} - y_{o,k}, \; x_{t,k} - x_{o,k}) + v_k, \quad v_k \sim \mathcal{N}(0, \sigma_\theta^2)$$

### 4.1 Why EKF and UKF Fail on Bearings-Only Tracking
1. **Range Unobservability without Maneuver**:
   Along the line-of-sight vector, range $r = \sqrt{\Delta x^2 + \Delta y^2}$ is completely unobservable from a stationary observer until the observer executes an orthogonal acceleration maneuver.
2. **Banana-Shaped / Curved True Posterior**:
   The true posterior distribution over Cartesian space $(x, y)$ is a curved, arc-shaped crescent.
3. **EKF Taylor Linearization Breakdown**:
   The Jacobian $H = \left[ -\frac{\Delta y}{r^2}, \frac{\Delta x}{r^2} \right]$ assumes a local hyperplane. The Gaussian ellipse extends into negative range or diverges orthogonal to the bearing line, leading to severe filter divergence.
4. **Particle Filter Advantage**:
   The particle cloud naturally spans the curved line-of-sight cone and collapses into a tight unimodal cluster as soon as the observer executes an S-turn maneuver.

---

## 5. Summary of Estimator Trade-Offs

| Estimator | Distribution Representation | Computational Complexity | Non-Gaussian / Multimodal | Constraint Handling |
| :--- | :---: | :---: | :---: | :---: |
| **EKF** | Gaussian $(\hat{x}, P)$ | $\mathcal{O}(n^3)$ | Poor (Linearized) | None |
| **UKF** | Sigma points Gaussian | $\mathcal{O}(n^3)$ | Moderate (2nd order) | None |
| **MHE** | Constrained MAP trajectory | $\mathcal{O}(N \cdot n^3)$ | Good (Constrained) | **Hard bounds $x \in \mathcal{X}$** |
| **Particle Filter (PF)** | Empirical $N_p$ point-masses | $\mathcal{O}(N_p \cdot n)$ | **Exact / Arbitrary** | Support truncation |
