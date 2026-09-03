# LQR Tuning, Bryson's Rule & Robustness Guarantees

> **Module 05: Optimal Control** | Theory Note 04  
> Focus: Bryson's tuning rule, Kalman's return difference inequality, guaranteed gain/phase margins, and Doyle's LQG robustness counterexample.

---

## 1. Systematic Weight Tuning: Bryson's Rule

Choosing state weighting matrix $Q \succeq 0$ and control weighting matrix $R \succ 0$ is often viewed as a black art. **Bryson's Rule** provides a physically motivated normalization:

$$Q_{ii} = \frac{1}{(x_{i, \max})^2}, \qquad R_{jj} = \frac{\rho}{(u_{j, \max})^2}$$

where:
- $x_{i, \max}$: Maximum allowable / acceptable deviation of state variable $x_i(t)$.
- $u_{j, \max}$: Maximum physical actuator authority for input $u_j(t)$.
- $\rho > 0$: Scalar tuning knob. Increasing $\rho$ penalizes control effort (yielding sluggish, smooth control), while decreasing $\rho$ prioritizes aggressive state convergence at the expense of large actuator commands.

---

## 2. Kalman's Return Difference Inequality

Let $L(s) \triangleq K(sI - A)^{-1}B$ be the open-loop transfer function of a continuous-time LQR system broken at the plant input.

### 2.1 Kalman's Algebraic Identity
By substituting the Riccati equation $A^T P + P A - P B R^{-1} B^T P + Q = 0$, Kalman proved:

$$\left( I + L(-s)^T \right) R \left( I + L(s) \right) = R + B^T (-sI - A^T)^{-1} Q (sI - A)^{-1} B$$

Evaluating this identity on the imaginary axis $s = j\omega$ with $Q = H^T H \succeq 0$:

$$\left( I + L(j\omega) \right)^* R \left( I + L(j\omega) \right) = R + \underbrace{\left( H(j\omega I - A)^{-1} B \right)^* \left( H(j\omega I - A)^{-1} B \right)}_{\ge 0 \quad (\text{Hermitian Positive Semi-Definite})}$$

Therefore:

$$\left( I + L(j\omega) \right)^* R \left( I + L(j\omega) \right) \ge R \quad \forall \omega \in \mathbb{R}$$

### 2.2 SISO Geometric Interpretation
For a single-input system ($R = r > 0$):

$$|1 + L(j\omega)| \ge 1 \quad \forall \omega \in \mathbb{R}$$

```
                          Im(L)
                            │
                            │      /|  |1 + L(jω)| ≥ 1
                         1  │     / |
                    ┌───────┼───/───┐
                    │       │  /    │
         (-1, 0)    │       │ /     │
────────────X───────┼───────O───────┼────────── Re(L)
           •        │   1   │       │
       Critical     │       │       │
        Point       └───────┼───────┘
                            │
                            │
```

The Nyquist curve of $L(j\omega)$ **can never enter the unit circle centered at the critical point $-1 + 0j$**.

---

## 3. Guaranteed Robustness Margins of Continuous LQR

Because the Nyquist plot stays strictly outside the unit circle centered at $-1 + 0j$:

1. **Guaranteed Gain Margin ($G_m$):**  
   $$G_m \in \left[ \frac{1}{2}, \; \infty \right) \iff -6\text{ dB} \le \Delta K \le +\infty\text{ dB}$$  
   The closed-loop system remains stable if the loop gain is reduced by up to $50\%$ or increased by any finite factor.
2. **Guaranteed Phase Margin ($\Phi_m$):**  
   $$\Phi_m \ge 60^\circ \quad (\text{in every control channel for diagonal } R)$$
3. **Guaranteed Time-Delay Margin:**  
   $$\tau_{\max} \ge \frac{\pi / 3}{\omega_{\text{gc}}}$$

---

## 4. Doyle's 1978 Warning: Loss of Robustness in LQG

In 1978, John Doyle published a landmark two-page paper titled *"Guaranteed Margins for LQG Regulators"*.

> **The LQG Robustness Paradox:**  
> While full-state LQR possesses infinite gain margin and $\ge 60^\circ$ phase margin, **output-feedback LQG controllers (LQR + Kalman Filter) possess zero guaranteed margins**.

When states are estimated via an observer, the combined loop gain $L_{\text{lqg}}(s) = K(sI - A + BK + LC)^{-1} L y$ no longer satisfies Kalman's return difference inequality. An LQG controller tuned aggressively can have:
- Gain margin $G_m < 0.1\text{ dB}$
- Phase margin $\Phi_m < 2^\circ$

### Modern Remedies:
- **Loop Transfer Recovery (LTR):** Artificially inject large fictional process noise $Q_w \to q_0 B B^T$ to force the Kalman filter transmission zeros to cancel plant dynamics, recovering LQR robustness.
- **$H_\infty$ Robust Control:** Directly minimize the worst-case sensitivity peak $\|S(j\omega)\|_\infty$.
