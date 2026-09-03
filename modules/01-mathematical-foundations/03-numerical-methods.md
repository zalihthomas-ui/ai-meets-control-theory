# Numerical Methods & Integration

> **Module 01: Mathematical Foundations** | Theory Note 03  
> Focus: Numerical integration of ODEs, Forward Euler, Runge-Kutta 4th Order (RK4), stability regions, and error analysis.

---

## 1. Initial Value Problems (IVPs) in Control

Simulating continuous-time dynamical systems $\dot{x} = f(t, x, u)$ with initial condition $x(t_0) = x_0$ requires numerical discretization over discrete time increments $\Delta t = t_{k+1} - t_k$:

$$x(t_{k+1}) = x(t_k) + \int_{t_k}^{t_{k+1}} f(\tau, x(\tau), u(\tau)) \, d\tau$$

Because the true state trajectory $x(\tau)$ inside the interval is unknown, numerical integrators approximate this definite integral using weighted polynomial evaluations.

---

## 2. Forward Euler Method

The simplest numerical integrator approximates the derivative at the start of the interval:

$$x_{k+1} = x_k + \Delta t \, f(t_k, x_k, u_k)$$

### 2.1 Taylor Series Error Analysis

Expanding the true solution $x(t_{k+1})$ in a Taylor series about $t_k$:

$$x(t_k + \Delta t) = x(t_k) + \Delta t \dot{x}(t_k) + \frac{\Delta t^2}{2} \ddot{x}(t_k) + \mathcal{O}(\Delta t^3)$$

Substituting $\dot{x}(t_k) = f(t_k, x_k)$:

$$x(t_{k+1}) = \underbrace{x(t_k) + \Delta t f(t_k, x_k)}_{\text{Forward Euler Step}} + \underbrace{\frac{\Delta t^2}{2} \ddot{x}(\xi)}_{\text{Local Truncation Error (LTE)}}$$

- **Local Truncation Error (LTE):** $\mathcal{O}(\Delta t^2)$ per single integration step.
- **Global Truncation Error (GTE):** Over $N = T/\Delta t$ steps, accumulated error scales as $N \cdot \mathcal{O}(\Delta t^2) = \mathcal{O}(\Delta t)$ (First-order accurate).

### 2.2 Numerical Stability Region

Applied to the scalar test equation $\dot{x} = \lambda x$ ($\lambda \in \mathbb{C}$):

$$x_{k+1} = x_k + \Delta t \lambda x_k = (1 + \lambda \Delta t) x_k$$

For stability, the amplification factor must satisfy $|1 + \lambda \Delta t| \le 1$.
- **Stability Boundary:** A disk of radius 1 centered at $-1 + 0j$ in the complex $z = \lambda \Delta t$ plane.
- For a lightly damped mechanical oscillator with $\lambda = \pm j\omega_n$, Forward Euler has $|1 \pm j\omega_n \Delta t| = \sqrt{1 + \omega_n^2 \Delta t^2} > 1$. **Forward Euler is unconditionally unstable for undamped physical systems**, adding artificial numerical energy at every step.

---

## 3. Explicit Runge-Kutta 4th Order (RK4)

The classical 4th-order Runge-Kutta method balances high numerical accuracy with computational simplicity, making it the workhorse simulation algorithm across this repository.

### 3.1 Mathematical Derivation & Stages

For constant control input $u_k$ held over the interval $[t_k, t_k + \Delta t]$:

$$\begin{aligned}
k_1 &= f(t_k, x_k, u_k) \\
k_2 &= f\left(t_k + \frac{\Delta t}{2}, \; x_k + \frac{\Delta t}{2} k_1, \; u_k\right) \\
k_3 &= f\left(t_k + \frac{\Delta t}{2}, \; x_k + \frac{\Delta t}{2} k_2, \; u_k\right) \\
k_4 &= f(t_k + \Delta t, \; x_k + \Delta t k_3, \; u_k)
\end{aligned}$$

The state update is computed via Simpson's 1/3-rule weighting:

$$x_{k+1} = x_k + \frac{\Delta t}{6} \left( k_1 + 2k_2 + 2k_3 + k_4 \right)$$

### 3.2 Butcher Tableau

$$\begin{array}{c|cccc}
0 & 0 & 0 & 0 & 0 \\
1/2 & 1/2 & 0 & 0 & 0 \\
1/2 & 0 & 1/2 & 0 & 0 \\
1 & 0 & 0 & 1 & 0 \\
\hline
& 1/6 & 1/3 & 1/3 & 1/6
\end{array}$$

### 3.3 Error Properties & Stability Region

- **Local Truncation Error (LTE):** $\mathcal{O}(\Delta t^5)$
- **Global Truncation Error (GTE):** $\mathcal{O}(\Delta t^4)$ (Fourth-order accurate). Halving $\Delta t$ reduces simulation error by a factor of 16.
- **Stability Polynomial:** For test equation $\dot{x} = \lambda x$, $x_{k+1} = R(\lambda \Delta t) x_k$, where:
  $$R(z) = 1 + z + \frac{z^2}{2!} + \frac{z^3}{3!} + \frac{z^4}{4!}$$
  The stability region $\{z \in \mathbb{C} \mid |R(z)| \le 1\}$ extends along the imaginary axis up to $|z| \le 2\sqrt{2} \approx 2.828$, enabling stable simulation of conservative / oscillatory mechanical systems.

---

## 4. Stiff Systems & Step-Size Selection Rules

A system $\dot{x} = Ax$ is **stiff** if its eigenvalues span multiple orders of magnitude:

$$\text{Stiffness Ratio} = \frac{\max_i |\text{Re}(\lambda_i)|}{\min_j |\text{Re}(\lambda_j)|} \gg 1$$

### Step-Size Rules of Thumb for Control Engineers:
1. **Bandwidth Rule:** Ensure sampling frequency $f_s = \frac{1}{\Delta t} \ge 20 \cdot f_{\max}$ (where $f_{\max}$ is the highest closed-loop natural frequency).
2. **Fastest Time Constant:** Set $\Delta t \le \frac{1}{5} \tau_{\min} = \frac{1}{5 \max_i |\text{Re}(\lambda_i)|}$.

---

## 5. Python Implementation

```python
import numpy as np
from typing import Callable

def rk4_step(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Perform a single step of explicit 4th-order Runge-Kutta integration."""
    k1 = f(t, x, u)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```
