# State-Space Dynamics, Controllability & Geometry

State-space representation is the mathematical foundation of modern multi-variable control theory. Unlike classical transfer functions $G(s) = Y(s)/U(s)$ which describe single-input single-output (SISO) input-output mappings, state-space models explicitly track the $n$-dimensional internal state vector $x(t) \in \mathbb{R}^n$.

---

## 1. Continuous & Discrete Representations

Physical dynamical systems in `aimct` are formulated as continuous nonlinear ordinary differential equations (ODEs):

$$\dot{x}(t) = f(x(t), u(t), d(t)), \qquad y(t) = g(x(t), u(t)) + v(t)$$

where $x(t) \in \mathbb{R}^n$ is the state vector, $u(t) \in \mathbb{R}^m$ is the control input, $d(t) \in \mathbb{R}^{n_d}$ is an exogenous disturbance, $y(t) \in \mathbb{R}^p$ is the measurement vector, and $v(t) \in \mathbb{R}^p$ is sensor noise.

### Numerical Integration & Energy Conservation
To simulate continuous physical dynamics without artificial numerical dissipation or spurious energy growth, `aimct.simulate` employs 4th-Order Runge-Kutta (RK4) integration:

$$x_{k+1} = x_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where:
$$\begin{aligned}
k_1 &= f(x_k, u_k), \\
k_2 &= f\left(x_k + \frac{\Delta t}{2}k_1, u_k\right), \\
k_3 &= f\left(x_k + \frac{\Delta t}{2}k_2, u_k\right), \\
k_4 &= f(x_k + \Delta t \, k_3, u_k).
\end{aligned}$$

As demonstrated empirically in [Experiment 01](../experiments/01_rk4_vs_euler.md), Forward Euler ($\mathcal{O}(\Delta t)$) introduces fictitious numerical energy that rapidly destabilizes conservative oscillatory systems (such as unforced pendulums or orbital dynamics). In contrast, RK4 ($\mathcal{O}(\Delta t^4)$) preserves Hamiltonian energy conservation with $< 10^{-7}\%$ drift rate over extended horizons.

---

## 2. Jacobian Linearization & Validity Envelopes

About an operating equilibrium point $(\bar{x}, \bar{u})$ where $f(\bar{x}, \bar{u}) = 0$, first-order Taylor expansion yields the linear state-space realization:

$$\dot{\delta x} = A\,\delta x + B\,\delta u, \qquad \delta y = C\,\delta x + D\,\delta u$$

with continuous Jacobian matrices:

$$A = \left.\frac{\partial f}{\partial x}\right|_{(\bar{x}, \bar{u})}, \quad B = \left.\frac{\partial f}{\partial u}\right|_{(\bar{x}, \bar{u})}, \quad C = \left.\frac{\partial g}{\partial x}\right|_{(\bar{x}, \bar{u})}, \quad D = \left.\frac{\partial g}{\partial u}\right|_{(\bar{x}, \bar{u})}$$

```python
from aimct.systems import Pendulum

sys = Pendulum(m=0.2, L=0.3, b=0.01)
A, B, C, D = sys.linearize()  # Continuous-time Jacobian matrices about upright
```

### The $23^\circ$ Divergence Law
Linear state-space approximations are tangent hyperplanes that hold only locally. In [Experiment 02](../experiments/02_jacobian_linearization.md), we systematically mapped the divergence envelope of the inverted pendulum. For initial angles $|\theta_0| \le 23^\circ$ ($0.40\,\text{rad}$), the linear approximation error remains below $5\%$; beyond $23^\circ$, higher-order trigonometric nonlinearities dominate, causing linear controllers to diverge unless paired with global nonlinear energy swing-up ([Exp 07](../experiments/07_cartpole_swingup_hybrid.md)).

---

## 3. Controllability, Observability & Kalman Duality

For a linear continuous system $(A, B, C, D)$:

### Controllability
The system is **controllable** (any state $x \in \mathbb{R}^n$ can be steered to the origin in finite time by some unconstrained control $u(t)$) if and only if the controllability matrix $\mathcal{C}$ has full row rank $n$:

$$\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \dots & A^{n-1}B \end{bmatrix} \in \mathbb{R}^{n \times nm}, \qquad \text{rank}(\mathcal{C}) = n$$

### Observability
The system is **observable** (the initial state $x(0)$ can be uniquely determined from output history $y(t)$ over finite time) if and only if the observability matrix $\mathcal{O}$ has full column rank $n$:

$$\mathcal{O} = \begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix} \in \mathbb{R}^{np \times n}, \qquad \text{rank}(\mathcal{O}) = n$$

### Kalman Duality Theorem
Controllability and observability are exact mathematical duals under matrix transposition:

$$\text{The pair } (A, B) \text{ is controllable} \iff (A^T, B^T) \text{ is observable}$$

This structural symmetry is the bridge between optimal state feedback control (LQR) and optimal state estimation (Kalman filtering).

---

## 4. Modal Decomposition & Gramian Energy

For asymptotically stable linear systems ($\text{Re}(\lambda_i(A)) < 0$), the **Controllability Gramian** $W_c$ and **Observability Gramian** $W_o$ satisfy the continuous Lyapunov matrix equations:

$$A W_c + W_c A^T + B B^T = 0, \qquad A^T W_o + W_o A + C^T C = 0$$

- **Input Energy:** The minimum control energy required to reach state $x$ from the origin is $E_{min}(x) = x^T W_c^{-1} x$. Small eigenvalues of $W_c$ correspond to "hard-to-reach" directions in state space.
- **Output Energy:** An autonomous initial state $x_0$ generates total measurement energy $E_{obs}(x_0) = x_0^T W_o x_0$. Small eigenvalues of $W_o$ correspond to "hard-to-observe" state modes.

In `aimct`, Gramian diagnostics are used for balanced truncation model order reduction and identifying ill-conditioned sensor/actuator configurations before controller synthesis.
