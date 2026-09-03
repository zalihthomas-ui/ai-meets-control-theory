# Operating Points & Jacobian Linearization

> **Module 02: Dynamic System Modeling** | Theory Note 03  
> Focus: Nonlinear state equations, equilibrium analysis, Taylor series linearization, and Cart-Pole Jacobian derivation.

---

## 1. Nonlinear Systems & Equilibrium Manifolds

Physical systems are inherently nonlinear:

$$\dot{x}(t) = f(x(t), u(t)), \quad y(t) = g(x(t), u(t))$$

where $f: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^n$ and $g: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^p$ are smooth ($C^1$) nonlinear vector functions.

### 1.1 Operating Equilibrium Points

An operating point $(x_0, u_0)$ is an **equilibrium point** if the system remains stationary when unperturbed:

$$\dot{x} = 0 \iff f(x_0, u_0) = 0$$

- **Upright Cart-Pole:** $x_0 = [x, \dot{x}, \theta, \dot{\theta}]^T = [0, 0, 0, 0]^T, \; u_0 = 0$ (unstable).
- **Downward Hanging Cart-Pole:** $x_0 = [0, 0, \pi, 0]^T, \; u_0 = 0$ (stable).

---

## 2. Multivariable Taylor Series Expansion

Define small perturbation state and input coordinates around $(x_0, u_0)$:

$$\delta x(t) \triangleq x(t) - x_0, \qquad \delta u(t) \triangleq u(t) - u_0$$

Expanding $f(x, u)$ in a Taylor series about $(x_0, u_0)$:

$$\dot{x} = f(x_0 + \delta x, u_0 + \delta u) = f(x_0, u_0) + \left. \frac{\partial f}{\partial x} \right|_{(x_0, u_0)} \delta x + \left. \frac{\partial f}{\partial u} \right|_{(x_0, u_0)} \delta u + \mathcal{O}(\|\delta x\|^2 + \|\delta u\|^2)$$

Since $\dot{x}_0 = 0$ and $f(x_0, u_0) = 0$, dropping higher-order terms ($\mathcal{O}$) yields the linear time-invariant system:

$$\dot{\delta x}(t) = A \, \delta x(t) + B \, \delta u(t)$$
$$\delta y(t) = C \, \delta x(t) + D \, \delta u(t)$$

where the **Jacobian Matrices** are:

$$A \triangleq \left. \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_n}{\partial x_1} & \dots & \frac{\partial f_n}{\partial x_n} \end{bmatrix} \right|_{(x_0, u_0)}, \qquad B \triangleq \left. \begin{bmatrix} \frac{\partial f_1}{\partial u_1} & \dots & \frac{\partial f_1}{\partial u_m} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_n}{\partial u_1} & \dots & \frac{\partial f_n}{\partial u_m} \end{bmatrix} \right|_{(x_0, u_0)}$$

---

## 3. Worked Derivation: Cart-Pole Linearization at Upright Equilibrium

Recall the nonlinear Cart-Pole equations of motion from Note 01 (with point mass pole $I = 0$ for simplicity):

$$\begin{aligned}
(M + m)\ddot{x} + m \ell \ddot{\theta} \cos\theta - m \ell \dot{\theta}^2 \sin\theta &= u \\
m \ell \ddot{x} \cos\theta + m \ell^2 \ddot{\theta} - m g \ell \sin\theta &= 0
\end{aligned}$$

State vector: $x = [x_1, x_2, x_3, x_4]^T = [p, \dot{p}, \theta, \dot{\theta}]^T$, where $p$ is cart position and $\theta$ is pole angle from upright vertical.

### 3.1 General Inertia Form & Small-Angle Approximations
Near the upright equilibrium $\theta_0 = 0, \dot{\theta}_0 = 0, u_0 = 0$:
- $\cos\theta \approx 1$
- $\sin\theta \approx \theta$
- $\dot{\theta}^2 \sin\theta \approx 0$ (second-order term)

For a pole with general rotational inertia $I$ about its center of mass ($I = 0$ for point mass, $I = \frac{1}{3}m\ell^2$ for uniform rod of half-length $\ell$):

$$\begin{bmatrix} M + m & m \ell \\ m \ell & I + m \ell^2 \end{bmatrix} \begin{bmatrix} \ddot{p} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix} u \\ m g \ell \theta \end{bmatrix}$$

Defining denominator $\mathcal{D} = (M + m)(I + m \ell^2) - (m \ell)^2$:

$$\ddot{\theta} = \frac{(M + m) m g \ell}{\mathcal{D}} \theta - \frac{m \ell}{\mathcal{D}} u$$
$$\ddot{p} = -\frac{(m \ell)^2 g}{\mathcal{D}} \theta + \frac{I + m \ell^2}{\mathcal{D}} u$$

### 3.2 The Barto/Sutton Benchmark Parameterization ($I = \frac{1}{3} m \ell^2$)
In the canonical Barto/Sutton benchmark implemented in `aimct.systems.CartPole`, the pole is modeled as a uniform rod with $I = \frac{1}{3} m \ell^2$. Dividing numerator and denominator by $(M+m) m \ell$:

$$\text{denom} = \ell \left( \frac{4}{3} - \frac{m}{M + m} \right)$$
$$\ddot{\theta} = \frac{g}{\text{denom}} \theta - \frac{1}{(M + m)\text{denom}} u$$
$$\ddot{p} = -\frac{m \ell}{M + m} \ddot{\theta} + \frac{1}{M + m} u$$

### 3.3 Linearized State-Space Matrices

$$\begin{bmatrix} \dot{p} \\ \ddot{p} \\ \dot{\theta} \\ \ddot{\theta} \end{bmatrix} = \underbrace{\begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & -\frac{m \ell}{M+m}\frac{g}{\text{denom}} & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & \frac{g}{\text{denom}} & 0 \end{bmatrix}}_{A} \begin{bmatrix} p \\ \dot{p} \\ \theta \\ \dot{\theta} \end{bmatrix} + \underbrace{\begin{bmatrix} 0 \\ \frac{1}{M+m}\left(1 + \frac{m\ell}{(M+m)\text{denom}}\right) \\ 0 \\ -\frac{1}{(M+m)\text{denom}} \end{bmatrix}}_{B} u$$

*(Note: For point-mass pole $I = 0$, $\text{denom} = \frac{M \ell}{M + m}$, reducing $A_{2,3} = -\frac{mg}{M}$ and $A_{4,3} = \frac{(M+m)g}{M\ell}$.)*

The pole dynamics matrix has eigenvalues:

$$\lambda = \pm \sqrt{\frac{g}{\text{denom}}}$$

One eigenvalue is strictly positive real ($\lambda_1 = +\omega_0 > 0$), proving that the upright equilibrium is **open-loop exponentially unstable**.

---

## 4. Numerical Central-Difference Jacobian Evaluation

In the `aimct.systems.DynamicalSystem` base class, Jacobians are computed via second-order central difference:

$$A_{:, j} = \frac{f(x_0 + \epsilon e_j, u_0) - f(x_0 - \epsilon e_j, u_0)}{2\epsilon} + \mathcal{O}(\epsilon^2)$$
$$B_{:, k} = \frac{f(x_0, u_0 + \epsilon e_k) - f(x_0, u_0 - \epsilon e_k)}{2\epsilon} + \mathcal{O}(\epsilon^2)$$

With $\epsilon = 10^{-6}$, central differencing achieves floating-point precision error $\approx 10^{-12}$ without requiring symbolic algebra packages.

---

## 5. Validity Domain and Linearization Breakdown

The linear model $A, B$ is an accurate approximation only within a neighborhood $\mathcal{B}_\delta(x_0)$.
- For the Cart-Pole, the linear model error exceeds $5\%$ when $|\theta| > 15^\circ$ ($0.26\text{ rad}$).
- Large-angle maneuvers (e.g., swing-up from $\theta = \pi$ to $\theta = 0$) **cannot** be achieved with linear controllers alone; they require energy-based swing-up, nonlinear MPC, or Reinforcement Learning policies.
