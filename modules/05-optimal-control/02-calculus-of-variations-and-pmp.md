# Calculus of Variations & Pontryagin's Minimum Principle

> **Module 05: Optimal Control** | Theory Note 02  
> Focus: Variational calculus, the Hamiltonian function, Pontryagin's Minimum Principle (PMP), and bang-bang control.

---

## 1. Calculus of Variations Fundamentals

The calculus of variations finds extremal curves that minimize functional integrals:

$$J(x) = \int_{t_0}^T L(t, x(t), \dot{x}(t)) \, dt$$

Let $x^*(t)$ be the optimal trajectory and $\delta x(t)$ an arbitrary smooth variation with fixed boundary conditions $\delta x(t_0) = \delta x(T) = 0$. Setting the first variation $\delta J = 0$:

$$\delta J = \int_{t_0}^T \left( \frac{\partial L}{\partial x} \delta x + \frac{\partial L}{\partial \dot{x}} \delta \dot{x} \right) dt = 0$$

Integrating by parts on the second term:

$$\int_{t_0}^T \frac{\partial L}{\partial \dot{x}} \delta \dot{x} \, dt = \left[ \frac{\partial L}{\partial \dot{x}} \delta x \right]_{t_0}^T - \int_{t_0}^T \frac{d}{dt}\left( \frac{\partial L}{\partial \dot{x}} \right) \delta x \, dt = -\int_{t_0}^T \frac{d}{dt}\left( \frac{\partial L}{\partial \dot{x}} \right) \delta x \, dt$$

Because this must vanish for any arbitrary perturbation $\delta x(t)$, we obtain the celebrated **Euler-Lagrange Equation**:

$$\frac{\partial L}{\partial x} - \frac{d}{dt}\left( \frac{\partial L}{\partial \dot{x}} \right) = 0$$

---

## 2. Pontryagin's Minimum Principle (PMP)

When system dynamics are constrained by differential equations $\dot{x} = f(x, u, t)$ and control inputs are constrained to a set $u \in \mathcal{U}$, we introduce the **Costate (Adjoint) Vector** $\lambda(t) \in \mathbb{R}^n$ and construct the **Hamiltonian Function**:

$$\mathcal{H}(x(t), u(t), \lambda(t), t) \triangleq L(x(t), u(t), t) + \lambda(t)^T f(x(t), u(t), t)$$

### 2.1 Necessary Conditions for Optimality (PMP)

If $(x^*(t), u^*(t))$ is an optimal pair for $t \in [t_0, T]$, there exists an adjoint trajectory $\lambda^*(t)$ such that:

1. **State Dynamics Equation:**
   $$\dot{x}^*(t) = \nabla_\lambda \mathcal{H} = f(x^*(t), u^*(t), t), \qquad x^*(t_0) = x_0$$
2. **Costate / Adjoint Dynamics Equation:**
   $$\dot{\lambda}^*(t) = -\nabla_x \mathcal{H} = -\left( \frac{\partial f}{\partial x} \right)^T \lambda^*(t) - \nabla_x L(x^*(t), u^*(t), t)$$
3. **Pointwise Hamiltonian Minimization:**
   $$u^*(t) = \arg\min_{u \in \mathcal{U}} \mathcal{H}(x^*(t), u, \lambda^*(t), t)$$
   If $u$ is unconstrained and $\mathcal{H}$ is strictly convex in $u$:
   $$\nabla_u \mathcal{H}(x^*(t), u^*(t), \lambda^*(t), t) = 0$$
4. **Transversality Conditions (Terminal Boundary):**
   - If terminal state is free with terminal cost $\Phi(x(T))$:
     $$\lambda^*(T) = \nabla_x \Phi(x^*(T))$$
   - If terminal time $T$ is free:
     $$\mathcal{H}(T) + \frac{\partial \Phi}{\partial T} = 0$$

---

## 3. Bang-Bang Control in Control-Affine Systems

Consider a system linear in control with bounded actuator magnitude $|u(t)| \le u_{\max}$:

$$\dot{x}(t) = f(x(t)) + g(x(t)) u(t)$$

Minimizing execution time $J = \int_0^T 1 \, dt$ ($L = 1$):

The Hamiltonian is:
$$\mathcal{H} = 1 + \lambda(t)^T f(x(t)) + \underbrace{\left( \lambda(t)^T g(x(t)) \right)}_{\sigma(t) \text{ (Switching Function)}} u(t)$$

To minimize $\mathcal{H}$ with respect to $u \in [-u_{\max}, u_{\max}]$:

$$u^*(t) = -u_{\max} \cdot \text{sign}(\sigma(t)) = -u_{\max} \cdot \text{sign}\left( \lambda(t)^T g(x(t)) \right)$$

- **Bang-Bang Principle:** The optimal control input switches instantaneously between its extreme limits $\pm u_{\max}$ based on the sign of the switching function $\sigma(t)$.
- **Singular Arcs:** If $\sigma(t) \equiv 0$ over a non-zero time interval, control is determined by differentiating $\sigma(t)$ until $u$ appears explicitly.
