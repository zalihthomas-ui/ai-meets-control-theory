# Optimal Control Problem Formulation

> **Module 05: Optimal Control** | Theory Note 01  
> Focus: Bolza, Lagrange, and Mayer cost formulations, state/control constraints, and classic engineering objectives.

---

## 1. General Optimal Control Problem

Given a continuous-time dynamical system:

$$\dot{x}(t) = f(x(t), u(t), t), \quad x(t_0) = x_0$$

where $x(t) \in \mathbb{R}^n$ is the state vector and $u(t) \in \mathcal{U} \subseteq \mathbb{R}^m$ is the control input vector constrained to an admissible control set $\mathcal{U}$.

The **Optimal Control Problem** is to find a control trajectory $u^*(t)$ for $t \in [t_0, T]$ that drives the system along an optimal state trajectory $x^*(t)$ while minimizing a scalar performance functional:

$$J(u) \triangleq \underbrace{\Phi(x(T), T)}_{\text{Terminal Cost}} + \int_{t_0}^T \underbrace{L(x(t), u(t), t)}_{\text{Running / Stage Cost}} \, dt$$

subject to:
$$\begin{aligned}
\dot{x}(t) &= f(x(t), u(t), t), \quad x(t_0) = x_0 \\
u(t) &\in \mathcal{U}, \quad \forall t \in [t_0, T] \\
x(t) &\in \mathcal{X}, \quad \forall t \in [t_0, T] \\
x(T) &\in \mathcal{X}_f \quad (\text{Terminal State Constraint Set})
\end{aligned}$$

---

## 2. Standard Cost Functional Classifications

| Formulation Name | Structure of $J(u)$ | Physical Application |
| :--- | :--- | :--- |
| **Bolza Form** | $\Phi(x(T)) + \int_{t_0}^T L(x(t), u(t)) \, dt$ | General tracking + terminal accuracy. |
| **Lagrange Form** | $\int_{t_0}^T L(x(t), u(t)) \, dt \quad (\Phi \equiv 0)$ | Continuous energy / error minimization (e.g., LQR). |
| **Mayer Form** | $\Phi(x(T), T) \quad (L \equiv 0)$ | Target rendezvous, terminal landing precision. |

---

## 3. Classic Objective Functions in Engineering

1. **Minimum-Time Control:**  
   $$J = \int_0^T 1 \, dt = T$$  
   Drives the system from $x_0$ to target set $\mathcal{X}_f$ in the shortest possible time (e.g., missile interception, high-speed robotics). Yields bang-bang control policies.
2. **Minimum-Energy / Minimum-Effort Control:**  
   $$J = \int_0^T u(t)^T R u(t) \, dt \quad (R \succ 0)$$  
   Minimizes electrical power / mechanical actuator strain.
3. **Minimum-Fuel Control:**  
   $$J = \int_0^T \|u(t)\|_1 \, dt = \int_0^T \sum_{j=1}^m |u_j(t)| \, dt$$  
   Minimizes chemical propellant in spacecraft thrusters.
4. **Quadratic Regulation & Tracking:**  
   $$J = \frac{1}{2} x(T)^T P_f x(T) + \frac{1}{2} \int_0^T \left( x(t)^T Q x(t) + u(t)^T R u(t) \right) \, dt$$  
   Balancing transient tracking errors ($Q \succeq 0$) against actuator demand ($R \succ 0$). Forms the basis of LQR and MPC.
