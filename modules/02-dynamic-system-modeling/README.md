# Module 02: Dynamic System Modeling

> **Curriculum Path:** [AI Meets Control Theory](../../README.md) $\rightarrow$ **Module 02: Dynamic System Modeling**

---

## Overview

A controller is only as good as the mathematical model upon which it is designed. This module covers the end-to-end physics-to-control pipeline: from first-principles Newtonian and Lagrangian mechanics to state-space realizations, nonlinear Jacobian linearization, and discrete-time conversion for digital microcontrollers.

```
+-----------------------------------------------------------------------------------+
|                           DYNAMIC SYSTEM MODELING                                 |
+-------------------------+-------------------------+-------------------------------+
| First Principles        | Representations         | Linearization & Discrete      |
| - Newton & Euler-Lagr.  | - First-Order ODEs      | - Operating Equilibria        |
| - Mass-Spring-Damper    | - State-Space (A,B,C,D) | - Jacobian Matrices           |
| - Pendulum & Cart-Pole  | - Transfer Function G(s)| - Exact ZOH Discretization    |
+-------------------------+-------------------------+-------------------------------+
```

---

## Module Topics

1. **[01. First-Principles Modeling (Newtonian & Lagrangian)](01-first-principles-modeling.md)**
   - Newtonian mechanics ($F = ma$, $\tau = I\ddot{\theta}$) and free-body diagrams.
   - Lagrangian mechanics ($L = T - V$) and Euler-Lagrange equations.
   - Complete derivations of benchmark systems:
     - Level 1: Mass-Spring-Damper (2nd order linear).
     - Level 2: Simple Pendulum (2nd order nonlinear).
     - Level 2: Cart-Pole / Inverted Pendulum on a Cart (4th order underactuated nonlinear).

2. **[02. State-Space Realizations & Transfer Functions](02-state-space-and-transfer-functions.md)**
   - Definition of state variables and state-space realization $\dot{x} = Ax + Bu, y = Cx + Du$.
   - Laplace transform and matrix transfer function derivation: $G(s) = C(sI - A)^{-1}B + D$.
   - Physical coordinates vs. controllable/observable canonical forms.
   - Poles, transmission zeros, and minimal realizations.

3. **[03. Operating Points & Jacobian Linearization](03-linearization.md)**
   - General nonlinear autonomous/forced dynamics $\dot{x} = f(x, u)$.
   - Identifying equilibrium manifolds $f(x_0, u_0) = 0$.
   - First-order Taylor series expansion and Jacobian evaluation ($A = \left.\frac{\partial f}{\partial x}\right|_0, B = \left.\frac{\partial f}{\partial u}\right|_0$).
   - Linearization of the Cart-Pole at the upright equilibrium ($\theta = 0$) vs. downward equilibrium ($\theta = \pi$).
   - Validity domain and linearization breakdown boundaries.

4. **[04. Discretization & Sampled-Data Systems](04-discretization.md)**
   - Continuous physical systems controlled by discrete digital computers.
   - Zero-Order Hold (ZOH) exact state-space discretization: $A_d = e^{A\Delta t}, B_d = \int_0^{\Delta t} e^{A\tau} B d\tau$.
   - Van Loan's matrix exponential method for exact numerical computation.
   - Comparison: Forward Euler, Tustin (Bilinear transform), and exact ZOH.

---

## Benchmark Systems Covered in this Module

- **`MassSpringDamper`**: $\ddot{x} + \frac{c}{m}\dot{x} + \frac{k}{m}x = \frac{1}{m}u$
- **`Pendulum`**: $\ddot{\theta} + \frac{b}{m\ell^2}\dot{\theta} + \frac{g}{\ell}\sin\theta = \frac{1}{m\ell^2}u$
- **`CartPole`**: 4-state nonlinear coupling cart position $x$ and pole angle $\theta$.
