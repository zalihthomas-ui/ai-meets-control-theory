# Module 01: Mathematical Foundations

> **Curriculum Path:** [AI Meets Control Theory](../../README.md) $\rightarrow$ **Module 01: Mathematical Foundations**

---

## Overview

Modern control engineering and machine learning share a common mathematical language. This module provides rigorous, self-contained foundations in linear algebra, dynamical systems, numerical methods, and optimization. Every theoretical concept is directly tied to its computational implementation in Python/NumPy.

```
+-----------------------------------------------------------------------------------+
|                            MATHEMATICAL FOUNDATIONS                               |
+-------------------------+-------------------------+-------------------------------+
| Linear Algebra          | Differential Equations  | Numerical Methods             |
| - State vectors x ∈ Rⁿ  | - Continuous ODEs ẋ=Ax  | - Runge-Kutta (RK4)           |
| - Eigenvalues & Jordan  | - State Transition eᴬᵗ  | - Numerical Stability         |
| - Matrix Exponential    | - Phase Space & Foci    | - Truncation Errors           |
+-------------------------+-------------------------+-------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
|                            OPTIMIZATION & CONTROL                                 |
| - Convex Optimization, Quadratic Forms, Lagrange Multipliers, KKT Conditions      |
+-----------------------------------------------------------------------------------+
```

---

## Module Topics

1. **[01. Linear Algebra for Dynamical Systems](01-linear-algebra.md)**
   - State-space coordinates, vector spaces, and linear transformations.
   - Spectral decomposition: Eigenvalues, eigenvectors, algebraic vs. geometric multiplicity, and Jordan canonical form.
   - Symmetric positive definite matrices and quadratic forms $x^T P x$.
   - Matrix exponential $e^{At}$: analytical definition, properties, and computation methods.
   - Singular Value Decomposition (SVD) and multi-input multi-output (MIMO) gain.

2. **[02. Differential Equations & Dynamical Systems](02-differential-equations.md)**
   - Linear continuous-time systems $\dot{x}(t) = Ax(t) + Bu(t)$.
   - The state transition matrix $\Phi(t, \tau) = e^{A(t-\tau)}$ and total trajectory solution (Duhamel's formula).
   - Phase portraits and classification of 2D equilibria (nodes, saddles, spirals, centers).
   - Internal vs. external stability: Hurwitz matrices, eigenvalues in open left-half plane, and continuous Lyapunov equations $A^T P + P A = -Q$.

3. **[03. Numerical Methods & Integration](03-numerical-methods.md)**
   - Initial Value Problems (IVPs) in simulation.
   - Forward Euler vs. Backward Euler: truncation errors and stability regions.
   - Explicit Runge-Kutta 4th Order (RK4): derivation, Butcher tableau, and error bounds ($\mathcal{O}(\Delta t^4)$).
   - Stiff systems, numerical damping, and step-size selection.

4. **[04. Optimization Foundations](04-optimization.md)**
   - Unconstrained minimization: gradients, Hessians, and convexity conditions.
   - Constrained optimization: Equality constraints and Lagrange multipliers.
   - Inequality constraints and Karush-Kuhn-Tucker (KKT) necessary conditions.
   - Quadratic Programming (QP): formulation $\min \frac{1}{2}x^T H x + c^T x$ subject to $Ax \le b$, active set and interior-point methods in control.

---

## Pedagogical Progression

Every topic follows the core cycle:
$$\text{THEORY} \longrightarrow \text{DERIVATION} \longrightarrow \text{IMPLEMENTATION} \longrightarrow \text{SIMULATION} \longrightarrow \text{VALIDATION}$$

All code snippets utilize pure standard scientific Python (`numpy`, `scipy`).
