# First-Principles Modeling (Newtonian & Lagrangian)

> **Module 02: Dynamic System Modeling** | Theory Note 01  
> Focus: First-principles physics, Lagrangian dynamics, and complete mathematical derivations of benchmark systems.

---

## 1. Principles of Mechanics

Physical modeling transforms conservation laws into ordinary differential equations. Two complementary formalisms are standard:

1. **Newtonian Mechanics (Vectorial):** Balances forces and moments on individual rigid bodies using free-body diagrams:
   $$\sum F = \frac{d}{dt}(m v) = m \ddot{x}, \qquad \sum \tau = \frac{d}{dt}(I \omega) = I \ddot{\theta}$$
2. **Lagrangian Mechanics (Variational / Energy-Based):** Eliminates internal constraint forces and scales cleanly to multi-body robotic chains.

### 1.1 The Euler-Lagrange Equations

For a system described by $k$ generalized coordinates $q = [q_1, q_2, \dots, q_k]^T$:
1. Compute the total **Kinetic Energy** $T(q, \dot{q})$ and **Potential Energy** $V(q)$.
2. Form the **Lagrangian Function**:
   $$\mathcal{L}(q, \dot{q}) \triangleq T(q, \dot{q}) - V(q)$$
3. Apply the Euler-Lagrange equations for each coordinate $i \in \{1, \dots, k\}$:
   $$\frac{d}{dt}\left( \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \right) - \frac{\partial \mathcal{L}}{\partial q_i} = Q_i$$
   where $Q_i$ represents generalized non-conservative forces (actuator forces/torques, friction).

---

## 2. Derivation 1: Mass-Spring-Damper (L1 Benchmark)

```
        k (spring)
      ┌───/\/\/\/\───┐
      │              │
══════╡              ╞══════[ M ] ───► u(t) (applied force)
      │              │         │
      └───[ c ]──────┘         ▼
         (damper)             x(t) (displacement)
```

### 2.1 Physical Equations
- Mass: $m\text{ [kg]}$
- Damping constant: $c\text{ [N}\cdot\text{s/m]}$
- Spring stiffness: $k\text{ [N/m]}$
- Applied force: $u(t)\text{ [N]}$

Using Newton's Second Law:
$$\sum F = u(t) - k x(t) - c \dot{x}(t) = m \ddot{x}(t)$$

$$m \ddot{x}(t) + c \dot{x}(t) + k x(t) = u(t)$$

### 2.2 First-Order State-Space Form
Choosing state vector $x = [x_1, x_2]^T = [x, \dot{x}]^T$:

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\frac{k}{m} & -\frac{c}{m} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} u$$

---

## 3. Derivation 2: Simple Pendulum with Damping (L2 Benchmark)

A point mass $m$ attached to a massless rigid rod of length $\ell$, rotating about a fixed pivot with viscous friction coefficient $b$ under gravitational acceleration $g$:

### 3.1 Lagrangian Derivation
- Generalized coordinate: Angle $\theta(t)$ measured from downward vertical ($\theta = 0$ is stable hanging down).
- Position of mass: $r = [\ell \sin\theta, -\ell \cos\theta]^T$.
- Velocity of mass: $\dot{r} = [\ell \dot{\theta} \cos\theta, \ell \dot{\theta} \sin\theta]^T \implies \|\dot{r}\|^2 = \ell^2 \dot{\theta}^2$.

Energies:
$$T = \frac{1}{2} m \ell^2 \dot{\theta}^2, \qquad V = -m g \ell \cos\theta$$
$$\mathcal{L}(\theta, \dot{\theta}) = \frac{1}{2} m \ell^2 \dot{\theta}^2 + m g \ell \cos\theta$$

Derivatives:
$$\frac{\partial \mathcal{L}}{\partial \dot{\theta}} = m \ell^2 \dot{\theta} \implies \frac{d}{dt}\left( \frac{\partial \mathcal{L}}{\partial \dot{\theta}} \right) = m \ell^2 \ddot{\theta}$$
$$\frac{\partial \mathcal{L}}{\partial \theta} = -m g \ell \sin\theta$$

Euler-Lagrange with non-conservative torque $Q_\theta = u - b\dot{\theta}$:
$$m \ell^2 \ddot{\theta} - (-m g \ell \sin\theta) = u - b\dot{\theta}$$

$$\ddot{\theta} = -\frac{g}{\ell}\sin\theta - \frac{b}{m \ell^2}\dot{\theta} + \frac{1}{m \ell^2}u$$

---

## 4. Derivation 3: Cart-Pole / Inverted Pendulum on a Cart (L2 Benchmark)

A cart of mass $M$ moves horizontally along a 1D track driven by control force $u(t)$. An inverted pole of mass $m$, length $2\ell$ (or point mass $m$ at distance $\ell$), and moment of inertia $I$ pivots freely atop the cart.

```
                  o (mass m, pole tip)
                 /
                /  length ℓ, angle θ (0 = upright)
               /
              o (pivot)
        [=== Cart M ===] ───► u(t) (Force)
        ════════════════ (horizontal track x)
```

### 4.1 Kinematics
- Generalized coordinates: $q = [x, \theta]^T$, where $x$ is cart position and $\theta$ is pole angle from **upright vertical** ($\theta = 0$ is upright unstable equilibrium).
- Cart position: $r_{\text{cart}} = [x, 0]^T$.
- Pole center of mass: $r_{\text{pole}} = [x + \ell \sin\theta, \ell \cos\theta]^T$.
- Pole COM velocity: $\dot{r}_{\text{pole}} = [\dot{x} + \ell \dot{\theta} \cos\theta, -\ell \dot{\theta} \sin\theta]^T$.

Squared velocity:
$$\|\dot{r}_{\text{pole}}\|^2 = (\dot{x} + \ell \dot{\theta} \cos\theta)^2 + (-\ell \dot{\theta} \sin\theta)^2 = \dot{x}^2 + 2\ell \dot{x} \dot{\theta} \cos\theta + \ell^2 \dot{\theta}^2$$

### 4.2 Energy Equations
$$T = \frac{1}{2} M \dot{x}^2 + \frac{1}{2} m \|\dot{r}_{\text{pole}}\|^2 + \frac{1}{2} I \dot{\theta}^2 = \frac{1}{2}(M + m)\dot{x}^2 + m \ell \dot{x} \dot{\theta} \cos\theta + \frac{1}{2}(I + m \ell^2)\dot{\theta}^2$$
$$V = m g \ell \cos\theta$$

$$\mathcal{L}(x, \theta, \dot{x}, \dot{\theta}) = \frac{1}{2}(M + m)\dot{x}^2 + m \ell \dot{x} \dot{\theta} \cos\theta + \frac{1}{2}(I + m \ell^2)\dot{\theta}^2 - m g \ell \cos\theta$$

### 4.3 Euler-Lagrange Equations of Motion

**1. For Cart Coordinate $x$:**
$$\frac{\partial \mathcal{L}}{\partial \dot{x}} = (M + m)\dot{x} + m \ell \dot{\theta} \cos\theta, \qquad \frac{\partial \mathcal{L}}{\partial x} = 0$$
$$\frac{d}{dt}\left( \frac{\partial \mathcal{L}}{\partial \dot{x}} \right) = (M + m)\ddot{x} + m \ell \ddot{\theta} \cos\theta - m \ell \dot{\theta}^2 \sin\theta = u$$

**2. For Pole Coordinate $\theta$:**
$$\frac{\partial \mathcal{L}}{\partial \dot{\theta}} = m \ell \dot{x} \cos\theta + (I + m \ell^2)\dot{\theta}, \qquad \frac{\partial \mathcal{L}}{\partial \theta} = -m \ell \dot{x} \dot{\theta} \sin\theta + m g \ell \sin\theta$$
$$\frac{d}{dt}\left( \frac{\partial \mathcal{L}}{\partial \dot{\theta}} \right) = m \ell \ddot{x} \cos\theta - m \ell \dot{x} \dot{\theta} \sin\theta + (I + m \ell^2)\ddot{\theta}$$
$$m \ell \ddot{x} \cos\theta + (I + m \ell^2)\ddot{\theta} - m g \ell \sin\theta = 0$$

### 4.4 Matrix Representation & Explicit Solvability

Combining the equations of motion into matrix form:

$$\begin{bmatrix} M + m & m \ell \cos\theta \\ m \ell \cos\theta & I + m \ell^2 \end{bmatrix} \begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix} u + m \ell \dot{\theta}^2 \sin\theta \\ m g \ell \sin\theta \end{bmatrix}$$

The mass matrix determinant is:
$$\mathcal{D}(\theta) = (M + m)(I + m \ell^2) - (m \ell \cos\theta)^2 > 0 \quad \forall \theta$$

Because $\mathcal{D}(\theta) > 0$ strictly, the mass matrix is invertible everywhere, yielding explicit accelerations for numerical simulation.
