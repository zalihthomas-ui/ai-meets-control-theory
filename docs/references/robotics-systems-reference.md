# Robotics Benchmark Systems Reference Specification

This document provides canonical physical parameters, differential equations, Euler-Lagrange formulations, and standard control configurations for the two Phase 2 robotics plants: **Differential-Drive Mobile Robot** (`DifferentialDriveRobot`) and **Two-Link Planar Robotic Manipulator** (`TwoLinkArm`).

---

## 1. Differential-Drive Mobile Robot (`DifferentialDriveRobot`)

### 1.1 Physical Architecture & Canonical Parameters
Based on standard mobile robotics research platforms (TurtleBot3 Burger / Pioneer 3-DX class):

| Parameter | Symbol | Nominal Value | Unit | Description |
| :--- | :---: | :---: | :---: | :--- |
| Robot Mass | $M_{\text{robot}}$ | `1.0` | $\text{kg}$ | Total chassis mass |
| Wheel Radius | $r$ | `0.033` | $\text{m}$ | Driven wheel radius ($33\text{ mm}$) |
| Wheelbase Track Width | $W$ | `0.160` | $\text{m}$ | Distance between left & right wheel contact points ($160\text{ mm}$) |
| Motor Time Constant | $\tau_m$ | `0.05` | $\text{s}$ | 1st-order actuator velocity response lag |
| Max Linear Velocity | $v_{\max}$ | `0.22` | $\text{m/s}$ | Maximum forward/backward speed |
| Max Angular Velocity | $\omega_{\max}$ | `2.84` | $\text{rad/s}$ | Maximum rotational speed ($162.7^\circ/\text{s}$) |
| Max Linear Acceleration | $a_{\max}$ | `1.5` | $\text{m/s}^2$ | Maximum longitudinal acceleration |
| Max Angular Acceleration | $\alpha_{\max}$ | `5.0` | $\text{rad/s}^2$ | Maximum rotational acceleration |

### 1.2 State Representation & Kinematic Equations
- **State Vector**: $x = [p_x, p_y, \theta, v, \omega]^T \in \mathbb{R}^5$
  - $p_x, p_y$: Cartesian position in inertial world frame [m]
  - $\theta$: Heading orientation angle [rad] (wrapped to $(-\pi, \pi]$)
  - $v$: Longitudinal linear speed [m/s]
  - $\omega$: Rotational yaw rate [rad/s]
- **Control Input**: $u = [v_{\text{cmd}}, \omega_{\text{cmd}}]^T \in \mathbb{R}^2$
- **Nonlinear Differential Equations**:
  $$\begin{aligned}
  \dot{p}_x &= v \cos\theta \\
  \dot{p}_y &= v \sin\theta \\
  \dot{\theta} &= \omega \\
  \dot{v} &= \frac{1}{\tau_m}(v_{\text{cmd}} - v) \\
  \dot{\omega} &= \frac{1}{\tau_m}(\omega_{\text{cmd}} - \omega)
  \end{aligned}$$

### 1.3 Differential Drive Wheel Velocity Conversion
Forward kinematics from wheel angular velocities $(\omega_R, \omega_L)$:
$$v = \frac{r}{2}(\omega_R + \omega_L), \quad \omega = \frac{r}{W}(\omega_R - \omega_L)$$
Inverse kinematics commanding individual wheel speeds:
$$\omega_R = \frac{v + \frac{W}{2}\omega}{r}, \quad \omega_L = \frac{v - \frac{W}{2}\omega}{r}$$

### 1.4 Nonholonomic Constraints & Control Properties
- **Nonholonomic Constraint**: $\dot{p}_x \sin\theta - \dot{p}_y \cos\theta = 0$ (no lateral slipping).
- **Brockett's Condition**: The unicycle kinematics cannot be stabilized to a fixed posture $(x^*, y^*, \theta^*)$ via smooth time-invariant static state feedback $u = K x$. Stabilization requires dynamic feedback linearization, polar coordinate transformations, receding-horizon MPC, or learned policies.

---

## 2. Two-Link Planar Robotic Manipulator (`TwoLinkArm`)

### 2.1 Physical Architecture & Canonical Parameters
Based on standard 2-DOF planar revolute manipulators (Quanser 2-DOF Planar Robot / Teaching Arm):

| Parameter | Symbol | Nominal Value | Unit | Description |
| :--- | :---: | :---: | :---: | :--- |
| Link 1 Mass | $m_1$ | `1.0` | $\text{kg}$ | Mass of link 1 |
| Link 1 Length | $l_1$ | `0.50` | $\text{m}$ | Total kinematic length of link 1 |
| Link 1 COM Offset | $l_{c1}$ | `0.25` | $\text{m}$ | Distance from joint 1 to link 1 center of mass |
| Link 1 Inertia | $I_1$ | `0.02083` | $\text{kg}\cdot\text{m}^2$ | Moment of inertia about COM ($\frac{1}{12} m_1 l_1^2$) |
| Link 2 Mass | $m_2$ | `0.8` | $\text{kg}$ | Mass of link 2 |
| Link 2 Length | $l_2$ | `0.40` | $\text{m}$ | Total kinematic length of link 2 |
| Link 2 COM Offset | $l_{c2}$ | `0.20` | $\text{m}$ | Distance from joint 2 to link 2 center of mass |
| Link 2 Inertia | $I_2$ | `0.01067` | $\text{kg}\cdot\text{m}^2$ | Moment of inertia about COM ($\frac{1}{12} m_2 l_2^2$) |
| Nominal Wrist Payload | $m_p$ | `0.0` | $\text{kg}$ | Payload point-mass at end-effector (perturbable to $0.5\text{ kg}$) |
| Joint 1 Friction | $b_1$ | `0.10` | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | Joint 1 viscous damping coefficient |
| Joint 2 Friction | $b_2$ | `0.10` | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | Joint 2 viscous damping coefficient |
| Joint 1 Max Torque | $\tau_{1, \max}$ | `15.0` | $\text{N}\cdot\text{m}$ | Shoulder actuator peak torque |
| Joint 2 Max Torque | $\tau_{2, \max}$ | `10.0` | $\text{N}\cdot\text{m}$ | Elbow actuator peak torque |
| Joint 1 Position Limits | $[\theta_{1, \min}, \theta_{1, \max}]$ | `[-pi, +pi]` | $\text{rad}$ | Shoulder travel bounds |
| Joint 2 Position Limits | $[\theta_{2, \min}, \theta_{2, \max}]$ | `[-2.8, +2.8]` | $\text{rad}$ | Elbow travel bounds (avoids mechanical collision) |

### 2.2 Generalized Coordinates & Euler-Lagrange Dynamics
- **State Vector**: $x = [\theta_1, \theta_2, \dot{\theta}_1, \dot{\theta}_2]^T \in \mathbb{R}^4$
- **Control Input**: $\tau = [\tau_1, \tau_2]^T \in \mathbb{R}^2$
- **Equations of Motion**:
  $$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) + F_v \dot{q} = \tau$$
  where $q = [\theta_1, \theta_2]^T$ and $\dot{q} = [\dot{\theta}_1, \dot{\theta}_2]^T$.

### 2.3 Explicit Dynamic Matrices

#### 1. Inertia Matrix $M(q) = M(q)^T > 0$:
$$M(q) = \begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{bmatrix}$$
where (including payload mass $m_p$ at distance $l_2$):
$$\begin{aligned}
M_{11} &= m_1 l_{c1}^2 + m_2(l_1^2 + l_{c2}^2 + 2 l_1 l_{c2}\cos\theta_2) + m_p(l_1^2 + l_2^2 + 2 l_1 l_2\cos\theta_2) + I_1 + I_2 \\
M_{12} = M_{21} &= m_2(l_{c2}^2 + l_1 l_{c2}\cos\theta_2) + m_p(l_2^2 + l_1 l_2\cos\theta_2) + I_2 \\
M_{22} &= m_2 l_{c2}^2 + m_p l_2^2 + I_2
\end{aligned}$$

#### 2. Coriolis and Centrifugal Matrix $C(q, \dot{q})$:
$$C(q, \dot{q}) = \begin{bmatrix} -h \dot{\theta}_2 & -h(\dot{\theta}_1 + \dot{\theta}_2) \\ h \dot{\theta}_1 & 0 \end{bmatrix}$$
where:
$$h = (m_2 l_1 l_{c2} + m_p l_1 l_2)\sin\theta_2$$
*(Note: $\dot{M}(q) - 2C(q, \dot{q})$ is skew-symmetric, guaranteeing energy passivity).*

#### 3. Gravity Vector $G(q)$ (Vertical Plane, $g = 9.81\text{ m/s}^2$):
$$G(q) = \begin{bmatrix} (m_1 l_{c1} + m_2 l_1 + m_p l_1)g\cos\theta_1 + (m_2 l_{c2} + m_p l_2)g\cos(\theta_1 + \theta_2) \\ (m_2 l_{c2} + m_p l_2)g\cos(\theta_1 + \theta_2) \end{bmatrix}$$
*(For horizontal-plane motion, $G(q) = [0, 0]^T$).*

#### 4. Viscous Friction:
$$F_v \dot{q} = \begin{bmatrix} b_1 \dot{\theta}_1 \\ b_2 \dot{\theta}_2 \end{bmatrix}$$

### 2.4 Forward Kinematics & Geometric Jacobian
- **End-Effector Cartesian Position**:
  $$\begin{aligned}
  p_x &= l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2) \\
  p_y &= l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2)
  \end{aligned}$$
- **Jacobian Matrix $J(q) = \frac{\partial p}{\partial q}$**:
  $$J(q) = \begin{bmatrix} -l_1\sin\theta_1 - l_2\sin(\theta_1+\theta_2) & -l_2\sin(\theta_1+\theta_2) \\ l_1\cos\theta_1 + l_2\cos(\theta_1+\theta_2) & l_2\cos(\theta_1+\theta_2) \end{bmatrix}$$
- **Singularities**: $\det(J(q)) = l_1 l_2 \sin\theta_2 = 0 \implies \theta_2 = 0\text{ (fully extended)}$ or $\theta_2 = \pm \pi\text{ (fully folded)}$.
