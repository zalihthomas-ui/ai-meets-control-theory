# Coupled Two-Tank Liquid Level System Reference Specification

This document provides physical principles, nonlinear Torricelli hydraulics, operating-point linearizations, canonical Quanser Coupled Tanks parameters, and control strategies for the **Coupled Two-Tank System** (`aimct.systems.TwoTank`).

---

## 1. Executive Summary & Physical Architecture

The Coupled Two-Tank system is a foundational benchmark in **process control**, chemical engineering, and fluid mechanics. It captures two core physical phenomena:
1. **Nonlinear Square-Root Discharge Dynamics** (Torricelli's Law $\sqrt{2 g \Delta h}$).
2. **Interactive Capacity Lag & Non-Minimum Phase Characteristics**: Liquid is pumped directly into Tank 1 and flows through an unactuated restriction orifice into Tank 2, where the primary process level $h_2$ is regulated.

```
                  +-------------------------+
                  | Variable-Speed DC Pump  |
                  |     Flow F_in(u)        |
                  +------------+------------+
                               |
                               v
                       +---------------+
                       |    TANK 1     |  Area A1
                       |   Level h1    |
                       +-------+-------+
                               | Inter-tank orifice a12 (Torricelli)
                               v
                       +---------------+
                       |    TANK 2     |  Area A2
                       |   Level h2    |
                       +-------+-------+
                               | Bottom discharge orifice a2
                               v (Drain)
```

---

## 2. Canonical Parameters (Quanser Coupled Tanks Standard)

| Component | Parameter | Symbol | Nominal Value | Unit | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tank 1** | Cross-Sectional Area | $A_1$ | `1.555e-3` | $\text{m}^2$ | Inside area of acrylic cylinder ($D_1 = 4.45\text{ cm}$) |
| | Maximum Liquid Height | $h_{1, \max}$ | `0.30` | $\text{m}$ | Full scale height ($30.0\text{ cm}$) |
| **Tank 2** | Cross-Sectional Area | $A_2$ | `1.555e-3` | $\text{m}^2$ | Inside area of acrylic cylinder ($D_2 = 4.45\text{ cm}$) |
| | Maximum Liquid Height | $h_{2, \max}$ | `0.30` | $\text{m}$ | Full scale height ($30.0\text{ cm}$) |
| **Orifices** | Inter-Tank Area | $a_{12}$ | `1.81e-5` | $\text{m}^2$ | Coupling restriction orifice ($d_{12} = 4.8\text{ mm}$) |
| | Tank 2 Bottom Drain Area | $a_2$ | `1.81e-5` | $\text{m}^2$ | Gravity outflow drain orifice ($d_2 = 4.8\text{ mm}$) |
| **Pump** | Pump Flow Constant | $K_p$ | `3.3e-6` | $\text{m}^3/(\text{s}\cdot\text{V})$ | Volumetric flow rate per volt ($3.3\text{ cm}^3/(\text{s}\cdot\text{V})$) |
| | Max Pump Voltage | $V_{\max}$ | `12.0` | $\text{V}$ | Peak supply voltage to submersible DC pump |
| **Environment**| Gravitational Acceleration | $g$ | `9.81` | $\text{m/s}^2$ | Earth gravity field |

---

## 3. Nonlinear Governing Equations

Let $x = [h_1, h_2]^T \in [0, h_{\max}]^2$ be the liquid heights in Tank 1 and Tank 2, and $u = V_p \in [0, V_{\max}]$ be the pump input voltage:

$$\begin{aligned}
\frac{dh_1}{dt} &= \frac{1}{A_1} \left[ K_p V_p - a_{12} \operatorname{sign}(h_1 - h_2) \sqrt{2 g |h_1 - h_2|} \right] \\
\frac{dh_2}{dt} &= \frac{1}{A_2} \left[ a_{12} \operatorname{sign}(h_1 - h_2) \sqrt{2 g |h_1 - h_2|} - a_2 \sqrt{2 g h_2} \right]
\end{aligned}$$

### 3.1 Fluid Flow Constraints
- **Unidirectional Pump**: $F_{\text{in}} = K_p V_p \ge 0$ (the pump can only add fluid; it cannot actively suck fluid out).
- **Physical Level Bounds**: $0 \le h_1(t) \le h_{1, \max}$ and $0 \le h_2(t) \le h_{2, \max}$.

---

## 4. Operating Point Equilibrium & Linearization

### 4.1 Steady-State Operating Point
For a desired steady-state target level in Tank 2, $h_{2, 0} = 0.10\text{ m}$ ($10.0\text{ cm}$):
$$\dot{h}_2 = 0 \implies a_{12} \sqrt{2 g (h_{1,0} - h_{2,0})} = a_2 \sqrt{2 g h_{2,0}} \implies h_{1,0} = h_{2,0} + \left(\frac{a_2}{a_{12}}\right)^2 h_{2,0} = 2 h_{2,0} = 0.20\text{ m}$$
$$\dot{h}_1 = 0 \implies K_p V_{p, 0} = a_2 \sqrt{2 g h_{2,0}} \implies V_{p, 0} = \frac{a_2 \sqrt{2 g h_{2,0}}}{K_p} = 7.6827\text{ V}$$

### 4.2 Linearized State-Space Matrices
Defining fluid flow resistances:
$$R_{12} = \frac{\sqrt{2 (h_{1,0} - h_{2,0})}}{a_{12} \sqrt{g}} = 7.8899 \times 10^3\text{ s/m}^2, \quad R_2 = \frac{\sqrt{2 h_{2,0}}}{a_2 \sqrt{g}} = 7.8899 \times 10^3\text{ s/m}^2$$

The Jacobian linear state-space system $\Delta \dot{x} = A \Delta x + B \Delta u$ is:
$$A = \begin{bmatrix} -\frac{1}{R_{12} A_1} & \frac{1}{R_{12} A_1} \\ \frac{1}{R_{12} A_2} & -\left(\frac{1}{R_{12} A_2} + \frac{1}{R_2 A_2}\right) \end{bmatrix} = \begin{bmatrix} -0.08152 & 0.08152 \\ 0.08152 & -0.16304 \end{bmatrix}\text{ s}^{-1}$$

$$B = \begin{bmatrix} \frac{K_p}{A_1} \\ 0 \end{bmatrix} = \begin{bmatrix} 2.1222 \times 10^{-3} \\ 0 \end{bmatrix}\text{ m}/(\text{s}\cdot\text{V})$$

- **Open-Loop Poles**: $\lambda(A) = \{-0.03114, \ -0.21342\}\text{ rad/s}$ (Two stable, overdamped time constants $\tau_1 = 32.1\text{ s}, \tau_2 = 4.68\text{ s}$).
- **Controllability**: $\operatorname{rank}(\mathcal{C}) = 2$ (Full rank).

---

## 5. Control Benchmark Comparison & Interaction Analysis

### 5.1 Why Naive SISO PI Struggles
- When a single-loop PI controller regulates $h_2(t)$ by manipulating $V_p(t)$ into Tank 1, the intermediate capacity of Tank 1 introduces a large phase lag ($e^{-\tau s}$).
- High proportional or integral gains cause severe liquid level overshoot and actuator windup against the $0\text{ V}$ pump cut-off (since the pump cannot drain Tank 1).

### 5.2 LQR State Feedback
By sensing both levels $[h_1, h_2]$, LQR coordinates the lead-tank level $h_1$ to smoothly throttle flow into Tank 2, eliminating overshoot:
$$Q = \operatorname{diag}([1.0, 50.0]), \ R = [1.0] \implies K = \begin{bmatrix} 0.2363 & 0.2240 \end{bmatrix}$$

### 5.3 Constrained Linear MPC
Linear MPC anticipates the hydraulic resistance and enforces hard physical constraints ($0 \le V_p \le 12\text{ V}$ and $0 \le h \le 0.30\text{ m}$) over a receding horizon ($N = 25, \Delta t = 0.5\text{ s}$), delivering optimal settling time without overflow.
