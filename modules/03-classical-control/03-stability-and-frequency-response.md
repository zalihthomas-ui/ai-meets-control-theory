# Stability & Frequency Response Analysis

> **Module 03: Classical Control** | Theory Note 03  
> Focus: Routh-Hurwitz, Root Locus, Bode diagrams, Nyquist stability criterion, and robustness margins.

---

## 1. Closed-Loop Stability & The Characteristic Equation

For a feedback loop with loop gain $L(s) = G(s)C(s) = \frac{N(s)}{D(s)}$, the closed-loop transfer function is:

$$T(s) = \frac{L(s)}{1 + L(s)} = \frac{N(s)}{D(s) + N(s)}$$

Closed-loop poles are the roots of the **Characteristic Equation**:

$$1 + L(s) = 0 \iff D(s) + N(s) = 0$$

The system is **Bounded-Input Bounded-Output (BIBO) stable** if and only if all roots of the characteristic equation lie strictly in the open Left-Half Plane (LHP): $\text{Re}(s_i) < 0$ for all $i$.

---

## 2. Routh-Hurwitz Stability Criterion

To evaluate stability without numerically finding polynomial roots:

Given characteristic polynomial $p(s) = a_n s^n + a_{n-1} s^{n-1} + \dots + a_1 s + a_0$ ($a_n > 0$):

1. **Necessary Condition (Stodola):** All coefficients $a_i > 0$. If any coefficient is zero or negative, the system is unstable or marginally stable.
2. **Sufficient Condition (Routh Array):** Construct the array:

$$\begin{array}{c|cccc}
s^n & a_n & a_{n-2} & a_{n-4} & \dots \\
s^{n-1} & a_{n-1} & a_{n-3} & a_{n-5} & \dots \\
s^{n-2} & b_1 & b_2 & b_3 & \dots \\
s^{n-3} & c_1 & c_2 & c_3 & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{array}$$

where $b_1 = \frac{a_{n-1} a_{n-2} - a_n a_{n-3}}{a_{n-1}}$, $b_2 = \frac{a_{n-1} a_{n-4} - a_n a_{n-5}}{a_{n-1}}$, and $c_1 = \frac{b_1 a_{n-3} - a_{n-1} b_2}{b_1}$.

**Theorem (Routh):** The number of roots in the Right-Half Plane (RHP) equals the number of sign changes in the first column of the Routh array. A system is stable if and only if all entries in the first column are strictly positive.

---

## 3. Evan's Root Locus Design Method

The Root Locus plots the paths of closed-loop poles in the complex $s$-plane as a scalar parameter $K \in [0, \infty)$ varies:

$$1 + K G(s) = 0 \iff G(s) = -\frac{1}{K}$$

### 3.1 Fundamental Criteria
- **Magnitude Condition:** $|K G(s)| = 1 \implies K = \frac{1}{|G(s)|}$
- **Angle Condition:** $\angle G(s) = \pm 180^\circ (2k+1), \quad k \in \mathbb{Z}$

### 3.2 Key Construction Rules
1. **Number of Branches:** Exactly $n$ branches (where $n$ is the number of open-loop poles).
2. **Starting and Ending Points:** Branches originate at open-loop poles ($K = 0$) and terminate at open-loop zeros ($K \to \infty$) or radiate to infinity.
3. **Real-Axis Segments:** A point on the real axis lies on the locus if the total number of open-loop poles and zeros to its right is **odd**.
4. **Asymptotes ($n > m$):** As $K \to \infty$, $n-m$ branches approach straight-line asymptotes intersecting at centroid $\sigma_a$ with angles $\theta_a$:
   $$\sigma_a = \frac{\sum_{i=1}^n p_i - \sum_{j=1}^m z_j}{n - m}, \qquad \theta_a = \frac{(2k+1)\pi}{n - m}, \quad k = 0, \dots, n-m-1$$
5. **Breakaway / Break-in Points:** Real-axis points where $\frac{dK}{ds} = 0$.

---

## 4. Frequency Response & Bode Diagrams

The steady-state response of an LTI system to a sinusoidal input $u(t) = U_0 \sin(\omega t)$ is $y(t) = Y_0 \sin(\omega t + \phi)$, where:

$$\frac{Y_0}{U_0} = |G(j\omega)|, \qquad \phi = \angle G(j\omega)$$

- **Bode Magnitude Plot:** $20 \log_{10}|G(j\omega)|\text{ [dB]}$ versus $\log_{10}\omega$.
- **Bode Phase Plot:** $\angle G(j\omega)\text{ [deg]}$ versus $\log_{10}\omega$.

---

## 5. Nyquist Stability Criterion

Based on Cauchy's Argument Principle, the Nyquist plot maps the closed contour enclosing the entire RHP (the Nyquist D-contour) through $L(s)$.

```
                     Im(L)
                       │
                       │    * -180 deg phase crossover
                       │
       (-1, 0)         │
──────────X────────────┼──────────────────────── Re(L)
   Critical Point      │
                       │
                       │
```

### 5.1 The Encirclement Formula

$$Z = N + P$$

where:
- $P$: Number of open-loop poles of $L(s)$ in the RHP (unstable open-loop modes).
- $N$: Number of **clockwise** encirclements of the critical point $-1 + 0j$.
- $Z$: Number of closed-loop poles in the RHP.

**Criterion for Closed-Loop Stability ($Z = 0$):**  
$$N = -P$$
The Nyquist plot of $L(s)$ must encircle the critical point $-1 + 0j$ in the **counter-clockwise** direction exactly $P$ times. If the open-loop plant is stable ($P = 0$), the plot must make **zero** net encirclements ($N = 0$).

---

## 6. Quantitative Robustness Margins

```
Gain Margin: Gm = -20 log10 |L(jω_pc)|  [dB]     where ∠L(jω_pc) = -180°
Phase Margin: Φm = 180° + ∠L(jω_gc)     [deg]    where |L(jω_gc)| = 1 (0 dB)
Delay Margin: τ_max = Φm / ω_gc          [s]      maximum tolerable loop delay
```

### 6.1 Engineering Rules of Thumb
- **Healthy Design Targets:** $G_m \ge 6\text{ dB}$ (factor of 2 gain margin), $\Phi_m \in [45^\circ, 60^\circ]$.
- **Damping Approximation:** For a dominant second-order system:
  $$\zeta \approx \frac{\Phi_m [\text{deg}]}{100}$$
  A $45^\circ$ phase margin provides damping ratio $\zeta \approx 0.45 \implies M_p \approx 20\%$. A $60^\circ$ phase margin provides $\zeta \approx 0.60 \implies M_p \approx 10\%$.
