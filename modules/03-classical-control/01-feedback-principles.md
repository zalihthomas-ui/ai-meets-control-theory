# Feedback Principles & Fundamental Limitations

> **Module 03: Classical Control** | Theory Note 01  
> Focus: Closed-loop architecture, sensitivity functions, the algebraic trade-off $S + T = 1$, and the Bode Sensitivity Integral (Waterbed Effect).

---

## 1. Canonical Feedback Architecture

Feedback modifies system dynamics to achieve reference tracking and disturbance rejection despite model uncertainty:

```
           d_in(t) (Input Disturbance)
              │
              ▼
r(t) ──(+)──► e(t) ──►[ C(s) ]──(+)──► u(t) ──►[ G(s) ]──(+)──(+)──► y(t)
 (ref)  ▲ -           (ctrl)     ▲              (plant)   ▲    ▲     (output)
        │                        │                        │    │
        │                        └────────────────────────┘    │ d_out(t) (Output Dist.)
        │                                                      │
        └──────────────────────────────────────────────[ - ]───┴──►(+)──► y_m(t)
                                                                    ▲
                                                                    │ n(t) (Noise)
```

---

## 2. Fundamental Closed-Loop Transfer Functions

Let the **Loop Transfer Function** be $L(s) \triangleq G(s) C(s)$. The system outputs and signals satisfy:

$$\begin{aligned}
y(s) &= \underbrace{\frac{L(s)}{1 + L(s)}}_{T(s)} r(s) + \underbrace{\frac{1}{1 + L(s)}}_{S(s)} d_{\text{out}}(s) + \underbrace{\frac{G(s)}{1 + L(s)}}_{S_{\text{in}}(s)} d_{\text{in}}(s) - \underbrace{\frac{L(s)}{1 + L(s)}}_{T(s)} n(s) \\
e(s) &= \underbrace{\frac{1}{1 + L(s)}}_{S(s)} r(s) - \underbrace{\frac{1}{1 + L(s)}}_{S(s)} d_{\text{out}}(s) - \underbrace{\frac{G(s)}{1 + L(s)}}_{S_{\text{in}}(s)} d_{\text{in}}(s) + \underbrace{\frac{L(s)}{1 + L(s)}}_{T(s)} n(s) \\
u(s) &= \underbrace{\frac{C(s)}{1 + L(s)}}_{S_u(s)} (r(s) - d_{\text{out}}(s) - n(s))
\end{aligned}$$

### 2.1 The Two Primary Sensitivity Operators
- **Sensitivity Function $S(s)$:**
  $$S(s) \triangleq \frac{1}{1 + G(s)C(s)}$$
  Quantifies tracking error, output disturbance attenuation, and relative sensitivity to plant variations $\frac{dG}{G}$.
- **Complementary Sensitivity Function $T(s)$:**
  $$T(s) \triangleq \frac{G(s)C(s)}{1 + G(s)C(s)}$$
  Quantifies reference tracking bandwidth, sensor noise transmission, and robust stability against multiplicative plant uncertainty.

---

## 3. The Algebraic Trade-Off: $S(s) + T(s) \equiv 1$

At every complex frequency $s \in \mathbb{C}$:

$$S(s) + T(s) = \frac{1}{1 + L(s)} + \frac{L(s)}{1 + L(s)} \equiv 1$$

### 3.1 Design Implications for Control Engineers
- **Low Frequencies ($\omega < \omega_c$):** We require excellent tracking ($e \approx 0$) and disturbance rejection $\implies |S(j\omega)| \ll 1 \implies |L(j\omega)| \gg 1 \implies |T(j\omega)| \approx 1$.
- **High Frequencies ($\omega > \omega_c$):** We must reject sensor noise $n(t)$ and remain robust to unmodeled structural dynamics $\implies |T(j\omega)| \ll 1 \implies |L(j\omega)| \ll 1 \implies |S(j\omega)| \approx 1$.
- **Crossover Region ($\omega \approx \omega_c$):** The controller must gracefully transition from high loop gain to low loop gain with adequate phase margin to avoid resonant amplification ($|S(j\omega)| \gg 1$).

---

## 4. The Bode Sensitivity Integral ("The Waterbed Effect")

Can we make $|S(j\omega)|$ arbitrarily small across all frequencies? **No.**

### 4.1 Stable Open-Loop Plants ($p_i \in \mathbb{C}^-$)
For any strictly proper transfer function $L(s)$ with relative degree $\ge 2$:

$$\int_0^\infty \ln |S(j\omega)| \, d\omega = 0$$

*Physical Meaning:* If a controller reduces sensitivity ($\ln |S| < 0$) in the low-frequency control band $[0, \omega_c]$, sensitivity **must increase** ($\ln |S| > 0$, i.e., $|S| > 1$) at higher frequencies. Like pushing down on a waterbed, reducing error in one frequency band causes it to pop up elsewhere.

### 4.2 Unstable Open-Loop Plants ($p_i \in \mathbb{C}^+$)
If the open-loop plant $G(s)$ has $N_{\text{RHP}}$ unstable poles $p_i$ with $\text{Re}(p_i) > 0$:

$$\int_0^\infty \ln |S(j\omega)| \, d\omega = \pi \sum_{i=1}^{N_{\text{RHP}}} \text{Re}(p_i) > 0$$

*Physical Meaning:* Unstable poles act as a severe mathematical penalty. Every unstable open-loop pole forces a net positive area under $\ln |S(j\omega)|$, causing substantial sensitivity peaking ($M_s \triangleq \max_\omega |S(j\omega)| > 2$), sluggish damping, and increased sensitivity to disturbances.
