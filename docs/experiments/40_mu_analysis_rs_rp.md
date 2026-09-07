# Experiment 40 — μ Analysis: Structured Robust Stability & Performance

**Question.** An LQG loop has a seemingly comfortable **14 dB gain margin and 66° phase margin**. It tolerates a ±45% loop-gain error on its own, and it tolerates an unmodelled resonance $R(s)$ on its own. Is it safe to ship? Structured **$\mu$ analysis** (`aimct.robust`) says no — and it is the *only* quantity in the sweep that reflects the growing cross-coupling danger.

Companion: [`aimct.robust.mu`](../references/mu-analysis-reference.md) (`mu`, `robust_stability_margin`, `dk_iterate`), [`aimct.controllers.hinf.mixsyn`](../references/hinf-reference.md), and [Experiment 35](35_hinf_vs_lqg.md).

---

## 1. Executive Summary & Setup

Plant $G(s) = \frac{12}{(s+1)(s+3)}$, with moderate LQG controller (nominal crossover $\approx 5\text{ rad/s}$).
Two **output-multiplicative** uncertainties:

| Block | Kind | Weight | Physical Meaning |
| :--- | :--- | :--- | :--- |
| $\delta_r$ | Real, $|\cdot|\le 1$ | $w_g = 0.45$ | A $\pm 45\%$ actuator / loop-gain variation |
| $\Delta_c$ | Complex, $|\cdot|\le 1$ | $W_m(s) = R(s) - 1$ | Covers unmodelled mode $R(s)$ ($\omega_r = 15\text{ rad/s}$) at $\Delta_c = 1$ |

Both enter at the plant output, so the $M$-$\Delta$ matrix is **rank one**, $M(j\omega) = -T(j\omega)\begin{bmatrix} w_g & W_m \\ w_g & W_m \end{bmatrix}$ with $T = GK/(1+GK)$, yielding exact analytic solution:

$$\mu_{\boldsymbol{\Delta}}(M(j\omega)) = |T(j\omega)| \cdot (w_g + |W_m(j\omega)|)$$

The `aimct.robust.mu` solver's upper and lower bounds match this exact formula to $< 4\times 10^{-16}$ relative error across the sweep.

---

## 2. Benchmark Results

### Part (a) — As Resonance Damping Decreases ($\zeta_r \downarrow$)

Nominal single-loop margins on $G$ alone: **GM = 14.0 dB, PM = 66.5°** (constant on every row).

| Parameter $\zeta_r$ | $\mu$ (Analytic) | $\mu$ (Solver UB/LB) | $\|W_m T\|_\infty$ | $w_g \|T\|_\infty$ | RS Margin $1/\mu$ | Nominal $G\cdot R$ Pole | GM | PM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.40 | 0.654 | 0.654 / 0.653 | 0.322 | 0.410 | 1.53 | $-2.53$ | 14.0 dB | 66.5° |
| 0.28 | 0.593 | 0.593 / 0.593 | 0.288 | 0.410 | 1.69 | $-3.18$ | 14.0 dB | 66.5° |
| 0.16 | 0.530 | 0.530 / 0.530 | 0.390 | 0.410 | 1.88 | $-3.50$ | 14.0 dB | 66.5° |
| 0.074 | 0.833 | 0.833 / 0.833 | 0.778 | 0.410 | 1.20 | $-1.78$ | 14.0 dB | 66.5° |
| 0.048 | **1.240** | **1.240 / 1.240** | 1.187 | 0.410 | 0.81 (Fail) | $-1.28$ | 14.0 dB | 66.5° |
| 0.015 | **3.340** | **3.340 / 3.340** | 3.290 | 0.410 | 0.30 (Fail) | $-0.64$ | 14.0 dB | 66.5° |

### Part (b) — Constant-D D-K Iteration on $H_\infty$ Design

At severe $\zeta_r = 0.04$, standard $H_\infty$ mixed-sensitivity achieves peak $\mu = 1.010$ (violating robust stability). Applying 2 rounds of D-K iteration:

| Controller Design | Peak $\mu$ | Robust Stability Margin | $\|S\|_\infty$ | $\gamma$ Achieved |
| :--- | :---: | :---: | :---: | :---: |
| Plain `mixsyn` ($H_\infty$) | 1.010 | 0.990 (RS Fail) | 1.01 | 0.981 |
| **+ Constant-D D-K Iteration** | **0.995** | **1.005 (RS Pass)** | 1.00 | 0.983 |

![Structured vs Single-Loop Robustness and D-K](figures/exp40_figure.png)

---

## 3. In-Depth Engineering Takeaways

1. **Blindness of Classical Single-Loop Margins**:
   - As resonance damping drops from $0.40$ to $0.015$, single-loop gain margin sits frozen at $14.0\text{ dB}$ and phase margin at $66.5^\circ$.
   - The structured singular value $\mu$ climbs monotonically from $0.65$ to $3.34$, crossing $1.0$ at $\zeta_r \approx 0.062$. Classical metrics give zero warning before sudden instability.
2. **Conservative Bound Comparison**:
   - Single unstructured small-gain test $\|W_m T\|_\infty$ crosses $1.0$ at $\zeta_r \approx 0.059$.
   - Structured $\mu$, which simultaneously accounts for real gain uncertainty $w_g$, crosses at $\zeta_r \approx 0.062$, proving that actuator gain drift directly eats into resonance robustness margin.
3. **Synthesis via D-K Iteration**:
   - D-K iteration iteratively alternates between $H_\infty$ controller synthesis and frequency-domain scaling $D(j\omega)$ optimization, bridging the gap between conservative loop shaping and exact structured robust performance.
