# H-infinity Mixed-Sensitivity (S / KS / T) Loop Shaping Reference

Companion to `aimct.controllers.hinf`
(`StateSpace`, `weight_S` / `weight_KS` / `weight_T`, `augment_plant`,
`hinf_syn`, `mixsyn`, `HinfController`) and
Experiment 35 (`experiments/35_hinf_vs_lqg_resonance/`).

---

## 1. What H-infinity control optimises

`LQR` / `LQG` minimise a *quadratic average* of the response to white noise (the
`H2` norm). `H-infinity` control minimises the **worst case** — the largest gain
the closed loop can present to *any* bounded-energy disturbance, at *any*
frequency:

$$
\|M\|_\infty \;=\; \sup_{\omega} \; \bar\sigma\!\big(M(j\omega)\big),
$$

where $\bar\sigma$ is the largest singular value. Making a transfer function
small in the $H_\infty$ sense means "no input direction, at no frequency, is
amplified more than this."

The quantity that matters for feedback is not one transfer function but the
**closed-loop gang** built from the loop $L = GK$:

| symbol | definition | shapes … |
| :-- | :-- | :-- |
| **S** (sensitivity) | $(I + GK)^{-1}$ | reference tracking, output-disturbance rejection, **peak $\overline M_S$ = a robustness margin** |
| **T** (compl. sensitivity) | $GK(I+GK)^{-1}$ | sensor-noise rejection, robustness to *multiplicative* plant error |
| **KS** (control sensitivity) | $K(I+GK)^{-1}$ | actuator effort, control-signal bandwidth, saturation risk |

Since $S + T = I$, they cannot both be small at the same frequency — the
design is a frequency-wise *trade*: `S` small in the low band (tracking),
`T` small in the high band (robustness), with `KS` bounding what the actuator
is asked to do in between.

---

## 2. The mixed-sensitivity objective

Pick frequency-dependent **weights** $W_S, W_{KS}, W_T$ that encode the desired
shape (Section 3), stack the three weighted closed-loop maps, and minimise the
$H_\infty$ norm of the stack over all stabilising controllers:

$$
\min_{K \text{ stab.}}\;
\left\|
\begin{bmatrix} W_S\, S \\ W_{KS}\, K S \\ W_T\, T \end{bmatrix}
\right\|_\infty
\;=\; \gamma_{\text{opt}} .
$$

Reading the result: if the synthesis returns $\gamma \le 1$, then
$\bar\sigma(S(j\omega)) \le \bar\sigma(W_S^{-1}(j\omega))$ at every $\omega$, and
likewise for `KS` and `T` — **every specification encoded in a weight is met**.
$\gamma$ slightly above 1 means the specs are met up to the factor $\gamma$;
$\gamma \gg 1$ means the weights are asking for something impossible for this
plant and must be relaxed.

---

## 3. Weight selection (first-order shaping filters)

The library uses the standard Skogestad & Postlethwaite (2005) first-order
forms. Each is scalar or an `I_k` block (`blocks=k`).

### 3.1 Performance weight $W_S$ — `weight_S(wb, A, M)`

$$
W_S(s) = \frac{s/M + \omega_b}{s + \omega_b A},
\qquad 0 < A \ll 1 < M .
$$

* low-frequency gain $\approx 1/A$ — a large number, forcing $|S| \lesssim A$ in
  band (tight tracking / disturbance rejection);
* crossover near $\omega_b$ (the desired closed-loop bandwidth);
* high-frequency gain $1/M$ — so $M$ is the allowed **sensitivity peak**
  $\overline M_S$ (a value of 2 corresponds to a $\ge 6\text{ dB}$ gain margin
  and $\ge 29^\circ$ phase margin).

`weight_S` is *bi-proper* ($D = 1/M \neq 0$). The DGKF synthesis needs
$D_{11}=0$; `mixsyn` therefore uses the **strictly-proper part** of $W_S$ (it
drops the $1/M$ feed-through). The high-frequency sensitivity bound is then
carried by $W_T$ instead — with $S = I - T$, bounding $|T|$ at high frequency
bounds $|S|$ near unity there anyway.

### 3.2 Robustness weight $W_T$ — `weight_T(wb, A, M)`

$$
W_T(s) = \frac{s + \omega_{bT}/M}{A\, s + \omega_{bT}},
\qquad 0 < A \ll 1 < M .
$$

* low-frequency gain $1/M$;
* rises through $\omega_{bT}$ to a high-frequency gain $\approx 1/A$, which
  **forces $|T|$ to roll off** above $\omega_{bT}$.

$W_T$ is where robustness is bought. If the true plant is
$G_p = G(I + \Delta W_m)$ with $\|\Delta\|_\infty \le 1$ and a known
multiplicative-error bound $|W_m(j\omega)|$, the small-gain theorem gives robust
stability iff $\|W_m T\|_\infty < 1$. Choosing $W_T \succeq |W_m|$ (in
particular, $1/A \gtrsim$ the high-frequency uncertainty level) certifies robust
stability against *every* such $\Delta$ — including an unmodelled resonance
(Experiment 35).

### 3.3 Control weight $W_{KS}$ — `weight_KS(wb, A, M)` or a scalar

A flat scalar $W_{KS} = 1/u_{\max}$ caps the control amplitude. The first-order
high-pass form (same shape as $W_T$) additionally limits the *actuator
bandwidth*, rolling the control off past $\omega_b$ so the loop does not command
fast actuator motion that a real drive cannot follow. `mixsyn`'s DGKF path
assumes $W_{KS}$ contributes a full-column-rank direct term (assumption A2), so a
**static** $W_{KS}$ (scalar or matrix) is the safe default; a dynamic $W_{KS}$
also works as long as $D_{12}$ stays full rank.

---

## 4. The generalised plant

`augment_plant(G, W_S, W_KS, W_T)` builds the interconnection with exogenous
input $w$ (reference), control $u$, error $v = w - Gu$:

$$
z_1 = W_S\, v,\qquad z_2 = W_{KS}\, u,\qquad z_3 = W_T\, (G u),\qquad y = v,
$$

as one state-space system $P$ with inputs $[w;\,u]$ and outputs $[z;\,y]$,
$z = [z_1;z_2;z_3]$. For a realisation
$G=(A_g,B_g,C_g,D_g)$, $W_i = (A_i,B_i,C_i,D_i)$ and state
$x = [x_g;x_1;x_2;x_3]$:

$$
A =
\begin{bmatrix}
A_g & 0 & 0 & 0\\
-B_1 C_g & A_1 & 0 & 0\\
0 & 0 & A_2 & 0\\
B_3 C_g & 0 & 0 & A_3
\end{bmatrix},\quad
B_1^{(w)} = \begin{bmatrix}0\\B_1\\0\\0\end{bmatrix},\quad
B_2^{(u)} = \begin{bmatrix}B_g\\-B_1 D_g\\B_2\\B_3 D_g\end{bmatrix},
$$

$$
C_1 =
\begin{bmatrix}
-D_1 C_g & C_1 & 0 & 0\\
0 & 0 & C_2 & 0\\
D_3 C_g & 0 & 0 & C_3
\end{bmatrix},\;
D_{11} = \begin{bmatrix}D_1\\0\\0\end{bmatrix},\;
D_{12} = \begin{bmatrix}-D_1 D_g\\D_2\\D_3 D_g\end{bmatrix},
$$

$$
C_2 = \begin{bmatrix}-C_g & 0 & 0 & 0\end{bmatrix},\qquad
D_{21} = I,\qquad D_{22} = -D_g .
$$

The defining identity — verified in `tests/test_hinf.py` to $10^{-6}$ — is that
the lower LFT of $P$ with **any** controller reproduces the stack:

$$
\mathcal{F}_\ell(P, K) \;=\;
\begin{bmatrix} W_S\,S \\ W_{KS}\,K S \\ W_T\,T \end{bmatrix},
\qquad u = K y .
$$

---

## 5. DGKF state-space solution ($\gamma$-iteration)

`hinf_syn` follows Doyle-Glover-Khargonekar-Francis (1989) / Zhou-Doyle-Glover
(1996) ch. 17. After normalising $D_{12}^\top D_{12} = I$,
$D_{21} D_{21}^\top = I$ by SVD scalings, and balancing the plant state, each
trial $\gamma$ solves two Hamiltonian Riccati equations:

$$
H_\infty =
\begin{bmatrix}
A - B_2 D_{12}^\top C_1 & \gamma^{-2} B_1 B_1^\top - B_2 B_2^\top\\
-C_1^\top (I - D_{12}D_{12}^\top) C_1 & -(A - B_2 D_{12}^\top C_1)^\top
\end{bmatrix},
\quad X_\infty = \operatorname{Ric}(H_\infty),
$$

$$
J_\infty =
\begin{bmatrix}
(A - B_1 D_{21}^\top C_2)^\top & \gamma^{-2} C_1^\top C_1 - C_2^\top C_2\\
-B_1 (I - D_{21}^\top D_{21}) B_1^\top & -(A - B_1 D_{21}^\top C_2)
\end{bmatrix},
\quad Y_\infty = \operatorname{Ric}(J_\infty),
$$

where $\operatorname{Ric}(\cdot)$ is the stabilising solution
$P = U_2 U_1^{-1}$ from the stable invariant subspace, extracted by an **ordered
real Schur** factorisation (numerically stable for the stiff Hamiltonians that
arise when a weight corner sits far from the plant band).

**A stabilising $H_\infty$ controller for this $\gamma$ exists** iff

$$
X_\infty \succeq 0, \qquad Y_\infty \succeq 0, \qquad
\rho(X_\infty Y_\infty) < \gamma^2 .
$$

The $\gamma^{-2}$ term is the only structural difference from the two LQR/LQG
Riccati equations: it injects the *worst-case disturbance feedforward*
$w_{\text{worst}} = \gamma^{-2} B_1^\top X_\infty x$. As $\gamma \to \infty$ that
term vanishes and the $H_\infty$ controller collapses onto the $H_2$ (LQG) one —
$H_\infty$ is LQG that has stopped assuming the disturbance is benign.

`hinf_syn` **bisects** $\gamma$ down to the smallest value for which the three
conditions hold and returns the **central controller**

$$
F = -(D_{12}^\top C_1 + B_2^\top X_\infty),\quad
L = -(B_1 D_{21}^\top + Y_\infty C_2^\top),\quad
Z = (I - \gamma^{-2} Y_\infty X_\infty)^{-1},
$$

$$
A_K = A - B_1 D_{21}^\top C_2 + \gamma^{-2}\tilde B_1 \tilde B_1^\top X_\infty
      + B_2 F - Z\,(Y_\infty C_2^\top)\,C_2,
\qquad \tilde B_1 = B_1 (I - D_{21}^\top D_{21}),
$$

$$
K = \big(A_K,\; -Z L,\; F,\; 0\big),
$$

un-scaled by the $D_{12}/D_{21}$ normalisations. `mixsyn`'s plant always has
$D_{21} = I$ (perfect reference measurement), so $Y_\infty = 0$, $Z = I$, and
$K$ reduces to $\big(A - B_1 C_2 + B_2 F,\; B_1,\; F,\; 0\big)$.

**Verification.** `tests/test_hinf.py` checks, without any external solver, that
the returned $K$ (i) makes the closed loop internally stable, (ii) achieves
$\|\mathcal{F}_\ell(P,K)\|_\infty \le \gamma$ (via the independent bounded-real
Hamiltonian $H_\infty$-norm), (iii) satisfies $X\succeq0$, $Y\succeq0$,
$\rho(XY)<\gamma^2$, and (iv) that $\gamma$ is a tight infimum (feasibility flips
just below it). A slycot-guarded test additionally matches $\gamma$ to
`control.hinfsyn` within 3 %.

**Limits.** The dense-Schur Riccati solver copes with weight corner frequencies
up to $\sim 4$–$5$ decades from the plant band; beyond that (e.g. a $W_T$ pole
$10^6$ rad/s above the dominant plant pole) it can fail to bracket $\gamma$ —
use gentler weights or a specialised solver. Bi-proper $W_S$ needs
`mixsyn` (which strips the feed-through); calling `hinf_syn` directly on a
$D_{11}\neq 0$ plant raises.

---

## 6. H-infinity vs LQG on an unmodelled mode (Experiment 35)

**Setup.** Design a controller for a nominal plant $G(s)$, then evaluate it on
the *true* plant $G_p(s) = G(s)\cdot R(s)$ where $R(s)$ is a **lightly-damped
high-frequency mode** ($\omega_r$ well above crossover, $\zeta_r \approx 0.02$)
that was **left out of the design model**.

**LQG** shapes only the nominal loop; it places no explicit constraint on $|T|$
past crossover, so $T$ typically has non-trivial gain out at $\omega_r$. When
$R(s)$ is present, that mode is inside the loop, $|W_m T|$ crosses 1, and the
gain / phase / **disk** margins collapse — often to instability.

**H-infinity** with a $W_T$ whose $1/A$ level exceeds the multiplicative-error
size $|W_m(j\omega_r)|$ forces $|T| < 1/|W_T| \le 1/|W_m|$ there. By the small
gain theorem the loop is then robustly stable for *every* $\|\Delta\|_\infty\le1$
— the resonance included, even though it never appeared in the design model. The
price is a slightly lower nominal bandwidth and a marginally higher nominal $S$
peak: robustness traded for nominal performance, explicitly and by the single
knob $A$ in $W_T$.

Experiment 35 reports, side by side for LQG and $H_\infty$: the nominal and
perturbed **gain / phase / disk margins**, the $\|S\|_\infty$ and $\|T\|_\infty$
peaks, and a 4-panel figure (loop $L$, $S$, $T$ with the $W_T^{-1}$ bound, and
the margin table).

---

## 7. References

* J. Doyle, K. Glover, P. Khargonekar & B. Francis, "State-space solutions to
  standard H2 and H-infinity control problems", *IEEE TAC* 34(8), 831-847, 1989.
* K. Zhou, J. Doyle & K. Glover, *Robust and Optimal Control*, Prentice Hall,
  1996 — chapters 13 (H-infinity norm), 16-17 (state-space H-infinity).
* S. Skogestad & I. Postlethwaite, *Multivariable Feedback Control: Analysis and
  Design*, 2nd ed., Wiley, 2005 — chapters 2-3 (S/T, weights), 9 (the S/KS/T
  stack and the general algorithm).
* J. Doyle, "Analysis of feedback systems with structured uncertainties",
  *IEE Proc. D* 129(6), 242-250, 1982 (disk margin / structured singular value).
