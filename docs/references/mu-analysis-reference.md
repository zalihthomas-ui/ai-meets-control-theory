# Structured Singular Value (μ) Analysis Reference

Companion to `aimct.robust`
(`BlockStructure`, `mu`, `robust_stability_margin`,
`robust_performance_margin`, `dk_iterate`) and
Experiment 40 (`experiments/40_mu_analysis_rs_rp/`). This is the *analysis*
counterpart to the H-infinity *synthesis* in
[`aimct.controllers.hinf`](hinf-reference.md).

---

## 1. Why μ

H-infinity loop shaping protects against an **unstructured** ball of dynamics —
one full complex `Δ` with `‖Δ‖∞ ≤ 1`. Real hardware uncertainty is almost never
that shapeless: it is a handful of *independent* effects — a mass known to ±10 %,
a neglected actuator lag, a lightly-damped mode of roughly-known frequency, a
delay — each with its own bound. Bundling them into one full block is
conservative: it lets the perturbations conspire in ways the physics forbids
(e.g. the mass and the delay perturbations phased to cancel a stabilising term).

The **structured singular value** `μ` answers the sharp question: given that the
uncertainty is block-diagonal with a *known structure*, exactly how much of it
can the loop absorb?

---

## 2. The M-Δ form and the block structure

Pull every uncertain element out of the closed loop into a single block-diagonal
perturbation and collapse everything known into one transfer matrix `M(s)`:

```
        ┌─────────────┐
    w ──►│             ├──► z          F_u(M, Δ) = M22 + M21 Δ (I − M11 Δ)⁻¹ M12
        │    M(s)     │
   ┌───►│             ├───┐
   │    └─────────────┘   │
   │                      │
   └──────  Δ  ◄───────────┘           Δ = blkdiag(Δ1, …, Δm)
```

The structure `𝚫` is an ordered list of blocks, each

| kind | form | meaning |
| :-- | :-- | :-- |
| `"R"` | `δ I_k`, `δ ∈ ℝ` | a repeated **real** scalar — a parametric uncertainty (a gain, a mass) |
| `"C"` | `δ I_k`, `δ ∈ ℂ` | a repeated **complex** scalar — a scalar dynamic uncertainty entering `k` places |
| `"F"` | any `k×k` complex | a **full complex** block — an unmodelled MIMO transfer, or a fictitious *performance* block |

`aimct.robust.BlockStructure` holds `[(k, kind), …]`; `BlockStructure.parse`
also accepts a bare int `n` (one `n×n` full block, the unstructured case).

**Definition.**

$$
\mu_{\mathbf{\Delta}}(M) \;=\;
\Big( \min\{\, \bar\sigma(\Delta) : \Delta \in \mathbf{\Delta},\;
             \det(I - M\Delta) = 0 \,\} \Big)^{-1},
$$

with `μ = 0` if no structured `Δ` makes `I − MΔ` singular. It is *not* a norm and
has no closed form for more than three blocks; it is computed by bracketing.

---

## 3. The bounds

$$
\rho(M) \;\le\;
\max_{\Delta\in\mathbf{\Delta},\,\bar\sigma(\Delta)\le 1}\rho(\Delta M)
\;\le\; \mu_{\mathbf{\Delta}}(M) \;\le\;
\inf_{D\in\mathbf{D}} \bar\sigma(D M D^{-1}) \;\le\; \bar\sigma(M).
$$

### 3.1 Upper bound — D-scaling

`𝐃` is the set of invertible matrices that **commute** with every `Δ ∈ 𝚫`: a
positive scalar `d_i I` on each full block, a full Hermitian positive-definite
`D_i` on each repeated-scalar block. Because `DΔD⁻¹ = Δ`,
`det(I − MΔ) = det(I − (DMD⁻¹)Δ)`, so `μ_𝚫(M) = μ_𝚫(DMD⁻¹) ≤ σ̄(DMD⁻¹)` for
*every* such `D` — minimise over `D` to tighten. The bound is **exact for three
or fewer blocks** and mildly conservative beyond.

`aimct.robust` uses the **diagonal** subset of `𝐃` (one positive scale per
block), found by **Osborne balancing** of the block-aggregated `|M|`: iterate
`d_p ← d_p · √(col_p / row_p)` on the `b×b` matrix of block Frobenius norms until
the row and column mass equalise, keeping the best `σ̄(DMD⁻¹)` seen (`D = I` is
always a fallback, so the returned bound never exceeds `σ̄(M)`).

The full `D_i` on repeated-scalar blocks needs an LMI / convex step and buys
little on the small structures the library targets; it is a later refinement.

### 3.2 Lower bound — power iteration

Any *specific* structured `Δ` with `‖Δ‖ ≤ 1` gives a valid lower bound
`μ ≥ ρ(ΔM)`. `aimct.robust` maximises `ρ(ΔM)` by an eigenvector-alignment
iteration: from the dominant right and left eigenvectors `x`, `w` of the current
`ΔM`, rebuild each block to drive `wᴴ Δ M x` up —

* **full block:** `Δ_i = x_i (Mx)_iᴴ / ‖(Mx)_i‖²`, scaled to `σ̄ ≤ 1`;
* **complex scalar:** `d_i = \overline{w_iᴴ (Mx)_i} / |w_iᴴ (Mx)_i|` (unit modulus);
* **real scalar:** `d_i = sign\,\mathrm{Re}\big(w_iᴴ (Mx)_i\big)`,

restarted from `I` and several random structured `Δ`. For a rank-one `M = uvᵀ`
with scalar blocks this recovers the exact `μ = Σ_k |u_k v_k|` (both bounds), and
it matches a brute-force `Δ`-grid search on small real-block problems — see
`tests/test_robust.py`.

Real blocks are handled **exactly in the lower bound**; the upper bound treats
them as complex — a valid over-bound. Tight real-`μ` needs the `G`-scaling of
Young–Newlin–Doyle and is deferred.

---

## 4. Robust stability and robust performance

Wrap the interconnection at each frequency into `M(jω)` (the closed loop, with
the controller already absorbed).

**Robust stability.** The loop is stable for *every* `Δ ∈ 𝚫` with
`‖Δ‖∞ < 1` iff

$$
\sup_\omega \mu_{\mathbf{\Delta}}\big(M(j\omega)\big) < 1,
\qquad
\text{margin} = \frac{1}{\sup_\omega \mu_{\mathbf{\Delta}}(M(j\omega))}.
$$

`robust_stability_margin(M_of_omega, structure, omega)` sweeps the grid and
returns `mu_lower(ω)`, `mu_upper(ω)`, the peak, the worst-case `ω`, the margin,
and a `robust` flag. A margin of 2 means every uncertainty bound can be doubled
and stability still holds; `1/peak_lower` is the size of a `Δ` that is *known* to
destabilise.

**Robust performance.** By the *main-loop theorem*, a performance requirement
`‖F_u(M,Δ)‖∞ < 1` for all admissible `Δ` is equivalent to robust *stability* of
an augmented problem: append a **fictitious full complex block** `Δ_p`
(size = performance channel) to the structure and test
`sup_ω μ_{[𝚫, Δ_p]}(M) < 1`. `robust_performance_margin(N, structure,
perf_shape, omega)` does exactly this — it is `robust_stability_margin` on the
augmented structure.

---

## 5. Constant-D D-K iteration

`μ`-*synthesis* alternates:

* **K-step** — hold `D(ω)` fixed, absorb it into the weights, and run
  H-infinity synthesis (`aimct.controllers.hinf.hinf_syn`) on the scaled plant;
* **D-step** — hold `K` fixed, sweep `μ`, and refit the `D(ω)` scalings that
  minimise the peak upper bound.

Full D-K fits a *rational* `D(s)` each round. `aimct.robust.dk_iterate` does the
**analysis-grade** version: a single frequency-flat scale per block, taken from
the worst-frequency `D` of the `μ` sweep. That still tightens a peak that lives
in one band — the common case — and shows the mechanism without the curve-fitting
machinery. It takes `resynthesise(d) -> K` and `mu_matrix(K, ω) -> M` callbacks
and returns `(K, history)` with the `robust_stability_margin` dict of each round.

---

## 6. What Experiment 40 shows

On Experiment 35's plant, with the resonance
`R(s)` re-expressed as a **multiplicative-uncertainty weight** `W_m(s)` and a
**second, real, parametric** uncertainty on the DC gain:

1. **`μ` predicts the coupled failure that single-loop margins miss.** Sweeping
   the resonance damping `ζ_r` down, the `μ`-based RS margin crosses 1 at exactly
   the `ζ_r` where LQG loses closed-loop stability — while the *nominal* loop's
   gain and phase margins, computed on `G` alone, barely move. Two moderate
   uncertainties, each individually survivable, are jointly not: `μ` of the
   2-block structure exceeds the max of the two one-block values.

2. **D-K iteration buys back robust-performance margin.** One or two constant-`D`
   rounds on the H-infinity design from Exp 35 lower the peak `μ` of the
   RP-augmented structure versus the plain `mixsyn` controller, at a small
   nominal-performance cost — the same robustness-for-bandwidth trade as Exp 35,
   now targeted at the frequency `μ` flags.

---

## 7. References

* J. Doyle, "Analysis of feedback systems with structured uncertainties",
  *IEE Proc. D* 129(6), 242–250, 1982.
* M. Safonov, "Stability margins of diagonally perturbed multivariable feedback
  systems", *IEE Proc. D* 129(6), 251–256, 1982 (D-scaling).
* A. Packard & J. Doyle, "The complex structured singular value",
  *Automatica* 29(1), 71–109, 1993.
* G. Balas, J. Doyle, K. Glover, A. Packard & R. Smith, *μ-Analysis and
  Synthesis Toolbox*, 1991 (D-K iteration).
* K. Zhou, J. Doyle & K. Glover, *Robust and Optimal Control*, 1996, ch. 10–11.
* S. Skogestad & I. Postlethwaite, *Multivariable Feedback Control*, 2nd ed.,
  2005, ch. 8 (robust stability / performance, `μ`).
