# Tube MPC Reference

Companion to `aimct.controllers.TubeMPC` / `mrpi_box` / `MRPISet`
and Experiment 37 (`experiments/37_tube_mpc/`).

---

## 1. The problem nominal MPC cannot handle

`LinearMPC` plans on the assumption that
the model is exact. Under a **persistent additive disturbance**

$$
x^+ = A_d x + B_d u + w, \qquad w \in W \;(\text{a box}),
$$

the true state drifts off the nominal prediction by an amount that never
decays, and a hard state constraint `x ∈ X` that the nominal plan just meets is
violated the moment `w` pushes the state across the boundary. Softening the
constraint (what `LinearMPC` does) hides the violation; it does not prevent it.

## 2. The tube idea (Mayne, Seron & Raković 2005)

Split the control into a **nominal** part and an **ancillary** feedback:

$$
u = u_{\text{nom}} + K\,(x - x_{\text{nom}}),
\qquad
x_{\text{nom}}^+ = A_d x_{\text{nom}} + B_d u_{\text{nom}} .
$$

The error `e = x − x_nom` then evolves as `e^+ = A_K e + w` with
`A_K = A_d + B_d K` (K chosen Schur — the discrete-LQR gain by default). If
`Z` is a **robust positively invariant** (RPI) set for `A_K` and `W`,

$$
A_K Z \oplus W \subseteq Z,
$$

then `e_0 ∈ Z ⟹ e_k ∈ Z` for all `k` and every disturbance sequence: the true
state stays inside a **tube** `x_nom + Z` around the nominal trajectory.

**Tightening.** Solve the nominal MPC on constraints shrunk by the tube:

$$
x_{\text{nom}} \in X \ominus Z, \qquad u_{\text{nom}} \in U \ominus K Z
$$

(`⊖` is the Pontryagin difference). Then *nominal* feasibility implies the
**true** state satisfies `x ∈ X` and the **true** input satisfies `u ∈ U`, for
all `w ∈ W`. Recursive feasibility follows from the RPI property plus the
nominal MPC's own terminal ingredients.

## 3. The mРPI set, as a box

The *minimal* RPI set is
`F_∞ = ⊕_{i=0}^{∞} A_K^i W`. `aimct` outer-approximates it by its **interval
hull** — the one representation `LinearMPC` consumes. For a symmetric box
`W = {|w| ≤ ŵ}` the support of `A_K^i W` on axis `j` is `(|A_K^i| ŵ)_j`
(`|·|` element-wise), so

$$
Z = \Big\{\, |x| \le \hat z \,\Big\},
\qquad
\hat z = \Big(\sum_{i \ge 0} |A_K^i|\Big)\,\hat w .
$$

The series converges for any Schur `A_K` (`‖A_K^i‖ ≤ C\rho^i`); `mrpi_box` sums
it until `‖A_K^k‖_∞ < term_tol` and adds a geometric-rate bound
`‖A_K^s‖_∞ /(1-\rho)·‖ŵ‖` for the negligible tail.

**Why not the Raković `(1-\alpha)^{-1}\sum_{i<s}` form?** That form needs
`A_K^s W ⊆ α W`, which requires `W` *full-dimensional*. A disturbance on only a
few channels (the common case) has a degenerate `W`, and the plain series
handles it directly. **Why not the fixed point `(I - |A_K|)^{-1} ŵ`?** That box
is exactly RPI, but only exists when `\rho(|A_K|) < 1` — which *fails for every
position/velocity plant under position feedback* (the `A_K[vel, pos]` entry is
negative, and `|A_K|` then has `\rho \ge 1`). The interval hull of `F_∞` still
**contains** `F_∞`, so constraint tightening by it is sound; `rpi_residual > 0`
merely records that the box hull is itself not invariant, which is expected and
harmless.

**Properties** (`tests/test_tube_mpc.py`):

* `W ⊆ Z` always;
* for a non-negative `A_K` the box *is* RPI and matches `(I − A_K)^{-1} ŵ`;
* `Z` scales **linearly** with `ŵ` — the conservatism is finite and vanishes as
  the disturbance shrinks;
* `TubeMPC` raises if `X ⊖ Z` or `U ⊖ K Z` is empty (the disturbance is too
  large for the constraints).

## 4. What Experiment 37 shows

A mass-spring-damper in a slot `|x| ≤ 0.40 m`, held near the wall while a
persistent bounded force pushes it toward the boundary. Nominal `LinearMPC`
parks its plan at the setpoint and the disturbance carries the *true* mass ~7 mm
**past** the wall, for essentially the whole run — a sustained box violation.
`TubeMPC` backs the *nominal* setpoint off the boundary by the tube half-width
`\hat z_x = 0.070 m`, so the true mass sits exactly on the wall and never
crosses it, for every disturbance realisation in the box. The price, measured in
the experiment: a nominal setpoint 50 mm short of the wall and a control
confined to the tightened `±3.7 N` input box (vs the full `±6 N`) — conservatism
bought for a guarantee, and it scales linearly with the disturbance bound, so it
shrinks as `W` is tightened and vanishes at `W = 0`.

(A strongly-coupled unstable plant like the cart-pole inflates the box hull too
far — `\rho(|A_K|)` is large — and would need a polytopic mРPI; the box method
is at its best for a moderately-coupled loop, which the MSD is.)

## 5. References

* D. Q. Mayne, M. M. Seron & S. V. Raković, "Robust model predictive control of
  constrained linear systems with bounded disturbances", *Automatica* 41(2),
  219–224, 2005.
* S. V. Raković, E. C. Kerrigan, K. I. Kouramas & D. Q. Mayne, "Invariant
  approximations of the minimal robust positively invariant set", *IEEE TAC*
  50(3), 406–410, 2005.
* J. B. Rawlings, D. Q. Mayne & M. M. Diehl, *Model Predictive Control: Theory,
  Computation, and Design*, 2nd ed., Nob Hill, 2017 — §3.5 (tube MPC).
