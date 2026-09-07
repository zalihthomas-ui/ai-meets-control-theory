# Experiment 37 — Tube MPC vs Nominal MPC under a Persistent Disturbance

**Question.** A mass on a spring must stay inside a slot $|x| \le 0.40\,\text{m}$ while an unmodelled force pushes on it every step—a bounded process disturbance $x^+ = A_d x + B_d u + w$, $w \in W$. Nominal `LinearMPC` plans as if $w = 0$. Does the constraint hold? And what does making it hold *robustly* cost?

Companion: [`aimct.controllers.TubeMPC`](../references/tube-mpc-reference.md), [`LinearMPC`](08_mpc_vs_lqr_constrained_cartpole.md), and the [Tube-MPC Reference](../references/tube-mpc-reference.md).

---

## 1. Setup

| Parameter | Value / Configuration |
| :--- | :--- |
| **Plant** | `MassSpringDamper(m=1, k=1, c=4)`, ZOH at $\Delta t = 0.05\,\text{s}$ |
| **Hard State Box** | Slot $|x| \le 0.40\,\text{m}$, Actuator input $|F| \le 6\,\text{N}$ |
| **Disturbance Set $W$** | Box half-widths $[0, 0.07]$—a persistent force on the velocity channel |
| **Realisation** | Worst case: a **constant** push toward the wall |
| **Setpoint** | $x_{\text{ref}} = 0.38\,\text{m}$ (hold near the boundary wall) |

- **Nominal `LinearMPC`**: $N = 30$, $Q = \text{diag}(60, 2)$, $R = 0.15$; state box soft, no disturbance model.
- **`TubeMPC`**: Same stage weights; ancillary gain $K$ = discrete-LQR gain; minimal robust positively invariant (mRPI) box $Z = \Big[\sum_{i=0}^\infty |A_K^i| \hat{w}\Big]$; nominal MPC solved on the state box tightened by $Z$ ($X \ominus Z$) and input box tightened by $|K|Z$ ($U \ominus K Z$); applied control law $u = u_{\text{nom}} + K (x - x_{\text{nom}})$.

```bash
python experiments/37_tube_mpc/run.py
```

---

## 2. Benchmark Results

mRPI box half-widths (computed at $\Delta t = 0.05\,\text{s}$): **$z = [0.070, 0.283]$** ($140$ series terms, per-step contraction $\approx 0.82$). Tightened position box $|x_{\text{nom}}| \le 0.330\,\text{m}$ (wall $0.40\,\text{m}$ minus $z_1 = 0.070\,\text{m}$); tightened input box $|u_{\text{nom}}| \le 3.73\,\text{N}$.

| Controller | Max $|x|$ [m] | Box Violation [mm] | Violates Constraint | Steps Outside | $x_{ss}$ [m] | Peak $|F|$ [N] | RMS $F$ [N] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal LinearMPC** | 0.407 | **7 mm** | **YES** | 124 / 140 | 0.405 | 6.00 | 1.30 |
| **TubeMPC (Robust)** | 0.400 | **0 mm** | **NO** | **0 / 140** | 0.400 | 3.73 | 1.18 |

![Tube MPC vs Nominal MPC under persistent disturbance](figures/exp37_figure.png)

---

## 3. Key Takeaways

1. **Nominal MPC Violates the Constraint Continuously**:
   - Nominal MPC parks its nominal plan at the $0.38\,\text{m}$ setpoint. The persistent disturbance carries the true mass to $0.405\,\text{m}$ steady state ($7\,\text{mm}$ past the $0.40\,\text{m}$ wall), violating constraints for 124 out of 140 steps. Soft penalties register the error but cannot arrest drift.
2. **Tube MPC Enforces Invariance Strictly**:
   - The mRPI box $z_1 = 0.070\,\text{m}$ bounds the maximum error $x - x_{\text{nom}}$ under any disturbance $w \in W$. Tightening the nominal box to $0.330\,\text{m}$ guarantees $x \le 0.330 + 0.070 = 0.400\,\text{m}$. The true mass sits exactly on the boundary without crossing it.
3. **Explicit and Bounded Conservatism Cost**:
   - Tube MPC backs off the nominal target to $0.330\,\text{m}$ ($50\,\text{mm}$ short of the wall). Both the state backoff $z_1$ and input backoff $|K|z$ scale **linearly** with the disturbance bound $\hat{w}$, vanishing when $W \to 0$ (where `TubeMPC` reduces identically to `LinearMPC`).
4. **Engineering Guidelines**:
   - Reach for Tube MPC whenever hard state boundaries must be certified under persistent unmodelled forces, actuator offset biases, or quantization ripple where soft quadratic penalties fail.
