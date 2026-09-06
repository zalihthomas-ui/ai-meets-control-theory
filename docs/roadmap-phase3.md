# Roadmap — Phase 3: Robust, Deployable, Reachable

> **Status:** active planning, opened 2026-09-06, immediately after the
> **v0.2.0** release. See [`docs/roadmap.md`](roadmap.md) for Phases 1–2.

Phase 2 turned this repository into a mature, installable simulation library
with 34 honest experiments. Phase 3 takes it in three directions at once:

1. **Robust & certified control** — close the last acknowledged Track-B gap
   (H∞ / μ) and go past LQR/LQG margins to *certificates*.
2. **The hardware bridge** — the flagship. Everything needed to take a
   controller designed in `aimct` and run it on a real machine, and a
   concrete buildable 2-DOF arm as the worked example.
3. **Reach** — a hosted documentation portal, a citable paper, and a
   performance-regression discipline so the library scales.

Priority order within each track is top-down. Engineering agreement is
unchanged: **clarity → correctness → reproducibility → performance.**

---

## Track A — Robust & Certified Control

The one Track-B stretch item Phase 2 consciously deferred was *"H∞ / μ-synthesis
loop shaping"*. Phase 3 delivers it and builds out the robust-control corner
of the library.

| # | Deliverable | Module | Notes |
| --- | --- | --- | --- |
| A1 | **H∞ mixed-sensitivity synthesis** | `aimct.controllers.hinf` | From-scratch Riccati-based S/KS/T loop shaping (γ-iteration on the two Hamiltonian AREs). Weighting-function helpers (`W_S`, `W_KS`, `W_T`). |
| A2 | **Structured-uncertainty analysis** | `aimct.robust` | μ (structured singular value) upper bound via D-scaling; robust-stability / robust-performance tests against an LFT uncertainty block. D–K iteration at least for analysis. |
| A3 | **Tube / robust MPC** | `aimct.controllers.LinearMPC` (extension) | Constraint tightening with a disturbance-invariant set; nominal-plus-ancillary-feedback tube. |
| A4 | *(stretch)* Control-contraction metrics | `aimct.controllers` | CCM-based nonlinear tracking with a certified contraction rate. |
| A5 | **`DirectCollocation` robustness pass** | `aimct.planning` | Optional variable/constraint scaling (`x_scale` / `u_scale`) + a sparse Hermite–Simpson constraint Jacobian so `trust-constr` is viable as the constrained-problem default. Surfaced by Exp 32 Task B: an *active nonconvex path constraint* (keep-out disk) on a badly-scaled ≥4-state plant (`PlanarQuadrotor`, pitch gain `l/Iyy ≈ 3e3`) makes SLSQP's LSQ subproblem singular; `trust-constr` copes but at 10–50 s. Keep the singular-row guard for a knot on a keep-out centre. Currently a documented limit in the Exp 32 README. |
| **Exp 35** | **H∞ vs LQG under unmodeled dynamics** | `experiments/35_hinf_vs_lqg/` | Plant with a lightly-damped high-frequency mode omitted from the design model. LQG chases the nominal and loses margin; H∞ trades nominal performance for the robustness the omission demands. Report the gain/phase/disk margins and the sensitivity peaks side by side. |
| **Exp 37** | **Tube MPC vs nominal MPC under bounded disturbance** | `experiments/37_tube_mpc/` | Persistent bounded process disturbance on the constrained cart-pole; nominal MPC violates the state box, tube MPC does not — at a measured conservatism cost. |

---

## Track B — The Hardware Bridge  *(flagship)*

The recurring user question — *"can I build a two-arm robot in real life and
use this program to balance it?"* — becomes the organising goal. The answer
must be **yes, here is exactly how**, with every step reproducible in
simulation first.

| # | Deliverable | Module | Notes |
| --- | --- | --- | --- |
| B1 | **HIL harness** | `aimct.hil` | A fixed-rate real-time loop runner (`RealTimeLoop`, deadline-miss accounting), a transport abstraction (`InProcess`, `UDP`, `Serial`), and a **plant-emulator node** with the effects that break a clean sim: encoder quantisation (n-bit), torque saturation + slew limit, transport delay, sample jitter, and sensor noise. The controller code path is *identical* to the pure-sim path. |
| B2 | **Identification-from-logs pipeline** | `aimct.sysid` (extension) | `identify_manipulator(log)` — a CSV / rosbag of `(t, q, q̇, q̈, τ)` → least-squares fit of the linear-in-parameters manipulator regressor `Y(q,q̇,q̈)·π = τ` → a populated `TwoLinkArm`. Train/validation split, condition-number report, and a residual-torque plot. |
| B3 | **Controller export path** | `aimct.deploy` | Serialise a designed discrete-time controller (gains, integrator state layout, update law, limits) to a portable `controller.json`, plus a ~120-line MicroPython / C reference executor that the **HIL harness itself runs** — so "simulated" and "deployed" are the same artefact. |
| B4 | **Buildable 2-DOF arm** | `docs/hardware/two-link-arm.md` | Datasheet-grade build doc: BOM (2× Dynamixel XM430 **or** 2× hobby servo + 2× AS5600 magnetic encoders + an RP2040/Teensy), wiring, safe-torque limits, and the end-to-end `aimct` workflow: *identify → design (LQR / computed-torque / Slotine–Li adaptive) → HIL-validate → deploy*. Ties directly to Experiments 17, 23, and `live_arm_balance`. |
| **Exp 36** | **Balance through the HIL emulator** | `experiments/36_hil_arm_balance/` | The `live_arm_balance` double-inverted-pendulum controllers (stiff / integral / soft LQR) run through `aimct.hil` at a realistic 1 kHz loop with 12-bit encoders, an 8 ms round-trip comms delay, and torque slew limits. Which survive? By how much does the delay margin shrink vs. the ideal sim? This is the experiment that says whether the real build would actually stand up. |

---

## Track C — Estimation & Scale

| # | Deliverable | Module | Notes |
| --- | --- | --- | --- |
| C1 | **Moving-horizon estimation (MHE)** | `aimct.estimation.MHE` | The optimisation-based dual of MPC; arrival cost from the EKF covariance. Compare to EKF/UKF on a system with an active state constraint the Gaussian filters cannot respect. |
| C2 | **Particle filter** | `aimct.estimation.ParticleFilter` | Bootstrap + systematic resampling, for a genuinely non-Gaussian / multi-modal posterior (bearing-only tracking). |
| C3 | **Multi-agent consensus & formation** | `aimct.systems.MultiAgent`, `aimct.controllers.formation` | N differential-drive robots, a communication graph, consensus + rigid-formation control; a partial-observability variant as a stretch. |
| **Exp 38** | **MHE vs EKF with a state constraint** | `experiments/38_mhe_vs_ekf/` | |
| **Exp 39** | **Formation control on a switching comm graph** | `experiments/39_formation/` | |

---

## Track D — Reach: Portal, Paper, Performance

| # | Deliverable | Where | Notes |
| --- | --- | --- | --- |
| D1 | **Hosted documentation portal** | `mkdocs.yml`, `.github/workflows/docs.yml` | mkdocs-material → GitHub Pages. API reference via `mkdocstrings`; `DECISION-GUIDE`, `RESULTS`, `GETTING-STARTED`, `USAGE`, `VISUALIZATION` become first-class pages; notebooks rendered; CLI reference. Deploys on every push to `main`. |
| D2 | **JOSS paper** | `paper.md`, `paper.bib` | A Journal of Open Source Software submission draft — the library as a citable research artefact. Statement of need, a summary of the 34-experiment evidence base, comparison to `python-control` / `do-mpc` / `casadi`. |
| D3 | **Performance-regression suite** | `benchmarks/perf/`, `.github/workflows/perf.yml` | Time the hot paths (CARE solve, dense QP, one iLQR iteration, a `simulate` step) across problem sizes; commit a baseline; CI flags a >20% regression. |
| D4 | **Vectorised batch simulation** | `aimct.simulate` (extension) | `simulate_batch(system, x0s, controller)` for Monte-Carlo robustness sweeps without a Python loop over trials; optional numba path for the integrator. |

---

## Ownership

| Agent | Phase-3 lane | Status |
| --- | --- | --- |
| **toku** | Track A (robust control) | ✅ Exp-32 Task B (keep-out disk, done on a well-scaled point mass — the quad case needs **A5**). **Now:** **A1** `aimct.controllers.hinf` + **Exp 35**, then **A5** + **A2**. |
| **famo** | Track B (HIL) | ✅ **B1** `aimct.hil` (`ee6cf32`) + **Exp 36** (`ee6cf32`) + deploy↔HIL bridge test (`31d7614`). Next: HIL for a second plant / telemetry playback. |
| **lava** | Track D (reach) | ✅ **D1** docs portal (`a2bdae0`, live on `gh-pages`) + **D2** `paper.md`/`paper.bib`. Next: **D3** perf-regression CI (with puma), experiment-page auto-gen. |
| **puma** | Track B support + infra | ✅ **B2** `aimct.sysid.identify_manipulator` (`b6a0686`) + **B3** `aimct.deploy` (`7ffa78d`). **B4** buildable-arm doc lives in the portal (lava). Next: **D3** perf suite, roadmap/CI/release upkeep. |

`H∞` / HIL / portal ran in parallel. Track C opens once **A1** has landed.

---

## Not in Phase 3

Full ROS 2 node generation, a real flight-controller firmware target, a
GUI application, cloud-hosted interactive sandboxes. Revisit after the
hardware bridge has been exercised on an actual build.
