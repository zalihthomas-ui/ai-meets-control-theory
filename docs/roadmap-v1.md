# Roadmap — the road to v1.0

> **Status: ✅ SHIPPED.** `aimct 1.0.0` was tagged and published to PyPI on
> 2026-09-07 (`236e51b` → `v1.0.0`), with a full GitHub release. All six exit
> criteria below were met. This page is kept as the record of how it was done;
> the post-1.0 backlog is at the bottom.
>
> Interim: `v0.3.0` (Phase 3 feature checkpoint) shipped the same day.

**v1.0 means:** the library is *feature-complete against its stated vision*
and its *public API is frozen with a stability guarantee*. Concretely, all
six exit criteria below are met and `aimct 1.0.0` is published via the OIDC
pipeline with a full GitHub release.

---

## Exit criteria

1. **Feature tracks closed.** Every item in "Work remaining" below has
   landed on `main` with tests and a docs entry. (CCM / contraction metrics
   — Phase-3 A4 — is explicitly *post-1.0*.)
2. **Green, non-flaky CI.** The full matrix (3.10–3.12) passes on `main` and
   has passed the last 10 consecutive runs with no flaky failures. Coverage
   gate raised to **85 %** and met.
3. **API frozen.** Every public name is in a module `__all__`; constructors,
   return dataclasses, and keyword names are consistent across subpackages;
   `docs/STABILITY.md` states the semver + deprecation policy; the API
   reference renders every public class.
4. **Docs complete.** Portal has a Concepts narrative in addition to the API
   reference and the 40+ experiment pages; every reference doc builds under
   `mkdocs --strict`; `CHANGELOG.md` covers 0.3.0 and 1.0.0.
5. **Report + paper final.** The living technical report covers every
   experiment through the v1.0 set; `paper.md` is JOSS-submission-ready
   (all refs carry DOIs, the checklist passes).
6. **Released.** `v0.3.0` cut when the feature tracks land; a stabilization
   pass; then `v1.0.0` tagged, published, and GitHub-released.

---

## Work remaining

### Track A — robust control & numerics  (owner: **toku**)

| # | Deliverable | Notes |
| --- | --- | --- |
| A5 | `DirectCollocation` robustness pass | *in progress* — `x_scale`/`u_scale` + sparse Hermite–Simpson Jacobian so `trust-constr` is the constrained-problem default; keep the singular-row guard. |
| A3 | **Tube / robust MPC** + **Exp 37** | `aimct.controllers` — constraint tightening with a disturbance-invariant set; nominal + ancillary-feedback tube. Exp 37: tube vs nominal MPC under a persistent bounded disturbance on the constrained cart-pole — nominal violates the state box, tube doesn't, at a measured conservatism cost. |
| A6 | **API-freeze audit** of `controllers` / `planning` / `robust` | naming, `from_system` signatures, return dataclasses, `__all__`, docstring completeness → feed `docs/STABILITY.md`. |

### Track C — estimation & multi-agent  (owner: **famo**)

| # | Deliverable | Notes |
| --- | --- | --- |
| C2 | **`aimct.estimation.ParticleFilter`** + **Exp 41** | bootstrap PF, systematic resampling, adaptive resample on ESS; log-space weights. Exp 41: bearings-only tracking (genuinely non-Gaussian / bimodal posterior) — PF converges where EKF/UKF lock onto the wrong mode. |
| C3 | **`aimct.systems.MultiAgent`** + **`aimct.controllers.formation`** + **Exp 39** | N differential-drive robots, a weighted communication graph (Laplacian), consensus + rigid-formation control; a leader-follower variant. Exp 39: formation acquisition and maintenance on a *switching* comm graph (edges drop and recover) — show the Laplacian's algebraic connectivity gating convergence rate. |
| C4 | **API-freeze audit** of `estimation` / `systems` / `hil` | as A6, for these subpackages. |

### Track D — docs, report, release  (owner: **lava**)

| # | Deliverable | Notes |
| --- | --- | --- |
| D7 | **Concepts narrative** for the portal | `docs/concepts/` — a guided path through the ideas (state-space → estimation → optimal/constrained → robust → learning → hardware), each linking to the experiments and API that realise it. Not a re-hash of the report; the "how the pieces fit" the API reference can't give. |
| D8 | **Report finalisation** | fold in Exp 37, 39, 41; add "Part IV — Robustness & Deployment"; regenerate the PDF; a one-page executive summary at the front. |
| D9 | **`paper.md` to submission standard** | JOSS checklist compliance; every `paper.bib` entry with a DOI; the statement-of-need tightened; a "State of the field" para vs `python-control` / `do-mpc` / `casadi` / `stable-baselines3`. |
| D10 | **`docs/STABILITY.md`** + **CONTRIBUTING refresh** + issue/PR templates | puma drafts the stability/semver/deprecation policy; lava writes it up and refreshes `CONTRIBUTING.md` for a 1.0 project; add `.github/ISSUE_TEMPLATE/` + a PR template. |
| D11 | **`CHANGELOG` for 0.3.0 and 1.0.0** | Keep-a-Changelog; 0.3.0 = the Phase-3 feature set, 1.0.0 = the API freeze + stabilisation. |

### Lead — integration, releases, remaining code  (owner: **puma**)

| # | Deliverable |
| --- | --- |
| L1 | **`aimct.simulate.simulate_batch`** — vectorised Monte-Carlo rollout (`x0s` array, shared controller), optional numba integrator path. + a `benchmarks/perf` case. |
| L2 | **Coverage to ≥ 85 %** — fill the gaps the `--cov-report` names; raise the CI gate. |
| L3 | **Flaky-test hunt** — the py3.12 test-suite failure on `9faf350` (green on 3.10/3.11, green again on `c40057f`) is a nondeterministic test; find and pin it. No flaky failures is a v1.0 gate. |
| L4 | **API-freeze consolidation** — collect A6 + C4 + the ml/rl/viz/dev/deploy/benchmarks audits; add a top-level `aimct.__all__`; produce the stability matrix; `docs/STABILITY.md` policy draft. |
| L5 | **perf-baseline noise** — only auto-commit `baseline.json` when a metric moves > 8 %. |
| L6 | **Releases** — cut `v0.3.0` when A3/A5/C2/C3/L1 are in; run the stabilisation pass (criteria 2–5); tag + publish + GitHub-release `v1.0.0`. |

---

## How it went

```
toku:  A5 ✅ → A3 (TubeMPC) + Exp 37 ✅ → A6 audit ✅
       (session wedged twice; puma landed the A6 code changes directly)
famo:  C2 (ParticleFilter) + Exp 41 ✅ → C3 (MultiAgent + formation) + Exp 39 ✅ → C4 ✅
lava:  D7 Concepts ✅ → D8 report ✅ → D9 paper ✅ → D10 governance ✅ → D11 changelog ✅
       → STABILITY.md public-surface finalisation ✅
nero:  three full QA sweeps ✅ — found the py3.12 flake root cause and the SyntaxWarnings
puma:  L1 simulate_batch ✅ · L3 pinned BOTH flaky tests (test_mhe seed, DOB drift bound) ✅
       · L4 top-level aimct.__all__ + lazy loading ✅ · L5 perf drift-gate ✅
       · EKF/UKF from_system, solve_qp/QPResult public, aimct.robust dataclasses,
         api/robust.md + api/simulate.md, hybrid in the package ✅
       · cut v0.3.0 ✅ · cut v1.0.0 ✅

exit criteria at release:  574 tests green · coverage 90.2 % (gate 85 %) ·
mkdocs --strict clean · report + paper final · STABILITY.md frozen · PyPI + GitHub release live
```

## Post-1.0 backlog

- **Polytopic (H-rep) mRPI for tube MPC** — the box interval-hull mRPI in
  `TubeMPC` inflates for strongly-coupled plants (cart-pole `rho(|A_K|)` ~
  3.5–7 → empty tightened set); a polytopic set with more normals would carry
  it. Exp 37 uses a well-scaled mass-spring-damper for now.
- **Control-contraction metrics (CCM)** — certified nonlinear tracking with a
  contraction rate (was Phase-3 A4, deferred).
- A hosted **interactive sandbox** (WebAssembly / server), **ROS 2 node
  generation**, a real flight-controller firmware target.
- Wider RL: model-based RL, offline RL on the logged experiment data.

These are `1.x` minor-release material — none breaks the frozen API.
