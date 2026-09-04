# Roadmap — Phase 2 (post-capstone depth)

The curriculum (modules 01–08), a real drone, the live sandboxes, the
Intelligent Control Challenge, and the grand bake-off are done. Phase 2 adds
**breadth** (more real systems), **depth** (methods the capstone showed we were
missing), and turns the repo into an installable package.

Priority order within each track is top-down.

---

## Track A — more real systems

Each is a new `aimct.systems` model with real datasheet / spec parameters,
analytic (or numeric) linearisation, a test file, and at least one comparison
experiment.

| system | task | methods to compare | roadmap level |
| --- | --- | --- | --- |
| **Differential-drive robot** (unicycle + wheel/motor lag) | waypoint path following through a cluttered map | pure-pursuit · Stanley · LQR (linearised about the path) · kinematic MPC | L3 |
| **2-link planar arm** (Euler–Lagrange, real link masses/inertias) | joint-space trajectory tracking with a payload step | PD+gravity-comp · **computed-torque** · LQR · MPC; then MRAC when the payload is unknown | L3 |
| **Bicycle-model ground vehicle** | lane-keeping / double-lane-change at speed | Stanley · LQR · kinematic MPC · (stretch) an RL policy | L4 |
| *(stretch)* Furuta pendulum, ball-and-beam, coupled two-tank | classic lab benchmarks | swing-up / loop-shaping / PI vs MPC | L2–L3 |

Reuse the harness, metrics, sweep, and `plot_style`. Trajectory tracking needs
the "reference trajectory as a first-class task" helper (Track C).

## Track B — method depth

| method | where | why (capstone gap) |
| --- | --- | --- |
| **Nonlinear MPC** — real-time-iteration / iLQR (SQP over a rollout) | `aimct.controllers.ILQR` + an experiment vs `SamplingMPC` on the cart-pole swing-up and the quad | Exp 21: the only nonlinear planner we had was CEM, which is loose and 20× the real-time deadline |
| **SAC** (off-policy continuous RL) + a proper PPO hyper-parameter search | `aimct.rl.sac` + re-run Exp 18 / a quad task | Exp 21: pure from-scratch on-policy RL did not bootstrap on the quad |
| **Direct trajectory optimisation** — collocation / multiple shooting | `aimct.trajopt` + swing-up / quad-through-a-gate | we plan online but never *design* an optimal open-loop trajectory offline |
| **Behaviour cloning + DAgger** as a named module | `aimct.rl.imitation` (formalise what Exp 21's policy did ad hoc) | Exp 21 hand-rolled a clone; make it a first-class, tested thing |
| *(stretch)* H∞ / μ loop shaping; disturbance-observer control for the quad | `aimct.controllers` | robustness certificates beyond LQR/LQG margins; the unmatched-wind gap in the live sandbox |

## Track C — infrastructure

- **Trajectory-tracking task in the harness** — reference generators + tracking
  metrics (RMS along-track / cross-track error, path completion), so Track A's
  experiments do not hand-roll their own scoring.
- **`aimct.trajectories`** — lemniscate, minimum-jerk, Dubins path, spline —
  shared by the RL env, the sandboxes, and the new experiments.

## Track D — PyPI packaging

1. Pick the distribution name (`aimct` on PyPI — check availability; fall back to
   `ai-meets-control-theory` if taken).
2. `pyproject.toml`: full metadata (authors, description, `readme`, `license`,
   `keywords`, trove `classifiers`, `project.urls`), a `[project.scripts]`
   console entry point `aimct = "aimct.__main__:main"`.
3. `python -m build` → wheel + sdist; install into a clean venv and smoke-test
   `aimct compare --system pendulum`.
4. `CHANGELOG.md` (Keep-a-Changelog), a version policy (start `0.1.0`).
5. `docs/PACKAGING.md` — the release runbook.
6. A `release.yml` GitHub Action: build on tag, publish via **PyPI Trusted
   Publishing (OIDC)** — no tokens in the repo. Keep it dormant until the first
   tag.
7. Trim the sdist (`MANIFEST.in` / `tool.setuptools`): ship `src/aimct` + the
   module notes, exclude `experiments/`, `docs/papers/*.pdf`, the report build
   artifacts.

## Not in Phase 2

Hardware-in-the-loop, real flight logs, a hosted docs site, multi-agent /
partially-observable systems (Level 6). Noted for Phase 3.
