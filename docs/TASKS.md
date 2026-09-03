# Task Board — Phase 0

Coordination happens over hcom. Lead: **puma**. Coding: **puma** + **toku**.
Docs / design / research: **lava** + **famo**.

## puma (lead + code)

- Own `src/aimct/` architecture and the common interfaces.
- Build `systems/` (`LinearSystem`, `MassSpringDamper`, `Pendulum`, `CartPole`) and `simulate.py` (RK4 + rollout).
- Integrate everyone's work; run the repo; own the git history and the GitHub publish.
- Review toku's controller PRs for numerical correctness.

## toku (code)

- Build `src/aimct/controllers/`: `PID`, `StateFeedback` (pole placement), `LQR` — from scratch, then cross-check against `python-control`.
- Build `src/aimct/benchmarks/`: metrics (settling time, overshoot, RMS tracking error, control effort) + comparison harness (N controllers × 1 system → table + plots).
- Write `tests/` for controllers and benchmark metrics.

## lava (docs + design)

- Own `modules/01`–`modules/05` theory notes: concise derivations, consistent notation, references. Match the "build-first" tone in `docs/vision.md`.
- Design the standard **comparison report** layout (the table + figure set every benchmark emits) and a plotting style guide for `assets/`.
- Draft `modules/03-classical-control` worked-example writeup once puma+toku land the code.

## famo (research + design)

- Research pass: for each benchmark system (L1–L2), collect canonical parameters, reference controller gains, and known results from the literature into `docs/references/`.
- Survey existing open frameworks (python-control, do-mpc, Stable-Baselines3, Gymnasium classic control) — what we reuse vs. build from scratch. Output: `docs/references/prior-art.md`.
- Draft the "Intelligent Control Challenge" evaluation spec (metrics, scoring, submission format).

## Definition of done for Phase 0

Unstable system stabilized by PID, plus LQR on cart-pole, both run from one
comparison harness that emits a committed table + figures, with tests passing and
theory notes written.
