# Roadmap

## Phase 0 — Foundation (current)

Goal: a working, tested library skeleton + one end-to-end worked example that
demonstrates the full core cycle (theory → ... → comparison).

- [ ] `src/aimct/systems`: `LinearSystem`, `MassSpringDamper`, `Pendulum`, `CartPole` with a common interface (`dynamics(x, u, t) -> xdot`, `step`, `linearize`).
- [ ] `src/aimct/simulate.py`: fixed-step RK4 integrator + rollout helper returning time/state/input trajectories.
- [ ] `src/aimct/controllers`: `PID`, `StateFeedback` (pole placement), `LQR` — from scratch, then cross-checked against `python-control`.
- [ ] `src/aimct/benchmarks`: metrics (settling time, overshoot, RMS tracking error, control effort, robustness sweep) + a comparison harness that runs N controllers on 1 system and emits a table + plots.
- [ ] `tests/`: numerical correctness tests for each of the above.
- [ ] `modules/03-classical-control`: first worked example — stabilize an unstable system with PID (theory notes + runnable script + figures).
- [ ] Reproducibility convention: every experiment is a folder with `config.yaml`, `run.py`, `README.md`, and committed metrics.

## Phase 1 — Classical & Modern Control

Modules 02–04 filled in: modeling, PID design, root locus / Bode tooling,
observers, Kalman filter. Benchmark L1 + L2 systems.

## Phase 2 — Optimal Control & MPC

Module 05: LQR deep-dive, Riccati solvers, a from-scratch linear MPC with
constraints, comparison vs. LQR on cart-pole and a constrained vehicle.

## Phase 3 — ML & RL

Modules 06–07: learned dynamics models, DQN + PPO on cart-pole / pendulum,
classical baselines for every RL result.

## Phase 4 — AI + Control & Capstones

Module 08 hybrids + capstone projects. Standardized "Intelligent Control
Challenge" harness for community benchmarking.

## Working agreement

Priority order for all contributions: **clarity → correctness → reproducibility → performance.**
