# Examples Gallery

A collection of minimal (~15–25 line), self-contained, and verified runnable entry points for `aimct`.

```bash
pip install -e ".[dev,ml]"
```

---

## The Gallery

| Script | Capability Demonstrated | Description |
| :--- | :--- | :--- |
| [`01_simulate_single_system.py`](01_simulate_single_system.py) | **Simulate One System** | Linearize `CartPole`, solve continuous CARE for LQR gains, and simulate forward in time with RK4 integration. |
| [`02_compare_controllers.py`](02_compare_controllers.py) | **Compare Controllers** | Benchmark single-loop `PID` against optimal `LQR` on `Pendulum` under identical perturbations and torque bounds. |
| [`03_track_trajectory.py`](03_track_trajectory.py) | **Track a Trajectory** | Steer a `DifferentialDriveRobot` along a cubic `Spline` waypoint path, computing cross-track and along-track metrics. |
| [`04_run_challenge.py`](04_run_challenge.py) | **Intelligent Control Challenge** | Evaluate a custom `ChallengeController` against a blind hidden plant (`track1-msd`) across performance, effort, safety, and robustness axes. |
| [`05_replay_animation.py`](05_replay_animation.py) | **Replay Animation** | Generate a simulated `PlanarQuadrotor` trajectory and build a synchronized visual animation with telemetry HUD via `aimct.viz.animate`. |
| [`06_live_sandbox_headless.py`](06_live_sandbox_headless.py) | **Live Sandbox (Headless)** | Instantiate an interactive `Sandbox` with `TwoLinkArm`, switchable controllers, and dynamic `Disturbance` models without opening a GUI window. |
| [`07_full_workflow_gantry_crane.py`](07_full_workflow_gantry_crane.py) | **The whole loop, one hard problem** | Define a brand-new `GantryCrane` system, validate it with `aimct.dev`, design LQR / input-shaper / constrained-MPC / iLQR, benchmark all four across nominal / wind-gust / model-mismatch scenarios, and animate it with a custom `SystemArtist`. Longer (~150 lines, ~2 min) — the capstone walkthrough, see [`docs/GETTING-STARTED.md`](../docs/GETTING-STARTED.md). |
| [`08_multisystem_relay_handoff.py`](08_multisystem_relay_handoff.py) | **A system of systems** | One package relayed 20 m by three different machines — a `GantryCrane`, a `DifferentialDriveRobot`, and a freshly-derived 8-state `SlungLoadQuad` (planar quadrotor + pendulum load) — each with its own controller (ZV-shaped LQR / cruise + heading LQR / iLQR-RTI). A supervisor sequences the legs and gates each hand-off on *aligned ∧ slow ∧ not-swinging*. Longest (~330 lines, ~5 min); shows how the library's pieces compose into a pipeline. |

---

## Running the Examples

Run any script directly with Python:

```bash
python examples/01_simulate_single_system.py
python examples/02_compare_controllers.py
python examples/03_track_trajectory.py
python examples/04_run_challenge.py
python examples/05_replay_animation.py
python examples/06_live_sandbox_headless.py
python examples/07_full_workflow_gantry_crane.py   # heavier: ~2 min, writes examples/_out/
python examples/08_multisystem_relay_handoff.py    # heaviest: ~5 min, writes examples/_out/
```
