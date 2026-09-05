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
```
