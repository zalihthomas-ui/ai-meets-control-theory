# Using `aimct`

How to drive the framework once you have cloned the repo — the entry points, the
five axes you can vary, and copy-paste recipes for "simulate and see".

```bash
git clone https://github.com/zalihthomas-ui/ai-meets-control-theory.git
cd ai-meets-control-theory
pip install -e ".[dev,ml]"        # dev = tests + python-control; ml = torch/gymnasium/sb3
pytest -m "not slow"             # ~90 s sanity check
```

There are two ways in: the **CLI** (`python -m aimct …`) for the built-in
presets and the live sandboxes, and the **library** (`import aimct…`) when you
want to choose the pieces yourself.

---

## 1. The command line

| command | what it does |
| :-- | :-- |
| `python -m aimct list` | every built-in system; the four with a `compare` preset are listed first |
| `python -m aimct compare --system cartpole` | LQR + MPC bake-off on one preset system → prints a table, `--out DIR` also writes `table.md` / `figure.png` |
| `python -m aimct compare --system quadrotor --t-final 12 --dt 0.004` | same, overriding the horizon / step |
| `python -m aimct live` | interactive 2-D drone-vs-wind sandbox (drag the target, switch controllers) |
| `python -m aimct live3d` | 6-DOF sandbox — PyVista if installed, else matplotlib-3D; `--web` serves the WebGL build |
| `python -m aimct live3d --headless` | physics smoke-check, no GUI |

`compare` presets exist for `mass_spring_damper`, `pendulum`, `cartpole`,
`quadrotor`. Every other system is library-only (for now).

---

## 2. The five axes

A study is **system × controller × trajectory × disturbance × parameters**.
Vary any one, hold the rest.

### System — `aimct.systems`

| system | state | input | notes |
| :-- | :-- | :-- | :-- |
| `MassSpringDamper()` | `[x, v]` | `[F]` | the "hello world" plant |
| `Pendulum(m, L, b, g)` | `[θ, ω]` | `[τ]` | `θ = 0` down, `θ = π` up |
| `CartPole(mc, mp, l, g)` | `[x, ẋ, θ, θ̇]` | `[F]` | `θ = 0` up |
| `DCMotor(R, L, Kt, Ke, J, b, v_max)` | `[θ, ω, i]` | `[V]` | `.reduced()` → 2-state textbook model |
| `PlanarQuadrotor(m, Iyy, arm, g, thrust_max)` | `[x, z, θ, ẋ, ż, θ̇]` | `[T₁, T₂]` | `.u_hover`, `.linearize()` about hover |
| `Quadrotor3D(...)` | 12-state | `[T, τx, τy, τz]` | full 6-DOF |
| `DifferentialDriveRobot(wheel_radius, wheel_base, tau_v, tau_omega, v_max, omega_max)` | `[x, y, θ, v, ω]` | `[v_cmd, ω_cmd]` | first-order actuator lag; `.wheel_speeds()` |
| `TwoLinkArm(m1, l1, lc1, I1, m2, …, b, payload, tau_max)` | `[q1, q2, q̇1, q̇2]` | `[τ1, τ2]` | public `M(q) / C(q,dq) / G(q)`, settable `.payload` |

Every constructor argument is a real physical parameter with a documented
default. Building your own plant: subclass `DynamicalSystem`, set
`n_states` / `n_inputs`, implement `dynamics(t, x, u) -> xdot` — that is the
whole contract (numeric `linearize()` comes for free). See
[`pendulum.py`](../src/aimct/systems/pendulum.py) and
[`twolink_arm.py`](../src/aimct/systems/twolink_arm.py). While you're writing
one, `python -m aimct.dev mymodule.py:MyPlant --watch` gives a live pole
map / controllability / Jacobian-residual / response-trace dashboard that
rebuilds on every save — see [`docs/DEV_PREVIEW.md`](DEV_PREVIEW.md).

### Controller — `aimct.controllers`

| controller | build it with |
| :-- | :-- |
| `PID(kp, ki, kd, setpoint=…, output_limits=…)` | scalar/vector gains, anti-windup |
| `LQR(A, B, Q, R, x_ref=…, u_ref=…)` | pass `*system.linearize()` |
| `StateFeedback(K, x_ref=…)` / `StateFeedback.from_poles(A, B, poles)` | any gain |
| `LinearMPC(A, B, Q, R, N=…, u_bounds=…)` | condensed active-set QP; `x_ref` / `u_ref` accept a scalar, `(N,·)` array, or `callable(t)` |
| `SamplingMPC(step, running_cost, …)` | CEM / MPPI, model-agnostic |
| `ILQR(step, cost, N)` / RTI-NMPC wrapper | gradient nonlinear MPC |
| `ObserverFeedback(observer, gain)` | any `aimct.estimation` filter + state feedback |
| `EnergyShapingSwingUp` / `HybridSwingUpLQR` | pendulum / cart-pole swing-up |
| `ShieldedController(inner, shield)` | wrap any policy in a safety filter |
| `MRAC(A_m, B_m, B, …)` / `GainScheduledLQR(…)` | adaptive |

A controller is anything with `update(measurement, dt) -> u` (and optionally
`reset()`); a bare `lambda y, dt: ...` works too.

### Trajectory — `aimct.trajectories`

`Setpoint(p)`, `Circle(radius, period)`, `Lemniscate(A, B, period)`,
`MinimumJerk(p0, p1, T)`, `Spline(waypoints)`, `Dubins(...)`. Each is
`traj(t) -> (pos, vel, acc)`; path-likes also expose `.length` and
`.closest(p)`. Used by `track_trajectory` and the tracking experiments.

### Disturbance — `aimct.benchmarks` + `simulate`

- `simulate(..., input_disturbance=lambda t: d)` — additive plant-input push (not counted in recorded `u`)
- `perturbed_system(factory, seed, frac=0.30)` — ±30 % random parameter draw
- `ActuatorLag(base, tau_a)` — first-order actuator dynamics on top of a plant
- `ImpulseInjector(seed, b_scale=, rate_hz=, ...)` — random shove train, returns `d(t)`
- `measurement_fn=lambda t, x, u: x[[0]] + noise` — sensor model / partial output

### Parameters

Anything above: constructor kwargs, `Q`/`R` weights, horizon `N`, `dt`,
`t_final`, `u_bounds`, gains, adaptation rates. `aimct.benchmarks.sweep` runs a
whole grid in one call (see recipe 3).

---

## 3. Recipes

### Simulate one controller

```python
import numpy as np
from aimct.systems import CartPole
from aimct.controllers import LQR
from aimct.simulate import simulate

sys = CartPole()
K = LQR(*sys.linearize(), np.diag([10, 1, 100, 10]), np.array([[0.1]]))
traj = simulate(sys, K, x0=[0, 0, 0.2, 0], dt=0.01, t_final=5.0,
                u_bounds=(-20, 20))
print(traj.x[-1], "diverged" if traj.diverged else "ok")   # traj.t, traj.x, traj.u, traj.y
```

### Compare several, scored on identical conditions

```python
from aimct.systems import Pendulum
from aimct.controllers import PID, LQR
from aimct.benchmarks import compare
import numpy as np

sys = Pendulum()
A, B = sys.linearize()                       # about upright (θ = π)
res = compare(
    sys,
    {"PID": PID(40, 8, 6, setpoint=np.pi, output_limits=(-8, 8)),
     "LQR": LQR(A, B, np.diag([10, 1]), np.array([[0.5]]), x_ref=[np.pi, 0])},
    x0=[np.pi - 0.3, 0.0], dt=0.01, t_final=6.0, reference=np.pi,
    u_bounds=(-8, 8), output_index=0,
    # PID is a single-channel loop: give it just the angle, not the full state
    measurement_fns={"PID": lambda t, x, u: x[[0]]},
)
print(res.to_markdown())
res.save("out/pendulum")                     # table.md + table.csv + figure.png
```

Controllers are handed `system.output` (full state for the built-in plants) by
default; `measurement_fns` overrides that per controller — the hook for
single-loop PID, output feedback, or a noisy sensor model.

### Sweep a parameter

```python
from aimct.benchmarks import sweep

def case(noise):
    sys = Pendulum(b=0.1)
    A, B = sys.linearize()
    return dict(system=sys,
                controllers={"LQR": LQR(A, B, np.diag([10, 1]), [[0.5]], x_ref=[np.pi, 0])},
                x0=[np.pi - 0.3, 0], dt=0.01, t_final=6.0, reference=np.pi,
                measurement_fns={"LQR": lambda t, x, u, s=noise: x + s*np.random.randn(2)})

sw = sweep([0.0, 0.01, 0.05, 0.1], case, param_name="sensor_sigma")
print(sw.table("rmse"))          # or "settling_time", "control_energy", …
sw.save("out/noise_sweep")       # per-metric tables + CSV + a trend figure
```

### Track a moving reference

```python
from aimct.trajectories import Lemniscate
from aimct.benchmarks import track_trajectory
# controllers here must already be configured to *follow* the trajectory
# (hold their own clock) — see experiments 14 / 22 / 23 for the pattern.
res = track_trajectory(quad, {"LQR+ff": ctrl}, Lemniscate(0.6, 0.35, 6.0),
                       x0, dt=0.01, t_final=12, pos_index=(0, 1))
res.save("out/fig8")
```

### Estimate state (output feedback)

```python
from aimct.estimation import ExtendedKalmanFilter
from aimct.controllers import ObserverFeedback
ekf = ExtendedKalmanFilter(f, h, Q, R, dt=0.02)
ctrl = ObserverFeedback(ekf, K)                # K from an LQR on the same model
traj = simulate(sys, ctrl, x0, dt=0.02, t_final=8,
                measurement_fn=lambda t, x, u: h(x) + noise)
```

### Blind benchmark (Intelligent Control Challenge)

```python
from aimct.benchmarks.challenge import Challenge
ch = Challenge("track1-msd")          # controller sees only spec dims, never A/B
result = ch.evaluate(my_controller_factory, seed=0)
print(result.report())               # 5 metric dims + composite + PASS/FAIL/DQ
```

---

## 4. Simulate and *see*

- **`python -m aimct live [drone|drone3d|arm|diffdrive]`** — an interactive
  sandbox per system: drag the target, switch controllers, toggle a
  disturbance, watch the telemetry HUD. `live3d` is an alias for `live drone3d`.
- **`aimct.viz.animate(traj, system)`** — replay *any* `simulate()` or
  `track_trajectory()` run as an animation (reference drawn alongside, a
  breadcrumb trail, the same telemetry HUD); `.save("run.mp4" | "run.gif")`.
  `TrackingResult.animate(system, controller=...)` is a one-line shortcut.
  Works in a notebook via `_repr_html_`.
- Every `experiments/NN_*/run.py` writes a publication-ready figure next to
  itself; heavy ones are gated by `AIMCT_EXP_FULL=1`.
- `res.figure()` / `res.save(dir)` on any `ComparisonResult` / `SweepResult` /
  `TrackingResult`.
- `notebooks/01_tour.ipynb` — build → design → simulate → plot → study, top to
  bottom in < 30 s.

```python
from aimct.viz import animate
from aimct.simulate import simulate

traj = simulate(sys, controller, x0, dt=0.01, t_final=6.0)
animate(traj, sys).save("run.gif")     # or just `animate(traj, sys)` in a notebook
```

See [`docs/VISUALIZATION.md`](VISUALIZATION.md) for the `SystemArtist`
contract — the one class you add to give a new system its own replay and
sandbox.

---

## 5. The experiments as worked examples

The 24 studies in [`experiments/`](../experiments/) are the canonical usage
reference — each is one `run.py` with a `config.yaml`, a table, a figure and a
`README.md` stating the question and the verdict. Pick the one closest to your
problem from the matrix in the top-level [`README`](../README.md) /
[`docs/RESULTS.md`](RESULTS.md) and start from its `run.py`.
