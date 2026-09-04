# Visualization — `aimct.viz`

## The idea

Every dynamical system in this library implements one interface (`dynamics`,
`linearize`), which is why every benchmark runs through one harness —
`simulate`, `compare`, `track_trajectory`, the Intelligent Control Challenge.
`aimct.viz` extends that same discipline to **pictures**:

> Say once how to *draw* a system's state, and every experiment gets a replay
> animation and every system gets an interactive sandbox — for free, with one
> consistent visual language.

Concretely, one small contract — `SystemArtist` — is consumed by two front ends:

| | what it is | entry point |
| --- | --- | --- |
| **Replay** | play back a finished run (`simulate` / `track_trajectory` output) as an animation, reference drawn alongside, telemetry HUD in the system's own units | `aimct.viz.animate(traj, system)` |
| **Sandbox** | the same picture driven in real time, with sliders / hot-keys for disturbances and a controller you switch on the fly | `aimct.viz.Sandbox(system, controllers, …)` |

## Design commitments

1. **Honest before pretty.** The reference is always drawn; tracking error is
   always visible (ghost target, HUD read-out, breadcrumb trail). A diverged run
   animates as readily as a clean one — you should be able to *see* the failure,
   not just read a red "Diverged" in a table.
2. **One contract, many systems.** Adding a view for a new 2-D system is a
   single class (`bounds`, `build`, `draw`, plus `hud_lines` / `position`) and a
   `register_artist(...)` call. No per-experiment plotting code.
3. **Composable with the harness.** `animate` takes the *unmodified* output of
   `simulate` or a `TrackingResult` (`result.animate(system)` is a shortcut).
   The reference comes along automatically for a `TrackingResult`.
4. **Reproducible and shareable.** Deterministic playback, headless-safe
   (`matplotlib.use("Agg")`), `Replay.save("run.mp4" | "run.gif")` (falls back to
   GIF when ffmpeg is absent), `_repr_html_` for notebooks.
5. **One identity.** The Okabe–Ito `aimct` palette, a monospace telemetry HUD, a
   progress bar, the same framing and aspect handling across every system.

## Drawable systems

| system | artist | trail traces | notes |
| --- | --- | --- | --- |
| `Pendulum` | `PendulumArtist` | the bob | torque shown as an arc |
| `CartPole` | `CartPoleArtist` | the cart | force arrow, rail, wheels |
| `TwoLinkArm` | `TwoLinkArmArtist` | the wrist | payload drawn as a disc; target marker |
| `DifferentialDriveRobot` | `DiffDriveArtist` | the body | path + start/goal + look-ahead point |
| `PlanarQuadrotor` | `PlanarQuadrotorArtist` | the c.o.m. | per-rotor thrust arrows |

`aimct.viz.has_artist(system)` / `get_artist(system)` look one up;
`register_artist(MySystem, MyArtist)` adds your own.

## Replay — `animate`

```python
from aimct.simulate import simulate
from aimct.viz import animate

tr = simulate(cartpole, controller, x0=x0, dt=0.01, t_final=5.0)
animate(tr, cartpole, title="LQR swing-down").save("cartpole.gif")
```

```python
# straight from a trajectory-tracking benchmark — reference comes for free
res = track_trajectory(quad, {"LQR+ff": ctrl}, Lemniscate(0.5, 0.3, 8.0),
                       x0, dt=0.02, t_final=12, pos_index=(0, 1))
res.animate(quad, controller="LQR+ff").save("figure8.mp4")
```

Key arguments: `ref` (an `aimct.trajectories` object, a callable `t -> pos`, or
an array), `target` (a fixed goal marker), `path` (a polyline to trace),
`fps`, `speed`, `trail`, `hud`.

## Sandbox — real-time, interactive

```python
from aimct.viz import Sandbox, Disturbance

Sandbox(
    system, {"stiff": k_stiff, "adaptive": k_adaptive},
    x0=x0, target=goal, dt=0.02,
    disturbance=Disturbance(
        sliders=[("wind [N]", -0.1, 0.1, 0.0)],
        hotkeys=[("g", "gust", lambda s: s.kick([0, 0, 2.0, 0]))],
        xdot_extra=lambda t, x, u, knobs: np.array([0, 0, knobs["wind [N]"] / m, 0]),
    ),
).run()                       # opens the window
```

- **Controllers** switch with the number keys `1…N` or the radio buttons;
  each is `reset()` on switch.
- **`Disturbance`** carries `sliders` (continuous knobs read by `xdot_extra`),
  `hotkeys` (`fn(sandbox)` — usually `sandbox.kick(Δstate)`), `xdot_extra` (a
  steady external term added to `ẋ` each sub-step), and `on_slider`.
- **`headless(steps=…)`** runs the same loop with no GUI and returns/prints the
  settled error — every shipped sandbox uses it as a smoke test.

Shipped sandboxes (`python -m aimct live <name>`):

| name | system | the question |
| --- | --- | --- |
| `drone` | `PlanarQuadrotor` | which controller holds hover against a wind you steer? (`experiments/live_drone`) |
| `drone3d` | `Quadrotor3D` | the same, in full 6-DOF (`experiments/live_drone_3d`, PyVista) |
| `arm` | `TwoLinkArm` | hang an unknown payload — fixed computed-torque breaks, adaptive identifies it (`experiments/live_arm`) |
| `diffdrive` | `DifferentialDriveRobot` | shove the robot off its path — pure-pursuit vs. Stanley vs. path-LQR recovery (`experiments/live_diffdrive`) |

## Extending

```python
from aimct.viz.artists import SystemArtist, register_artist

class TankArtist(SystemArtist):
    label = "two-tank"
    aspect_equal = False

    def bounds(self, states=None):
        return (-0.1, 1.1), (0.0, self.system.h_max * 1.1)

    def build(self, ax):
        from matplotlib.patches import Rectangle
        self.level = Rectangle((0.1, 0), 0.3, 0, color=self.accent)
        ax.add_patch(self.level)
        return self._remember(self.level)

    def draw(self, x, u=None, t=0.0, aux=None):
        self.level.set_height(float(x[1]))          # second tank

register_artist(TwoTank, TankArtist)
```

That is the whole cost of a new view — `animate(...)` and `Sandbox(...)` work
immediately.
