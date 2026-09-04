# 3-D live sandbox — renderer spec

`sim3d.py` is a **renderer-agnostic** real-time physics + control engine for
`aimct.systems.Quadrotor3D` (Crazyflie 2.0). A minimal matplotlib-3D renderer
ships in the file so it runs standalone; this document is for building a richer
one — a real drone mesh, WebGL, a game engine — **without touching the physics**.

## What the engine gives you

```python
from experiments.live_drone_3d.sim3d import Engine, HOVER, GUST

eng = Engine()                       # optional: Engine("LQR (soft)")
eng.controllers                      # dict name -> control law  (3 entries)
frame = eng.step_frame()             # advance ~16 ms of physics, return a Frame
```

`step_frame()` runs `SUBSTEPS` (8) RK4 steps of `PHYS_DT` (2 ms) and returns a
`Frame` (a `NamedTuple`):

| field | shape | meaning |
| --- | --- | --- |
| `pos` | `(3,)` | world position `[x, y, z]` m, ENU, z up |
| `R` | `(3, 3)` | body → world rotation matrix (orthonormal) |
| `rotors` | `(4,)` | per-rotor thrust (N) — front-right, back-right, back-left, front-left; use for prop-blur / arrow length |
| `wind` | `(3,)` | current wind force vector (N) |
| `trail` | `(N, 3)` | recent positions (≤ 240) for a path ribbon |
| `hud` | `str` | 5 pre-formatted monospace lines (controller, wind, pos error, attitude, status) |
| `tumbling` | `bool` | drone has lost control — prompt a reset |

Target frame rate ~60 Hz. `Engine.reset()` recentres; the engine is pure
NumPy, no global state.

## What you drive into the engine

| control | call |
| --- | --- |
| steady wind | `eng.steady_wind = np.array([wx, wy, wz])`  (± ~0.08 N is plenty) |
| a gust | `eng.add_gust([gx, gy, gz])`  → ~0.3 s impulse, `GUST ≈ 0.06 N` |
| switch controller | `eng.set_controller(name)`  (`name` from `eng.controllers`) |
| reset | `eng.reset()` |

Suggested bindings: sliders for `wx/wy/wz`, arrow keys + PgUp/PgDn for gusts,
a radio/menu for the controller, `R` to reset. Mouse drag on the scene → a gust
toward the drag is a nice touch (the 2-D sandbox does this).

## Recommended approaches (pick one)

1. **Web / WebGL artifact (best for sharing).** A single self-contained HTML page
   with Three.js. Port `Quadrotor3D.dynamics` + the three LQR gains to JS
   (they're ~40 lines each — the A/B/K matrices can be pre-computed here and
   pasted in as constants), run the same 2 ms RK4 loop in `requestAnimationFrame`.
   Load a **CC0 / CC-BY quadrotor `.glb`** (e.g. from Poly Haven, Sketchfab CC0,
   Quaternius, or Kenney) for the drone body; four spinning discs for props.
   Deliver as an Artifact.
2. **Local, high quality: `pyvista`** (`pip install pyvista`, VTK-backed).
   `pyvista.Plotter(off_screen=False)` with `add_mesh` for a loaded `.obj`/`.stl`
   drone, `plotter.add_callback` on a timer calling `eng.step_frame()` and
   updating the actor's `user_matrix` from a 4×4 built from `frame.pos` +
   `frame.R`. Smooth, real 3-D lighting, ~60 Hz easily.
3. **Local, no new deps: `vispy`** (already scientific-Python friendly) or keep
   improving the shipped matplotlib-3D renderer (works now, just not pretty).

## Asset requirements

- A quadrotor model: `.glb`/`.gltf` for the web path, `.obj`+`.mtl` or `.stl`
  for pyvista. **X-configuration**, body ~10 cm, licence **CC0 or CC-BY**
  (record the source + licence in this folder's `README`). Nose along `+x`.
- Optional: a ground plane / grid, a skybox, four translucent prop discs.
- Keep total committed asset size < ~5 MB; if bigger, add a `fetch_assets.py`
  and gitignore the binaries (as `docs/papers/` does).

## Do not change

`sim3d.py` physics (`Engine`, `Quadrotor3D`, the RK4 loop, the controllers) and
the `Frame` contract. If the renderer needs something more from a frame, add a
field to `Frame` and populate it in `step_frame` — don't fork the loop.
