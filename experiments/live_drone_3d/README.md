# 3-D live drone-vs-wind sandbox

The 6-DOF counterpart of [`experiments/live_drone`](../live_drone/). A real-time
`aimct.systems.Quadrotor3D` (full 12-state Crazyflie 2.0) holds a hover point
while **you** drive a 3-D wind vector; switch controllers on the fly and watch
which droop, which recover, and which reject a steady wind outright.

```bash
python experiments/live_drone_3d/sim3d.py        # or:  python -m aimct live3d
python -m aimct live3d --headless                # GUI-less physics smoke check
```

## Controls (matplotlib-3D renderer)

| input | effect |
| --- | --- |
| **wind x / y / z** sliders | steady wind force [N] |
| **← → ↑ ↓ / PgUp PgDn** | gust in ±x, ±y, ±z |
| **gust** button | random gust |
| **radio buttons** | switch controller live |
| **reset** / **R** | recentre the drone |

## Controllers

Same three as the 2-D sandbox, now regulating all six degrees of freedom:

| name | headless steady-state error under a `(0.03, −0.02, 0.015) N` wind |
| --- | --- |
| LQR (stiff) | 111 mm |
| **LQR + integral (wind-adaptive)** | **2 mm** |
| LQR (soft) | 234 mm |

The integral-augmented LQR nulls the steady wind in 3-D exactly as in the plane —
a constant disturbance needs an integrator, no amount of state-feedback tuning
replaces it.

## Rendering

The shipped renderer is a functional-but-plain matplotlib 3-D view. A richer one
(a real drone mesh, WebGL, pyvista) plugs into the same physics via the `Frame`
contract — see [`RENDERER_SPEC.md`](RENDERER_SPEC.md).
