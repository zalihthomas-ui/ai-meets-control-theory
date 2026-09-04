# 3-D Live Drone-vs-Wind Sandbox

The 6-DOF counterpart of [`experiments/live_drone`](../live_drone/). A real-time
`aimct.systems.Quadrotor3D` (full 12-state Crazyflie 2.0) holds a hover point
while **you** drive a 3-D wind vector; switch controllers on the fly and watch
which droop, which recover, and which reject a steady wind outright.

🚁 **[Live Interactive 3D WebGL Sandbox (Claude Artifact)](https://claude.ai/code/artifact/69b12b78-d7b2-4732-a7af-2af14930139b)**

---

## 🎮 Launching the Visualizer

### Option 1: WebGL / Three.js 3-D Renderer (Interactive Web App)
```bash
# Launch the rich 3-D WebGL visualizer in your browser
python -m aimct live3d --web
# or open the live Claude artifact: https://claude.ai/code/artifact/69b12b78-d7b2-4732-a7af-2af14930139b
# or run local server: python experiments/live_drone_3d/web.py
```

### Option 2: Standalone Matplotlib 3-D Renderer
```bash
python -m aimct live3d
```

### Option 3: Headless Physics Verification
```bash
python -m aimct live3d --headless
```

---

## Controls & Hotkeys

| Input | Action |
| :--- | :--- |
| **Wind X / Y / Z Sliders** | Continuous steady wind force [N] ($\pm 0.08\,\text{N}$) |
| **Arrow Keys (← → ↑ ↓)** | Impulse gusts in $\pm X$ (East/West) and $\pm Y$ (North/South) |
| **PgUp / PgDn** or **W / S** | Vertical impulse gusts in $\pm Z$ (Up/Down) |
| **Right-Click Drag / Shift-Drag** | Direct mouse-drag gust on the 3-D canvas towards drag direction |
| **Spacebar** | Instantly zero out all wind components |
| **R** | Reset drone to hover with initial tilt perturbation |
| **1 / 2 / 3** | Quick switch between LQR Stiff, LQI Adaptive, and LQR Soft |

---

## Controllers & The 3-Way Story

Same three controllers as the 2-D planar sandbox, now regulating all six degrees of freedom:

| Controller | Headless Steady-State Error Under $(0.03, -0.02, 0.015)\,\text{N}$ Wind | Key Physical Mechanism |
| :--- | :--- | :--- |
| **LQR (stiff)** | $111\,\text{mm}$ droop | High-gain proportional feedback resists transients quickly, but static state feedback lacks an integrator to eliminate steady wind forces. |
| **LQR + integral (wind-adaptive)** | **$2\,\text{mm}$ (Zero Droop)** | Integral augmentation on $(x, y, z)$ position errors tilts the rotor disc into the oncoming wind vector, perfectly nulling the disturbance. |
| **LQR (soft)** | $234\,\text{mm}$ droop | Relaxed control weights consume minimal actuator effort but allow substantial position drift under persistent wind. |

---

## Renderer Architecture & Assets

1. **WebGL / Three.js Renderer (`index.html`):**
   - **Physics:** 12-state nonlinear aerodynamics with fixed-step 4th-order Runge--Kutta integration ($h = 2\,\text{ms}$, 8 substeps per frame, 60 FPS target).
   - **Model:** Procedural PBR carbon-fiber cross frame ($d = \text{arm}/\sqrt{2} \approx 32.5\,\text{mm}$), mainboard PCB with status LEDs, LiPo battery pack, brushed motor cans, and 4 translucent spinning propeller discs with dynamic blur speed.
   - **Visualizations:** Live per-rotor thrust vectors ($T_1, T_2, T_3, T_4$), 240-step trajectory ribbon, 3-D wind particle streamer field, target hover beacon, and real-time telemetry HUD overlay.
   - **License:** MIT (part of the AI Meets Control Theory framework).
2. **Matplotlib Renderer (`sim3d.py`):**
   - Headless and lightweight fallback using standard `matplotlib.animation`.
