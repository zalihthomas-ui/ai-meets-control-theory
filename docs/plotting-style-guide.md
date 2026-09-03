# Plotting & Visualization Style Guide

> **AI Meets Control Theory — Visual Standard**  
> Version 1.0 | Maintainer: docs & design (`lava`)

---

## 1. Visual Philosophy

Control engineering data must communicate **physical truth with zero ambiguity**. Every plot produced across the repository (benchmarks, module examples, experimental reports, publication assets) must adhere to these four core tenets:

1. **Information Density with Absolute Clarity:** Never sacrifice readability for decoration. Clean gridlines, clear axis ticks, and legible legends.
2. **Explicit Physical Units:** Every axis, colorbar, and table column **must** include its physical dimension (e.g., $t\text{ [s]}$, $\theta\text{ [rad]}$, $u\text{ [N}\cdot\text{m]}$, $\omega\text{ [rad/s]}$).
3. **Color-Blind Accessibility:** Never rely solely on red/green contrast. Use colorblind-safe palettes with distinct line styles (solid, dashed, dash-dot, dotted) and markers for multi-controller comparisons.
4. **Reproducibility & Publication Quality:** Figures must look sharp both on high-DPI screens (300 DPI PNG / SVG) and in print / PDF papers.

---

## 2. Color Palette & Line Styles

### 2.1 The Standard Palette (Okabe-Ito Color-Universal Design)

We standardize on the accessible **Okabe-Ito** palette supplemented with high-contrast neutrals.

| Name | Hex Code | Swatch / Role | Recommended Controller Association |
| :--- | :--- | :--- | :--- |
| **Reference / Target** | `#555555` | Neutral Dark Grey | Setpoints $r(t)$, target trajectories, equilibrium lines (dashed) |
| **PID (Classical)** | `#0072B2` | Deep Blue | Standard PID, cascaded PID |
| **State Feedback / Pole Placement** | `#009E73` | Bluish Green | Full-state feedback, Luenberger observer |
| **LQR (Optimal)** | `#D55E00` | Vermilion / Rust Orange | Continuous / Discrete LQR |
| **MPC (Constrained Optimal)** | `#CC79A7` | Reddish Purple | Linear / Nonlinear MPC |
| **RL / Neural (AI)** | `#E69F00` | Amber / Gold | PPO, SAC, DQN, learned policies |
| **Hybrid (AI + Control)** | `#56B4E9` | Sky Blue | Residual RL + LQR, MPC + Neural Dynamics |
| **Limits / Saturation** | `#D62728` | Crimson / Warning Red | Actuator saturation boundaries (dotted/dashed) |

### 2.2 Line Styles & Hierarchy

When plotting multiple controllers on the same axes:

```python
CONTROLLER_STYLES = {
    "reference": {"color": "#555555", "linestyle": "--", "linewidth": 1.5, "label": "Reference $r(t)$", "zorder": 1},
    "pid": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.0, "label": "PID", "zorder": 4},
    "state_feedback": {"color": "#009E73", "linestyle": "-.", "linewidth": 2.0, "label": "State Feedback", "zorder": 3},
    "lqr": {"color": "#D55E00", "linestyle": "-", "linewidth": 2.2, "label": "LQR", "zorder": 5},
    "mpc": {"color": "#CC79A7", "linestyle": "--", "linewidth": 2.2, "label": "MPC", "zorder": 6},
    "rl": {"color": "#E69F00", "linestyle": "-", "linewidth": 1.8, "label": "RL (PPO)", "zorder": 2},
    "hybrid": {"color": "#56B4E9", "linestyle": ":", "linewidth": 2.2, "label": "Hybrid (LQR + RL)", "zorder": 7},
}
```

- **Reference Signal:** Always thin dashed (`--`), dark grey (`#555555`), drawn behind controllers (`zorder=1`).
- **Actuator Limits:** Dotted or dashed red lines (`#D62728`) at $u_{\max}$ and $u_{\min}$. If the region beyond is prohibited, use an optional light red fill (`alpha=0.08`).
- **Settling Envelope:** Light grey shaded band (`#000000`, `alpha=0.06`) indicating the $\pm 2\%$ tolerance band around the setpoint.

---

## 3. Typography & Formatting

### 3.1 Font Families

- **Primary Font:** Sans-serif clean typeface (`DejaVu Sans`, `Helvetica`, `Arial`, or `system-ui`).
- **Math Engine:** Matplotlib `mathtext` (default) or LaTeX when available.
- **Font Sizes:**
  - Figure Title: `14 pt` (bold)
  - Subplot Titles: `11 pt` (medium/bold)
  - Axis Labels: `10 pt` (medium)
  - Tick Labels: `9 pt` (regular)
  - Legends: `9 pt` (regular)
  - Inset / Annotations: `8 pt` (regular)

### 3.2 Labeling Conventions

- **Time Axis:** Always labeled as `Time $t$ [s]` or `Time [s]` (never just `t` or `Time`).
- **State Axes:** State symbol followed by standard SI units, e.g.:
  - Position: `Position $x$ [m]`
  - Velocity: `Velocity $\dot{x}$ [m/s]`
  - Angle: `Angle $\theta$ [rad]` (or `[deg]` if explicitly converted, with note in caption)
  - Angular Velocity: `Angular Velocity $\dot{\theta}$ [rad/s]`
  - Control Action: `Control Input $u(t)$ [N]` or `Torque $\tau$ [N$\cdot$m]`
  - Error: `Tracking Error $e(t) = r(t) - y(t)$ [m]`

---

## 4. Standard Figure Dimensions & Formats

| Use Case | Dimensions ($W \times H$) | DPI | Formats | Typical Content |
| :--- | :--- | :--- | :--- | :--- |
| **Benchmark Standard (4-Panel)** | `12.0 x 8.0 in` | 300 | `.svg`, `.png` | State tracking, Control action, Error, Phase portrait |
| **Single Column / Summary** | `6.5 x 4.0 in` | 300 | `.svg`, `.png` | Step response or Bode plot |
| **Phase Portrait (Square)** | `5.5 x 5.0 in` | 300 | `.svg`, `.png` | State-space trajectory ($x_1$ vs $x_2$) with vector field |
| **Wide Comparison (2x3 Grid)** | `15.0 x 9.0 in` | 300 | `.svg`, `.png` | Multi-state systems (e.g., Cart-Pole $[x, \dot{x}, \theta, \dot{\theta}, u]$) |

---

## 5. Standard Multi-Panel Layouts

### 5.1 The Canonical 4-Panel Benchmark Figure

Every benchmark comparing $N$ controllers on a dynamic system should generate the standard 4-panel figure:

```
+------------------------------------+------------------------------------+
|  (a) Output Tracking y(t) vs r(t)  |  (b) Control Effort u(t)           |
|      - Reference setpoint (dashed) |      - Actuator bounds (red dash)  |
|      - +/- 2% settling band (fill) |      - Saturation highlighted      |
|      - Controller traces           |      - Integrated energy in legend |
+------------------------------------+------------------------------------+
|  (c) Tracking Error e(t) = r - y   |  (d) Phase Portrait (x vs xdot)    |
|      - Zero-error baseline (dash)  |      - Vector field streamlines    |
|      - Transient peaks             |      - Initial point & equilibrium |
|      - Steady-state convergence    |      - Limit cycles or attractors  |
+------------------------------------+------------------------------------+
```

### 5.2 Multi-State Systems (e.g., Cart-Pole)

For higher-order systems ($n \ge 4$), layout panels by physical sub-systems:
- **Row 1:** Cart position $x(t)\text{ [m]}$ and Cart velocity $\dot{x}(t)\text{ [m/s]}$.
- **Row 2:** Pole angle $\theta(t)\text{ [rad]}$ and Pole angular velocity $\dot{\theta}(t)\text{ [rad/s]}$.
- **Row 3:** Control Force $u(t)\text{ [N]}$ with limits $u \in [-u_{\max}, u_{\max}]$ and Phase portrait $(\theta, \dot{\theta})$.

---

## 6. Python Implementation (`src/aimct/plot_style.py`)

A reusable styling helper is integrated into `aimct` to apply these standards in a single call:

```python
import matplotlib.pyplot as plt
from aimct.plot_style import set_aimct_style, get_controller_color

# Apply global styling
set_aimct_style()

# Get standardized colors and line configurations
color_lqr = get_controller_color("lqr")
```

See [`src/aimct/plot_style.py`](../src/aimct/plot_style.py) for the complete implementation.

---

## 7. Quality Checklist Before Exporting Figures

- [ ] Are all axes labeled with variable symbol and physical units in brackets $[ \cdot ]$?
- [ ] Is the setpoint/reference signal clearly distinguishable (dashed neutral grey)?
- [ ] Are actuator saturation limits visibly marked if constraints exist?
- [ ] Is the legend placed without obscuring dynamic transients?
- [ ] Is the font size legible when viewed at 100% zoom on a standard screen?
- [ ] Has the figure been saved in both `.png` (for web/preview) and `.svg` (for vector lossless scaling)?
- [ ] Is the background clean white (`#FFFFFF`) with subtle gridlines (`alpha=0.3`)?
