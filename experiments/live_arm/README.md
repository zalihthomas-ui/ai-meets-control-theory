# Live 2-link-arm sandbox — computed torque vs. an unknown payload

The interactive counterpart of [Experiment 23B](../23_twolink_arm_tracking/).
A planar `aimct.systems.TwoLinkArm` holds its wrist on a target while **you**
hang an unknown payload on it with the slider. Switch controllers on the fly.

```bash
python -m aimct live arm            # or:  python experiments/live_arm/run.py
python -m aimct live arm3d          # the same sandbox as a real 3-D PyVista scene
python -m aimct live arm --headless # physics + controller smoke check, no GUI
```

The 3-D view (`arm3d`, `pv_arm.py`) is the *same* physics and controllers —
`aimct.viz.pv_arm.run_pyvista_arm` reads the sandbox's own slider/hotkey
declarations, so both views share one control panel definition. It renders
real cylinders/spheres with lighting and an orbit camera; the arm still moves
in one plane, which spinning the camera edge-on makes plain.

## Controls

Sliders, hotkeys, controller keys, `r` reset, and the `h`/`g`/`c` additives
(help overlay / surprise-me / snapshot — see
[docs/VISUALIZATION.md](../../docs/VISUALIZATION.md)) all work the same in
both views. The one exception: **mouse-click-to-place-target** is 2-D-only —
PyVista's click coordinates aren't wired to the sandbox in `arm3d`, so use the
arrow keys there instead.

| input | action |
| :--- | :--- |
| **payload slider** | wrist payload 0 → 0.6 kg — the disturbance none of the controllers' models include |
| **mouse click** | move the hold target (clamped inside the reach) |
| **arrow keys** | nudge the target |
| **p** | poke the arm (velocity impulse) |
| **1 / 2 / 3** | PD + gravity comp / Computed torque (0 kg model) / Adaptive computed torque |
| **r** | reset |

## The three controllers

All three know only the **nominal (0 kg)** arm model.

| controller | with no payload | when you hang 0.4 kg on the wrist |
| :--- | :--- | :--- |
| **PD + gravity comp** | holds, high feedback gain | a **bounded standing droop** (~15 mm) — it never models the load, it just fights it |
| **Computed torque (0 kg model)** | perfect — it inverts the exact dynamics | **worst** (~55 mm) — it now confidently inverts a model that is wrong |
| **Adaptive computed torque** (Slotine–Li, σ-modified, projection-bounded) | holds, `m̂ ≈ 0` | droops briefly, then `m̂ → ~0.4 kg` and the error returns to **< 1 mm** |

Headless numbers (payload stepped on at t = 2.4 s):

```
PD + gravity comp              droop peak  13.7 mm   settled  13.7 mm
Computed torque (0 kg model)   droop peak  60.9 mm   settled  54.1 mm
Adaptive computed torque       droop peak   5.6 mm   settled   0.3 mm   m̂ -> 0.39 kg (true 0.40)
```

## Honest notes

* The adaptive law here identifies the payload cleanly because it acts against a
  **step** disturbance while holding a point — the estimator sees a persistent,
  informative error. Set `RADIUS > 0` in `run.py` to make the target circle
  instead: the arm then tracks a moving reference, tracking *lag* starts to leak
  into `m̂`, and the projection bound (0.6 kg = the arm's rated max payload) and
  the σ-modification leakage term are what keep it well-behaved. That trade-off
  — rich motion identifies, but also confounds — is the Experiment 23B lesson.
* Gains are ~⅓ of Experiment 23B's: this is a small teaching arm with only
  ±15 / ±10 N·m of joint torque, and a hand-driven target asks more of the
  controller than a pre-planned spline. They are set just below saturation on
  the default hold so the comparison is about *modelling*, not who clips first.
