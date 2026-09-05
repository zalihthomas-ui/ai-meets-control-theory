# Getting Started

This repository is three things at once — an **evidence base** (34 experiments
that each answer one "does X beat Y?" question with a measured verdict), a
**course** (a 10-module curriculum with a living technical report), and a
**library** (`pip install aimct`). This page is the map: pick the lane that
matches what you're here to do.

| I want to… | Start here |
| --- | --- |
| **decide** what controller/estimator to use for a real problem | [`docs/DECISION-GUIDE.md`](DECISION-GUIDE.md) → [`docs/RESULTS.md`](RESULTS.md) |
| **learn** the material | [the report PDF](report/ai-meets-control-theory.pdf) + [`notebooks/01_tour.ipynb`](../notebooks/01_tour.ipynb) |
| **build** something with the library | `pip install aimct` + [`examples/`](../examples/) + [`docs/USAGE.md`](USAGE.md) |
| **reproduce or extend** an experiment | `python experiments/NN_name/run.py` + `python -m aimct preview` |

---

## 1. Use it to make a decision

Every experiment is framed as a falsifiable question and ends with a verdict
and a *boundary* — where the method stops working.

- **[`docs/DECISION-GUIDE.md`](DECISION-GUIDE.md)** is the structured entry
  point:
  - **§1** — a flowchart: answer a few questions about your plant (stable?
    constrained? real-time budget? model known? disturbances matched or
    unmatched?) and it routes you to a method.
  - **§2** — a matrix by problem class → recommended method → *the experiment
    that proves it*.
  - **§3** — for each method, its measured "works here / breaks here" envelope.
  - **§4** — the invariant laws that recur across all experiments (e.g. *an
    integrator is reactive on an unmatched disturbance — it has to let the
    error build before it responds; a disturbance observer estimates the
    disturbance directly and doesn't*, Exp 34).
- **[`docs/RESULTS.md`](RESULTS.md)** is the flat lookup table: every
  experiment, its metrics, its one-line finding. Use it to check "has anyone
  measured LQR vs constrained MPC under actuator limits?" → yes, Exp 08.

Then apply the recommendation with the library (lane 3).

---

## 2. Use it to learn

- **[The living technical report](report/ai-meets-control-theory.pdf)** (~36
  pages) has a worked write-up per experiment, organised as Modules 01–10:
  foundations → classical → modern/estimation → optimal/constrained →
  system-ID/ML → RL → hybrid AI+control → robotics → the Intelligent Control
  Challenge → grand synthesis.
- **[`notebooks/01_tour.ipynb`](../notebooks/01_tour.ipynb)** is a 2-minute
  guided run: model a system from first principles, design an LQR, do a hybrid
  swing-up, compare PID/LQR/MPC — all executing live.
- The **[`docs/references/`](references/)** folder has a datasheet-grade spec
  (equations, parameters, linearisation) for every real system in the library.

---

## 3. Use it as a library

```bash
pip install aimct                 # or:  pip install -e ".[dev,ml,viz]"  from a clone
```

The 34 experiments are demonstrations of the same handful of components you'd
use on your own problem:

```python
from aimct.systems       import DynamicalSystem, CartPole, PlanarQuadrotor, ...
from aimct.controllers   import LQR, LinearMPC, ILQR, PID, MRAC, DisturbanceObserver
from aimct.estimation    import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from aimct.benchmarks    import compare, track_trajectory
from aimct.trajectories  import MinimumJerk, Lemniscate, Dubins, Spline
from aimct.viz           import animate, Sandbox
```

The **[`examples/`](../examples/)** gallery is the on-ramp — each script is
~20 lines:

| script | shows |
| --- | --- |
| `01_simulate_single_system.py` | one system + one controller through `simulate` |
| `02_compare_controllers.py` | `compare()` — PID vs LQR vs MPC on identical conditions |
| `03_track_trajectory.py` | `track_trajectory()` — a path-following benchmark |
| `04_run_challenge.py` | the Intelligent Control Challenge scoring |
| `05_replay_animation.py` | `aimct.viz.animate` — replay a run as a video/GIF |
| `06_live_sandbox_headless.py` | driving a `Sandbox` without a GUI |
| **`07_full_workflow_gantry_crane.py`** | **the whole loop on one hard problem — see below** |

[`docs/USAGE.md`](USAGE.md) covers the axes you can vary
(*system × controller × trajectory × disturbance × parameters*) with
copy-paste recipes.

---

## 4. Reproduce or extend

- **Re-run any experiment:** `python experiments/NN_name/run.py` regenerates
  its table and 4-panel figure. Some train models and take a while — those
  honour `AIMCT_EXP_FULL=1` for the committed high-resolution version.
- **Author a new system:** subclass `DynamicalSystem` (implement `dynamics`;
  optionally override `linearize` with an analytic Jacobian), then
  `python -m aimct preview yourmod:YourPlant --watch` gives you a live
  design dashboard — pole map, controllability/observability, an
  analytic-vs-numeric Jacobian check, and response traces — that re-renders
  every time you save the file. It catches the three bugs that usually bite
  early: a sign error in `dynamics`, a wrong `linearize`, and an
  uncontrollable model.
- **Add an experiment:** copy an `experiments/NN_*/` folder's structure
  (`config.yaml`, `run.py`, `README.md`, and the generated `table.*` /
  `*.png`).

---

## Worked example — the whole loop on one hard problem

**[`examples/07_full_workflow_gantry_crane.py`](../examples/07_full_workflow_gantry_crane.py)**
runs the entire pipeline end-to-end on a problem the library does **not**
ship: **gantry-crane anti-sway** — move a payload on a rigid cable 3 m and
stop it dead, under a force limit, without residual swing. (It's the opposite
of cart-pole: the pendulum is open-loop stable, but every trolley move pumps
energy into it.)

```bash
python examples/07_full_workflow_gantry_crane.py
```

It:

1. **Defines a new system** — `class GantryCrane(DynamicalSystem)` with the
   trolley-pendulum Euler-Lagrange dynamics (state `[p, θ, ṗ, θ̇]`, input `[F]`).
2. **Sanity-checks it** with `aimct.dev.build_report` — confirms the linearised
   model is controllable and observable, and reports the open-loop poles
   (a marginally-stable rigid mode + a lightly-damped 2.2 rad/s pendulum mode).
3. **Linearises** about the target and checks controllability directly.
4. **Designs four controllers** on the *same* minimum-jerk reference move:
   - **LQR** tracking the reference (feedback baseline),
   - **LQR + Zero-Vibration input shaper** — the classic crane feed-forward
     trick: pre-shape the set-point so the commanded motion doesn't excite the
     pendulum mode,
   - **Linear MPC** with a *hard* in-transit constraint `|θ| ≤ 8°`,
   - **iLQR / RTI-NMPC** on the true nonlinear dynamics.
5. **Benchmarks them honestly across three scenarios** — nominal, a mid-move
   wind gust, and a plant whose cable is 40 % longer than the model. The
   verdicts:
   - *Nominal*: all four land the payload with sway well under 8°. MPC is
     quickest, iLQR is the smoothest (lowest peak sway, lowest force), the
     input-shaper uses the least energy but leaves a small steady offset (its
     feedback gain is deliberately weak — the shaper is doing the work).
   - *Wind gust*: the external force kicks **every** controller briefly past
     the 8° cap — a hard state constraint stops the *controller* from
     violating it, not the wind. What separates them is the recovery: the
     feedback controllers damp back to zero; the shaper-based one rings for
     seconds afterward (~6.9° residual sway vs ~2–3° for the others), because
     feed-forward has nothing to push back with.
   - *+40 % cable*: near-identical to nominal for all four. The feedback loop
     closes on the *true* state error regardless of the model, so a 40 %
     parameter error barely registers — the recurring lesson that a crude
     model with feedback beats a perfect model open-loop.
6. **Draws it** — registers a `CraneArtist` (subclass `SystemArtist`, ~15
   lines) so `aimct.viz.animate` can render the trolley, cable, and payload,
   and saves a GIF of the best performer riding out the gust.

Outputs land in `examples/_out/`: `crane_table.md`, `crane_scenarios.png`,
`crane_animation.gif`. That's the shape of applying this repo to *your*
problem: define the plant, validate it, throw the method menu at it, and let
the honest comparison — including the scenarios where things break — pick the
winner.
