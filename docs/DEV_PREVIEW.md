# Design-time preview — `aimct.dev`

## The idea

Authoring a new `DynamicalSystem` today is: edit → hand-write a scratch
`simulate()` script → plot → eyeball → repeat. `aimct.dev` shortens that loop
to one call, and gives fast feedback on the three things that go wrong early
and are otherwise invisible until an LQR blows up much later:

1. a **sign error** in `dynamics()` — shows up as an unexpectedly unstable or
   divergent response,
2. a **wrong analytic** `linearize()` — caught by comparing it against the
   base class's numeric central-difference Jacobian,
3. a model that is **not controllable** (or not observable) about the chosen
   operating point.

This is deliberately **separate from [`aimct.viz`](VISUALIZATION.md)**: viz is
the *runtime* story (replay and interactive sandboxes for a finished system's
users); `aimct.dev` is an *authoring-time* tool for whoever is writing the
system. It may reuse `aimct.viz.animate` / `SystemArtist` for a replay panel
in a future version; it has no dependency the other way.

## Use it

```python
from aimct.dev import build_report, render

report = build_report(MyPlant())            # or build_report(MyPlant(), x_eq=[...], u_eq=[...])
print(report.summary())
render(report).savefig("design_preview.png")
```

Or watch a file while you edit it — a preview image is rewritten on every
save, so point an editor's image-preview pane (or just reopen it) at the file:

```bash
python -m aimct preview mymodule.py:MyPlant --watch
python -m aimct preview aimct.systems.pendulum:Pendulum      # one-shot, an installed system too
```

`target` is `"module:ClassName"` for an importable module or
`"path/to/file.py:ClassName"` for a system not yet packaged — either way a
fresh instance is built (extra CLI args pass through as constructor kwargs via
the Python API's `system_kwargs=`) and, for a file target, the source is
re-read and re-executed on every rebuild (never served from a stale
`__pycache__`), so the preview always reflects the latest save.

Also callable as `python -m aimct.dev ...` directly (same CLI, one module down)
if you don't want to go through the top-level entry point.

## What the dashboard shows

- **Pole map** of `linearize()` about the chosen operating point (default: the
  zero state and zero input — pass `x_eq=`/`u_eq=` for a system whose
  interesting equilibrium is elsewhere, e.g. `x_eq=[np.pi, 0]` for a
  pendulum's upright).
- **Controllability / observability** at that point (`aimct.controllers` /
  `aimct.estimation`; observability uses a numeric Jacobian of `output()`).
- **Analytic-vs-numeric Jacobian residual** — only computed when `linearize()`
  is overridden (skipped, with a note, when a system relies on the inherited
  numeric fallback, since there is then nothing independent to check it
  against).
- **Four response traces** — free, step, impulse, sinusoid — each state
  component plotted, with divergent runs called out in red.
- A **warnings** list surfacing exactly the three failure modes above in
  plain language.

## Design commitments

Same discipline as the rest of the repo: honest before pretty (a divergent
response is drawn, not hidden), one small contract (`DesignReport` is pure
data — computation is fully separate from rendering, so it is cheap to unit
test), reproducible (deterministic given the same system and operating
point), headless-safe (`matplotlib.use("Agg")`).
