# Contributing

## Principles

Priority order for every contribution: **clarity → correctness → reproducibility → performance.**

- Implement fundamental algorithms **from scratch** first; use libraries to verify and extend.
- Every AI method ships with an appropriate **classical baseline** and a comparison.
- Report **failure modes** honestly, not just successes.

## Structure

- Reusable code lives in `src/aimct/` with unit tests in `tests/`.
- Teaching material lives in `modules/<NN-topic>/` — short theory notes (`README.md` or `notes.md`), a runnable script or notebook, and generated figures.
- Research runs live in `experiments/<name>/` — a folder containing `config.yaml`, `run.py`, a `README.md` stating the question, and committed metrics. Set and record random seeds.

## Style

- Python, PEP 8, type hints on public functions.
- `numpy` for math, `matplotlib` for plots, `python-control` for cross-checks.
- Docstrings state the equations / references being implemented.
- Keep functions small; prefer pure functions for dynamics and control laws.

## Workflow

1. `pytest` must pass (`pip install -e ".[dev,xcheck]"` for the full set, incl.
   `python-control` cross-checks). CI runs the suite on 3.10–3.12 and smoke-runs
   every `experiments/*/run.py`. For a fast local loop, skip the heavy
   RL / learned-model training tests with `pytest -m "not slow"` (a few seconds
   vs a couple of minutes); run the full `pytest` before pushing.
2. `git pull --rebase` before you push. Small commits; one logical change each.
3. `experiments/*/run.py` must be non-interactive — set `matplotlib.use("Agg")`
   before importing `pyplot`, write outputs next to the script, commit the
   generated `table.md` / `table.csv` / `figure.png`.
4. Describe the question a change answers or the capability it adds.
