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

1. Branch from `main`.
2. `pytest` must pass.
3. One logical change per PR; describe the question it answers or the capability it adds.
