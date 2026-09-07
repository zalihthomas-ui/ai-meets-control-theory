# Contributing to AIMCT

Thank you for your interest in contributing to **AIMCT (`ai-meets-control-theory`)**!

AIMCT is an open-source research and educational platform unifying classical control theory, robust synthesis, state estimation, and data-driven reinforcement learning.

---

## 1. Guiding Principles

Priority order for every contribution:
$$\textbf{Clarity} \longrightarrow \textbf{Correctness} \longrightarrow \textbf{Reproducibility} \longrightarrow \textbf{Performance}$$

- **Self-Contained Implementations:** Implement fundamental algorithms from first principles (`numpy`, `scipy`) first; use external reference packages (`python-control`, `slycot`) for cross-validation tests.
- **Fair Baselines:** Every learning-based or heuristic method must be benchmarked against an appropriate **classical control baseline** (e.g. LQR, Kalman Filter, or MPC) under identical initial states, noise distributions, and disturbance bounds.
- **Transparent Failure Modes:** Document where algorithms break down (e.g., divergence envelopes, actuator saturation limits, or singular Jacobians) rather than only cherry-picked successes.
- **Strict Stability Guarantee:** Adhere to our [API Stability & Deprecation Policy](docs/STABILITY.md) (SemVer 2.0.0).

---

## 2. Developer Environment Setup

Clone the repository and install the development environment with editable extras:

```bash
# Clone the repository
git clone https://github.com/zalihthomas-ui/ai-meets-control-theory.git
cd ai-meets-control-theory

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install editable package with all development dependencies
pip install -e ".[dev,xcheck,ml,viz,doc]"
```

### Dependency Groups:
- `dev`: `pytest`, `pytest-cov`, `ruff`, `mypy`.
- `xcheck`: `control`, `slycot` (used in cross-validation test suites).
- `ml`: `torch`, `gymnasium`, `stable-baselines3` (for RL and neural ODE benchmarks).
- `viz`: `pyvista`, `matplotlib` (for 3D visualization and simulation animations).
- `doc`: `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter`.

---

## 3. Development & Testing Workflow

### Running Unit Tests
```bash
# Fast local test loop (skips long-running training loops)
pytest -m "not slow"

# Run full test suite with coverage
pytest tests/ -v

# Run targeted subsystem tests
pytest tests/test_lqr.py tests/test_kalman.py tests/test_hinf.py
```

### Linting & Formatting
We enforce strict linting and code quality standards:
```bash
# Check code style and formatting
ruff check src/ tests/
ruff format --check src/ tests/

# Type checking
mypy src/aimct
```

### Building Documentation
The documentation portal is powered by MkDocs Material:
```bash
# Run local live-reload server
mkdocs serve

# Validate strict build (must pass with 0 warnings/errors)
mkdocs build --strict
```

---

## 4. Codebase Architecture

- `src/aimct/`: Core package source code:
  - `systems/`: Dynamical plant ODEs and multi-body kinematics (`InvertedPendulum`, `CartPole`, `Quadrotor`, `TwoLinkArm`, `MultiAgentSystem`).
  - `controllers/`: State feedback, LQR, MPC, iLQR, adaptive control, $H_\infty$ loop shaping (`mixsyn`), and formation consensus.
  - `estimation/`: Luenberger, Kalman Filter, EKF, UKF, Particle Filter, and Moving-Horizon Estimator (MHE).
  - `planning/`: Trajectory optimization and Direct Collocation.
  - `robust/`: Structured singular value ($\mu$-analysis), disk margin calculations, and D-K iteration.
  - `simulate/`: 4th-order Runge-Kutta integrators and batch Monte-Carlo rollout engines.
  - `deploy/` & `hil/`: C99 zero-alloc code generation and hardware-in-the-loop emulation.
- `experiments/`: Self-contained empirical benchmarks (`01` through `41`), each containing `config.yaml`, `run.py`, `table.md`, `table.csv`, and `figure.png`.
- `docs/`: Markdown documentation portal, math references, and API documentation.
- `docs/report/`: LaTeX technical report source (`main.tex`).

---

## 5. Adding New Experiments

Every experiment in `experiments/` must follow this structure:
1. `config.yaml`: Explicit simulation parameters, noise covariances, and initial states.
2. `run.py`: Self-contained, non-interactive script (`matplotlib.use("Agg")` before importing `pyplot`).
3. `table.md` & `table.csv`: Quantitative metrics generated during execution.
4. `figure.png`: High-resolution comparison figure saved in the experiment folder and copied to `docs/experiments/figures/`.
5. `README.md`: Problem statement, math formulation, table, and takeaways.
6. Documentation page added to `docs/experiments/` and wired into `mkdocs.yml`.

---

## 6. Pull Request Process

1. Fork the repo and create a topic branch (`git checkout -b feature/my-new-controller`).
2. Implement your feature, write unit tests in `tests/`, and document docstrings with mathematical equations.
3. Ensure all tests and linting pass (`pytest`, `ruff check`, `mypy`, `mkdocs build --strict`).
4. Commit your changes with conventional commit messages (`feat:`, `fix:`, `docs:`, `perf:`).
5. Open a Pull Request following the `.github/pull_request_template.md`.
