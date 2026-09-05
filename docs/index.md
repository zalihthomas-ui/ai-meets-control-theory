# AI Meets Control Theory

<p align="center">
  <strong>From Classical Feedback Control to Intelligent Autonomous Systems</strong>
</p>

<p align="center">
  <a href="https://github.com/zalihthomas-ui/ai-meets-control-theory/actions/workflows/ci.yml"><img src="https://github.com/zalihthomas-ui/ai-meets-control-theory/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/aimct/"><img src="https://img.shields.io/pypi/v/aimct.svg?color=blue" alt="PyPI"></a>
  <a href="https://github.com/zalihthomas-ui/ai-meets-control-theory/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/zalihthomas-ui/ai-meets-control-theory/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.10+"></a>
  <a href="report/ai-meets-control-theory.pdf"><img src="https://img.shields.io/badge/Report-Living%20PDF-brightgreen" alt="Report: Living PDF"></a>
</p>

---

**AI Meets Control Theory (`aimct`)** is a rigorous, from-scratch experimentation framework and Python library that systematically bridges classical control theory, modern state-space methods, constrained Model Predictive Control (MPC), Kalman filtering, adaptive control, and modern machine learning/reinforcement learning on physical dynamical systems.

Under the core discipline:

> **"Derive it, build it from scratch, simulate it, visualise it, and compare it honestly."**

Every controller—from PID and LQR to active-set condensed MPC, EKF/UKF, real-time iteration iLQR, Hermite--Simpson direct collocation, Soft Actor-Critic (SAC), and supervisory safety shields—is evaluated on identical dynamical plants, sensor noise profiles, external disturbances, and actuator limits.

---

## Key Highlights

- **34 Benchmark Experiments:** Empirical side-by-side Pareto evaluations across 10 physical systems under standardized metrics.
- **472 Passing Unit Tests:** 100% custom, zero-black-box implementations of CARE solvers, active-set QPs, Kalman filters, neural MLPs, and RL actors.
- **Unified Visualisation (`aimct.viz`):** Real-time interactive physics sandboxes (`python -m aimct live`) and animation replay engines with telemetry HUDs.
- **Design-Time Preview (`aimct.dev`):** Live model inspection dashboard computing pole maps, controllability/observability, and Jacobian residual validations (`python -m aimct preview`).
- **Hardware Bridge & HIL Harness (`aimct.hil`):** Real-time loop execution, sensor quantization, delay modeling, and log-based manipulator identification.

---

## Quick Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](GETTING-STARTED.md)**
    ---
    Install `aimct`, run the test suite, and follow the 4-minute Gantry Crane anti-sway walkthrough.

-   :material-compass: **[Decision Guide](DECISION-GUIDE.md)**
    ---
    Interactive branching flowchart and 13 invariant engineering laws across 34 empirical experiments.

-   :material-chart-box: **[Master Results](RESULTS.md)**
    ---
    Complete quantitative metrics and Pareto ranking tables for all 34 benchmark experiments.

-   :material-book-open-page-variant: **[API Recipes & Usage](USAGE.md)**
    ---
    The 5-axis framework guide (*system × controller × trajectory × disturbance × parameters*) and copy-paste recipes.

-   :material-eye: **[Visualization & Sandboxes](VISUALIZATION.md)**
    ---
    Replay animation engine and 7 interactive real-time 2D/3D physics sandboxes.

-   :material-code-tags: **[API Reference](api/index.md)**
    ---
    Comprehensive auto-generated documentation for all submodules in `src/aimct/`.

</div>
