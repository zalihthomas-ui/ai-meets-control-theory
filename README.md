# AI Meets Control Theory

### From Classical Control to Intelligent Autonomous Systems

> A rigorous, open-source learning and experimentation framework exploring the
> intersection of control theory, machine learning, and autonomous systems.

---

## Mission

**Understand the mathematics. Build the controller. Test it against reality.
Learn where it succeeds — and where it fails.**

Rather than treating AI as a replacement for established engineering methods,
this project investigates AI as *another tool in the control engineer's toolbox*.

Through progressively challenging simulations we implement, analyze, and compare:

- Classical control
- Modern state-space control
- Optimal control & Model Predictive Control
- State estimation
- Machine learning & neural-network-based control
- Reinforcement learning
- Hybrid AI / control architectures

The objective is **engineering judgment, not technological hype** — to understand
*why* controllers work, *when* they fail, and *which* methodology fits a problem.

## The Central Question

> When should we use classical control, when should we use AI, and when should we use both?

## Core Cycle

Every major topic follows the same path:

```
THEORY → DERIVATION → IMPLEMENTATION → SIMULATION → VISUALIZATION → VALIDATION → COMPARISON → EXPERIMENT
```

Fundamental algorithms are implemented **from scratch** first; libraries are then
introduced to verify, accelerate, and extend.

## Learning Path

| Module | Topic |
| ------ | ----- |
| [01](modules/01-mathematical-foundations) | Mathematical Foundations |
| [02](modules/02-dynamic-system-modeling)  | Dynamic System Modeling |
| [03](modules/03-classical-control)        | Classical Control |
| [04](modules/04-modern-control)           | Modern Control |
| [05](modules/05-optimal-control)          | Optimal Control |
| [06](modules/06-machine-learning)         | Machine Learning for Dynamical Systems |
| [07](modules/07-reinforcement-learning)   | Reinforcement Learning |
| [08](modules/08-ai-plus-control)          | AI + Control (hybrid architectures) |

## Repository Layout

```
src/aimct/          reusable library
  systems/          dynamical-system models (mass-spring-damper, pendulum, cart-pole, ...)
  controllers/      pid, lqr, mpc, neural, rl policies
  estimation/       observers, Kalman filters
  ml/               learned dynamics, surrogate models
  rl/               agents, environments
  benchmarks/       standardized systems + comparison harness
modules/            curriculum, one folder per topic (theory + notebooks + experiments)
experiments/        research-oriented runs with configs, seeds, metrics
notebooks/          exploratory notebooks
tests/              unit tests for the library
docs/               vision, roadmap, methodology
```

See [docs/vision.md](docs/vision.md) for the full manifesto and
[docs/roadmap.md](docs/roadmap.md) for current priorities.

## Status

🚧 **Early Development.** Curriculum, simulation framework, benchmark environments,
and experimental methodology are under active construction.

## License

MIT — see [LICENSE](LICENSE).
