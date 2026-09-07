# Track C API Audit & Stability Surface (v1.0.0 Freeze)

## 1. `aimct.estimation` Surface Audit

All classes adhere to the common estimation protocol:
- Initialization with system dynamics (continuous `f(x, u)` or `discrete=True`), measurement `h(x)`, covariances $Q, R$, and sampling time $dt$.
- Standard interface: `.predict(u) -> x_hat`, `.update(y) -> x_hat`, `.step(y, u) -> x_hat`, `.reset(x0, P0)`.
- State attributes: `.x_hat` (current state estimate), `.P` (current error covariance or empirical particle covariance).
- Factory classmethod: `.from_system(system, Q, R, dt=..., **kwargs)`.

### Public Classes & Exports
| Public Symbol | Module | Interface & Guarantees |
| :--- | :--- | :--- |
| `LuenbergerObserver` | `aimct.estimation.luenberger` | Continuous/discrete Luenberger state observer with pole placement (`place_observer`). |
| `place_observer` | `aimct.estimation.luenberger` | Ackermann duality-based observer gain placement $L$. |
| `solve_fare` | `aimct.estimation.kalman` | Continuous Filter Algebraic Riccati Equation solver. |
| `KalmanFilter` | `aimct.estimation.kalman` | Continuous-time Kalman-Bucy filter with steady-state Riccati gain. |
| `DiscreteKalmanFilter` | `aimct.estimation.kalman` | Discrete-time linear Kalman filter with exact van Loan matrix exponential discretization. |
| `ExtendedKalmanFilter` | `aimct.estimation.ekf` | Nonlinear EKF with Joseph stabilized covariance updates and analytic/FD Jacobians. |
| `UnscentedKalmanFilter` | `aimct.estimation.ukf` | Additive-noise scaled unscented transform $(\alpha, \beta, \kappa)$ with $2n+1$ sigma points. |
| `MovingHorizonEstimator` (alias `MHE`) | `aimct.estimation.mhe` | Constrained nonlinear MAP estimation over sliding horizon $N$ with hard state bounds $x \in \mathcal{X}$, disturbance bounds $w \in \mathcal{W}$, general constraints $g(x) \le 0$, and EKF Riccati arrival cost. |
| `ParticleFilter` (alias `PF`) | `aimct.estimation.particle_filter` | Sequential Importance Resampling (SIR) Bootstrap filter with Log-Sum-Exp weight updates and adaptive systematic resampling ($ESS < \gamma N_p$). |
| `systematic_resample` | `aimct.estimation.particle_filter` | $\mathcal{O}(N)$ systematic low-variance resampling algorithm. |
| `finite_diff_jacobian` | `aimct.estimation.ekf` | High-precision central difference Jacobian matrix calculation. |
| `observability_matrix`, `observability_rank`, `is_observable` | `aimct.estimation.observability` | Kalman observability rank condition $\text{rank}(\mathcal{O}) = n$. |

---

## 2. `aimct.systems` Multi-Agent Surface Audit

| Public Symbol | Module | Interface & Guarantees |
| :--- | :--- | :--- |
| `MultiAgentSystem` (alias `MultiAgent`) | `aimct.systems.multi_agent` | Multi-agent dynamical system composition subclassing `DynamicalSystem`. State $x \in \mathbb{R}^{N \cdot n_a}$, input $u \in \mathbb{R}^{N \cdot m_a}$. Supports single/double integrators and custom agents. Exposes `.adjacency`, `.degree_matrix`, `.laplacian`, `.algebraic_connectivity` ($\lambda_2(L)$), `.is_connected`, `.get_agent_positions()`, `.get_agent_velocities()`, `.pairwise_distances()`, `.min_pairwise_distance()`. |
| `complete_graph` | `aimct.systems.multi_agent` | Generates $K_N$ all-to-all connected adjacency matrix ($\lambda_2 = N$). |
| `cycle_graph` | `aimct.systems.multi_agent` | Generates $C_N$ ring / cycle adjacency matrix ($\lambda_2 = 2 - 2\cos(2\pi/N)$). |
| `line_graph` | `aimct.systems.multi_agent` | Generates $P_N$ linear path adjacency matrix ($\lambda_2 = 2 - 2\cos(\pi/N)$). |
| `star_graph` | `aimct.systems.multi_agent` | Generates $S_N$ star hub adjacency matrix ($\lambda_2 = 1$). |
| `disconnected_graph` | `aimct.systems.multi_agent` | Generates adjacency matrix with isolated nodes ($\lambda_2 = 0$). |

---

## 3. `aimct.controllers` Formation Control Surface Audit

| Public Symbol | Module | Interface & Guarantees |
| :--- | :--- | :--- |
| `ConsensusFormationController` (alias `FormationController`) | `aimct.controllers.formation` | Distributed consensus-based formation controller with relative offset consensus ($k_p$), velocity damping ($k_v$), leader pinning ($k_{\text{ref}}, k_{\text{vref}}$), artificial potential field collision barriers ($k_{\text{coll}}, d_{\text{safe}}$), and input saturation box $[u_{\min}, u_{\max}]$. |
| `polygon_formation` | `aimct.controllers.formation` | Generates regular $N$-gon relative offset vectors centered at origin. |
| `line_formation` | `aimct.controllers.formation` | Generates 1D equidistant relative offset vectors along specified axis. |
| `diamond_formation` | `aimct.controllers.formation` | Generates 4-agent diamond formation offsets. |
| `wedge_formation` | `aimct.controllers.formation` | Generates inverted V-shape / wedge formation offsets. |

---

## 4. Verification & Testing Summary

- 100% test pass across Track C test suites:
  - `tests/test_mhe.py` (7 tests)
  - `tests/test_particle_filter.py` (6 tests)
  - `tests/test_formation.py` (5 tests)
  - `tests/test_ekf.py` (8 tests)
  - `tests/test_ukf.py` (8 tests)
- Benchmark evidence artifacts versioned in `experiments/`:
  - `experiments/38_mhe_vs_ekf/` (TwoTank state bounds)
  - `experiments/39_formation_switching_graph/` (Switching graph formation tracking)
  - `experiments/41_particle_filter_bearings_only/` (Bearings-only target tracking)
