# Annotated Research Papers Survey — AI Meets Control Theory

This survey synthesizes 10 foundational and frontier research papers bridging **Classical Control Theory, Optimal Control, Reinforcement Learning, Physics-Informed ML, Operator Theory, and Safe Control**. All full-text PDFs are archived in [`docs/papers/`](../papers/README.md).

---

## 🗺️ Roadmap & Module Mapping

```
                                [ THE FOUNDATIONS ]
       Linear Systems, LQR, Riccati Solvers, Frequency Margins (Modules 01-05)
                                        |
     +----------------------------------+----------------------------------+
     |                                  |                                  |
     v                                  v                                  v
[ DATA-DRIVEN DYNAMICS ]     [ REINFORCEMENT LEARNING ]     [ SAFE & HYBRID CONTROL ]
  - SINDy (Brunton 2016)       - RL Tour (Recht 2019)         - CBF Theory (Ames 2019)
  - Neural ODEs (Chen 2018)    - ARS Baseline (Mania 2018)    - Safe Learn (Taylor 2020)
  - Koopman (Brunton 2022)     - MB vs MF LQR (Tu 2019)       - Diff MPC (Amos 2018)
  (Module 06)                  (Module 07)                    - MPPI (Williams 2017)
                                                              (Module 08 & Capstones)
```

---

## 1. Classical Control vs. Reinforcement Learning Foundations

### 1.1 *A Tour of Reinforcement Learning: The View from Continuous Control*
- **Author**: Benjamin Recht (2019)
- **Venue**: *Annual Review of Control, Robotics, and Autonomous Systems*, 2:253–279
- **arXiv**: [`1806.09460`](https://arxiv.org/abs/1806.09460) | **PDF**: [`recht_2019_tour_of_rl_continuous_control.pdf`](../papers/recht_2019_tour_of_rl_continuous_control.pdf)
- **Core Insights**:
  - Unifies the language of MDPs (states $s$, actions $a$, rewards $r$, value functions $V$) with dynamical systems (states $x$, controls $u$, costs $c$, Hamilton-Jacobi-Bellman $V$).
  - Explores the trade-off between **Nominal Control** (certainty equivalence: system ID $\to$ Riccati solver) vs. **Robust Control** ($H_\infty$ / min-max) vs. **Direct Policy Search** (model-free policy gradient).
  - Demonstrates that on linear quadratic problems, coarse system identification followed by standard Riccati control achieves $\mathcal{O}(\sqrt{d/N})$ parameter error, vastly outperforming model-free policy gradients in sample efficiency.
- **AIMCT Takeaway**: Always compare RL against a system identification + LQR baseline. Do not claim RL "solves" a control problem without benchmarking against certainty-equivalent linear control.

---

### 1.2 *The Gap Between Model-Based and Model-Free Methods on the Linear Quadratic Regulator*
- **Authors**: Stephen Tu, Benjamin Recht (2019)
- **Venue**: *International Conference on Machine Learning (ICML 2019)*
- **arXiv**: [`1812.03565`](https://arxiv.org/abs/1812.03565) | **PDF**: [`tu_2019_model_based_vs_model_free_lqr.pdf`](../papers/tu_2019_model_based_vs_model_free_lqr.pdf)
- **Core Insights**:
  - Rigorously bounds the sample complexity of learning an $\epsilon$-suboptimal policy for continuous-time/discrete-time LQR.
  - **Model-Based (Plugin / Certainty Equivalence)**: Requires $\tilde{\mathcal{O}}((n + m)/\epsilon)$ trajectory samples (where $n$ is state dimension and $m$ is input dimension).
  - **Model-Free (Policy Gradient / REINFORCE)**: Requires $\tilde{\mathcal{O}}((n + m)^3/\epsilon^2)$ or worse trajectory samples.
  - Proves an intrinsic dimensional gap: model-free policy gradients waste samples estimating gradients that can be derived directly from plant matrix estimates.
- **AIMCT Takeaway**: In Module 07, our benchmark comparison between model-based (Least-Squares System ID + LQR) and model-free (PPO/DQN) will verify this $\mathcal{O}(n^2)$ sample-efficiency gap empirically on Cart-Pole.

---

### 1.3 *Simple Random Search Provides a Competitive Approach to Reinforcement Learning*
- **Authors**: Horia Mania, Aurelia Guy, Benjamin Recht (2018)
- **Venue**: *NeurIPS 2018*
- **arXiv**: [`1803.07055`](https://arxiv.org/abs/1803.07055) | **PDF**: [`mania_2018_augmented_random_search.pdf`](../papers/mania_2018_augmented_random_search.pdf)
- **Core Insights**:
  - Proposes **Augmented Random Search (ARS)**: a derivative-free optimization algorithm that perturbs static linear policies $\pi(x) = W x$ along random Gaussian directions, updates weights via finite differences, and normalizes state inputs.
  - Outperforms or matches complex deep RL actor-critic methods (TRPO, PPO, DDPG) on standard MuJoCo continuous control benchmarks (HalfCheetah, Hopper, Walker2d) while being **15x faster to compute**.
- **AIMCT Takeaway**: Linear state-feedback policies $\pi(x) = K x$ have immense expressive power for continuous locomotion when properly scaled; deep networks are often unnecessarily complex for basic stabilization.

---

## 2. Data-Driven System Identification & Physics-Informed ML

### 2.1 *Discovering Governing Equations from Data: SINDy*
- **Authors**: Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz (2016)
- **Venue**: *Proceedings of the National Academy of Sciences (PNAS)*, 113(15):3932–3937
- **arXiv**: [`1509.03580`](https://arxiv.org/abs/1509.03580) | **PDF**: [`brunton_2016_sindy_governing_equations.pdf`](../papers/brunton_2016_sindy_governing_equations.pdf)
- **Core Insights**:
  - Formulates **Sparse Identification of Nonlinear Dynamics (SINDy)**:
    $$\dot{X} = \Theta(X, U) \Xi$$
    where $\Theta(X, U)$ is a library of candidate nonlinear functions (polynomials, trigonometric terms $1, x, x^2, \sin x, \cos x, u, x u$), and $\Xi$ is a sparse coefficient matrix determined via Sequential Thresholded Least Squares (STLSQ) or LASSO.
  - Accurately recovers the exact governing ODEs for chaotic Lorenz systems, non-linear pendulums, and fluid flows from noisy trajectories.
  - Highly interpretable, parsimonious, and prevents overfitting by finding the simplest physical model consistent with data (Occam's razor).
- **AIMCT Takeaway**: Module 06 will implement from-scratch STLSQ to discover the nonlinear pendulum ODE $\ddot{\theta} = -\frac{g}{l}\sin\theta - \frac{b}{m l^2}\dot{\theta}$ from raw trajectory data.

---

### 2.2 *Neural Ordinary Differential Equations*
- **Authors**: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud (2018)
- **Venue**: *NeurIPS 2018 (Best Paper Award)*
- **arXiv**: [`1806.07366`](https://arxiv.org/abs/1806.07366) | **PDF**: [`chen_2018_neural_odes.pdf`](../papers/chen_2018_neural_odes.pdf)
- **Core Insights**:
  - Replaces discrete neural network layers with continuous hidden state evolution:
    $$\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad z(t_1) = z(t_0) + \int_{t_0}^{t_1} f_\theta(z(t), t) dt$$
  - Trains parameters $\theta$ in $\mathcal{O}(1)$ memory via the continuous **Adjoint Sensitivity Method** (solving a reverse-time adjoint ODE $a(t) = \frac{\partial L}{\partial z(t)}$ derived from Pontryagin's Minimum Principle).
  - Handles irregularly sampled time-series data seamlessly and allows adaptive trade-offs between numerical precision and compute cost.
- **AIMCT Takeaway**: Connects deep learning directly to classical continuous-time ODE integration (our `aimct.simulate.rk4_step`).

---

### 2.3 *Modern Koopman Theory for Dynamical Systems*
- **Authors**: Steven L. Brunton, Marko Budišić, Eurika Kaiser, J. Nathan Kutz (2022)
- **Venue**: *SIAM Review*, 64(2):229–340
- **arXiv**: [`2102.12086`](https://arxiv.org/abs/2102.12086) | **PDF**: [`brunton_2022_modern_koopman_theory.pdf`](../papers/brunton_2022_modern_koopman_theory.pdf)
- **Core Insights**:
  - The **Koopman Operator $\mathcal{K}$** is an infinite-dimensional linear operator that advances observable functions $g(x)$ forward in time along nonlinear dynamics:
    $$\mathcal{K}_t g(x_0) = g(F_t(x_0))$$
  - Transforms nonlinear finite-dimensional dynamics $\dot{x} = f(x)$ into linear infinite-dimensional dynamics $\dot{g} = \mathcal{A} g$.
  - Uses Dynamic Mode Decomposition (DMD / Extended DMD) to find finite-dimensional invariant subspaces, allowing linear control techniques (LQR, Pole Placement, Kalman Filters) to control global nonlinear systems without Taylor linearization.
- **AIMCT Takeaway**: Enables global linear optimal control for nonlinear systems (e.g. large-angle pendulum) by lifting states into Koopman observable coordinates.

---

## 3. Safe Control, Barrier Functions & Differentiable Optimization

### 3.1 *Control Barrier Functions: Theory and Applications*
- **Authors**: Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, Paulo Tabuada (2019)
- **Venue**: *European Control Conference (ECC 2019)*, pp. 3420–3431
- **arXiv**: [`1903.11199`](https://arxiv.org/abs/1903.11199) | **PDF**: [`ames_2019_control_barrier_functions.pdf`](../papers/ames_2019_control_barrier_functions.pdf)
- **Core Insights**:
  - Formulates **Control Barrier Functions (CBFs)** $h(x) \ge 0$ that guarantee forward invariance of safe sets $\mathcal{C} = \{x \in \mathbb{R}^n \mid h(x) \ge 0\}$.
  - Nagumo's theorem & safety condition:
    $$\sup_{u \in U} \left[ L_f h(x) + L_g h(x) u + \alpha(h(x)) \right] \ge 0$$
  - **CBF-QP Safety Filter**: Given any nominal controller (untrusted RL policy, neural network, or aggressive human driver) $u_{\text{nom}}(x)$, filters the action in real time via a convex Quadratic Program:
    $$\min_{u \in U} \frac{1}{2} \|u - u_{\text{nom}}(x)\|^2 \quad \text{s.t.} \quad L_f h(x) + L_g h(x) u + \alpha(h(x)) \ge 0$$
- **AIMCT Takeaway**: The gold standard for safe AI control: wrap black-box neural/RL agents inside a real-time CBF-QP safety filter to provide mathematical guarantees against state constraint violations (rail limits, obstacle collisions).

---

### 3.2 *Learning for Safety-Critical Control with Control Barrier Functions*
- **Authors**: Andrew J. Taylor, Andrew K. Singletary, Yisong Yue, Aaron D. Ames (2020)
- **Venue**: *Learning for Dynamics & Control (L4DC 2020)*
- **arXiv**: [`2004.09559`](https://arxiv.org/abs/2004.09559) | **PDF**: [`taylor_2020_learning_safety_critical_cbf.pdf`](../papers/taylor_2020_learning_safety_critical_cbf.pdf)
- **Core Insights**:
  - Addresses the fundamental flaw in standard CBFs: model mismatch ($\dot{x} = f(x) + g(x)u + d(x)$ where $d(x)$ is unmodeled).
  - Uses machine learning (Gaussian Processes / Neural Networks) to estimate the model error $d(x)$ while maintaining a robustified CBF-QP constraint that prevents safety violations during the learning process.
- **AIMCT Takeaway**: Directly applicable to Track 4 of our Intelligent Control Challenge (Safe Black-Box Adaptive Control).

---

### 3.3 *Differentiable MPC for End-to-End Planning and Control*
- **Authors**: Brandon Amos, Ivan Jimenez, Jacob Sacks, Byron Boots, J. Zico Kolter (2018)
- **Venue**: *NeurIPS 2018*
- **arXiv**: [`1810.13400`](https://arxiv.org/abs/1810.13400) | **PDF**: [`amos_2018_differentiable_mpc.pdf`](../papers/amos_2018_differentiable_mpc.pdf)
- **Core Insights**:
  - Implements Model Predictive Control (MPC) as a differentiable layer in deep neural networks (DiffMPC).
  - Backpropagates analytical gradients through the Karush-Kuhn-Tucker (KKT) optimality conditions of the quadratic program using implicit differentiation.
  - Allows end-to-end learning of cost matrices ($Q, R$) and transition models directly from expert demonstrations or reinforcement signals.
- **AIMCT Takeaway**: Solves the long-standing "cost design" problem in optimal control by learning cost functions that optimize downstream tracking metrics.

---

### 3.4 *Information-Theoretic Model Predictive Control: MPPI*
- **Authors**: Grady Williams, Paul Drews, Brian Goldfain, James M. Rehg, Evangelos A. Theodorou (2017)
- **Venue**: *IEEE Transactions on Robotics*, 34(6):1603–1619
- **arXiv**: [`1707.02342`](https://arxiv.org/abs/1707.02342) | **PDF**: [`williams_2017_mppi_autonomous_driving.pdf`](../papers/williams_2017_mppi_autonomous_driving.pdf)
- **Core Insights**:
  - Formulates **Model Predictive Path Integral (MPPI)** control using information-theoretic free energy and Feynman-Kac path integrals.
  - Samples thousands of noisy control trajectories in parallel on GPUs, computes their costs, and computes the optimal control update as an importance-weighted average:
    $$u^*(t) = u_{\text{base}}(t) + \sum_{k=1}^K w_k \delta u_k(t), \quad w_k = \frac{\exp(-\frac{1}{\lambda} S(\tau_k))}{\sum_j \exp(-\frac{1}{\lambda} S(\tau_j))}$$
  - Operates on arbitrary non-convex, discontinuous, non-differentiable physics engines without requiring gradient computations or linearizations.
- **AIMCT Takeaway**: Enables aggressive autonomous vehicle drifting, obstacle avoidance, and extreme maneuver control for our Phase 4 Capstone projects.

---

## 4. Synthesis & Cross-Comparison Matrix

| Paper | Primary Paradigm | Model Requirement | Stability / Safety Certificate | Compute Cost | Sample Efficiency | Key Role in AIMCT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recht (2019)** | Survey / LTI Analysis | Known / Estimated $A, B$ | Lyapunov / Riccati | Low ($\mathcal{O}(n^3)$ ARE) | $\mathcal{O}(\sqrt{d/N})$ (Optimal) | Foundational philosophy & baseline guide |
| **Tu & Recht (2019)** | Sample Complexity | Linear $A, B$ | ARE Hurwitz stability | Low | Proof: Model-Based $\gg$ Model-Free | Theoretical foundation for Mod 05 vs 07 |
| **Mania et al. (2018)** | Derivative-Free RL | Black-Box $f(x, u)$ | None (Empirical) | Low (Linear policy) | High for model-free | Linear baseline against deep RL |
| **Brunton et al. (2016)** | Sparse System ID | Data Trajectories | Identifies ODEs | Low (STLSQ) | High (Sparse regression) | Module 06 data-driven modeling engine |
| **Chen et al. (2018)** | Continuous Deep ML | Neural ODE $f_\theta$ | Empirical | Medium (ODE Solver) | Medium | Module 06 learned dynamics representations |
| **Brunton et al. (2022)** | Operator Theory | Data / Functions | Linearized Invariance | Medium (DMD / SVD) | High | Module 08 global nonlinear linearizer |
| **Ames et al. (2019)** | Safe Optimization | Control-Affine $f(x)+g(x)u$ | **Rigorous Forward Invariance** | Low (Real-time QP) | Infinite (Zero-shot safety) | Module 08 safety filter for RL & MPC |
| **Taylor et al. (2020)** | Adaptive Safe ML | Learned $f(x) + d(x)$ | **Robust CBF Invariance** | Medium (QP + GP/NN) | Zero safety violations online | Capstone safe exploration |
| **Amos et al. (2018)** | Differentiable MPC | Linear / Convex QP | KKT Optimality | Medium (Implicit QP) | High | Module 08 end-to-end tuning |
| **Williams et al. (2017)**| Path Integral MPC | Any Simulator (GPU) | Statistical (Softmax) | High (Parallel GPU) | Real-time sampling | Capstone autonomous racing / drifting |
