# API Reference Overview

Welcome to the aimct (AI Meets Control Theory) API Reference.

aimct provides a unified, mathematically rigorous Python framework bridging classical control theory, modern state-space control, optimal control, system identification, and machine learning/reinforcement learning.

## Module Organization

| Module | Description | Core Abstractions |
| :--- | :--- | :--- |
| [aimct.systems](systems.md) | Physical and benchmark dynamical systems | DynamicalSystem, ContinuousSystem, DiscreteSystem |
| [aimct.controllers](controllers.md) | Classical, robust, adaptive, and optimal controllers | BaseController, PIDController, LQRController, MPCController, SMCController |
| [aimct.planning](planning.md) | Trajectory planning & optimization | TrajectoryPlanner, RRTPlanner, MinimumJerkPlanner |
| [aimct.estimation](estimation.md) | State estimation & filtering | StateEstimator, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter |
| [aimct.trajectories](trajectories.md) | Standard trajectory generation & references | StepTrajectory, SinusoidTrajectory, ChirpTrajectory |
| [aimct.sysid](sysid.md) | System identification & model learning | LeastSquaresSysId, SubspaceSysId, ManipulatorSysId |
| [aimct.ml](ml.md) | Physics-Informed Neural Networks & Neural ODEs | PINNModel, NeuralODE, DeepKoopman |
| [aimct.rl](rl.md) | Reinforcement learning algorithms & wrappers | PPOAgent, SACAgent, GymnasiumEnvWrapper |
| [aimct.hybrid](hybrid.md) | Hybrid, neuro-symbolic, and residual control | ResidualRLController, AdaptiveNeuralController |
| [aimct.hil](hil.md) | Hardware-in-the-loop bridges & streaming protocols | SerialHardwareBridge, UDPStreamer, CANBridge |
| [aimct.viz](viz.md) | Publication-quality plotting & dashboard widgets | PlotEngine, LiveVisualizer, PhasePortrait |
| [aimct.dev](dev.md) | Developer utilities, logging & verification | MetricLogger, ContractChecker, Timer |
| [aimct.benchmarks](benchmarks.md) | Standardized multi-system evaluation suite | BenchmarkSuite, ExperimentRunner |
