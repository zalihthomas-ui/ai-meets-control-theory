# API Reference Overview

Welcome to the imct (AI Meets Control Theory) API Reference.

imct provides a unified, mathematically rigorous Python framework bridging classical control theory, modern state-space control, optimal control, system identification, and machine learning/reinforcement learning.

## Module Organization

| Module | Description | Core Abstractions |
| :--- | :--- | :--- |
| [imct.systems](systems.md) | Physical and benchmark dynamical systems | DynamicalSystem, ContinuousSystem, DiscreteSystem |
| [imct.controllers](controllers.md) | Classical, robust, adaptive, and optimal controllers | BaseController, PIDController, LQRController, MPCController, SMCController |
| [imct.planning](planning.md) | Trajectory planning & optimization | TrajectoryPlanner, RRTPlanner, MinimumJerkPlanner |
| [imct.estimation](estimation.md) | State estimation & filtering | StateEstimator, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter |
| [imct.trajectories](trajectories.md) | Standard trajectory generation & references | StepTrajectory, SinusoidTrajectory, ChirpTrajectory |
| [imct.sysid](sysid.md) | System identification & model learning | LeastSquaresSysId, SubspaceSysId, ManipulatorSysId |
| [imct.ml](ml.md) | Physics-Informed Neural Networks & Neural ODEs | PINNModel, NeuralODE, DeepKoopman |
| [imct.rl](rl.md) | Reinforcement learning algorithms & wrappers | PPOAgent, SACAgent, GymnasiumEnvWrapper |
| [imct.hybrid](hybrid.md) | Hybrid, neuro-symbolic, and residual control | ResidualRLController, AdaptiveNeuralController |
| [imct.hil](hil.md) | Hardware-in-the-loop bridges & streaming protocols | SerialHardwareBridge, UDPStreamer, CANBridge |
| [imct.viz](viz.md) | Publication-quality plotting & dashboard widgets | PlotEngine, LiveVisualizer, PhasePortrait |
| [imct.dev](dev.md) | Developer utilities, logging & verification | MetricLogger, ContractChecker, Timer |
| [imct.benchmarks](benchmarks.md) | Standardized multi-system evaluation suite | BenchmarkSuite, ExperimentRunner |
