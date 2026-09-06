"""AI Meets Control Theory — reusable library.

Subpackages
-----------
systems      : dynamical-system models with a common interface
controllers  : PID, state feedback, LQR, MPC, neural, RL policies
planning     : direct trajectory optimisation (Hermite-Simpson collocation)
robust       : structured-uncertainty analysis (mu / structured singular value)
estimation   : observers, Kalman filters
ml           : learned dynamics, surrogate models
rl           : agents and environments
benchmarks   : standardized systems + controller comparison harness
sysid        : identify linear models / manipulator inertial parameters from logs
hil          : hardware-in-the-loop harness (real-time loop, plant emulator)
deploy       : export a static controller to JSON + C / MicroPython
"""

__version__ = "0.2.0"
