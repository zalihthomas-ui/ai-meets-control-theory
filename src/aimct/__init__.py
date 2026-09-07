"""AI Meets Control Theory — reusable library.

Subpackages
-----------
systems      : dynamical-system models with a common interface
controllers  : PID, state feedback, LQR, MPC, neural, RL policies
planning     : direct trajectory optimisation (Hermite-Simpson collocation)
robust       : structured-uncertainty analysis (mu / structured singular value)
estimation   : observers, Kalman filters, moving-horizon & particle filters
ml           : learned dynamics, surrogate models
rl           : agents and environments
benchmarks   : standardized systems + controller comparison harness
sysid        : identify linear models / manipulator inertial parameters from logs
hil          : hardware-in-the-loop harness (real-time loop, plant emulator)
deploy       : export a static controller to JSON + C / MicroPython

``import aimct`` stays lightweight; subpackages load on first attribute
access (``aimct.controllers.LQR``) so heavy optional dependencies
(``matplotlib``, ``gymnasium``) are only imported when their subpackage is
used. See ``docs/STABILITY.md`` for the public-API and versioning policy.
"""

import importlib as _importlib

__version__ = "1.0.0"

_SUBPACKAGES = (
    "systems", "controllers", "planning", "robust", "estimation", "sysid",
    "trajectories", "simulate", "benchmarks", "hybrid", "ml", "rl", "viz",
    "dev", "deploy", "hil",
)

__all__ = [*_SUBPACKAGES, "__version__"]


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        mod = _importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
