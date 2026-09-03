"""Dynamical-system models with a common interface (see :mod:`aimct.systems.base`)."""

from .base import DynamicalSystem
from .cartpole import CartPole
from .dc_motor import DCMotor, DCMotor2
from .linear import LinearSystem
from .mass_spring_damper import MassSpringDamper
from .pendulum import Pendulum
from .quadrotor import PlanarQuadrotor

__all__ = [
    "DynamicalSystem",
    "LinearSystem",
    "MassSpringDamper",
    "Pendulum",
    "CartPole",
    "PlanarQuadrotor",
    "DCMotor",
    "DCMotor2",
]
