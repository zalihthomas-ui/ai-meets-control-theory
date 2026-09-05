"""Dynamical-system models with a common interface (see :mod:`aimct.systems.base`)."""

from .base import DynamicalSystem
from .bicycle import BicycleVehicle
from .cartpole import CartPole
from .dc_motor import DCMotor, DCMotor2
from .diffdrive import DifferentialDriveRobot
from .linear import LinearSystem
from .mass_spring_damper import MassSpringDamper
from .pendulum import Pendulum
from .quadrotor import PlanarQuadrotor
from .quadrotor3d import Quadrotor3D, rotation_matrix
from .twolink_arm import TwoLinkArm

__all__ = [
    "DynamicalSystem",
    "BicycleVehicle",
    "LinearSystem",
    "MassSpringDamper",
    "Pendulum",
    "CartPole",
    "PlanarQuadrotor",
    "Quadrotor3D",
    "rotation_matrix",
    "DCMotor",
    "DCMotor2",
    "DifferentialDriveRobot",
    "TwoLinkArm",
]
