"""Reinforcement learning: a Gymnasium adapter for ``aimct.systems`` and
from-scratch tabular Q-learning.

``env``     : :class:`ControlEnv`, the :data:`TASKS` registry, :func:`make`.
``tabular`` : :class:`Discretizer`, :class:`QLearning`, :func:`train`,
              :func:`evaluate`.
"""

from .env import TASKS, ControlEnv, make, wrap_to_pi
from .tabular import Discretizer, QLearning, evaluate, train

__all__ = [
    "ControlEnv",
    "TASKS",
    "make",
    "wrap_to_pi",
    "Discretizer",
    "QLearning",
    "train",
    "evaluate",
]
