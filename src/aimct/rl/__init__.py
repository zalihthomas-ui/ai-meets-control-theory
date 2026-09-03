"""Reinforcement learning, all from scratch on NumPy.

``env``             : :class:`ControlEnv`, the :data:`TASKS` registry, :func:`make`.
``tabular``         : :class:`Discretizer`, :class:`QLearning`, :func:`train`.
``policy_gradient`` : :class:`GaussianPolicy`, :func:`reinforce` (REINFORCE).
``dqn``             : :class:`DQN` (deep Q-network with replay + target net).
"""

from .dqn import DQN, QNetwork, ReplayBuffer, dqn
from .env import (FIGURE8_PERIOD, TASKS, ControlEnv, figure8_obs,
                  figure8_reference, make, wrap_to_pi)
from .policy_gradient import GaussianPolicy, evaluate_policy, reinforce
from .ppo import PPO, ppo
from .tabular import Discretizer, GreedyPolicy, QLearning, evaluate, train

__all__ = [
    "ControlEnv",
    "TASKS",
    "make",
    "wrap_to_pi",
    "figure8_reference",
    "figure8_obs",
    "FIGURE8_PERIOD",
    "Discretizer",
    "QLearning",
    "GreedyPolicy",
    "train",
    "evaluate",
    "GaussianPolicy",
    "reinforce",
    "evaluate_policy",
    "DQN",
    "QNetwork",
    "ReplayBuffer",
    "dqn",
    "PPO",
    "ppo",
]
