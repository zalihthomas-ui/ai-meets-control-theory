r"""Multi-agent dynamical system composition and communication topologies.

Represents a collective of :math:`N` interacting autonomous agents (e.g. 2D
point masses, single/double integrators, unicycles, or robotic vehicles) coupled
via an algebraic communication graph :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})`.

The communication network is characterized by its Adjacency matrix :math:`A`,
Degree matrix :math:`D = \text{diag}(\sum_j a_{ij})`, and Graph Laplacian
:math:`L = D - A`. Algebraic connectivity is quantified by the Fiedler eigenvalue
:math:`\lambda_2(L)`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .base import ArrayLike, DynamicalSystem

__all__ = [
    "MultiAgentSystem",
    "MultiAgent",
    "complete_graph",
    "cycle_graph",
    "line_graph",
    "star_graph",
    "disconnected_graph",
]


# ------------------------------------------------------------------ Graph utilities

def complete_graph(n: int) -> np.ndarray:
    """Generate adjacency matrix for an all-to-all connected complete graph :math:`K_n`."""
    n = int(n)
    A = np.ones((n, n), dtype=float) - np.eye(n, dtype=float)
    return A


def cycle_graph(n: int) -> np.ndarray:
    """Generate adjacency matrix for a ring / cycle graph :math:`C_n`."""
    n = int(n)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[i, (i - 1) % n] = 1.0
    return A


def line_graph(n: int) -> np.ndarray:
    """Generate adjacency matrix for a linear path graph :math:`P_n`."""
    n = int(n)
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def star_graph(n: int, center: int = 0) -> np.ndarray:
    """Generate adjacency matrix for a star graph :math:`S_n` with a central hub node."""
    n = int(n)
    center = int(center) % n
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i != center:
            A[center, i] = 1.0
            A[i, center] = 1.0
    return A


def disconnected_graph(n: int, isolated_nodes: Sequence[int]) -> np.ndarray:
    """Generate a partially disconnected graph by removing all links to specified nodes."""
    A = complete_graph(n)
    for node in isolated_nodes:
        idx = int(node) % n
        A[idx, :] = 0.0
        A[:, idx] = 0.0
    return A


# ------------------------------------------------------------------ MultiAgentSystem

class MultiAgentSystem(DynamicalSystem):
    r"""Continuous-time multi-agent dynamical system.

    Parameters
    ----------
    n_agents:
        Number of interacting agents :math:`N \ge 2`. Defaults to 4.
    agent_type:
        Agent dynamic model type: ``'double_integrator'`` (default: state
        :math:`[p_x, p_y, v_x, v_y]`, input :math:`[u_x, u_y]`),
        ``'single_integrator'`` (state :math:`[p_x, p_y]`, input :math:`[u_x, u_y]`),
        or a custom :class:`~aimct.systems.DynamicalSystem` instance.
    adjacency:
        Initial :math:`(N, N)` communication graph adjacency matrix. Defaults
        to a cycle graph :math:`C_N` if omitted.
    damping:
        Linear velocity drag damping coefficient :math:`c_d` for double integrators (:math:`\ddot{p} = u - c_d v`).
    """

    def __init__(
        self,
        n_agents: int = 4,
        agent_type: str | DynamicalSystem = "double_integrator",
        adjacency: np.ndarray | None = None,
        damping: float = 0.0,
    ) -> None:
        self.n_agents = int(max(2, n_agents))
        self.damping = float(damping)

        if isinstance(agent_type, str):
            t_lower = agent_type.lower().strip()
            if t_lower in {"double_integrator", "double", "2nd_order"}:
                self.agent_type = "double_integrator"
                self.agent_dim = 4
                self.agent_input_dim = 2
                self._agent_sys = None
            elif t_lower in {"single_integrator", "single", "1st_order"}:
                self.agent_type = "single_integrator"
                self.agent_dim = 2
                self.agent_input_dim = 2
                self._agent_sys = None
            else:
                raise ValueError(f"Unknown agent_type string '{agent_type}'. Expected 'double_integrator' or 'single_integrator'.")
        elif isinstance(agent_type, DynamicalSystem):
            self.agent_type = "custom"
            self.agent_dim = agent_type.n_states
            self.agent_input_dim = agent_type.n_inputs
            self._agent_sys = agent_type
        else:
            raise TypeError(f"agent_type must be str or DynamicalSystem, got {type(agent_type)}")

        self.n_states = self.n_agents * self.agent_dim
        self.n_inputs = self.n_agents * self.agent_input_dim
        self.n_outputs = self.n_states

        if adjacency is not None:
            self.set_topology(adjacency)
        else:
            self.set_topology(cycle_graph(self.n_agents))

    # ------------------------------------------------------------------ Topology

    def set_topology(self, adjacency: np.ndarray) -> None:
        """Update communication graph adjacency matrix :math:`A`."""
        A = np.asarray(adjacency, dtype=float)
        if A.shape != (self.n_agents, self.n_agents):
            raise ValueError(f"Adjacency matrix must have shape ({self.n_agents}, {self.n_agents}), got {A.shape}")
        self._adjacency = A.copy()
        # Enforce zero diagonal
        np.fill_diagonal(self._adjacency, 0.0)

    @property
    def adjacency(self) -> np.ndarray:
        """Adjacency matrix :math:`A \in \mathbb{R}^{N \times N}`."""
        return self._adjacency.copy()

    @property
    def degree_matrix(self) -> np.ndarray:
        """Degree matrix :math:`D = \text{diag}(\sum_j A_{ij})`."""
        deg = np.sum(self._adjacency, axis=1)
        return np.diag(deg)

    @property
    def laplacian(self) -> np.ndarray:
        """Graph Laplacian matrix :math:`L = D - A`."""
        return self.degree_matrix - self._adjacency

    @property
    def algebraic_connectivity(self) -> float:
        """Fiedler eigenvalue :math:`\lambda_2(L)` (second smallest eigenvalue of Laplacian)."""
        L = self.laplacian
        # Symmetrize for numerical eigenvalue stability
        L_sym = 0.5 * (L + L.T)
        eigvals = np.sort(np.linalg.eigvalsh(L_sym))
        if len(eigvals) >= 2:
            return float(max(0.0, eigvals[1]))
        return 0.0

    @property
    def fiedler_eigenvalue(self) -> float:
        """Alias for :attr:`algebraic_connectivity`."""
        return self.algebraic_connectivity

    @property
    def is_connected(self) -> bool:
        """Return True if the communication graph is connected (:math:`\lambda_2(L) > 10^{-6}`)."""
        return bool(self.algebraic_connectivity > 1e-6)

    # ------------------------------------------------------------------ State helpers

    def get_agent_states(self, x: ArrayLike) -> np.ndarray:
        """Reshape stacked state vector :math:`x` into shape ``(n_agents, agent_dim)``."""
        x_arr = np.asarray(x, dtype=float).reshape(self.n_states)
        return x_arr.reshape((self.n_agents, self.agent_dim))

    def get_agent_positions(self, x: ArrayLike) -> np.ndarray:
        """Extract 2D Cartesian positions :math:`[p_{x,i}, p_{y,i}]` of shape ``(n_agents, 2)``."""
        states = self.get_agent_states(x)
        return states[:, :2].copy()

    def get_agent_velocities(self, x: ArrayLike) -> np.ndarray:
        """Extract 2D Cartesian velocities :math:`[v_{x,i}, v_{y,i}]` of shape ``(n_agents, 2)``."""
        states = self.get_agent_states(x)
        if self.agent_type == "double_integrator":
            return states[:, 2:4].copy()
        elif self.agent_dim >= 4:
            return states[:, 2:4].copy()
        return np.zeros((self.n_agents, 2), dtype=float)

    def pairwise_distances(self, x: ArrayLike) -> np.ndarray:
        """Compute :math:`(N, N)` symmetric matrix of pairwise Euclidean distances :math:`\|p_i - p_j\|`."""
        pos = self.get_agent_positions(x)
        diff = pos[:, None, :] - pos[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def min_pairwise_distance(self, x: ArrayLike) -> float:
        """Return the minimum Euclidean distance between any distinct agent pair :math:`i \neq j`."""
        dist_mat = self.pairwise_distances(x)
        np.fill_diagonal(dist_mat, np.inf)
        return float(np.min(dist_mat))

    # ------------------------------------------------------------------ Dynamics

    def dynamics(self, t: float, x: ArrayLike, u: ArrayLike) -> np.ndarray:
        """Compute time derivative of stacked multi-agent state vector."""
        x, u = self._prep(x, u)
        x_agents = x.reshape((self.n_agents, self.agent_dim))
        u_agents = u.reshape((self.n_agents, self.agent_input_dim))
        xdot = np.empty_like(x_agents)

        if self.agent_type == "double_integrator":
            for i in range(self.n_agents):
                # State: [px, py, vx, vy]
                # Input: [ux, uy]
                vx, vy = x_agents[i, 2], x_agents[i, 3]
                ax = u_agents[i, 0] - self.damping * vx
                ay = u_agents[i, 1] - self.damping * vy
                xdot[i] = [vx, vy, ax, ay]
        elif self.agent_type == "single_integrator":
            for i in range(self.n_agents):
                # State: [px, py]
                # Input: [ux, uy]
                xdot[i] = [u_agents[i, 0], u_agents[i, 1]]
        else:
            # Custom system instance
            assert self._agent_sys is not None
            for i in range(self.n_agents):
                xdot[i] = self._agent_sys.dynamics(t, x_agents[i], u_agents[i])

        return xdot.ravel()


# Convenient alias
MultiAgent = MultiAgentSystem
