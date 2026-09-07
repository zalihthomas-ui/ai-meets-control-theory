r"""Distributed consensus formation controllers for multi-agent systems.

Provides distributed formation control laws based on algebraic graph theory
and relative displacement consensus:

.. math::

    u_i = -k_p \sum_{j \in \mathcal{N}_i} a_{ij} (p_i - p_j - d_{ij})
          - k_v \sum_{j \in \mathcal{N}_i} a_{ij} (v_i - v_j)
          - b_i \left[ k_{\text{ref}} (p_i - d_i^* - r_0(t)) + k_{\text{vref}} (v_i - v_0(t)) \right]
          + u_{\text{coll}, i}

where :math:`d_{ij} = d_i^* - d_j^*` are the target relative offset vectors,
:math:`a_{ij}` are adjacency weights, :math:`b_i \ge 0` is leader pinning,
and :math:`u_{\text{coll}, i}` is an artificial potential field collision avoidance barrier.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "ConsensusFormationController",
    "FormationController",
    "polygon_formation",
    "line_formation",
    "diamond_formation",
    "wedge_formation",
]


# ------------------------------------------------------------------ Geometry generators

def polygon_formation(n: int, radius: float = 2.0) -> np.ndarray:
    """Generate 2D relative offset vectors :math:`d_i^*` for a regular :math:`N`-gon."""
    n = int(n)
    r = float(radius)
    angles = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
    offsets = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    # Center at origin
    return offsets - np.mean(offsets, axis=0)


def line_formation(n: int, spacing: float = 1.5, axis: int = 0) -> np.ndarray:
    """Generate 2D relative offset vectors along the specified axis (0 for X, 1 for Y)."""
    n = int(n)
    s = float(spacing)
    coords = (np.arange(n) - (n - 1) / 2.0) * s
    offsets = np.zeros((n, 2), dtype=float)
    offsets[:, int(axis) % 2] = coords
    return offsets


def diamond_formation(radius: float = 2.0) -> np.ndarray:
    """Generate 2D relative offset vectors for a 4-agent diamond formation."""
    r = float(radius)
    return np.array([
        [r, 0.0],    # Leader (East)
        [0.0, r],    # North
        [-r, 0.0],   # West
        [0.0, -r],   # South
    ], dtype=float)


def wedge_formation(n: int, spacing: float = 1.5, angle_deg: float = 60.0) -> np.ndarray:
    """Generate 2D relative offset vectors for an inverted V / wedge formation."""
    n = int(n)
    s = float(spacing)
    half_angle = np.radians(angle_deg / 2.0)
    offsets = np.zeros((n, 2), dtype=float)

    # Leader at tip (0, 0)
    for i in range(1, n):
        side = 1 if i % 2 == 1 else -1
        rank = (i + 1) // 2
        dx = -rank * s * np.cos(half_angle)
        dy = side * rank * s * np.sin(half_angle)
        offsets[i] = [dx, dy]

    return offsets - np.mean(offsets, axis=0)


# ------------------------------------------------------------------ FormationController

class ConsensusFormationController:
    r"""Distributed consensus-based multi-agent formation controller.

    Parameters
    ----------
    offsets:
        Desired relative formation geometry offsets :math:`d_i^*` of shape ``(n_agents, 2)``.
    kp:
        Relative position consensus gain :math:`k_p > 0`. Defaults to 2.0.
    kv:
        Relative velocity damping gain :math:`k_v \ge 0`. Defaults to 1.2.
    k_ref:
        Leader navigational reference tracking gain. Defaults to 1.0.
    kv_ref:
        Leader velocity reference feedforward gain. Defaults to 0.8.
    k_coll:
        Repulsive collision avoidance potential field gain. Defaults to 2.0.
    d_safe:
        Safety distance threshold for collision avoidance barrier [m]. Defaults to 0.6.
    u_min:
        Lower saturation limit on control input per coordinate. Defaults to -10.0.
    u_max:
        Upper saturation limit on control input per coordinate. Defaults to 10.0.
    pinned_agents:
        List of agent indices with access to navigational leader reference.
        Defaults to ``[0]`` (Agent 0 acts as leader).
    """

    def __init__(
        self,
        offsets: np.ndarray,
        *,
        kp: float = 2.0,
        kv: float = 1.2,
        k_ref: float = 1.0,
        kv_ref: float = 0.8,
        k_coll: float = 2.0,
        d_safe: float = 0.6,
        u_min: float | None = -10.0,
        u_max: float | None = 10.0,
        pinned_agents: Sequence[int] | None = None,
    ) -> None:
        self.offsets = np.asarray(offsets, dtype=float)
        if self.offsets.ndim != 2 or self.offsets.shape[1] != 2:
            raise ValueError(f"offsets must have shape (N, 2), got {self.offsets.shape}")

        self.n_agents = len(self.offsets)
        self.kp = float(kp)
        self.kv = float(kv)
        self.k_ref = float(k_ref)
        self.kv_ref = float(kv_ref)
        self.k_coll = float(k_coll)
        self.d_safe = float(d_safe)
        self.u_min = None if u_min is None else float(u_min)
        self.u_max = None if u_max is None else float(u_max)

        if pinned_agents is not None:
            if isinstance(pinned_agents, str) and pinned_agents.lower() == "all":
                self.pinned_agents = list(range(self.n_agents))
            else:
                self.pinned_agents = [int(idx) % self.n_agents for idx in pinned_agents]
        else:
            self.pinned_agents = list(range(self.n_agents))

    # ------------------------------------------------------------------ Metrics

    def formation_error(self, positions: np.ndarray) -> float:
        """Compute mean pairwise formation shape distortion error."""
        pos = np.asarray(positions, dtype=float).reshape((self.n_agents, 2))
        n = self.n_agents
        err_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                actual_rel = pos[i] - pos[j]
                target_rel = self.offsets[i] - self.offsets[j]
                err_sum += float(np.linalg.norm(actual_rel - target_rel))
                count += 1
        return float(err_sum / max(1, count))

    def formation_centroid(self, positions: np.ndarray) -> np.ndarray:
        """Return 2D Cartesian center of the current agent positions."""
        pos = np.asarray(positions, dtype=float).reshape((self.n_agents, 2))
        return np.mean(pos, axis=0)

    # ------------------------------------------------------------------ Control Law

    def compute_control(
        self,
        t: float,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        adjacency: np.ndarray | None = None,
        r_ref: np.ndarray | None = None,
        v_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute stacked control input vector for all :math:`N` agents."""
        pos = np.asarray(positions, dtype=float).reshape((self.n_agents, 2))
        if velocities is not None:
            vel = np.asarray(velocities, dtype=float).reshape((self.n_agents, 2))
        else:
            vel = np.zeros((self.n_agents, 2), dtype=float)

        if adjacency is not None:
            A = np.asarray(adjacency, dtype=float)
        else:
            A = np.ones((self.n_agents, self.n_agents)) - np.eye(self.n_agents)

        r0 = np.zeros(2, dtype=float) if r_ref is None else np.asarray(r_ref, dtype=float).reshape(2)
        v0 = np.zeros(2, dtype=float) if v_ref is None else np.asarray(v_ref, dtype=float).reshape(2)

        u_out = np.zeros((self.n_agents, 2), dtype=float)

        for i in range(self.n_agents):
            u_i = np.zeros(2, dtype=float)

            # 1. Distributed relative consensus with neighbors
            for j in range(self.n_agents):
                if i != j and A[i, j] > 0.0:
                    weight = float(A[i, j])
                    d_ij = self.offsets[i] - self.offsets[j]
                    # Relative position alignment
                    u_i -= self.kp * weight * (pos[i] - pos[j] - d_ij)
                    # Relative velocity damping
                    u_i -= self.kv * weight * (vel[i] - vel[j])

            # 2. Leader navigational pinning
            if i in self.pinned_agents:
                target_p_i = r0 + self.offsets[i]
                u_i -= self.k_ref * (pos[i] - target_p_i)
                u_i -= self.kv_ref * (vel[i] - v0)

            # 3. Inter-agent collision avoidance barrier
            if self.k_coll > 0.0 and self.d_safe > 0.0:
                for j in range(self.n_agents):
                    if i != j:
                        dist_ij = float(np.linalg.norm(pos[i] - pos[j]))
                        if 1e-6 < dist_ij < self.d_safe:
                            # Repulsive force directed away from agent j
                            rep_mag = self.k_coll * (1.0 / dist_ij - 1.0 / self.d_safe) * (1.0 / (dist_ij ** 2))
                            rep_dir = (pos[i] - pos[j]) / dist_ij
                            u_i += rep_mag * rep_dir

            u_out[i] = u_i

        # Apply actuator saturation limits
        if self.u_min is not None:
            u_out = np.maximum(u_out, self.u_min)
        if self.u_max is not None:
            u_out = np.minimum(u_out, self.u_max)

        return u_out.ravel()


# Convenient alias
FormationController = ConsensusFormationController
