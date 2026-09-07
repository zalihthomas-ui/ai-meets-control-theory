"""Tests for MultiAgentSystem and FormationController."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import (
    ConsensusFormationController,
    FormationController,
    diamond_formation,
    line_formation,
    polygon_formation,
    wedge_formation,
)
from aimct.simulate import rk4_step, simulate
from aimct.systems import (
    MultiAgent,
    MultiAgentSystem,
    complete_graph,
    cycle_graph,
    disconnected_graph,
    line_graph,
    star_graph,
)


# -------------------------------------------------- Graph Laplacian properties

def test_graph_laplacian_and_algebraic_connectivity():
    n = 5
    # Complete graph K_5
    A_comp = complete_graph(n)
    sys_comp = MultiAgentSystem(n_agents=n, adjacency=A_comp)
    L_comp = sys_comp.laplacian
    assert np.allclose(L_comp @ np.ones(n), np.zeros(n))
    assert sys_comp.is_connected
    assert sys_comp.algebraic_connectivity == pytest.approx(5.0)

    # Cycle graph C_5
    A_cyc = cycle_graph(n)
    sys_cyc = MultiAgentSystem(n_agents=n, adjacency=A_cyc)
    assert sys_cyc.is_connected
    assert sys_cyc.algebraic_connectivity > 0.5

    # Line graph P_5
    A_line = line_graph(n)
    sys_line = MultiAgentSystem(n_agents=n, adjacency=A_line)
    assert sys_line.is_connected
    assert sys_line.algebraic_connectivity > 0.1

    # Disconnected graph (isolated node 4)
    A_disc = disconnected_graph(n, isolated_nodes=[4])
    sys_disc = MultiAgentSystem(n_agents=n, adjacency=A_disc)
    assert not sys_disc.is_connected
    assert sys_disc.algebraic_connectivity == pytest.approx(0.0, abs=1e-5)


# -------------------------------------------------- Formation Geometry Generators

def test_formation_geometry_generators():
    # Polygon formation
    p4 = polygon_formation(4, radius=2.0)
    assert p4.shape == (4, 2)
    assert np.allclose(np.mean(p4, axis=0), [0.0, 0.0])
    # Equilateral pairwise distances
    d01 = np.linalg.norm(p4[0] - p4[1])
    d12 = np.linalg.norm(p4[1] - p4[2])
    assert d01 == pytest.approx(d12)

    # Diamond formation
    d4 = diamond_formation(radius=3.0)
    assert d4.shape == (4, 2)

    # Line formation
    l5 = line_formation(5, spacing=1.0, axis=0)
    assert l5.shape == (5, 2)
    assert np.allclose(l5[:, 1], 0.0)

    # Wedge formation
    w5 = wedge_formation(5, spacing=2.0, angle_deg=60.0)
    assert w5.shape == (5, 2)


# -------------------------------------------------- Formation Convergence (Double Integrator)

def test_formation_convergence_double_integrators():
    """N=4 agents in random initial positions converge to a target diamond formation."""
    n = 4
    offsets = diamond_formation(radius=2.0)
    sys = MultiAgentSystem(n_agents=n, agent_type="double_integrator", adjacency=complete_graph(n))
    ctrl = ConsensusFormationController(offsets=offsets, kp=3.0, kv=2.0, k_ref=1.0, kv_ref=1.0)

    # Initial scattered positions with zero velocity
    rng = np.random.default_rng(42)
    p0 = rng.uniform(-10.0, 10.0, size=(n, 2))
    v0 = np.zeros((n, 2))
    x0 = np.hstack([p0, v0]).ravel()

    r_target = np.array([5.0, 5.0])
    dt = 0.05
    x = x0.copy()

    for k in range(200):
        pos = sys.get_agent_positions(x)
        vel = sys.get_agent_velocities(x)
        u = ctrl.compute_control(k * dt, pos, vel, sys.adjacency, r_ref=r_target)
        x = rk4_step(lambda t, xx, uu: sys.dynamics(t, xx, uu), k * dt, x, u, dt)

    # Verify final formation shape error is small (< 0.1 m)
    final_pos = sys.get_agent_positions(x)
    form_err = ctrl.formation_error(final_pos)
    assert form_err < 0.15

    # Verify formation centroid reached target [5.0, 5.0]
    centroid = ctrl.formation_centroid(final_pos)
    assert np.allclose(centroid, r_target, atol=0.25)


# -------------------------------------------------- Switching Topology Stability

def test_formation_switching_topology_stability():
    """Formation remains stable and converges even as graph switches between Complete -> Cycle -> Line."""
    n = 4
    offsets = polygon_formation(4, radius=2.5)
    sys = MultiAgentSystem(n_agents=n, agent_type="double_integrator")
    ctrl = FormationController(offsets=offsets, kp=2.5, kv=1.8, k_ref=0.8, kv_ref=0.8)

    topologies = [
        complete_graph(n),
        cycle_graph(n),
        line_graph(n),
        star_graph(n),
    ]

    p0 = np.array([
        [0.0, 0.0],
        [1.0, 5.0],
        [-5.0, 2.0],
        [3.0, -4.0],
    ])
    v0 = np.zeros((n, 2))
    x = np.hstack([p0, v0]).ravel()

    dt = 0.05
    # Run for 8 seconds, switching topology every 2 seconds
    for k in range(160):
        t = k * dt
        topo_idx = int(t / 2.0) % len(topologies)
        sys.set_topology(topologies[topo_idx])

        pos = sys.get_agent_positions(x)
        vel = sys.get_agent_velocities(x)
        u = ctrl.compute_control(t, pos, vel, sys.adjacency, r_ref=[0.0, 0.0])
        x = rk4_step(lambda t, xx, uu: sys.dynamics(t, xx, uu), t, x, u, dt)

    final_pos = sys.get_agent_positions(x)
    assert ctrl.formation_error(final_pos) < 0.20


# -------------------------------------------------- Collision Avoidance Barrier

def test_collision_avoidance_barrier():
    """Two agents launched directly towards each other repel and avoid collision."""
    n = 2
    offsets = line_formation(2, spacing=4.0)
    sys = MultiAgentSystem(n_agents=n, agent_type="double_integrator", adjacency=complete_graph(n))
    ctrl = FormationController(
        offsets=offsets, kp=1.0, kv=0.5, k_coll=10.0, d_safe=1.0, u_min=-15.0, u_max=15.0
    )

    # Initial conditions on collision course
    p0 = np.array([[-2.0, 0.0], [2.0, 0.0]])
    v0 = np.array([[2.0, 0.0], [-2.0, 0.0]])
    x = np.hstack([p0, v0]).ravel()

    dt = 0.02
    min_dist = np.inf
    for k in range(100):
        pos = sys.get_agent_positions(x)
        vel = sys.get_agent_velocities(x)
        d_curr = sys.min_pairwise_distance(x)
        min_dist = min(min_dist, d_curr)

        u = ctrl.compute_control(k * dt, pos, vel, sys.adjacency)
        x = rk4_step(lambda t, xx, uu: sys.dynamics(t, xx, uu), k * dt, x, u, dt)

    # Pairwise distance should never drop to zero (barrier repels before impact)
    assert min_dist > 0.25
