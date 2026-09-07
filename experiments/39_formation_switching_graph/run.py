"""Experiment 39 - Multi-Agent Formation Tracking under Switching Graph Topologies.

Evaluates distributed consensus formation control of N=5 double-integrator agents
under dynamic communication network switching:
  1. Complete Graph K5 (Full Mesh, lambda2 = 5.00)
  2. Cycle Graph C5 (Ring, lambda2 = 1.38)
  3. Line Path Graph P5 (Linear Chain, lambda2 = 0.38)
  4. Disconnected Graph (Link Drop / Isolated Agent, lambda2 = 0.00)
  5. Star Graph S5 (Central Hub Recovery, lambda2 = 1.00)

Run:
    python experiments/39_formation_switching_graph/run.py
Outputs:
    table.md, table.csv, formation_switching.png, formation_switching.svg, figure.png
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aimct.controllers import ConsensusFormationController, polygon_formation
from aimct.plot_style import set_aimct_style
from aimct.simulate import rk4_step
from aimct.systems import (
    MultiAgentSystem,
    complete_graph,
    cycle_graph,
    disconnected_graph,
    line_graph,
    star_graph,
)

HERE = Path(__file__).parent


def run_experiment():
    full_mode = bool(os.getenv("AIMCT_EXP_FULL", "0") == "1")
    dt = 0.02 if full_mode else 0.05
    t_final = 40.0
    steps = int(round(t_final / dt))

    n_agents = 5
    formation_radius = 3.0
    offsets = polygon_formation(n_agents, radius=formation_radius)

    sys = MultiAgentSystem(n_agents=n_agents, agent_type="double_integrator", damping=0.1)
    ctrl = ConsensusFormationController(
        offsets=offsets,
        kp=2.8,
        kv=2.0,
        k_ref=1.2,
        kv_ref=1.0,
        k_coll=4.0,
        d_safe=1.0,
        u_min=-12.0,
        u_max=12.0,
    )

    # Leader sinusoidal trajectory
    vx_leader = 1.0
    amp_y = 4.0
    freq_y = 0.15

    def get_leader_ref(t: float) -> tuple[np.ndarray, np.ndarray]:
        r0 = np.array([vx_leader * t, amp_y * np.sin(freq_y * t)])
        v0 = np.array([vx_leader, amp_y * freq_y * np.cos(freq_y * t)])
        return r0, v0

    # Switching topology schedule
    def get_topology(t: float) -> tuple[np.ndarray, str, float]:
        if t < 8.0:
            return complete_graph(n_agents), "Complete ($K_5$)", 5.0
        elif t < 18.0:
            return cycle_graph(n_agents), "Cycle ($C_5$)", 1.382
        elif t < 26.0:
            return line_graph(n_agents), "Line ($P_5$)", 0.382
        elif t < 32.0:
            return disconnected_graph(n_agents, isolated_nodes=[4]), "Disconnected (Link Loss)", 0.0
        else:
            return star_graph(n_agents, center=0), "Star ($S_5$ Hub)", 1.0

    # Initial condition: scattered starting positions around origin
    rng = np.random.default_rng(123)
    p0 = rng.uniform(-6.0, 6.0, size=(n_agents, 2))
    v0 = np.zeros((n_agents, 2))
    x_curr = np.hstack([p0, v0]).ravel()

    # Data logging
    t_hist = np.empty(steps)
    x_hist = np.empty((steps, n_agents * 4))
    pos_hist = np.empty((steps, n_agents, 2))
    lambda2_hist = np.empty(steps)
    form_err_hist = np.empty(steps)
    centroid_err_hist = np.empty(steps)
    min_dist_hist = np.empty(steps)
    u_energy_hist = np.empty(steps)
    phase_names = []

    snapshot_times = [2.0, 12.0, 22.0, 29.0, 38.0]
    snapshots: list[tuple[float, np.ndarray, np.ndarray]] = []

    latencies = []

    for k in range(steps):
        t_curr = k * dt
        t_hist[k] = t_curr

        A_curr, phase_name, _ = get_topology(t_curr)
        sys.set_topology(A_curr)
        lambda2_hist[k] = sys.algebraic_connectivity
        phase_names.append(phase_name)

        r0, v0 = get_leader_ref(t_curr)

        pos = sys.get_agent_positions(x_curr)
        vel = sys.get_agent_velocities(x_curr)

        pos_hist[k] = pos
        x_hist[k] = x_curr

        # Control step
        t0 = time.perf_counter()
        u = ctrl.compute_control(t_curr, pos, vel, sys.adjacency, r_ref=r0, v_ref=v0)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        u_energy_hist[k] = np.sum(u ** 2)

        # Record metrics
        form_err_hist[k] = ctrl.formation_error(pos)
        centroid = ctrl.formation_centroid(pos)
        centroid_err_hist[k] = float(np.linalg.norm(centroid - r0))
        min_dist_hist[k] = sys.min_pairwise_distance(x_curr)

        # Snapshot capture
        for st in snapshot_times:
            if abs(t_curr - st) < dt / 2.0 and len([s for s in snapshots if abs(s[0] - st) < 0.1]) == 0:
                snapshots.append((t_curr, pos.copy(), r0.copy()))

        # System RK4 step
        x_curr = rk4_step(lambda t, xx, uu: sys.dynamics(t, xx, uu), t_curr, x_curr, u, dt)

    # ------------------------------------------------------------------ Metrics by Phase
    phases_info = [
        ("Phase 1: Complete Mesh (K5)", 0.0, 8.0, 5.0),
        ("Phase 2: Ring Topology (C5)", 8.0, 18.0, 1.38),
        ("Phase 3: Line Chain (P5)", 18.0, 26.0, 0.38),
        ("Phase 4: Disconnected Link Loss", 26.0, 32.0, 0.0),
        ("Phase 5: Star Hub Recovery (S5)", 32.0, 40.0, 1.0),
    ]

    phase_metrics = []
    for name, t_start, t_end, lam_nom in phases_info:
        mask = (t_hist >= t_start) & (t_hist < t_end)
        mean_form_err = float(np.mean(form_err_hist[mask]))
        final_form_err = float(form_err_hist[mask][-1])
        mean_cent_err = float(np.mean(centroid_err_hist[mask]))
        min_p_dist = float(np.min(min_dist_hist[mask]))
        phase_metrics.append({
            "phase": name,
            "lambda2_nom": lam_nom,
            "mean_form_err_m": mean_form_err,
            "final_form_err_m": final_form_err,
            "mean_cent_err_m": mean_cent_err,
            "min_dist_m": min_p_dist,
        })

    # Overall Summary Table
    headers = [
        "Communication Phase",
        "lambda2(L)",
        "Mean Form. Error [m]",
        "Final Form. Error [m]",
        "Centroid Error [m]",
        "Min Separation [m]",
        "Safety Status",
    ]

    rows = []
    for pm in phase_metrics:
        safe_str = "Safe (d > d_safe)" if pm["min_dist_m"] >= 0.95 * ctrl.d_safe else "Proximity Alert"
        rows.append([
            pm["phase"],
            f"{pm['lambda2_nom']:.2f}",
            f"{pm['mean_form_err_m']:.3f}",
            f"{pm['final_form_err_m']:.3f}",
            f"{pm['mean_cent_err_m']:.3f}",
            f"{pm['min_dist_m']:.3f}",
            safe_str,
        ])

    # Write CSV
    with open(HERE / "table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Write Markdown Table
    md_content = f"""# Experiment 39 Benchmark Results: Multi-Agent Formation Control under Switching Topologies

| {' | '.join(headers)} |
| {' | '.join([':---' if i == 0 or i == len(headers)-1 else ':---:' for i in range(len(headers))])} |
"""
    for r in rows:
        md_content += f"| {' | '.join(r)} |\n"

    with open(HERE / "table.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Generated table.csv and table.md successfully.")

    # ------------------------------------------------------------------ Plots
    set_aimct_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax_traj, ax_err = axes[0, 0], axes[0, 1]
    ax_lam, ax_dist = axes[1, 0], axes[1, 1]

    agent_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    # (a) 2D Multi-Agent Trajectories & Formation Snapshots
    leader_traj = np.array([get_leader_ref(t)[0] for t in t_hist])
    ax_traj.plot(leader_traj[:, 0], leader_traj[:, 1], "k--", linewidth=1.5, label="Virtual Leader Reference $r_0(t)$", zorder=2)

    for i in range(n_agents):
        ax_traj.plot(
            pos_hist[:, i, 0],
            pos_hist[:, i, 1],
            color=agent_colors[i],
            linewidth=1.8,
            label=f"Agent {i+1}",
            zorder=3,
        )
        ax_traj.scatter(pos_hist[0, i, 0], pos_hist[0, i, 1], color=agent_colors[i], marker="o", s=30, zorder=4)

    # Plot snapshot polygons
    for st, p_snap, r_snap in snapshots:
        # Close polygon for plotting
        poly_pts = np.vstack([p_snap, p_snap[0]])
        ax_traj.plot(poly_pts[:, 0], poly_pts[:, 1], color="#333333", linestyle=":", linewidth=1.2, alpha=0.7, zorder=5)
        ax_traj.scatter(r_snap[0], r_snap[1], marker="*", color="#000000", s=60, zorder=6)
        ax_traj.text(r_snap[0], r_snap[1] + 1.2, f"$t={st:.0f}\\mathrm{{s}}$", fontsize=8, ha="center", weight="bold")

    ax_traj.set_title("(a) 2D Formation Navigation & Pentagon Snapshots")
    ax_traj.set_xlabel("X Position [m]")
    ax_traj.set_ylabel("Y Position [m]")
    ax_traj.legend(loc="upper left", fontsize=8)

    # (b) Formation Distortion Error Over Time
    ax_err.plot(t_hist, form_err_hist, color="#0072B2", linewidth=2.0, label=r"Formation Shape Error $e_{\mathrm{form}}(t)$")
    ax_err.plot(t_hist, centroid_err_hist, color="#D55E00", linestyle="--", linewidth=1.6, label=r"Centroid Tracking Error $\|\bar{p} - r_0\|$")
    for _, t_s, _, _ in phases_info[1:]:
        ax_err.axvline(t_s, color="#999999", linestyle=":", linewidth=1.0)
    ax_err.set_title(r"(b) Formation Shape Distortion & Centroid Error")
    ax_err.set_xlabel("Time $t$ [s]")
    ax_err.set_ylabel("Error [m]")
    ax_err.legend(loc="upper right", fontsize=8)

    # (c) Algebraic Connectivity lambda2(L(t)) & Topology Switches
    ax_lam.step(t_hist, lambda2_hist, color="#009E73", linewidth=2.2, where="post", label=r"Fiedler Eigenvalue $\lambda_2(L(t))$")
    ax_lam.axhline(0.0, color="#D62728", linestyle=":", linewidth=1.2, label=r"Disconnection Boundary $\lambda_2 = 0$")
    for name, t_s, t_e, _ in phases_info:
        mid_t = (t_s + min(t_e, t_final)) / 2.0
        ax_lam.text(mid_t, max(lambda2_hist) * 0.88, name.split(":")[0], fontsize=7.5, ha="center", rotation=0, weight="bold")
        if t_s > 0:
            ax_lam.axvline(t_s, color="#999999", linestyle=":", linewidth=1.0)
    ax_lam.set_title(r"(c) Algebraic Connectivity $\lambda_2(L)$ & Switching Topologies")
    ax_lam.set_xlabel("Time $t$ [s]")
    ax_lam.set_ylabel(r"$\lambda_2(L)$ [Connectivity]")
    ax_lam.set_ylim(-0.2, 5.5)
    ax_lam.legend(loc="upper right", fontsize=8)

    # (d) Inter-Agent Minimum Pairwise Distance & Safety
    ax_dist.plot(t_hist, min_dist_hist, color="#CC79A7", linewidth=2.0, label=r"Min Pairwise Distance $\min_{i \neq j} \|p_i - p_j\|$")
    ax_dist.axhline(ctrl.d_safe, color="#D62728", linestyle="--", linewidth=1.5, label=f"Safety Margin $d_{{\\mathrm{{safe}}}} = {ctrl.d_safe}\\mathrm{{ m}}$")
    ax_dist.fill_between(t_hist, 0.0, ctrl.d_safe, color="#D62728", alpha=0.08, label="Collision Danger Zone")
    for _, t_s, _, _ in phases_info[1:]:
        ax_dist.axvline(t_s, color="#999999", linestyle=":", linewidth=1.0)
    ax_dist.set_title(r"(d) Inter-Agent Minimum Separation & Collision Safety")
    ax_dist.set_xlabel("Time $t$ [s]")
    ax_dist.set_ylabel("Distance [m]")
    ax_dist.set_ylim(0.0, max(min_dist_hist) * 1.2)
    ax_dist.legend(loc="upper right", fontsize=8)

    fig.suptitle("Experiment 39: Multi-Agent Formation Control under Switching Graph Topologies", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    fig.savefig(HERE / "formation_switching.png", dpi=300)
    fig.savefig(HERE / "formation_switching.svg")
    fig.savefig(HERE / "figure.png", dpi=300)
    plt.close(fig)
    print("Saved formation_switching.png, formation_switching.svg, figure.png successfully.")


if __name__ == "__main__":
    run_experiment()
