"""Plotting and visualization utilities for AI Meets Control Theory.

Provides standardized color palettes, line styles, figure geometry helpers,
and multi-panel comparison plotting adhering to docs/plotting-style-guide.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe Okabe-Ito palette and semantic mapping
PALETTE: Dict[str, str] = {
    "reference": "#555555",       # Neutral Dark Grey
    "pid": "#0072B2",             # Deep Blue
    "state_feedback": "#009E73",  # Bluish Green
    "lqr": "#D55E00",             # Vermilion / Rust Orange
    "mpc": "#CC79A7",             # Reddish Purple
    "rl": "#E69F00",              # Amber / Gold
    "hybrid": "#56B4E9",          # Sky Blue
    "saturation": "#D62728",      # Crimson / Limit Red
    "background_band": "#000000", # Shading envelope
}

CONTROLLER_STYLES: Dict[str, Dict[str, Any]] = {
    "reference": {
        "color": PALETTE["reference"],
        "linestyle": "--",
        "linewidth": 1.5,
        "label": "Reference $r(t)$",
        "zorder": 1,
    },
    "pid": {
        "color": PALETTE["pid"],
        "linestyle": "-",
        "linewidth": 2.0,
        "label": "PID",
        "zorder": 4,
    },
    "state_feedback": {
        "color": PALETTE["state_feedback"],
        "linestyle": "-.",
        "linewidth": 2.0,
        "label": "State Feedback",
        "zorder": 3,
    },
    "lqr": {
        "color": PALETTE["lqr"],
        "linestyle": "-",
        "linewidth": 2.2,
        "label": "LQR",
        "zorder": 5,
    },
    "mpc": {
        "color": PALETTE["mpc"],
        "linestyle": "--",
        "linewidth": 2.2,
        "label": "MPC",
        "zorder": 6,
    },
    "rl": {
        "color": PALETTE["rl"],
        "linestyle": "-",
        "linewidth": 1.8,
        "label": "RL (PPO)",
        "zorder": 2,
    },
    "hybrid": {
        "color": PALETTE["hybrid"],
        "linestyle": ":",
        "linewidth": 2.2,
        "label": "Hybrid",
        "zorder": 7,
    },
}


def set_aimct_style() -> None:
    """Apply standard matplotlib rcParams for clear, publication-grade figures."""
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#D0D0D0",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def get_controller_style(name: str) -> Dict[str, Any]:
    """Return styling dict for a given controller name (case-insensitive lookup)."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key in CONTROLLER_STYLES:
        return CONTROLLER_STYLES[key].copy()
    # Fallback to default cyclic styling
    return {"linewidth": 1.8, "label": name}


def plot_benchmark_comparison(
    t: np.ndarray,
    reference: np.ndarray,
    trajectories: Dict[str, Dict[str, np.ndarray]],
    title: str = "Controller Benchmark Comparison",
    state_label: str = "State $x(t)$ [m]",
    control_label: str = "Control Input $u(t)$ [N]",
    u_limits: Optional[Tuple[float, float]] = None,
    tolerance_pct: float = 0.02,
    figsize: Tuple[float, float] = (12.0, 8.0),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generate canonical 4-panel benchmark comparison figure.

    Panels:
    (a) Output Tracking y(t) vs r(t)
    (b) Control Effort u(t)
    (c) Tracking Error e(t) = r(t) - y(t)
    (d) Phase Portrait / Error derivative

    Parameters
    ----------
    t : np.ndarray
        Time vector of shape (N,).
    reference : np.ndarray
        Reference target signal of shape (N,).
    trajectories : dict
        Mapping controller_name -> dict with keys:
            'state': ndarray of shape (N,) or (N, n_states) (first state used for output tracking)
            'input': ndarray of shape (N,) or (N, n_inputs)
            'state_dot': optional ndarray of shape (N,) for phase portrait
    title : str
        Figure super title.
    state_label : str
        Label and unit for output tracking plot.
    control_label : str
        Label and unit for control effort plot.
    u_limits : tuple of (u_min, u_max), optional
        Actuator saturation limits.
    tolerance_pct : float
        Settling tolerance band (default 0.02 = 2%).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, axes : Figure and 2x2 array of Axes
    """
    set_aimct_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_track, ax_ctrl = axes[0, 0], axes[0, 1]
    ax_err, ax_phase = axes[1, 0], axes[1, 1]

    # Reference signal in tracking plot
    ax_track.plot(
        t, reference,
        color=PALETTE["reference"],
        linestyle="--",
        linewidth=1.6,
        label="Target $r(t)$",
        zorder=1,
    )

    # Settling band
    ref_final = reference[-1] if len(reference) > 0 else 1.0
    band = abs(ref_final) * tolerance_pct if abs(ref_final) > 1e-6 else tolerance_pct
    ax_track.axhspan(
        ref_final - band, ref_final + band,
        color=PALETTE["background_band"],
        alpha=0.06,
        label=f"$\\pm {int(tolerance_pct*100)}\\%$ Settling Band",
        zorder=0,
    )

    # Zero line for error
    ax_err.axhline(0.0, color=PALETTE["reference"], linestyle="--", linewidth=1.2, zorder=1)

    # Actuator limits
    if u_limits is not None:
        u_min, u_max = u_limits
        ax_ctrl.axhline(u_max, color=PALETTE["saturation"], linestyle=":", linewidth=1.5, label="Limit $u_{\\max}$")
        ax_ctrl.axhline(u_min, color=PALETTE["saturation"], linestyle=":", linewidth=1.5)
        ax_ctrl.fill_between(t, u_max, u_max * 1.2, color=PALETTE["saturation"], alpha=0.08)
        ax_ctrl.fill_between(t, u_min * 1.2, u_min, color=PALETTE["saturation"], alpha=0.08)

    # Plot each controller
    for name, data in trajectories.items():
        style = get_controller_style(name)
        state = data["state"]
        y = state[:, 0] if state.ndim > 1 else state
        u = data["input"]
        u_val = u[:, 0] if u.ndim > 1 else u
        err = reference - y

        # (a) Output tracking
        ax_track.plot(t, y, **style)

        # (b) Control effort
        ctrl_style = {k: v for k, v in style.items() if k != "label"}
        ctrl_style["label"] = name
        ax_ctrl.plot(t, u_val, **ctrl_style)

        # (c) Error
        ax_err.plot(t, err, **ctrl_style)

        # (d) Phase portrait
        if "state_dot" in data:
            ydot = data["state_dot"]
        elif state.ndim > 1 and state.shape[1] >= 2:
            ydot = state[:, 1]
        else:
            ydot = np.gradient(y, t)
        ax_phase.plot(y, ydot, **ctrl_style)
        ax_phase.scatter(y[0], ydot[0], marker="o", color=style.get("color", "#333333"), s=30, zorder=6)

    # Equilibrium in phase space
    ax_phase.scatter(ref_final, 0.0, marker="*", color="#000000", s=100, label="Equilibrium", zorder=7)

    # Labels and Titles
    ax_track.set_title("(a) Output Tracking $y(t)$ vs $r(t)$")
    ax_track.set_xlabel("Time $t$ [s]")
    ax_track.set_ylabel(state_label)
    ax_track.legend(loc="best")

    ax_ctrl.set_title("(b) Control Effort $u(t)$")
    ax_ctrl.set_xlabel("Time $t$ [s]")
    ax_ctrl.set_ylabel(control_label)
    ax_ctrl.legend(loc="best")

    ax_err.set_title("(c) Tracking Error $e(t) = r(t) - y(t)$")
    ax_err.set_xlabel("Time $t$ [s]")
    ax_err.set_ylabel("Error $e(t)$")
    ax_err.legend(loc="best")

    ax_phase.set_title("(d) Phase Portrait $(y, \\dot{y})$")
    ax_phase.set_xlabel(state_label)
    ax_phase.set_ylabel("Rate $\\dot{y}(t)$")
    ax_phase.legend(loc="best")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    return fig, axes
