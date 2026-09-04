r"""Trajectory-tracking benchmark - the path-following counterpart of
:func:`aimct.benchmarks.compare` (which is step-response oriented).

    from aimct.trajectories import Lemniscate
    from aimct.benchmarks.tracking import track_trajectory
    res = track_trajectory(quad, {"LQR+ff": ctrl_a, "MPC": ctrl_b},
                           Lemniscate(0.6, 0.35, 6.0), x0, dt=0.01, t_final=12,
                           pos_index=(0, 1))
    print(res.to_markdown())

Each controller must already be configured to *follow* ``trajectory`` (hold its
own clock, like the trackers in Experiments 14/20/21); this scores the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from ..simulate import simulate
from ..trajectories import Trajectory

__all__ = ["track_trajectory", "TrackingResult"]


@dataclass
class TrackingResult:
    metrics: dict = field(default_factory=dict)          # name -> {metric: value}
    trajectories: dict = field(default_factory=dict)     # name -> aimct Trajectory
    trajectory: Trajectory | None = None
    pos_index: tuple = (0, 1)
    title: str = "Trajectory tracking"

    _COLS = ("rms_err_mm", "max_err_mm", "rms_cross_track_mm",
             "completion_pct", "ctrl_energy", "status")

    def to_markdown(self) -> str:
        head = "| controller | " + " | ".join(self._COLS) + " |"
        sep = "| --- |" + " --- |" * len(self._COLS)
        rows = []
        for name, m in self.metrics.items():
            rows.append("| " + name + " | " + " | ".join(
                (m[c] if c == "status" else f"{m[c]:.4g}") for c in self._COLS) + " |")
        return "\n".join([f"# {self.title}", "", head, sep, *rows]) + "\n"

    def to_csv(self) -> str:
        import io
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["controller", *self._COLS])
        for name, m in self.metrics.items():
            w.writerow([name, *(m[c] for c in self._COLS)])
        return buf.getvalue()

    def figure(self, **kw):
        import matplotlib.pyplot as plt
        from ..plot_style import PALETTE, set_aimct_style

        set_aimct_style()
        ix, iy = self.pos_index
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        traj = self.trajectory
        ts = np.linspace(0.0, traj.duration, 400)
        ref = np.array([traj.pos(t) for t in ts])
        ax[0].plot(ref[:, 0], ref[:, 1], "--", color=PALETTE["reference"],
                   lw=1.4, label="reference")
        cyc = [PALETTE["lqr"], PALETTE["mpc"], PALETTE["state_feedback"],
               PALETTE["rl"], PALETTE["hybrid"]]
        for i, (name, tr) in enumerate(self.trajectories.items()):
            c = cyc[i % len(cyc)]
            ax[0].plot(tr.x[:, ix], tr.x[:, iy], color=c, lw=1.6, label=name)
            e = np.array([np.linalg.norm(p - traj.closest(p)[0])
                          for p in tr.x[:, [ix, iy]]]) * 1e3
            ax[1].plot(tr.t, e, color=c, lw=1.3, label=name)
        ax[0].set(title="(a) path", xlabel="x [m]", ylabel="y [m]")
        ax[0].set_aspect("equal", "box"); ax[0].legend(fontsize=8)
        ax[1].set(title="(b) cross-track error [mm]", xlabel="t [s]", ylabel="mm")
        ax[1].legend(fontsize=8)
        fig.suptitle(self.title, fontweight="bold")
        fig.tight_layout()
        return fig, ax

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tracking.md").write_text(self.to_markdown(), encoding="utf-8")
        (out / "tracking.csv").write_text(self.to_csv(), encoding="utf-8")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, _ = self.figure()
            fig.savefig(out / "tracking.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass


def track_trajectory(
    system,
    controllers: Mapping[str, object],
    trajectory: Trajectory,
    x0,
    *,
    dt: float,
    t_final: float,
    pos_index=(0, 1),
    u_bounds=None,
    title: str = "Trajectory tracking",
) -> TrackingResult:
    ix, iy = pos_index
    res = TrackingResult(trajectory=trajectory, pos_index=pos_index, title=title)
    for name, ctrl in controllers.items():
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        tr = simulate(system, ctrl, x0=np.asarray(x0, float), dt=dt,
                      t_final=t_final, u_bounds=u_bounds)
        P = tr.x[:, [ix, iy]]
        ref = np.array([trajectory.pos(min(t, trajectory.duration))[:2] for t in tr.t])
        err = np.linalg.norm(P - ref, axis=1)
        cross = np.array([np.linalg.norm(p - trajectory.closest(p)[0]) for p in P])
        completion = trajectory.closest(P[-1])[2] * 100.0
        du = tr.u - (system.u_hover if hasattr(system, "u_hover") else 0.0)
        energy = float(np.trapezoid(np.sum(np.atleast_2d(du) ** 2, axis=1), tr.t))
        res.metrics[name] = {
            "rms_err_mm": float(np.sqrt(np.mean(err ** 2)) * 1e3),
            "max_err_mm": float(np.max(err) * 1e3),
            "rms_cross_track_mm": float(np.sqrt(np.mean(cross ** 2)) * 1e3),
            "completion_pct": float(min(completion, 100.0)),
            "ctrl_energy": energy,
            "status": "Diverged" if tr.diverged else "OK",
        }
        res.trajectories[name] = tr
    return res
