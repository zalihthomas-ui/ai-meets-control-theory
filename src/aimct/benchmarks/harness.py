r"""Multi-controller comparison harness.

``compare(system, controllers, x0, dt, t_final, reference, ...)`` rolls every
controller on the *same* system, initial condition, reference, disturbance and
actuator limits (the fairness conditions of ``docs/comparison-report-spec.md``
§6), scores each rollout with :func:`aimct.benchmarks.metrics.compute_all_metrics`,
and returns a :class:`ComparisonResult` that can emit

* the canonical metrics table (Markdown + CSV, spec §3), and
* the canonical 4-panel comparison figure (spec §4) via
  :func:`aimct.plot_style.plot_benchmark_comparison`.

The harness only *scores and plots*; it does not configure the controllers.
Give each controller its set-point / reference state before passing it in, and
pass the same target here as ``reference`` so the metrics and figure agree.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ..simulate import Trajectory, simulate
from ..systems.base import DynamicalSystem
from .metrics import compute_all_metrics

__all__ = ["compare", "ComparisonResult", "SPEC_COLUMNS"]


# spec §3.1 column order:  (header, metrics-key, printf format)
SPEC_COLUMNS: list[tuple[str, str, str]] = [
    ("Rise $t_r$ [s]", "rise_time", "{:.3g}"),
    ("Settling $t_s$ [s]", "settling_time", "{:.3g}"),
    ("Overshoot $M_p$ [%]", "peak_overshoot_pct", "{:.3g}"),
    ("Steady error $e_{ss}$", "steady_state_error", "{:.3g}"),
    ("RMSE", "rmse", "{:.3g}"),
    ("Energy $E_u$", "control_energy", "{:.3g}"),
    ("Peak $u_{max}$", "peak_control", "{:.3g}"),
    ("Saturation [%]", "saturation_pct", "{:.3g}"),
]


ReferenceLike = float | Sequence[float] | np.ndarray | Callable[[float], float]


def _as_reference_array(reference: ReferenceLike, t: np.ndarray) -> np.ndarray:
    """Normalise ``reference`` to a float array aligned with ``t``."""
    if callable(reference):
        return np.array([float(reference(tk)) for tk in t], dtype=float)
    arr = np.asarray(reference, dtype=float)
    if arr.ndim == 0:
        return np.full(t.shape, float(arr))
    if arr.shape != t.shape:
        raise ValueError(
            f"reference array has shape {arr.shape}, expected {t.shape} (one per sample)"
        )
    return arr


def _pad_nan(a: np.ndarray, n: int) -> np.ndarray:
    """Right-pad ``a`` along axis 0 with NaN up to length ``n`` (no-op if already
    that long)."""
    if len(a) >= n:
        return a[:n]
    pad = np.full((n - len(a),) + a.shape[1:], np.nan)
    return np.concatenate([a, pad], axis=0)


def _finite_prefix_len(x: np.ndarray) -> int:
    """Number of leading rows of ``x`` that are entirely finite (``simulate``
    keeps one non-finite blow-up sample on a divergent run)."""
    finite_row = np.all(np.isfinite(x), axis=tuple(range(1, x.ndim)))
    bad = np.flatnonzero(~finite_row)
    return int(bad[0]) if bad.size else len(x)


def _classify_status(traj: Trajectory, m: Mapping[str, float], t_final: float,
                     dt: float, ref_scale: float, n_ok: int) -> str:
    """`Diverged` (blew up) / `Stable` (settled in-horizon) / `Marginal` (bounded,
    never settled)."""
    if traj.diverged or n_ok < len(traj.x):
        return "Diverged"
    if np.max(np.abs(traj.x)) > 1e3 * ref_scale:
        return "Diverged"
    if np.isfinite(m["settling_time"]) and m["settling_time"] < t_final - dt:
        return "Stable"
    return "Marginal"


@dataclass
class ComparisonResult:
    """Outcome of :func:`compare` for one system and N controllers."""

    system_name: str
    t: np.ndarray
    reference: np.ndarray
    output_index: int
    x0: np.ndarray
    dt: float
    t_final: float
    u_bounds: tuple[float, float] | None
    names: list[str]
    trajectories: dict[str, Trajectory]
    metrics: dict[str, dict[str, float]]
    status: dict[str, str]
    deriv_index: int | None = None
    title: str = "Controller Benchmark Comparison"
    _extras: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ tables

    def rows(self, columns=SPEC_COLUMNS) -> list[list[str]]:
        """Formatted cell strings, one row per controller (spec column order)."""
        out = []
        for name in self.names:
            m = self.metrics[name]
            cells = [name]
            for _, key, fmt in columns:
                v = m.get(key)
                if v is None:
                    cells.append("--")
                elif isinstance(v, float) and not np.isfinite(v):
                    cells.append("inf")
                else:
                    cells.append(fmt.format(v))
            cells.append(self.status[name])
            out.append(cells)
        return out

    def to_markdown(self, columns=SPEC_COLUMNS) -> str:
        head = ["Controller", *(h for h, _, _ in columns), "Status"]
        align = [":---", *[":---:"] * (len(head) - 1)]
        lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(align) + " |"]
        for row in self.rows(columns):
            row = [f"**{row[0]}**", *row[1:]]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) + "\n"

    def to_csv(self, columns=SPEC_COLUMNS) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["controller", *(key for _, key, _ in columns), "status"])
        for name in self.names:
            m = self.metrics[name]
            w.writerow(
                [name, *(m.get(key, "") for _, key, _ in columns), self.status[name]]
            )
        return buf.getvalue()

    def full_metrics_csv(self) -> str:
        """Every metric :func:`compute_all_metrics` returned, not just spec columns."""
        keys = list(next(iter(self.metrics.values())).keys())
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["controller", *keys])
        for name in self.names:
            w.writerow([name, *(self.metrics[name].get(k, "") for k in keys)])
        return buf.getvalue()

    # ------------------------------------------------------------------ figure

    def _plot_trajectories(self) -> dict[str, dict[str, np.ndarray]]:
        di = self.deriv_index
        n_full = len(self.t)
        data: dict[str, dict[str, np.ndarray]] = {}
        for name in self.names:
            traj = self.trajectories[name]
            n_ok = _finite_prefix_len(traj.x)
            y = traj.y[:n_ok, self.output_index]
            if di is not None and 0 <= di < traj.x.shape[1]:
                ydot = traj.x[:n_ok, di]
            else:
                ydot = np.gradient(y, traj.t[:n_ok]) if n_ok > 1 else np.zeros(n_ok)
            u = traj.u[:n_ok]
            # pad a truncated (divergent) run with NaN so its lines stop cleanly
            data[name] = {
                "state": np.column_stack([_pad_nan(y, n_full), _pad_nan(ydot, n_full)]),
                "input": _pad_nan(u, n_full),
            }
        return data

    def figure(self, *, title: str | None = None, state_label: str = "Output $y(t)$",
               control_label: str = "Control $u(t)$", **kwargs):
        """Build the canonical 4-panel figure. Returns ``(fig, axes)``.

        Requires matplotlib; imported lazily so the rest of the harness works
        without it.
        """
        from ..plot_style import plot_benchmark_comparison

        return plot_benchmark_comparison(
            self.t,
            self.reference,
            self._plot_trajectories(),
            title=title or self.title,
            state_label=state_label,
            control_label=control_label,
            u_limits=self.u_bounds,
            **kwargs,
        )

    # ------------------------------------------------------------------- save

    def save(self, outdir: str | Path, *, stem: str = "comparison_benchmark",
             make_figure: bool = True, figure_formats: Sequence[str] = ("png", "svg"),
             figure_kwargs: dict | None = None) -> dict[str, Path]:
        """Write ``metrics.md``, ``metrics.csv`` (spec) + ``metrics_full.csv`` and,
        unless disabled, ``<stem>.<fmt>`` for each figure format. Returns the
        written paths."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        (outdir / "metrics.md").write_text(
            f"# Benchmark: {self.system_name} - {len(self.names)} controllers\n\n"
            f"Horizon $T = {self.t_final:g}$ s, $\\Delta t = {self.dt:g}$ s, "
            f"integrator RK4. Initial state $x_0 = {np.array2string(self.x0, precision=3)}$.\n\n"
            + self.to_markdown(),
            encoding="utf-8", newline="\n",
        )
        written["metrics_md"] = outdir / "metrics.md"
        (outdir / "metrics.csv").write_text(self.to_csv(), encoding="utf-8", newline="\n")
        written["metrics_csv"] = outdir / "metrics.csv"
        (outdir / "metrics_full.csv").write_text(
            self.full_metrics_csv(), encoding="utf-8", newline="\n"
        )
        written["metrics_full_csv"] = outdir / "metrics_full.csv"

        if make_figure:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, _ = self.figure(**(figure_kwargs or {}))
            for fmt in figure_formats:
                path = outdir / f"{stem}.{fmt}"
                fig.savefig(path, dpi=150)
                written[f"figure_{fmt}"] = path
            plt.close(fig)

        return written

    # ---------------------------------------------------------------- summary

    def summary(self) -> str:
        """Executive-summary bullet block (spec §5.1)."""
        live = [n for n in self.names if self.status[n] != "Diverged"]
        lines = []
        if live:
            best_rmse = min(live, key=lambda n: self.metrics[n]["rmse"])
            best_energy = min(live, key=lambda n: self.metrics[n]["control_energy"])
            lines.append(
                f"- **Precision:** {best_rmse} - lowest RMSE "
                f"({self.metrics[best_rmse]['rmse']:.3g})."
            )
            lines.append(
                f"- **Energy:** {best_energy} - least control effort "
                f"({self.metrics[best_energy]['control_energy']:.3g})."
            )
        diverged = [n for n in self.names if self.status[n] == "Diverged"]
        if diverged:
            lines.append(f"- **Diverged:** {', '.join(diverged)}.")
        return "\n".join(lines) + "\n"

    def __getitem__(self, name: str) -> dict[str, float]:
        return self.metrics[name]


def compare(
    system: DynamicalSystem,
    controllers: Mapping[str, object],
    x0: Sequence[float],
    dt: float,
    t_final: float,
    reference: ReferenceLike,
    *,
    disturbance: Callable[[float], np.ndarray] | None = None,
    u_bounds: tuple[float, float] | None = None,
    output_index: int = 0,
    deriv_index: int | None = None,
    measurement_fns: Mapping[str, Callable] | None = None,
    title: str | None = None,
) -> ComparisonResult:
    """Roll every controller on ``system`` under identical conditions and score them.

    Parameters
    ----------
    system:
        The plant (``aimct.systems`` interface).
    controllers:
        Ordered mapping ``name -> controller``. Each controller must already be
        configured with its target (PID ``setpoint``, ``StateFeedback`` ``x_ref``…).
    x0, dt, t_final:
        Shared initial state, fixed step, horizon.
    reference:
        Target for the scored output channel — scalar, per-sample array, or
        ``f(t) -> float``. Used for metrics and the figure only. Rise/settling/
        overshoot assume a constant (step) target; its final value is the metric
        ``target``.
    disturbance:
        Optional ``d(t)`` additive plant-input disturbance (passed straight to
        :func:`aimct.simulate.simulate`; not counted in recorded ``u``).
    u_bounds:
        Optional ``(low, high)`` actuator saturation, applied to every controller
        identically. Also sets ``u_limit`` for the saturation metric.
    output_index:
        Which channel of ``system.output`` is the tracked output (default 0).
    deriv_index:
        State index to use as :math:`\\dot y` for the phase-portrait panel.
        Defaults to ``output_index + 1`` when that is a valid state index, else a
        numerical derivative of the output.
    measurement_fns:
        Optional ``name -> measurement_fn`` for output-feedback controllers; see
        :func:`aimct.simulate.simulate`. Names absent from the mapping get
        full-state measurement.
    """
    if not controllers:
        raise ValueError("need at least one controller")

    names = list(controllers.keys())
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n_steps = int(round(t_final / dt))
    # provisional grid: sizes / validates the reference before any rollout
    ref = _as_reference_array(reference, np.arange(n_steps + 1) * dt)
    u_limit = None if u_bounds is None else max(abs(u_bounds[0]), abs(u_bounds[1]))

    if deriv_index is None:
        cand = output_index + 1
        deriv_index = cand if cand < system.n_states else None

    mfns = dict(measurement_fns or {})
    trajectories: dict[str, Trajectory] = {}
    for name in names:
        trajectories[name] = simulate(
            system,
            controllers[name],
            x0=x0,
            dt=dt,
            t_final=t_final,
            u_bounds=u_bounds,
            measurement_fn=mfns.get(name),
            input_disturbance=disturbance,
        )

    # Canonical grid: the longest trajectory produced (full length whenever at
    # least one controller completes the horizon; a divergent run is truncated
    # by simulate). target/ref follow that grid.
    t_grid = max((tr.t for tr in trajectories.values()), key=len)
    if callable(reference):
        ref = _as_reference_array(reference, t_grid)
    elif len(ref) != len(t_grid):           # every run diverged early
        ref = ref[: len(t_grid)]
    target = float(ref[-1])
    ref_scale = float(np.max(np.abs(x0)) + np.max(np.abs(ref)) + 1.0)

    metrics: dict[str, dict[str, float]] = {}
    status: dict[str, str] = {}
    for name in names:
        traj = trajectories[name]
        # Score only the finite prefix - the behaviour up to blow-up. Divergence
        # is reported through `status`, not by poisoning the metrics with inf.
        n_ok = max(2, _finite_prefix_len(traj.x)) if len(traj.x) >= 2 else len(traj.x)
        y = traj.y[:n_ok, output_index]
        # compute_all_metrics wants a 1-D u for single-input systems.
        u_scored = traj.u[:n_ok, 0] if traj.u.shape[1] == 1 else traj.u[:n_ok]
        m = compute_all_metrics(traj.t[:n_ok], y, u_scored, target=target, u_limit=u_limit)
        metrics[name] = m
        status[name] = _classify_status(traj, m, t_final, dt, ref_scale,
                                        _finite_prefix_len(traj.x))

    return ComparisonResult(
        system_name=type(system).__name__,
        t=t_grid,
        reference=ref,
        output_index=output_index,
        x0=x0,
        dt=dt,
        t_final=t_final,
        u_bounds=u_bounds,
        names=names,
        trajectories=trajectories,
        metrics=metrics,
        status=status,
        deriv_index=deriv_index,
        title=title or f"{type(system).__name__} - controller comparison",
    )
