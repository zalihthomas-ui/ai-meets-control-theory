r"""Parameter / robustness sweep on top of :func:`aimct.benchmarks.compare`.

``sweep(param_values, make_case)`` re-runs the comparison harness once per value
of a swept quantity - a plant parameter (mass, stiffness), a controller weight, a
disturbance magnitude, or an initial condition - and collects the per-controller
metrics and stability status across the grid.  The :class:`SweepResult` then
emits:

* a pivot table (one metric, rows = parameter value, columns = controller),
* a long-format CSV of every metric at every point, and
* the ``docs/comparison-report-spec.md`` §4.2 robustness plot: chosen metric on
  the left axis, control energy on the right, one line per controller, with the
  divergent operating region shaded.

``make_case(value) -> dict`` returns the keyword arguments for one
:func:`compare` call (``system``, ``controllers``, ``x0``, ``dt``, ``t_final``,
``reference`` and any optional keys).  The caller therefore controls exactly how
the swept value enters the problem - e.g. rebuild the plant and re-solve the LQR
gain for a mass change, or just move ``x0`` for a basin-of-attraction study.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .harness import ComparisonResult, compare

__all__ = ["sweep", "SweepResult"]

_REQUIRED_CASE_KEYS = {"system", "controllers", "x0", "dt", "t_final", "reference"}


@dataclass
class SweepResult:
    """Metrics and stability status for every controller across a parameter grid."""

    param_name: str
    param_values: list
    controllers: list[str]
    results: list[ComparisonResult]              # aligned with param_values
    records: list[dict] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------- accessors

    def result_at(self, value) -> ComparisonResult:
        return self.results[self._index(value)]

    def _index(self, value) -> int:
        for i, v in enumerate(self.param_values):
            if v == value or np.isclose(np.asarray(v, float), np.asarray(value, float)).all():
                return i
        raise KeyError(f"{value!r} is not a swept value of {self.param_name}")

    def series(self, metric: str, controller: str) -> np.ndarray:
        """Values of ``metric`` for one controller across the grid (NaN if absent)."""
        return np.array(
            [res.metrics[controller].get(metric, np.nan) for res in self.results],
            dtype=float,
        )

    def status(self, controller: str) -> list[str]:
        return [res.status[controller] for res in self.results]

    def stable_mask(self, controller: str) -> np.ndarray:
        return np.array([s == "Stable" for s in self.status(controller)], dtype=bool)

    def recovered_mask(self, controller: str) -> np.ndarray:
        """True where the run did not diverge (``Stable`` or ``Marginal``)."""
        return np.array([s != "Diverged" for s in self.status(controller)], dtype=bool)

    def max_stable(self, controller: str):
        """Largest swept value at which ``controller`` is ``Stable`` (assumes the
        grid is ordered and stability is lost monotonically - typical for a
        basin-of-attraction edge)."""
        stable = [v for v, ok in zip(self.param_values, self.stable_mask(controller)) if ok]
        return max(stable) if stable else None

    def first_unstable(self, controller: str):
        """First swept value at which ``controller`` is not ``Stable``."""
        for v, ok in zip(self.param_values, self.stable_mask(controller)):
            if not ok:
                return v
        return None

    def first_diverged(self, controller: str):
        """First swept value at which ``controller`` diverges (pole falls / blows
        up). ``None`` if it never does across the grid."""
        for v, s in zip(self.param_values, self.status(controller)):
            if s == "Diverged":
                return v
        return None

    def max_recoverable(self, controller: str):
        """Largest swept value strictly below the first divergence - the measured
        basin-of-attraction edge. Prints nothing about non-monotonicity; check
        :meth:`recovered_mask` if the grid may re-stabilise past a gap."""
        first_bad = self.first_diverged(controller)
        if first_bad is None:
            return self.param_values[-1]
        below = [v for v in self.param_values if v < first_bad]
        return max(below) if below else None

    # ---------------------------------------------------------------- tables

    def table(self, metric: str = "settling_time", fmt: str = "{:.3g}") -> str:
        """Markdown pivot: rows = parameter value, columns = controller."""
        head = [self.param_name, *self.controllers]
        lines = [
            "| " + " | ".join(head) + " |",
            "| " + " | ".join([":---"] + [":---:"] * len(self.controllers)) + " |",
        ]
        for i, v in enumerate(self.param_values):
            cells = [fmt.format(v) if isinstance(v, float) else str(v)]
            for c in self.controllers:
                val = self.results[i].metrics[c].get(metric)
                if val is None or (isinstance(val, float) and not np.isfinite(val)):
                    cells.append("inf" if val is not None else "--")
                else:
                    cells.append(fmt.format(val))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    def status_table(self) -> str:
        """Markdown pivot of the Stable / Marginal / Diverged label per point."""
        head = [self.param_name, *self.controllers]
        lines = [
            "| " + " | ".join(head) + " |",
            "| " + " | ".join([":---"] + [":---:"] * len(self.controllers)) + " |",
        ]
        for i, v in enumerate(self.param_values):
            vv = f"{v:.3g}" if isinstance(v, float) else str(v)
            lines.append("| " + " | ".join([vv, *(self.results[i].status[c] for c in self.controllers)]) + " |")
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        buf = io.StringIO()
        if not self.records:
            return ""
        keys = list(self.records[0].keys())
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(keys)
        for rec in self.records:
            w.writerow([rec.get(k, "") for k in keys])
        return buf.getvalue()

    # ----------------------------------------------------------------- plot

    def plot(
        self,
        *,
        left: str = "settling_time",
        right: str | None = "control_energy",
        controllers: Sequence[str] | None = None,
        title: str | None = None,
        left_label: str | None = None,
        right_label: str | None = None,
        shade_unstable: bool = True,
        figsize: tuple[float, float] = (9.0, 5.0),
    ):
        """Spec §4.2 robustness plot. Returns ``(fig, ax_left, ax_right)``
        (``ax_right`` is ``None`` when ``right`` is ``None``). Lazy matplotlib."""
        from ..plot_style import get_controller_style, set_aimct_style
        import matplotlib.pyplot as plt

        set_aimct_style()
        names = list(controllers or self.controllers)
        x = np.asarray(self.param_values, dtype=float)

        fig, ax_l = plt.subplots(figsize=figsize)
        ax_r = ax_l.twinx() if right else None

        if shade_unstable:
            bad = np.zeros(len(x), dtype=bool)
            for c in names:
                bad |= ~self.stable_mask(c)
            _shade_spans(ax_l, x, bad)

        for c in names:
            style = get_controller_style(c)
            color = style.get("color")
            yl = self.series(left, c)
            ax_l.plot(x, yl, marker="o", ms=4, color=color, lw=2.0, label=c)
            unstable = ~self.stable_mask(c)
            if unstable.any():
                ax_l.plot(x[unstable], yl[unstable], linestyle="none", marker="x",
                          ms=9, mew=2, color=color)
            if ax_r is not None:
                ax_r.plot(x, self.series(right, c), color=color, lw=1.3, ls=":")

        ax_l.set_xlabel(_nice(self.param_name))
        ax_l.set_ylabel(left_label or _nice(left))
        if ax_r is not None:
            ax_r.set_ylabel(right_label or f"{_nice(right)} (dotted)")
            ax_r.grid(False)
        ax_l.set_title(title or f"Robustness sweep vs {_nice(self.param_name)}")
        ax_l.legend(loc="best")
        fig.tight_layout()
        return fig, ax_l, ax_r

    # ----------------------------------------------------------------- save

    def save(self, outdir: str | Path, *, stem: str = "robustness_sweep",
             metric: str = "settling_time", figure_formats: Sequence[str] = ("png", "svg"),
             plot_kwargs: dict | None = None) -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        (outdir / "sweep.csv").write_text(self.to_csv(), encoding="utf-8", newline="\n")
        written["csv"] = outdir / "sweep.csv"
        (outdir / "sweep_summary.md").write_text(
            f"# Robustness sweep: {self.param_name}\n\n"
            f"Controllers: {', '.join(self.controllers)}. "
            f"{len(self.param_values)} grid points "
            f"[{self.param_values[0]!r} .. {self.param_values[-1]!r}].\n\n"
            f"## Stability\n\n{self.status_table()}\n"
            f"## {_nice(metric)}\n\n{self.table(metric)}",
            encoding="utf-8", newline="\n",
        )
        written["summary_md"] = outdir / "sweep_summary.md"

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, _, _ = self.plot(**{"left": metric, **(plot_kwargs or {})})
        for fmt in figure_formats:
            path = outdir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=150)
            written[f"figure_{fmt}"] = path
        plt.close(fig)
        return written


def sweep(
    param_values: Sequence,
    make_case: Callable[[object], dict],
    *,
    param_name: str = "param",
    progress: bool = False,
) -> SweepResult:
    """Run :func:`compare` once per entry of ``param_values``.

    Parameters
    ----------
    param_values:
        Ordered grid of the swept quantity (numbers, or opaque points such as
        tuples - the tables/CSV still work, the plot needs scalars).
    make_case:
        ``make_case(value) -> dict`` of keyword arguments for one :func:`compare`
        call. Must contain at least ``system``, ``controllers``, ``x0``, ``dt``,
        ``t_final`` and ``reference``.
    param_name:
        Column / axis label for the swept quantity.
    progress:
        Print ``param_name=value -> status`` after each point.
    """
    values = [v.item() if isinstance(v, np.generic) else v for v in param_values]
    if not values:
        raise ValueError("param_values is empty")

    results: list[ComparisonResult] = []
    records: list[dict] = []
    controller_order: list[str] | None = None

    for v in values:
        case = make_case(v)
        missing = _REQUIRED_CASE_KEYS - set(case)
        if missing:
            raise ValueError(
                f"make_case({v!r}) is missing compare() keys: {sorted(missing)}"
            )
        res = compare(**case)
        results.append(res)
        if controller_order is None:
            controller_order = list(res.names)
        elif list(res.names) != controller_order:
            raise ValueError(
                "controller set/order changed across the sweep: "
                f"{controller_order} -> {list(res.names)}"
            )
        for name in res.names:
            records.append(
                {param_name: v, "controller": name, "status": res.status[name],
                 **res.metrics[name]}
            )
        if progress:
            worst = ", ".join(f"{n}:{res.status[n]}" for n in res.names)
            print(f"{param_name}={v!r} -> {worst}")

    return SweepResult(
        param_name=param_name,
        param_values=values,
        controllers=controller_order or [],
        results=results,
        records=records,
    )


# --------------------------------------------------------------------- helpers

def _shade_spans(ax, x: np.ndarray, mask: np.ndarray) -> None:
    """Shade the x-intervals where ``mask`` is True (contiguous runs merged)."""
    if not mask.any():
        return
    edges = np.r_[0, np.flatnonzero(np.diff(mask.astype(int))) + 1, len(mask)]
    labelled = False
    for a, b in zip(edges[:-1], edges[1:]):
        if not mask[a]:
            continue
        lo = x[a] if a == 0 else 0.5 * (x[a - 1] + x[a])
        hi = x[b - 1] if b == len(x) else 0.5 * (x[b - 1] + x[b])
        ax.axvspan(lo, hi, color="#D62728", alpha=0.10,
                   label=None if labelled else "unstable / divergent")
        labelled = True


def _nice(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()
