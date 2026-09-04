r"""``aimct.dev`` — a design-time preview for a new :class:`DynamicalSystem`.

Authoring a system today is edit -> hand-write a scratch ``simulate()`` script
-> plot -> eyeball -> repeat. This module shortens that loop to one call and
one file-watch, and gives fast feedback on the three things that go wrong
early and are otherwise invisible until an LQR blows up much later:

1. a **sign error** in ``dynamics()`` — usually shows up as an unexpectedly
   unstable or divergent response,
2. a **wrong analytic** ``linearize()`` — caught by comparing it against the
   base class's numeric central-difference Jacobian,
3. a model that is **not controllable** (or not observable) about the chosen
   operating point.

This is an *authoring-time* tool for whoever is writing the system, deliberately
kept separate from :mod:`aimct.viz` (which is the *runtime* story — replay and
interactive sandboxes for a finished system's users). It may reuse
:mod:`aimct.viz` for the replay panel when useful, but has no dependency the
other way.

    from aimct.dev import build_report, render
    report = build_report(MyPlant())
    print(report.summary())
    render(report).savefig("design_preview.png")

or watch a file while you edit it::

    python -m aimct preview mymodule.py:MyPlant --watch
"""

from __future__ import annotations

import importlib
import itertools
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..controllers.state_feedback import is_controllable
from ..estimation.observability import is_observable
from ..simulate import Trajectory, simulate
from ..systems.base import DynamicalSystem

__all__ = [
    "DesignReport", "build_report", "render", "load_system",
    "preview_once", "watch",
]

_counter = itertools.count()          # unique sys.modules keys for reloaded file targets

_RESIDUAL_TOL = 1e-3          # relative Jacobian-mismatch flag threshold


# --------------------------------------------------------------------- report

@dataclass
class DesignReport:
    """Everything the preview computes about one system at one operating
    point. Pure data — no plotting, no I/O — so it is cheap to test."""

    system: DynamicalSystem
    name: str
    x_eq: np.ndarray
    u_eq: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray                          # d(output)/dx at (x_eq, u_eq)
    poles: np.ndarray                      # eigenvalues of A, complex
    stable: bool                           # all Re(poles) < 0
    controllable: bool
    observable: bool
    has_analytic_linearize: bool
    jacobian_residual: float | None        # None when linearize() is the numeric fallback
    responses: dict[str, Trajectory] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== {self.name} - design preview ===",
            f"n_states={self.system.n_states}  n_inputs={self.system.n_inputs}"
            f"  x_eq={np.round(self.x_eq, 3)}  u_eq={np.round(self.u_eq, 3)}",
            f"poles: {np.round(self.poles, 3)}",
            f"stable (all Re<0): {self.stable}   controllable: {self.controllable}"
            f"   observable: {self.observable}",
        ]
        if self.has_analytic_linearize:
            lines.append(f"analytic vs numeric Jacobian residual: "
                        f"{self.jacobian_residual:.3g}"
                        f"{'  [MISMATCH]' if self.warnings else ''}")
        else:
            lines.append("linearize() not overridden - using the numeric fallback"
                         " (no residual check to run)")
        if self.warnings:
            lines.append("")
            lines.append("warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def _numeric_jacobian_output(system, t, x_eq, u_eq, eps=1e-6):
    """``d(output)/dx`` at ``(x_eq, u_eq)`` by central difference — a linear
    ``C`` even when ``output()`` is nonlinear (a local approximation)."""
    n = system.n_states
    y0 = np.atleast_1d(np.asarray(system.output(t, x_eq, u_eq), dtype=float))
    C = np.zeros((y0.size, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        yp = np.atleast_1d(np.asarray(system.output(t, x_eq + dx, u_eq), dtype=float))
        ym = np.atleast_1d(np.asarray(system.output(t, x_eq - dx, u_eq), dtype=float))
        C[:, j] = (yp - ym) / (2 * eps)
    return C


def build_report(
    system: DynamicalSystem,
    *,
    x_eq: np.ndarray | None = None,
    u_eq: np.ndarray | None = None,
    dt: float = 0.01,
    t_final: float = 4.0,
    u_scale: float | np.ndarray | None = None,
    name: str | None = None,
) -> DesignReport:
    """Compute poles, controllability/observability, the analytic-vs-numeric
    Jacobian residual, and free/step/impulse/sinusoid responses about
    ``(x_eq, u_eq)`` (default: the zero state and zero input — pass an
    equilibrium explicitly for a system whose interesting operating point is
    not the origin, e.g. ``x_eq=[np.pi, 0]`` for a pendulum's upright)."""
    n, m = system.n_states, system.n_inputs
    x_eq = np.zeros(n) if x_eq is None else np.asarray(x_eq, dtype=float)
    u_eq = np.zeros(m) if u_eq is None else np.asarray(u_eq, dtype=float)

    A, B = system.linearize(x_eq, u_eq)
    poles = np.linalg.eigvals(A)
    stable = bool(np.all(poles.real < 0))
    ctrb = bool(is_controllable(A, B))

    C = _numeric_jacobian_output(system, 0.0, x_eq, u_eq)
    obsv = bool(is_observable(A, C)) if C.shape[0] > 0 else False

    has_analytic = type(system).linearize is not DynamicalSystem.linearize
    residual = None
    warnings: list[str] = []
    if has_analytic:
        A_num, B_num = DynamicalSystem.linearize(system, x_eq, u_eq)
        scale = max(np.linalg.norm(A_num), np.linalg.norm(B_num), 1e-9)
        residual = float(np.linalg.norm(A - A_num) + np.linalg.norm(B - B_num)) / scale
        if residual > _RESIDUAL_TOL:
            warnings.append(
                f"analytic linearize() disagrees with the numeric check by "
                f"{residual:.3g} (relative, tol {_RESIDUAL_TOL:.0e}) - check the "
                f"Jacobian derivation")

    if not ctrb:
        warnings.append(
            "model is not controllable about (x_eq, u_eq) - LQR / pole "
            "placement will fail here; check B or pick a different operating point")
    if C.shape[0] > 0 and not obsv:
        warnings.append(
            "model is not observable about (x_eq, u_eq) through output() - a "
            "state observer will not converge from this sensor")

    u_scale = (0.1 * np.ones(m) if u_scale is None
              else np.broadcast_to(np.asarray(u_scale, float), (m,)))

    def _run(u_fn, disturbance=None):
        clock = {"t": 0.0}

        def ctrl(y, dt_):
            u = u_fn(clock["t"])
            clock["t"] += dt_
            return u

        return simulate(system, ctrl, x0=x_eq, dt=dt, t_final=t_final,
                        input_disturbance=disturbance)

    responses = {
        "free": _run(lambda t: np.zeros(m)),
        "step": _run(lambda t: u_eq + u_scale),
        "impulse": _run(lambda t: u_eq,
                        disturbance=lambda t: u_scale / dt if t < dt else np.zeros(m)),
        "sinusoid": _run(lambda t: u_eq + u_scale * np.sin(2 * np.pi * t / max(t_final, 1e-9))),
    }
    for label, tr in responses.items():
        if tr.diverged:
            warnings.append(
                f"the '{label}' response diverged - check for a sign error or "
                f"a missing saturation in dynamics()")

    return DesignReport(
        system=system, name=name or type(system).__name__,
        x_eq=x_eq, u_eq=u_eq, A=A, B=B, C=C, poles=poles, stable=stable,
        controllable=ctrb, observable=obsv, has_analytic_linearize=has_analytic,
        jacobian_residual=residual, responses=responses, warnings=warnings,
    )


# -------------------------------------------------------------------- render

def render(report: DesignReport, *, figsize=(13, 8)):
    """A one-screen matplotlib dashboard: pole map, four response traces
    (state components), and the numeric summary as figure text."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from ..plot_style import PALETTE, set_aimct_style
    set_aimct_style()

    p = report.poles
    lim = max(1.0, float(np.max(np.abs(p))) * 1.3) if p.size else 1.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    ax_p = fig.add_subplot(gs[0, 0])
    ax_p.axvline(0.0, color="0.6", lw=1.0, ls="--")
    ax_p.scatter(p.real, p.imag, s=60, color=PALETTE.get("lqr", "C0"),
                marker="x", linewidths=2)
    ax_p.set(title="poles", xlabel="Re", ylabel="Im")
    ax_p.set_xlim(-lim, lim); ax_p.set_ylim(-lim, lim)
    ax_p.set_aspect("equal", "box")

    ax_t = fig.add_subplot(gs[0, 1:])
    ax_t.axis("off")
    ax_t.text(0.0, 1.0, report.summary(), family="monospace", fontsize=8,
             va="top", ha="left", transform=ax_t.transAxes)

    labels = ["free", "step", "impulse", "sinusoid"]
    for i, label in enumerate(labels):
        ax = fig.add_subplot(gs[1, i])
        tr = report.responses[label]
        for k in range(tr.x.shape[1]):
            ax.plot(tr.t, tr.x[:, k], lw=1.3, label=f"x{k}")
        if tr.diverged:
            ax.set_title(label + " [diverged]", color="crimson")
        else:
            ax.set_title(label)
        ax.set_xlabel("t [s]")
        if i == 0:
            ax.legend(fontsize=6)

    fig.suptitle(f"{report.name} - design preview", fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- loading / CLI

def load_system(target: str, **kwargs) -> DynamicalSystem:
    """``"pkg.module:ClassName"`` (installed) or ``"path/to/file.py:ClassName"``
    (not yet packaged) -> a fresh instance, built with ``**kwargs``. A fresh
    module object is created every call (never cached in ``sys.modules``) so a
    file-path target always reflects the latest save."""
    if ":" not in target:
        raise ValueError(f"target must be 'module:Class' or 'path.py:Class', got {target!r}")
    mod_ref, cls_name = target.rsplit(":", 1)

    if mod_ref.endswith(".py") or "/" in mod_ref or "\\" in mod_ref:
        path = Path(mod_ref)
        mod_name = f"_aimct_dev_preview_{path.stem}_{next(_counter)}"
        # Read + compile + exec by hand rather than importlib's file loader: the
        # loader's __pycache__ bytecode cache is keyed on mtime, and two saves
        # inside the same mtime tick (coarse on some filesystems) would replay
        # stale bytecode - defeating the whole point of a live preview.
        source = path.read_text(encoding="utf-8")
        code = compile(source, str(path), "exec")
        module = types.ModuleType(mod_name)
        module.__file__ = str(path)
        sys.modules[mod_name] = module
        exec(code, module.__dict__)
    else:
        module = importlib.import_module(mod_ref)
        importlib.reload(module)

    cls = getattr(module, cls_name)
    return cls(**kwargs)


def preview_once(target: str, *, out: str | Path | None = "design_preview.png",
                 **build_kwargs) -> DesignReport:
    """Load ``target``, build one :class:`DesignReport`, optionally render it
    to ``out``. Returns the report (``.summary()`` for a text-only view)."""
    sys_kwargs = build_kwargs.pop("system_kwargs", {})
    system = load_system(target, **sys_kwargs)
    report = build_report(system, **build_kwargs)
    if out is not None:
        fig = render(report)
        fig.savefig(out, dpi=130)
        import matplotlib.pyplot as plt
        plt.close(fig)
    return report


def watch(target: str, *, out: str | Path = "design_preview.png",
          poll: float = 1.0, max_ticks: int | None = None,
          on_report=None, **build_kwargs) -> None:
    """Poll the source file behind ``target`` for changes; on every change (and
    once immediately), rebuild the report and overwrite ``out`` — point an
    image viewer / editor preview pane at it for a live loop. ``max_ticks``
    bounds the loop (used by tests); ``None`` runs until interrupted.
    ``on_report(report)`` is called after every rebuild (printing a one-line
    status is the default)."""
    mod_ref = target.rsplit(":", 1)[0]        # rsplit: Windows paths carry a drive colon
    path = Path(mod_ref) if (mod_ref.endswith(".py") or "/" in mod_ref
                             or "\\" in mod_ref) else None
    on_report = on_report or (lambda r: print(
        f"[{time.strftime('%H:%M:%S')}] rebuilt {out} - "
        f"{'STABLE' if r.stable else 'unstable'}, "
        f"{'controllable' if r.controllable else 'NOT CONTROLLABLE'}"
        + (f", {len(r.warnings)} warning(s)" if r.warnings else "")))

    last_mtime = None
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        mtime = path.stat().st_mtime if path is not None else None
        if last_mtime is None or mtime != last_mtime:
            try:
                report = preview_once(target, out=out, **build_kwargs)
                on_report(report)
            except Exception as exc:                     # pragma: no cover - interactive
                print(f"[{time.strftime('%H:%M:%S')}] preview failed: {exc}")
            last_mtime = mtime
        ticks += 1
        if max_ticks is None or ticks < max_ticks:
            time.sleep(poll)
