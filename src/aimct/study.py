r"""One-call controller bake-off: give it a system and a task, get the standard
comparison table + 4-panel figure back.

This is the "bring your own problem" entry point. Pass any
:class:`aimct.systems.DynamicalSystem` (a built-in or your own subclass) and an
initial state; ``run_study`` auto-designs an LQR and a linear MPC on the
system's ``linearize()``, runs them all under identical conditions through
:func:`aimct.benchmarks.compare`, and (optionally) writes the artifacts to a
folder.

    from aimct.study import run_study
    from aimct.systems import CartPole
    res = run_study(CartPole(), x0=[0, 0, 0.2, 0], dt=0.01, t_final=5,
                    output_index=2, out_dir="my_study")
    print(res.to_markdown())

Command line:  ``python -m aimct compare --system cartpole --out my_study``
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .benchmarks import compare
from .controllers import LQR, LinearMPC
from .systems import DynamicalSystem

__all__ = ["auto_controllers", "run_study"]


def auto_controllers(
    system: DynamicalSystem,
    *,
    Q=None,
    R=None,
    x_ref=None,
    u_bounds=None,
    mpc_horizon: int = 20,
) -> dict:
    """LQR + linear MPC designed on ``system.linearize()``. Silently skips a
    controller if the pair is uncontrollable or the design fails."""
    A, B = system.linearize()
    n, m = system.n_states, system.n_inputs
    Q = np.eye(n) if Q is None else np.atleast_2d(Q)
    R = np.eye(m) if R is None else np.atleast_2d(R)

    out: dict = {}
    try:
        out["LQR"] = LQR(A, B, Q, R, x_ref=x_ref)
    except Exception as exc:  # noqa: BLE001 - report and move on
        out["_lqr_error"] = repr(exc)
    try:
        out["LinearMPC"] = LinearMPC(A, B, Q=Q, R=R, N=mpc_horizon,
                                     x_ref=x_ref, u_bounds=u_bounds)
    except Exception as exc:  # noqa: BLE001
        out["_mpc_error"] = repr(exc)
    return out


def run_study(
    system: DynamicalSystem,
    x0,
    *,
    dt: float,
    t_final: float,
    Q=None,
    R=None,
    reference=0.0,
    output_index: int = 0,
    deriv_index: int | None = None,
    u_bounds=None,
    extra_controllers: Mapping[str, object] | None = None,
    out_dir: str | Path | None = None,
    title: str | None = None,
):
    """Design LQR + MPC for ``system``, add any ``extra_controllers``, run the
    standard :func:`aimct.benchmarks.compare`, optionally save to ``out_dir``.

    Returns the :class:`~aimct.benchmarks.harness.ComparisonResult`.
    """
    ctrls = {k: v for k, v in
             auto_controllers(system, Q=Q, R=R, u_bounds=u_bounds).items()
             if not k.startswith("_")}
    if extra_controllers:
        ctrls.update(extra_controllers)
    if not ctrls:
        raise RuntimeError("no controller could be designed for this system; "
                           "pass extra_controllers=")

    result = compare(
        system, ctrls, x0=np.asarray(x0, dtype=float), dt=dt, t_final=t_final,
        reference=reference, u_bounds=u_bounds,
        output_index=output_index, deriv_index=deriv_index,
        title=title or f"Study: {type(system).__name__}",
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if hasattr(result, "save"):
            result.save(out)
        else:  # minimal fallback
            (out / "table.md").write_text(result.to_markdown(), encoding="utf-8")
            (out / "table.csv").write_text(result.to_csv(), encoding="utf-8")
    return result
