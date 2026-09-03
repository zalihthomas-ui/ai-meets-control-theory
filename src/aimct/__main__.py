"""``python -m aimct`` — run a controller bake-off on a built-in system.

    python -m aimct compare --system cartpole --out my_study
    python -m aimct compare --system quadrotor --t-final 12
    python -m aimct list
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .study import run_study

# name -> (factory, default study kwargs)
_PRESETS = {
    "mass_spring_damper": (
        lambda: _msd(), dict(x0=[1.0, 0.0], dt=0.01, t_final=20.0, output_index=0),
    ),
    "pendulum": (
        lambda: _pendulum(),
        dict(x0=[np.pi - 0.3, 0.0], dt=0.01, t_final=6.0, reference=np.pi,
             output_index=0,
             Q=np.diag([10.0, 1.0]), R=np.array([[0.5]]),
             u_bounds=(-8.0, 8.0)),
    ),
    "cartpole": (
        lambda: _cartpole(),
        dict(x0=[0.0, 0.0, 0.2, 0.0], dt=0.01, t_final=5.0, output_index=2,
             deriv_index=3, Q=np.diag([10.0, 1.0, 100.0, 10.0]),
             R=np.array([[0.1]]), u_bounds=(-20.0, 20.0)),
    ),
    "quadrotor": (
        lambda: _quad(),
        dict(x0=[0.3, 1.0, 0.15, 0.0, 0.0, 0.0], dt=0.004, t_final=6.0,
             output_index=0,
             Q=np.diag(1.0 / np.array([0.1, 0.1, 0.2, 0.5, 0.5, 3.0]) ** 2),
             R=np.diag(1.0 / np.array([0.15, 0.15]) ** 2),
             u_bounds=(0.0, 0.30)),
    ),
}


def _msd():
    from .systems import MassSpringDamper
    return MassSpringDamper()


def _pendulum():
    from .systems import Pendulum
    return Pendulum()


def _cartpole():
    from .systems import CartPole
    return CartPole()


def _quad():
    from .systems import PlanarQuadrotor
    return PlanarQuadrotor()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aimct")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compare", help="run LQR + MPC (+ your own) on a system")
    pc.add_argument("--system", required=True, choices=sorted(_PRESETS))
    pc.add_argument("--out", default=None, help="directory for table.md / figure.png")
    pc.add_argument("--t-final", type=float, default=None)
    pc.add_argument("--dt", type=float, default=None)

    sub.add_parser("list", help="list the built-in systems")

    pl = sub.add_parser("live", help="interactive drone-vs-wind sandbox")
    pl.add_argument("--headless", action="store_true",
                    help="run the physics smoke check without a GUI")

    args = p.parse_args(argv)

    if args.cmd == "list":
        for name in sorted(_PRESETS):
            print(name)
        return 0

    if args.cmd == "live":
        import runpy
        from pathlib import Path
        script = (Path(__file__).resolve().parents[2]
                  / "experiments" / "live_drone" / "live.py")
        if args.headless:
            sys.argv = [str(script), "--headless"]
        else:
            sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
        return 0

    factory, kw = _PRESETS[args.system]
    kw = dict(kw)
    if args.t_final is not None:
        kw["t_final"] = args.t_final
    if args.dt is not None:
        kw["dt"] = args.dt
    kw["out_dir"] = args.out

    res = run_study(factory(), title=f"{args.system} bake-off", **kw)
    print(res.to_markdown())
    if hasattr(res, "summary"):
        print()
        print(res.summary())
    if args.out:
        print(f"\nwrote artifacts to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
