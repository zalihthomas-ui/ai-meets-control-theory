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
        # regulate a disturbed pendulum back to rest (theta = 0 hangs down);
        # auto_controllers designs the LQR about x = 0, so the reference is 0.
        lambda: _pendulum(),
        dict(x0=[0.6, 0.0], dt=0.01, t_final=6.0, reference=0.0,
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

    sub.add_parser("list", help="list the built-in systems (* = has a compare preset)")

    pl = sub.add_parser("live", help="interactive sandbox: drone / arm / diffdrive")
    pl.add_argument("target", nargs="?", default="drone",
                    choices=["drone", "drone3d", "arm", "arm3d", "diffdrive",
                             "armbalance", "armbalance3d"],
                    help="which sandbox (default: drone)")
    pl.add_argument("--headless", action="store_true",
                    help="run the physics smoke check without a GUI")
    pl3 = sub.add_parser("live3d", help="alias for `live drone3d` (6-DOF drone sandbox)")
    pl3.add_argument("--headless", action="store_true",
                     help="run the physics smoke check without a GUI")
    pl3.add_argument("--matplotlib", action="store_true",
                     help="force the lightweight matplotlib-3D renderer")
    pl3.add_argument("--web", action="store_true",
                     help="serve the experimental WebGL/Three.js visualizer")

    args = p.parse_args(argv)

    if args.cmd == "list":
        from . import systems as _sys

        print("runnable via `aimct compare --system <name>`:")
        for name in sorted(_PRESETS):
            print(f"  {name}")
        skip = {"DynamicalSystem", "LinearSystem"}
        extra = [c for c in sorted(getattr(_sys, "__all__", []))
                 if c not in skip and isinstance(getattr(_sys, c, None), type)]
        print("\nalso in aimct.systems (import directly; no compare preset yet):")
        for c in extra:
            print(f"  aimct.systems.{c}")
        return 0

    if args.cmd in ("live", "live3d"):
        import runpy
        from pathlib import Path

        target = "drone3d" if args.cmd == "live3d" else args.target
        d = Path(__file__).resolve().parents[2] / "experiments"
        if not d.is_dir():
            print(
                "The interactive sandboxes live under experiments/ in the source\n"
                "checkout and are not shipped in the installed package. Clone the\n"
                "repo and run from there:\n"
                "  git clone https://github.com/zalihthomas-ui/ai-meets-control-theory\n"
                "  cd ai-meets-control-theory\n"
                f"  python -m aimct live {target}",
                file=sys.stderr,
            )
            return 1
        headless = getattr(args, "headless", False)
        if target == "drone":
            script = d / "live_drone" / "live.py"
        elif target in ("arm", "arm3d"):
            # same physics either way; --headless never needs PyVista
            sub = "run.py" if (target == "arm" or headless) else "pv_arm.py"
            script = d / "live_arm" / sub
        elif target in ("armbalance", "armbalance3d"):
            sub = "run.py" if (target == "armbalance" or headless) else "pv_arm.py"
            script = d / "live_arm_balance" / sub
        elif target == "diffdrive":
            script = d / "live_diffdrive" / "run.py"
        elif getattr(args, "web", False):
            script = d / "live_drone_3d" / "web.py"
        elif getattr(args, "matplotlib", False) or getattr(args, "headless", False):
            script = d / "live_drone_3d" / "sim3d.py"
        else:
            # prefer the PyVista renderer; fall back to matplotlib if not installed
            try:
                import pyvista  # noqa: F401
                script = d / "live_drone_3d" / "pv3d.py"
            except Exception:
                script = d / "live_drone_3d" / "sim3d.py"
        sys.argv = [str(script)] + (["--headless"] if getattr(args, "headless", False) else [])
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
