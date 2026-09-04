"""``python -m aimct.dev TARGET`` — design-time preview, standalone.

Not yet wired into ``python -m aimct`` (holding for puma's __main__.py pass —
see docs/DEV_PREVIEW.md); this entry point works today without touching it.

    python -m aimct.dev aimct.systems.pendulum:Pendulum
    python -m aimct.dev my_system.py:MyPlant --watch
"""

from __future__ import annotations

import argparse
import sys

from .preview import preview_once, watch


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m aimct.dev")
    p.add_argument("target", help="'module:Class' or 'path/to/file.py:Class'")
    p.add_argument("--out", default="design_preview.png",
                   help="PNG path to (re)write (default: design_preview.png)")
    p.add_argument("--watch", action="store_true",
                   help="poll the source file and rebuild on every change")
    p.add_argument("--poll", type=float, default=1.0, help="watch poll interval [s]")
    p.add_argument("--t-final", type=float, default=4.0)
    p.add_argument("--dt", type=float, default=0.01)
    args = p.parse_args(argv)

    kw = dict(dt=args.dt, t_final=args.t_final)
    if args.watch:
        print(f"watching {args.target} -> {args.out} (Ctrl+C to stop)")
        try:
            watch(args.target, out=args.out, poll=args.poll, **kw)
        except KeyboardInterrupt:
            print("\nstopped")
        return 0

    report = preview_once(args.target, out=args.out, **kw)
    print(report.summary())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
