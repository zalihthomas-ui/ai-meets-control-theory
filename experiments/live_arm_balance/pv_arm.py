"""3-D PyVista renderer for the live_arm_balance sandbox (double inverted
pendulum). Same physics and controllers as ``run.py`` - this only changes how
it is drawn, via the shared :mod:`aimct.viz.pv_arm` driver.

    python experiments/live_arm_balance/pv_arm.py    # or: python -m aimct live armbalance3d
    python experiments/live_arm_balance/pv_arm.py --headless
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from aimct.viz.pv_arm import run_pyvista_arm

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("live_arm_balance_run", _HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
sys.modules["live_arm_balance_run"] = run
_spec.loader.exec_module(run)


def main() -> int:
    if "--headless" in sys.argv:
        return run._headless()
    box = run.build()
    return run_pyvista_arm(box, title="Live double-inverted-pendulum (3-D)",
                           show_payload=False)


if __name__ == "__main__":
    raise SystemExit(main())
