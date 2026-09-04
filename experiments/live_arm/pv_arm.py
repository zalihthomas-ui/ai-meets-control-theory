"""3-D PyVista renderer for the live_arm sandbox (payload identification).

Same physics and controllers as ``run.py`` - see its module docstring and
``docs/VISUALIZATION.md``. This only changes how it is drawn: real cylinders,
lighting, and an orbit camera, via the shared :mod:`aimct.viz.pv_arm` driver.

    python experiments/live_arm/pv_arm.py         # or: python -m aimct live arm3d
    python experiments/live_arm/pv_arm.py --headless
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from aimct.viz.pv_arm import run_pyvista_arm

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("live_arm_run", _HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
sys.modules["live_arm_run"] = run
_spec.loader.exec_module(run)


def main() -> int:
    if "--headless" in sys.argv:
        return run._headless()
    box, _path = run.build()
    return run_pyvista_arm(box, title="Live 2-link arm (3-D) - payload identification",
                           show_payload=True)


if __name__ == "__main__":
    raise SystemExit(main())
