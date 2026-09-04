"""3-D live drone-vs-wind sandbox - PyVista (VTK) renderer.

Real 3-D lighting, a procedural quadrotor, and smooth 60 Hz interaction, driving
the exact physics in :mod:`sim3d` (no port, no browser). This is the primary
3-D visualiser; ``sim3d.py`` keeps a matplotlib fallback and the physics engine.

    python experiments/live_drone_3d/pv3d.py           # or: python -m aimct live3d --pyvista
    python experiments/live_drone_3d/pv3d.py --headless

Controls
--------
  * wind X / Y / Z sliders (bottom of the window)
  * arrow keys / z / x : impulse gusts (+/- x, +/- y, +/- z)
  * 1 2 3             : switch controller
  * r                : reset      c : clear wind
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sim3d", _HERE / "sim3d.py")
sim3d = importlib.util.module_from_spec(_spec)
sys.modules["sim3d"] = sim3d
_spec.loader.exec_module(sim3d)

HOVER = sim3d.HOVER
GUST = sim3d.GUST


def _drone_meshes(pv, arm):
    """A procedural X-quad: body box, 4 arms, 4 prop discs. Returns
    (static_parts, prop_discs) as PyVista meshes in the body frame."""
    d = arm / np.sqrt(2.0)
    body = pv.Box(bounds=(-0.03, 0.03, -0.03, 0.03, -0.012, 0.012))
    arms = []
    tips = []
    for sx, sy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
        p0 = np.zeros(3)
        p1 = np.array([sx * d, sy * d, 0.0])
        arms.append(pv.Tube(pointa=p0, pointb=p1, radius=0.004))
        tips.append(p1)
    props = [pv.Disc(center=t + [0, 0, 0.006], inner=0.0, outer=0.028,
                     normal=(0, 0, 1), r_res=1, c_res=40) for t in tips]
    return body, arms, tips, props


def _pose_matrix(pos, R):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pos
    return M


def _headless() -> int:
    return sim3d._headless()


def main() -> int:
    if "--headless" in sys.argv:
        return _headless()
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover
        print(f"pyvista not available ({exc}); try  pip install pyvista  "
              f"or use  python -m aimct live3d  (matplotlib).")
        return 1

    eng = sim3d.Engine()
    arm = eng.quad.arm

    pl = pv.Plotter(window_size=(1180, 820))
    pl.set_background("#0d1017", top="#1b2740")
    pl.add_text("AI Meets Control Theory - 3D drone vs wind",
                position="upper_left", font_size=11, color="#7fb0ff")

    # ground grid + hover beacon
    g = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=6, j_size=6,
                 i_resolution=24, j_resolution=24)
    pl.add_mesh(g, style="wireframe", color="#243049", line_width=1)
    pl.add_mesh(pv.Sphere(radius=0.03, center=HOVER), color="#ffd166")
    pl.add_mesh(pv.Line((HOVER[0], HOVER[1], 0), tuple(HOVER)),
                color="#ffd166", line_width=1)

    body, arms, tips, props = _drone_meshes(pv, arm)
    body_a = pl.add_mesh(body, color="#12203a", pbr=True, metallic=0.6, roughness=0.4)
    arm_as = [pl.add_mesh(a, color="#20304f") for a in arms]
    prop_as = [pl.add_mesh(p, color="#00d2ff", opacity=0.35) for p in props]
    thrust_as = [pl.add_mesh(pv.Arrow(), color="#ff9100") for _ in range(4)]
    trail_a = [None]
    hud = pl.add_text("", position="lower_right", font_size=9,
                      color="#cfe3ff", name="hud")
    wind_a = [pl.add_mesh(pv.Arrow(), color="#00e676")]

    # --- input ---------------------------------------------------------------
    def set_wx(v): eng.steady_wind[0] = v
    def set_wy(v): eng.steady_wind[1] = v
    def set_wz(v): eng.steady_wind[2] = v
    pl.add_slider_widget(set_wx, [-0.08, 0.08], 0.0, title="wind x [N]",
                         pointa=(0.03, 0.09), pointb=(0.32, 0.09), style="modern")
    pl.add_slider_widget(set_wy, [-0.08, 0.08], 0.0, title="wind y [N]",
                         pointa=(0.36, 0.09), pointb=(0.65, 0.09), style="modern")
    pl.add_slider_widget(set_wz, [-0.06, 0.06], 0.0, title="wind z [N]",
                         pointa=(0.69, 0.09), pointb=(0.98, 0.09), style="modern")

    names = list(eng.controllers)
    for i, nm in enumerate(names[:3], start=1):
        pl.add_key_event(str(i), (lambda nm=nm: eng.set_controller(nm)))
    for key, vec in {"Up": [0, GUST, 0], "Down": [0, -GUST, 0],
                     "Left": [-GUST, 0, 0], "Right": [GUST, 0, 0],
                     "z": [0, 0, GUST], "x": [0, 0, -GUST]}.items():
        pl.add_key_event(key, (lambda v=vec: eng.add_gust(v)))
    pl.add_key_event("r", eng.reset)
    pl.add_key_event("c", lambda: eng.steady_wind.__setitem__(slice(None), 0.0))

    # --- animation ---------------------------------------------------------------
    def tick():
        fr = eng.step_frame()
        M = _pose_matrix(fr.pos, fr.R)
        body_a.user_matrix = M
        for a in arm_as:
            a.user_matrix = M
        for a in prop_as:
            a.user_matrix = M
        up = fr.R[:, 2]
        for a, tip, T in zip(thrust_as, tips, fr.rotors):
            base = fr.pos + fr.R @ tip
            a.mapper.SetInputData(pv.Arrow(start=base, direction=up,
                                           scale=float(0.05 + 6 * T)))
        w = fr.wind
        if np.linalg.norm(w) > 1e-4:
            wind_a[0].mapper.SetInputData(pv.Arrow(
                start=fr.pos + [0, 0, 0.3], direction=w, scale=float(8 * np.linalg.norm(w))))
        if fr.trail.shape[0] > 2:
            if trail_a[0] is not None:
                pl.remove_actor(trail_a[0])
            trail_a[0] = pl.add_mesh(pv.Spline(fr.trail, 200), color="#56b4e9",
                                     line_width=2)
        pl.add_text(fr.hud, position="lower_right", font_size=9,
                    color="#cfe3ff", name="hud")

    pl.add_callback(tick, interval=16)
    pl.camera_position = [(3.2, -3.2, 2.6), tuple(HOVER), (0, 0, 1)]
    pl.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
