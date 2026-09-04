r"""A real 3-D PyVista scene for any :class:`~aimct.viz.sandbox.Sandbox` whose
system is a planar :class:`aimct.systems.TwoLinkArm`.

The physics stays exactly what it is — one vertical plane — this only changes
how it is drawn: cylindrical links, spherical joints, real lighting and an
orbit camera you can spin freely (which is also the honest way to *see* that
the motion is planar: spin it edge-on and the arm collapses to a line).  Every
slider / hotkey a sandbox declares on its :class:`~aimct.viz.sandbox.
Disturbance` is wired up automatically, so `live_arm` and `live_arm_balance`
each get a 3-D view with zero extra control-panel code.

    from aimct.viz.pv_arm import run_pyvista_arm
    run_pyvista_arm(build(), title="my arm sandbox")
"""

from __future__ import annotations

import numpy as np

__all__ = ["run_pyvista_arm", "embed"]

# matplotlib's lower-case arrow-key names -> PyVista/VTK's capitalised ones
_KEYMAP = {"up": "Up", "down": "Down", "left": "Left", "right": "Right"}


def embed(p2d) -> np.ndarray:
    """The arm's 2-D ``(x, y)`` plane, laid into 3-D as the ``x``-``z`` plane
    (``y = 0``) so "up" in 2-D is "up" in the 3-D scene too."""
    p2d = np.asarray(p2d, float)
    return np.array([p2d[0], 0.0, p2d[1]])


def _translate(p3):
    M = np.eye(4)
    M[:3, 3] = p3
    return M


def run_pyvista_arm(box, *, title: str, show_payload: bool = False,
                    camera=None) -> int:
    """Open the 3-D window; blocks until closed. Returns 0."""
    try:
        import pyvista as pv
    except Exception as exc:                                   # pragma: no cover
        print(f"pyvista not available ({exc}); try  pip install pyvista  "
              f"or run this sandbox's plain run.py (matplotlib).")
        return 1

    arm = box.system
    r1, r2 = 0.045 * arm.l1 / 0.5, 0.038 * arm.l2 / 0.4         # link radii

    pl = pv.Plotter(window_size=(1180, 860), lighting="light_kit")
    pl.set_background("#0d1017", top="#1b2740")
    try:
        pl.enable_anti_aliasing("ssaa")
        pl.enable_ssao(radius=0.15, bias=0.005)
        pl.enable_depth_peeling(8)
    except Exception:
        pass
    pl.add_text(title, position="upper_left", font_size=11, color="#7fb0ff")

    reach = arm.l1 + arm.l2
    grid = pv.Plane(center=(0, 0, -0.02), direction=(0, 1, 0),
                    i_size=2.4 * reach, j_size=2.4 * reach,
                    i_resolution=16, j_resolution=16)
    pl.add_mesh(grid, style="wireframe", color="#243049", line_width=1)
    envelope = pv.Circle(radius=reach, resolution=64).rotate_x(90, inplace=False)
    pl.add_mesh(envelope, style="wireframe", color="#2c3a55", line_width=1,
               opacity=0.5)

    mount = pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0),
                        radius=0.06 * reach, height=0.05 * reach)
    pl.add_mesh(mount, color="#2b2b2b")

    link1 = pl.add_mesh(pv.Tube(pointa=(0, 0, 0), pointb=(1e-3, 0, 0), radius=r1),
                        color="#d55e00", smooth_shading=True, specular=0.3)
    link2 = pl.add_mesh(pv.Tube(pointa=(0, 0, 0), pointb=(1e-3, 0, 0), radius=r2),
                        color="#eda876", smooth_shading=True, specular=0.3)
    elbow = pl.add_mesh(pv.Sphere(radius=1.3 * r1), color="#ffffff")
    tip = pl.add_mesh(pv.Sphere(radius=1.5 * r1), color="#d55e00")
    target_actor = None
    if box.target is not None:
        target_actor = pl.add_mesh(pv.Sphere(radius=0.03 * reach), color="#56b4e9",
                                   opacity=0.85)
    payload_actor = None
    if show_payload:
        payload_actor = pl.add_mesh(pv.Sphere(radius=1e-4), color="#d62728",
                                    opacity=0.55, name="payload")
    trail_actor = [None]
    hud = pl.add_text("", position="lower_right", font_size=9,
                      color="#cfe3ff", name="hud")

    # -- wire every slider / hotkey the sandbox already declares -------------
    slider_widgets = []                      # (widget, name, lo, hi, cb) for "surprise me"
    for i, (name, lo, hi, init) in enumerate(box.dist.sliders):
        def _cb(v, nm=name):
            box.knobs[nm] = v
            if box.dist.on_slider:
                box.dist.on_slider(box, nm, v)
        w = pl.add_slider_widget(_cb, [lo, hi], value=init, title=name,
                                 pointa=(0.03 + 0.34 * (i % 2), 0.06 + 0.05 * (i // 2)),
                                 pointb=(0.31 + 0.34 * (i % 2), 0.06 + 0.05 * (i // 2)),
                                 style="modern")
        slider_widgets.append((w, name, lo, hi, _cb))
    for i, name in enumerate(box.names[:9], start=1):
        pl.add_key_event(str(i), (lambda nm=name: box.set_controller(nm)))
    for key, _label, fn in box.dist.hotkeys:
        pl.add_key_event(_KEYMAP.get(key, key), (lambda fn=fn: fn(box)))
    pl.add_key_event("r", box.reset)

    # -- the same creative/helpful additives as the matplotlib Sandbox --------
    help_lines = [title, ""]
    if box.dist.help_text:
        help_lines += [box.dist.help_text, ""]
    help_lines.append(f"controllers  (keys 1..{len(box.names)}):")
    help_lines += [f"  {i + 1}. {n}" for i, n in enumerate(box.names)]
    if box.dist.sliders:
        help_lines += ["", "sliders:"]
        help_lines += [f"  {n}  [{lo:g} .. {hi:g}]" for n, lo, hi, _ in box.dist.sliders]
    if box.dist.hotkeys:
        help_lines += ["", "hotkeys:"]
        help_lines += [f"  {k}  {lbl}" for k, lbl, _ in box.dist.hotkeys]
    help_lines += ["", "r  reset", "h  toggle this help",
                  "g  surprise me (randomise every disturbance)",
                  "c  save a snapshot PNG"]
    help_actor = [None]

    def _toggle_help():
        if help_actor[0] is None:
            help_actor[0] = pl.add_text("\n".join(help_lines), position=(0.27, 0.15),
                                        viewport=True, font_size=10, color="#eaf2ff",
                                        shadow=True, name="help")
        else:
            pl.remove_actor(help_actor[0])
            help_actor[0] = None

    def _surprise_me():
        rng = np.random.default_rng()
        for w, name, lo, hi, cb in slider_widgets:
            v = float(rng.uniform(lo, hi))
            w.GetRepresentation().SetValue(v)
            cb(v)
        if box.dist.hotkeys:
            _, _, fn = box.dist.hotkeys[int(rng.integers(len(box.dist.hotkeys)))]
            fn(box)

    def _snapshot():
        from datetime import datetime
        from pathlib import Path as _P

        out = _P("snapshots"); out.mkdir(exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in title)
        path = out / f"{slug}_{datetime.now():%Y%m%d_%H%M%S}.png"
        pl.screenshot(str(path))
        print(f"saved {path}")

    pl.add_key_event("h", _toggle_help)
    pl.add_key_event("g", _surprise_me)
    pl.add_key_event("c", _snapshot)

    trail: list = []

    def tick(_step=None):
        u = box.step()
        q = np.asarray(box.x, float)[:2]
        base3 = np.zeros(3)
        elbow2 = np.array([arm.l1 * np.cos(q[0]), arm.l1 * np.sin(q[0])])
        tip2 = elbow2 + np.array([arm.l2 * np.cos(q[0] + q[1]),
                                  arm.l2 * np.sin(q[0] + q[1])])
        e3, w3 = embed(elbow2), embed(tip2)
        link1.mapper.SetInputData(pv.Tube(pointa=base3, pointb=e3, radius=r1))
        link2.mapper.SetInputData(pv.Tube(pointa=e3, pointb=w3, radius=r2))
        elbow.user_matrix = _translate(e3)
        tip.user_matrix = _translate(w3)
        if target_actor is not None:
            target_actor.user_matrix = _translate(embed(box.target))
        if payload_actor is not None:
            mp = getattr(arm, "payload", 0.0)
            rad = 0.0 if mp <= 0 else (0.03 + 0.10 * mp) * reach
            payload_actor.mapper.SetInputData(pv.Sphere(radius=max(rad, 1e-4),
                                                         center=tuple(w3)))
        trail.append(w3)
        if len(trail) > 3:
            if trail_actor[0] is not None:
                pl.remove_actor(trail_actor[0])
            trail_actor[0] = pl.add_mesh(pv.Spline(np.array(trail[-240:]), 150),
                                         color="#d55e00", opacity=0.35, line_width=2)
        text = "\n".join(_hud_text(box, u))
        pl.add_text(text, position="lower_right", font_size=9,
                    color="#cfe3ff", name="hud")

    # PyVista's Plotter has no add_callback in every version; add_timer_event
    # is the stable public API (its callback takes the step index).
    pl.add_timer_event(max_steps=10_000_000, duration=33, callback=tick)
    if camera is not None:
        pl.camera_position = camera
    else:
        # frame the whole reachable workspace (a fixed circle of radius
        # `reach`), not the current pose - so the view never crops as the
        # arm moves, at a view angle that also reads as a plane when orbited.
        view_dir = np.array([1.0, -1.35, 0.9]); view_dir /= np.linalg.norm(view_dir)
        dist = (1.05 * reach) / np.tan(np.radians(15))
        look_at = (0.0, 0.0, 0.35 * reach)
        pl.camera_position = [tuple(view_dir * dist), look_at, (0, 0, 1)]
    pl.show()
    return 0


def _hud_text(box, u):
    from .artists import get_artist

    art = get_artist(box.system)
    aux = {}
    if box.target is not None:
        aux["target"] = box.target
    return [f"[ {box.active} ]", ""] + art.hud_lines(box.x, u, box.t, aux)
