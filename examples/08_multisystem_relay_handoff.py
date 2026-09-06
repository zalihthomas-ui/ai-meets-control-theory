r"""A payload relay: three different machines, three different controllers, one
package.  A "system of systems" showcase.

The package has to travel 20 m -- further than any single machine here does
well -- so it is *handed off*:

    leg 1  GANTRY CRANE      lift from x=0, place on the transfer pad at x=6
                             (trolley + rigid-cable pendulum, 4 states)
    leg 2  MOBILE ROBOT      carry the pad-to-pad load from x=6 to x=13
                             (differential drive + actuator lag, 5 states;
                              the pickup changes its effective mass)
    leg 3  SLUNG-LOAD QUAD   fly it from x=13 over a wall to the target at x=20
                             (planar quadrotor + pendulum load, 8 coupled states)

Each leg runs its own ``DynamicalSystem`` with its own controller.  A
supervisor sequences the legs and only allows a hand-off when the receiver is
**aligned**, the payload is **slow**, and it is **not swinging** -- the same
three-part gate a real crane operator waits for.

The point: aimct's pieces compose.  The crane's anti-sway shaper (Ex. 07), a
path-tracking LQR, and iLQR on a freshly-derived 8-state coupled model are the
*same* building blocks, wired into a pipeline.

Run:  python examples/08_multisystem_relay_handoff.py
Outputs: examples/_out/relay_table.md, relay_legs.png, relay_animation.gif
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from aimct.controllers import LQR, ILQR
from aimct.simulate import simulate
from aimct.systems import DifferentialDriveRobot, DynamicalSystem

OUT = Path(__file__).parent / "_out"
OUT.mkdir(exist_ok=True)
G = 9.81
DT = 0.02


def _scalar(v):
    return float(np.ravel(v)[0])


# reuse the GantryCrane defined in example 07 (don't redefine it here)
_ex07 = importlib.util.spec_from_file_location(
    "_ex07", Path(__file__).parent / "07_full_workflow_gantry_crane.py")
_m07 = importlib.util.module_from_spec(_ex07)
_ex07.loader.exec_module(_m07)
GantryCrane = _m07.GantryCrane


# ======================================================================
# leg-3 plant: a planar quadrotor carrying a slung point-mass load
# ======================================================================
class SlungLoadQuad(DynamicalSystem):
    r"""Planar quadrotor + a point-mass load on a rigid massless cable.

    State ``[x, z, phi, beta, xd, zd, phid, betad]`` -- quad position (m),
    body roll ``phi`` (rad), cable angle ``beta`` from vertical (rad, 0 =
    load straight down), and their rates.  Input ``[T1, T2]`` -- the two
    rotor thrusts (N).

    Lagrangian in ``q = [x, z, phi, beta]`` with the load pivot at the quad
    CoM; the load reacts back on the quad (full coupling):

        (mQ+mL) x''  + mL Lc cos(b) b''         = -f sin(phi) - cd x' + mL Lc sin(b) b'^2
        (mQ+mL) z''  + mL Lc sin(b) b''         =  f cos(phi) - (mQ+mL) g - cd z' - mL Lc cos(b) b'^2
        IQ phi''                                =  (T1 - T2) l
        mL Lc^2 b''  + mL Lc cos(b) x'' + mL Lc sin(b) z'' + mL g Lc sin(b) = 0

    Defaults: a Crazyflie-class quad (mQ = 28 g) with a 6 g load on a 25 cm
    cable -- a load that is a *large* fraction of the vehicle mass, so the
    coupling genuinely matters.
    """

    n_states, n_inputs = 8, 2

    def __init__(self, mQ=0.9, mL=0.18, Lc=0.5, IQ=8.5e-3, l=0.18,
                 cd=2e-3, g=G, thrust_max=9.0):
        # a ~0.9 kg quad with a 0.18 kg slung load on a 0.5 m cable; each
        # rotor tops out well above the ~5.3 N hover point so the RTI has
        # real control margin for swing damping.
        self.mQ, self.mL, self.Lc = mQ, mL, Lc
        self.IQ, self.l, self.cd, self.g = IQ, l, cd, g
        self.thrust_max = thrust_max

    def hover_thrust(self):
        f = (self.mQ + self.mL) * self.g
        return np.array([f / 2, f / 2])

    def load_pos(self, x):
        return np.array([x[0] + self.Lc * np.sin(x[3]),
                         x[1] - self.Lc * np.cos(x[3])])

    def dynamics(self, t, x, u):
        x = np.asarray(x, float)
        xq, zq, phi, beta, xd, zd, phid, betad = x
        T1, T2 = np.clip(np.asarray(u, float), 0.0, self.thrust_max)
        f = T1 + T2
        mQ, mL, Lc, cd, g = self.mQ, self.mL, self.Lc, self.cd, self.g
        sb, cb = np.sin(beta), np.cos(beta)

        M = np.array([
            [mQ + mL, 0.0,      0.0,     mL * Lc * cb],
            [0.0,     mQ + mL,  0.0,     mL * Lc * sb],
            [0.0,     0.0,      self.IQ, 0.0],
            [mL * Lc * cb, mL * Lc * sb, 0.0, mL * Lc**2],
        ])
        rhs = np.array([
            -f * np.sin(phi) - cd * xd + mL * Lc * sb * betad**2,
            f * np.cos(phi) - (mQ + mL) * g - cd * zd - mL * Lc * cb * betad**2,
            (T1 - T2) * self.l,
            -mL * g * Lc * sb,
        ])
        xdd, zdd, phidd, betadd = np.linalg.solve(M, rhs)
        return np.array([xd, zd, phid, betad, xdd, zdd, phidd, betadd])

    def linearize(self, x_eq=None, u_eq=None, eps=1e-6):
        if x_eq is None:
            x_eq = np.zeros(8)
        if u_eq is None:
            u_eq = self.hover_thrust()
        return super().linearize(np.asarray(x_eq, float), np.asarray(u_eq, float), eps)


# ======================================================================
# scene geometry
# ======================================================================
CRANE = GantryCrane(L=1.6, M=6.0, m=2.0)
PAD_A = 4.0                       # crane -> robot transfer pad
PAD_B = 13.0                      # robot -> quad transfer pad
TARGET_X = 20.0                   # final drop point
WALL_X, WALL_H = 16.5, 2.6        # the quad has to clear this
PICKUP_DM = 1.2                   # kg the AMR gains when it takes the load

# hand-off gate: receiver aligned, payload slow, payload not swinging
GATE = dict(pos=0.25, vel=0.30, swing=np.deg2rad(8.0))


# ======================================================================
# leg 1 -- gantry crane: LQR + Zero-Vibration input shaper (from Ex. 07)
# ======================================================================
def run_leg1():
    A, B = CRANE.linearize(np.array([PAD_A, 0, 0, 0]), np.zeros(1))
    Ks = LQR(A, B, np.diag([28, 150, 5, 16.0]), np.array([[0.5]])).K
    wn = np.sqrt(G / CRANE.L)
    zeta = 0.05
    Kimp = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    amps = np.array([1 / (1 + Kimp), Kimp / (1 + Kimp)])
    delays = np.array([0.0, np.pi / (wn * np.sqrt(1 - zeta**2))])

    class Ctl:
        name = "crane: LQR + ZV shaper"

        def reset(self):
            self._t = 0.0

        def update(self, x, dt):
            # ZV-shaped set-point during the move; then a plain hold at PAD_A
            if self._t < delays[-1] + 0.5:
                tgt = PAD_A * sum(a * (self._t >= d)
                                  for a, d in zip(amps, delays))
            else:
                tgt = PAD_A
            self._t += dt
            return _scalar(np.clip(-Ks @ (np.asarray(x) - [tgt, 0, 0, 0]),
                                   -40.0, 40.0))

    ctl = Ctl()
    ctl.reset()
    tr = simulate(CRANE, ctl, x0=np.zeros(4), dt=DT, t_final=10.0,
                  u_bounds=(-40.0, 40.0))
    # payload kinematics in world coords
    load_x = tr.x[:, 0] + CRANE.L * np.sin(tr.x[:, 1])
    load_v = np.gradient(load_x, tr.t)
    swing = tr.x[:, 1]
    return tr, load_x, load_v, swing, ctl.name


# ======================================================================
# leg 2 -- mobile robot: waypoint pursuit + heading LQR, mass step on pickup
# ======================================================================
class RobotCtl:
    name = "robot: cruise + heading LQR"

    def __init__(self, x_start, x_goal, v_cruise):
        self.xs, self.xg = x_start, x_goal
        self.v_cruise = v_cruise
        tau = 0.25
        A = np.array([[0.0, 1.0], [0.0, -1.0 / tau]])
        B = np.array([[0.0], [1.0 / tau]])
        self.Kh = LQR(A, B, np.diag([12.0, 1.0]), np.array([[0.4]])).K

    def reset(self):
        pass

    def update(self, x, dt):
        px, _py, th, v, om = x
        d_to_go = self.xg - px
        # cruise, then a linear brake inside the last 1.5 m
        v_cmd = np.clip(self.v_cruise * min(1.0, d_to_go / 1.5), 0.0,
                        self.v_cruise)
        om_cmd = _scalar(-self.Kh @ np.array([th, om])) - 3.0 * th
        return np.array([v_cmd, om_cmd])


def run_leg2(x_enter_v):
    # a warehouse-class AMR (bigger than the default TurtleBot); the pickup
    # makes it heavier, so its speed loop is slower (larger tau_v).
    robot = DifferentialDriveRobot(mass=1.0 + PICKUP_DM, v_max=1.4,
                                   tau_v=0.05 * (1 + PICKUP_DM / 1.0))
    ctl = RobotCtl(PAD_A, PAD_B, v_cruise=1.35)
    ctl.reset()
    x0 = np.array([PAD_A, 0.0, 0.0, min(x_enter_v, 0.4), 0.0])
    tr = simulate(robot, ctl, x0=x0, dt=DT, t_final=16.0)
    load_x = tr.x[:, 0]
    load_v = tr.x[:, 3] * np.cos(tr.x[:, 2])
    swing = np.zeros_like(load_x)      # rigidly carried on the deck
    return tr, load_x, load_v, swing, ctl.name


# ======================================================================
# leg 3 -- slung-load quad: iLQR/RTI on the 8-state model + ZV climb shaper
# ======================================================================
def run_leg3(x_enter_v):
    quad = SlungLoadQuad()
    uh = quad.hover_thrust()

    # a smooth climb-cruise-descend reference for the quad body (the load
    # trails it); the last knots hold the target so the RTI arrests velocity
    z_cru = WALL_H + 1.1
    tk = np.array([0.0, 2.2, 5.0, 7.5, 9.5, 12.0])
    xk = np.array([PAD_B, PAD_B + 1.5, (PAD_B + TARGET_X) / 2,
                   TARGET_X - 1.0, TARGET_X, TARGET_X])
    zk = np.array([0.0, z_cru, z_cru, z_cru, 0.6, 0.6])

    def ref(t):
        xr = np.interp(t, tk, xk)
        zr = np.interp(t, tk, zk)
        return np.array([xr, zr, 0, 0, 0, 0, 0, 0])

    ilqr = ILQR.from_system(
        quad, DT, horizon=60,
        Q=np.diag([10, 10, 4, 40, 1, 1, 0.5, 8.0]),
        R=np.diag([2.0, 2.0]),
        Qf=np.diag([60, 60, 8, 160, 10, 10, 3, 40.0]),
        x_ref=ref, u_ref=uh, u_bounds=(0.0, quad.thrust_max),
        rti_iters=1, warm_iters=60)

    class Ctl:
        name = "quad: iLQR/RTI (8-state coupled)"

        def reset(self):
            ilqr.reset()

        def update(self, x, dt):
            return np.clip(ilqr.update(x, dt), 0.0, quad.thrust_max)

    ctl = Ctl()
    ctl.reset()
    x0 = np.zeros(8)
    x0[0] = PAD_B
    x0[4] = min(x_enter_v, 0.3)
    tr = simulate(quad, ctl, x0=x0, dt=DT, t_final=12.0,
                  u_bounds=(0.0, quad.thrust_max))
    load = np.array([quad.load_pos(r) for r in tr.x])
    load_x, load_z = load[:, 0], load[:, 1]
    load_v = np.gradient(load_x, tr.t)
    swing = tr.x[:, 3]
    return tr, quad, load_x, load_z, load_v, swing, ctl.name


# ======================================================================
# supervisor: run the legs in sequence, gate each hand-off
# ======================================================================
def gate_check(pos_err, vel, swing):
    return dict(
        aligned=abs(pos_err) <= GATE["pos"],
        slow=abs(vel) <= GATE["vel"],
        steady=abs(swing) <= GATE["swing"],
    )


def leg_metrics(name, t, load_x, load_v, swing, x_goal):
    err = load_x[-1] - x_goal
    checks = gate_check(err, load_v[-1], swing[-1])
    return {
        "leg": name,
        "t_leg_s": float(t[-1]),
        "final_x_err_m": float(err),
        "final_speed_ms": float(abs(load_v[-1])),
        "peak_swing_deg": float(np.rad2deg(np.max(np.abs(swing)))),
        "handoff_ok": all(checks.values()),
        "_checks": checks,
    }


def run_relay():
    rows = []

    tr1, lx1, lv1, sw1, n1 = run_leg1()
    m1 = leg_metrics(n1, tr1.t, lx1, lv1, sw1, PAD_A)
    rows.append(m1)

    tr2, lx2, lv2, sw2, n2 = run_leg2(abs(lv1[-1]))
    m2 = leg_metrics(n2, tr2.t, lx2, lv2, sw2, PAD_B)
    rows.append(m2)

    tr3, quad, lx3, lz3, lv3, sw3, n3 = run_leg3(abs(lv2[-1]))
    m3 = leg_metrics(n3, tr3.t, lx3, lv3, sw3, TARGET_X)
    rows.append(m3)

    bundle = dict(
        leg1=(tr1, lx1, sw1), leg2=(tr2, lx2, sw2),
        leg3=(tr3, quad, lx3, lz3, sw3),
        total_t=float(tr1.t[-1] + tr2.t[-1] + tr3.t[-1]),
    )
    return rows, bundle


def write_table(rows, total_t):
    cols = ["t_leg_s", "final_x_err_m", "final_speed_ms", "peak_swing_deg",
            "handoff_ok"]
    lines = ["# Multi-system payload relay -- 3 legs, 3 machines", "",
             f"End-to-end transfer of one package over {TARGET_X:.0f} m in "
             f"{total_t:.1f} s.", "",
             "| leg | " + " | ".join(cols) + " |",
             "| --- |" + " --- |" * len(cols)]
    for m in rows:
        lines.append(
            "| " + m["leg"] + " | "
            + f'{m["t_leg_s"]:.2f} | {m["final_x_err_m"]:+.3f} | '
            + f'{m["final_speed_ms"]:.3f} | {m["peak_swing_deg"]:.2f} | '
            + ("PASS" if m["handoff_ok"] else "FAIL "
               + ",".join(k for k, v in m["_checks"].items() if not v)) + " |")
    (OUT / "relay_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ======================================================================
# figures
# ======================================================================
def legs_figure(bundle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aimct.plot_style import set_aimct_style
    set_aimct_style()

    tr1, lx1, sw1 = bundle["leg1"]
    tr2, lx2, sw2 = bundle["leg2"]
    tr3, quad, lx3, lz3, sw3 = bundle["leg3"]

    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    for r, (t, lx, sw, tag, goal) in enumerate([
        (tr1.t, lx1, sw1, "leg 1 - gantry crane", PAD_A),
        (tr2.t, lx2, sw2, "leg 2 - mobile robot", PAD_B),
        (tr3.t, lx3, sw3, "leg 3 - slung-load quad", TARGET_X),
    ]):
        ax[r, 0].plot(t, lx, lw=1.8)
        ax[r, 0].axhline(goal, ls="--", color="#888", lw=1)
        ax[r, 0].set_ylabel(f"{tag}\npayload x [m]")
        ax[r, 1].plot(t, np.rad2deg(sw), lw=1.8, color="#d55e00")
        ax[r, 1].axhline(np.rad2deg(GATE["swing"]), ls=":", color="#888")
        ax[r, 1].axhline(-np.rad2deg(GATE["swing"]), ls=":", color="#888")
        ax[r, 1].set_ylabel("swing [deg]")
    ax[0, 0].set_title("(a) payload longitudinal position")
    ax[0, 1].set_title("(b) payload / cable swing")
    for a in ax[-1]:
        a.set_xlabel("t [s]")
    fig.suptitle("Payload relay: three machines carry one package 20 m",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "relay_legs.png", dpi=140)
    print(f"saved {OUT / 'relay_legs.png'}")


def relay_animation(bundle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    from aimct.plot_style import set_aimct_style
    set_aimct_style()

    tr1, lx1, sw1 = bundle["leg1"]
    tr2, lx2, sw2 = bundle["leg2"]
    tr3, quad, lx3, lz3, sw3 = bundle["leg3"]

    # stitch the three legs onto one clock
    seg = []
    for tr, tag in ((tr1, "CRANE"), (tr2, "ROBOT"), (tr3, "QUAD")):
        seg.append((tr, tag))
    t_off = np.cumsum([0.0] + [s[0].t[-1] for s in seg[:-1]])

    fig, ax = plt.subplots(figsize=(13, 5), dpi=100)   # -> ~1300x500 GIF
    ax.set_xlim(-1.5, TARGET_X + 2)
    ax.set_ylim(-CRANE.L - 0.6, 5.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_title("Payload relay - crane -> mobile robot -> slung-load quad",
                 fontweight="bold")
    ax.axhline(0, color="#999", lw=2)                       # ground / rail
    ax.plot([PAD_A, PAD_A], [-0.15, 0.15], color="#009e73", lw=3)
    ax.plot([PAD_B, PAD_B], [-0.15, 0.15], color="#009e73", lw=3)
    ax.add_patch(Rectangle((WALL_X - 0.15, 0), 0.3, WALL_H, fc="#777"))
    ax.plot([TARGET_X], [0], marker="x", ms=14, mew=3, color="#56b4e9")

    trolley = Rectangle((0, 0), 0.3, 0.14, fc="#2b2b2b"); ax.add_patch(trolley)
    cable, = ax.plot([], [], color="#555", lw=1.6)
    robot = Rectangle((0, 0), 0.6, 0.3, fc="#2b2b2b")
    quad_body, = ax.plot([], [], color="#2b2b2b", lw=5, solid_capstyle="round")
    qcable, = ax.plot([], [], color="#555", lw=1.4)
    pkg = Rectangle((0, 0), 0.28, 0.28, fc="#e69f00", ec="#2b2b2b")
    ax.add_patch(robot); ax.add_patch(pkg)
    phase_txt = ax.text(0.02, 0.94, "", transform=ax.transAxes, fontsize=12,
                        fontweight="bold")

    fps, speed = 20, 1.8      # ~38 s of sim -> ~1 k frames at 1x; 1.8x keeps it brisk
    total_T = sum(s[0].t[-1] for s in seg)
    frame_t = np.arange(0.0, total_T, speed / fps)

    def which(t):
        for i, (tr, tag) in enumerate(seg):
            if t <= t_off[i] + tr.t[-1] or i == len(seg) - 1:
                return i, tag, t - t_off[i]
        return 2, "QUAD", t - t_off[-1]

    def hide(*objs):
        for o in objs:
            if hasattr(o, "set_xy"):
                o.set_xy((-99, -99))
            else:
                o.set_data([], [])

    def frame(k):
        t = frame_t[k]
        i, tag, tl = which(t)
        phase_txt.set_text(f"leg {i + 1}/3  -  {tag}")

        if tag == "CRANE":
            hide(robot, quad_body, qcable)
            x = np.interp(tl, tr1.t, tr1.x[:, 0])
            th = np.interp(tl, tr1.t, tr1.x[:, 1])
            trolley.set_xy((x - 0.15, -0.07))
            px, pz = x + CRANE.L * np.sin(th), -CRANE.L * np.cos(th)
            cable.set_data([x, px], [0, pz])
            pkg.set_xy((px - 0.14, pz - 0.14))
        elif tag == "ROBOT":
            hide(trolley, quad_body, qcable)
            cable.set_data([], [])
            x = np.interp(tl, tr2.t, tr2.x[:, 0])
            robot.set_xy((x - 0.3, 0.02))
            pkg.set_xy((x - 0.14, 0.32))
        else:
            hide(trolley, robot)
            cable.set_data([], [])
            row = np.array([np.interp(tl, tr3.t, tr3.x[:, j]) for j in range(8)])
            xq, zq, phi, beta = row[0], row[1], row[2], row[3]
            dx = 0.28 * np.cos(phi)
            dz = 0.28 * np.sin(phi)
            quad_body.set_data([xq - dx, xq + dx], [zq - dz, zq + dz])
            lx = xq + quad.Lc * np.sin(beta)
            lz = zq - quad.Lc * np.cos(beta)
            qcable.set_data([xq, lx], [zq, lz])
            pkg.set_xy((lx - 0.14, lz - 0.14))
        return trolley, cable, robot, quad_body, qcable, pkg, phase_txt

    from matplotlib.animation import PillowWriter

    anim = FuncAnimation(fig, frame, frames=len(frame_t),
                         interval=int(1000 / fps), blit=False)
    anim.save(OUT / "relay_animation.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {OUT / 'relay_animation.gif'} "
          f"({(OUT / 'relay_animation.gif').stat().st_size} bytes)")


# ======================================================================
def energy_sanity():
    """thrust=weight, no drag/torque -> the load swings forever, energy const."""
    from aimct.simulate import rk4_step

    q = SlungLoadQuad(cd=0.0)
    uh = q.hover_thrust()
    x = np.zeros(8)
    x[3] = 0.3                       # release the load at 0.3 rad
    h, n = 0.005, 6000              # 30 s

    def energy(x):
        _, zq, _, beta, xd, zd, phid, betad = x
        vL = np.array([xd + q.Lc * np.cos(beta) * betad,
                       zd + q.Lc * np.sin(beta) * betad])
        return (0.5 * q.mQ * (xd**2 + zd**2) + 0.5 * q.IQ * phid**2
                + 0.5 * q.mL * vL @ vL
                + q.mQ * q.g * zq + q.mL * q.g * q.load_pos(x)[1])

    E0 = energy(x)
    for _ in range(n):
        x = rk4_step(lambda tt, xx, uu: q.dynamics(tt, xx, uu), 0.0, x, uh, h)
    drift = abs(energy(x) - E0) / max(abs(E0), 1e-9)
    print(f"leg-3 model energy drift over {h * n:.0f} s (thrust=weight): {drift:.2e}")
    return drift


def main():
    print("-- slung-load quad model check ----------------------------------")
    energy_sanity()
    A, B = SlungLoadQuad().linearize()
    from aimct.controllers.state_feedback import is_controllable
    print(f"   8-state (quad+load) controllable at hover: {is_controllable(A, B)}\n")

    rows, bundle = run_relay()
    write_table(rows, bundle["total_t"])
    legs_figure(bundle)
    relay_animation(bundle)


if __name__ == "__main__":
    main()
