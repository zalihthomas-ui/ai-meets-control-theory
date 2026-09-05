"""05. Replay and animate a simulation with aimct.viz.

Generates a simulated trajectory and creates a synchronized visual replay
with system-specific geometry, trail history, and real-time telemetry HUD.
"""

import numpy as np
from aimct.systems import PlanarQuadrotor
from aimct.controllers import LQR
from aimct.simulate import simulate
from aimct.viz import animate

# 1. Simulate a 2D quadrotor stabilizing from an offset position
quad = PlanarQuadrotor()
A, B = quad.linearize()  # Linearized about hover thrust [u_hover, u_hover]
Q = np.diag([10.0, 10.0, 50.0, 1.0, 1.0, 5.0])
R = np.eye(2) * 0.1
ctrl = LQR(A, B, Q, R, u_ref=quad.u_hover)

traj = simulate(quad, ctrl, x0=[0.5, 0.5, 0.1, 0, 0, 0], dt=0.02, t_final=3.0)

# 2. Build the visual replay animation
replay = animate(traj, quad, title="Quadrotor Hover Stabilization")
print(f"Replay animation created: {type(replay.anim).__name__} for {len(traj.t)} steps")

# 3. Export to GIF/MP4 or display interactively
# replay.save("quadrotor_hover.gif")
# replay.show()
print("Ready for rendering or display.")
