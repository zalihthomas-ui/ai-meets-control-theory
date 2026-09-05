"""01. Simulate a single dynamical system with optimal LQR control.

Runs a CartPole inverted pendulum under linear-quadratic regulation (LQR)
for 5.0 seconds from an initial 0.2 rad (11.5 deg) tilt.
"""

import numpy as np
from aimct.systems import CartPole
from aimct.controllers import LQR
from aimct.simulate import simulate

# 1. Instantiate the plant and linearize about upright equilibrium (x=0, theta=0)
sys = CartPole()
A, B = sys.linearize()

# 2. Design optimal state feedback gain via Algebraic Riccati Equation (CARE)
Q = np.diag([10.0, 1.0, 100.0, 10.0])  # State weights: [x, x_dot, theta, theta_dot]
R = np.array([[0.1]])                  # Control force weight
ctrl = LQR(A, B, Q, R)

# 3. Simulate forward in time with RK4 integration and actuator bounds
traj = simulate(sys, ctrl, x0=[0.0, 0.0, 0.2, 0.0], dt=0.01, t_final=5.0, u_bounds=(-20, 20))

print(f"Simulated {len(traj.t)} steps ({traj.t[-1]:.1f}s)")
print(f"Final Cart Position: {traj.x[-1, 0]:.4f} m, Final Pole Angle: {traj.x[-1, 2]:.4e} rad")
print(f"Trajectory Status: {'DIVERGED' if traj.diverged else 'STABLE'}")
