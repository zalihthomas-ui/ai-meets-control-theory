"""03. Track a geometric reference trajectory with a mobile robot.

Runs a differential-drive robot along a cubic spline waypoint path,
evaluating cross-track error with the standardized tracking harness.
"""

import numpy as np
from aimct.systems import DifferentialDriveRobot
from aimct.trajectories import Spline
from aimct.benchmarks.tracking import track_trajectory

# 1. Define a 5-waypoint cubic spline reference path
waypoints = np.array([[0.0, 0.0], [1.0, 0.6], [2.2, -0.2], [3.2, 0.8], [4.5, 0.0]])
path = Spline(waypoints, knots=np.linspace(0.0, 10.0, len(waypoints)))
robot = DifferentialDriveRobot(v_ref=0.15)

# 2. Define a simple geometric path-following controller
class PurePursuit:
    def __init__(self, target_path, lookahead: float = 0.25, v_cruise: float = 0.15):
        self.path = target_path
        self.L = lookahead
        self.v = v_cruise
        self.t = 0.0

    def update(self, x, dt: float):
        self.t += dt
        target = self.path.pos(min(self.t + self.L, self.path.duration))
        dx, dy = target[0] - x[0], target[1] - x[1]
        heading_err = np.arctan2(dy, dx) - x[2]
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        omega_cmd = 2.5 * heading_err
        return np.array([self.v, np.clip(omega_cmd, -robot.omega_max, robot.omega_max)])

# 3. Track and score trajectory performance
res = track_trajectory(robot, {"PurePursuit": PurePursuit(path)}, path,
                       x0=np.zeros(5), dt=0.02, t_final=path.duration, pos_index=(0, 1))
print(res.to_markdown())
