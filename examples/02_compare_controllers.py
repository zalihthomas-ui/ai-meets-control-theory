"""02. Compare multiple controllers under identical conditions.

Benchmarking single-loop PID vs optimal LQR on a nonlinear Inverted Pendulum
under identical initial tilt (theta0 = pi - 0.3 rad) and actuator torque limits.
"""

import numpy as np
from aimct.systems import Pendulum
from aimct.controllers import PID, LQR
from aimct.benchmarks import compare

# 1. Instantiate the plant and linearize about upright equilibrium (theta = pi)
sys = Pendulum()
A, B = sys.linearize()

# 2. Benchmark PID vs LQR under identical horizon, step size, and initial state
res = compare(
    sys,
    {
        "PID": PID(kp=40.0, ki=8.0, kd=6.0, setpoint=np.pi, output_limits=(-8.0, 8.0)),
        "LQR": LQR(A, B, Q=np.diag([10.0, 1.0]), R=np.array([[0.5]]), x_ref=[np.pi, 0.0]),
    },
    x0=[np.pi - 0.3, 0.0],
    dt=0.01,
    t_final=4.0,
    reference=np.pi,
    u_bounds=(-8.0, 8.0),
    output_index=0,
    # Hand PID only the measured angle; LQR receives full state [theta, theta_dot]
    measurement_fns={"PID": lambda t, x, u: x[[0]]},
)

# 3. Print the standardized Markdown comparison table
print(res.to_markdown())
