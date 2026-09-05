"""06. Run an interactive physics sandbox in headless evaluation mode.

Instantiates a real-time sandbox with switchable controllers and an external
disturbance model (wrist payload perturbation), executing physics steps headlessly.
"""

import numpy as np
from aimct.systems import TwoLinkArm
from aimct.viz import Sandbox, Disturbance

# 1. Instantiate the two-link robot manipulator
arm = TwoLinkArm()
target_q = np.array([0.4, 0.2])  # Desired joint angles [rad]

# 2. Define switchable joint-space feedback controllers
class GravityCompPD:
    def reset(self):
        pass
    def update(self, x, dt: float):
        q, dq = x[:2], x[2:]
        return arm.G(q) + 50.0 * (target_q - q) - 10.0 * dq

controllers = {
    "PD+Gravity": GravityCompPD(),
}

# 3. Attach interactive disturbance model (wrist payload perturbation)
dist = Disturbance(
    sliders=[("payload [kg]", 0.0, 0.5, 0.0)],
    on_slider=lambda s, name, val: setattr(s.system, "payload", val),
)

# 4. Create sandbox and evaluate headlessly (without opening GUI window)
box = Sandbox(arm, controllers, x0=np.array([0.2, 0.1, 0.0, 0.0]), target=target_q, disturbance=dist)
stats = box.headless(steps=200, quiet=False)

print(f"Active Controller: {stats['controller']}")
print(f"Mean Tail Tracking Error: {stats['mean_err_tail_mm']:.2f} mm")
