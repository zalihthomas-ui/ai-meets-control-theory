"""04. Evaluate a controller on the Intelligent Control Challenge (ICC).

Runs a state-feedback controller against a blind plant (Track 1: Mass-Spring-Damper)
under hidden disturbances, scoring performance, effort, safety, robustness, and latency.
"""

import numpy as np
from aimct.benchmarks.challenge import Challenge, ChallengeController
from aimct.systems import MassSpringDamper
from aimct.controllers import LQR

# 1. Implement the ChallengeController interface
class SetpointLQR(ChallengeController):
    def __init__(self, spec):
        super().__init__(spec)
        A, B = MassSpringDamper(m=1.0, c=0.4, k=1.0).linearize()
        self.K = LQR(A, B, np.eye(2), np.array([[0.1]])).K
        self._A, self._Bpinv = A, np.linalg.pinv(B)
        self._xr = np.zeros(2)
        self._uff = np.zeros(1)

    def reset(self, target):
        self._xr = np.asarray(target, float)
        self._uff = -(self._Bpinv @ (self._A @ self._xr))

    def compute_action(self, obs, t: float):
        return self._uff - self.K @ (np.asarray(obs, float) - self._xr)

# 2. Run blind evaluation across 5 standardized benchmark axes
challenge = Challenge("track1-msd")
result = challenge.evaluate(SetpointLQR, seed=0)

# 3. Print the formatted challenge scoring report
print(result.report())
