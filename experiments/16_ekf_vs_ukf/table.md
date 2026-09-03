# Experiment 16 - EKF vs UKF on a nonlinear pendulum measurement

`recovered` = mean state-estimate error over the final 50 steps < 0.3.

| case | filter | rms_err | final_err | recovered |
| --- | --- | --- | --- | --- |
| hard: sin/cos, bad init | EKF | 6.514 | 6.282 | False |
| hard: sin/cos, bad init | UKF | 1.382 | 0.068 | True |
| mild: sin/rate, good init | EKF | 0.021 | 0.026 | True |
| mild: sin/rate, good init | UKF | 0.021 | 0.026 | True |
