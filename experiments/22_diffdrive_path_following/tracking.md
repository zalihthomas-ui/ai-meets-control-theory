# Exp 22 - differential-drive path following

| controller | rms_err_mm | max_err_mm | rms_cross_track_mm | completion_pct | ctrl_energy | status |
| --- | --- | --- | --- | --- | --- | --- |
| Pure pursuit | 69.51 | 107.4 | 34.57 | 98.39 | 1.949 | OK |
| Stanley | 111.4 | 257 | 9.747 | 95.51 | 2.412 | OK |
| Path LQR | 96.45 | 229.6 | 9.248 | 96.15 | 2.278 | OK |
| Kinematic MPC | 128 | 290.4 | 19.16 | 94.88 | 2.366 | OK |
