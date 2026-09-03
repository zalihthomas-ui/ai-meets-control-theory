# Experiment 17 - adaptive vs fixed control, drifting plant

| controller | rms_err_early | rms_err_late | settle_err_final | ctrl_energy |
| --- | --- | --- | --- | --- |
| LQR (nominal k=1) | 0.2573 | 0.4013 | 0.4217 | 180.7 |
| LQR (worst-case k=5) | 0.3577 | 0.5194 | 0.5406 | 123.3 |
| MRAC | 0.0008333 | 0.0008333 | 0.0008333 | 413.5 |
| GainScheduled LQR | 0.2916 | 0.5075 | 0.5396 | 142.3 |
