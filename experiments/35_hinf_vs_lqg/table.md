# Experiment 35 - H-infinity vs LQG under an unmodelled resonance

nominal `G(s) = 12 / ((s+1)(s+3))`; true plant `G_p = G * R` with `R(s)` a mode at `omega_r = 12` rad/s, `zeta_r = 0.08` (peak multiplicative error ~ 6).

H-infinity achieved `gamma = 0.981`.  `disk_alpha` = skew-0 disk margin `1 / max_w |S - 0.5|`; `dm_gain_db` / `dm_phase_deg` are the simultaneous gain/phase variation it guarantees.

| controller | plant | stable | gm_db | pm_deg | disk_alpha | dm_gain_db | dm_phase_deg | Sinf | Tinf |
|---|---|---|---|---|---|---|---|---|---|
| LQG | nominal | yes | 12.22 dB | 50.94 deg | 0.82 | 19.94 dB | 78.50 deg | 1.63 | 1.16 |
| LQG | perturbed | **NO** | -- | -- | -- | -- | -- | 1.18 | 1.98 |
| H-infinity | nominal | yes | inf | 90.33 deg | 1.98 | inf | 180.00 deg | 1.01 | 1.00 |
| H-infinity | perturbed | yes | 4.17 dB | 89.41 deg | 0.47 | 8.75 dB | 49.89 deg | 2.65 | 1.66 |

