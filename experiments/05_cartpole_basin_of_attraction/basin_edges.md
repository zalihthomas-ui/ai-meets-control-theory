# Experiment 05 - measured basin edges vs reference envelope

Nonlinear CartPole, RK4 dt=0.004, T=6.0 s, actuator |F| <= 20 N. *Caught* = state stays finite and mean |theta| over the last 10 % of the run is < 0.1 rad. Edges read off the 19x19 grid along the axes; predicted values from `docs/references/swingup-and-basin.md` section 3.1.

| Tuning | R | theta0 edge (thetadot0=0) | predicted | thetadot0 edge (theta0=0) | predicted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| balanced | 0.1 | 0.83 rad (48 deg) | 0.80 rad (46 deg) | 4.4 rad/s | 4.0 rad/s |
| aggressive | 0.01 | 0.92 rad (53 deg) | 0.80 rad (46 deg) | 5.3 rad/s | 4.0 rad/s |
| soft | 1 | 1.00 rad (57 deg) | 0.33 rad (19 deg) | 5.3 rad/s | 1.3 rad/s |

The reference envelope is a conservative Lyapunov sub-level-set estimate; the simulated recoverable set is larger, and with a +/-20 N actuator the linear law catches the pole from rest almost to the horizontal. The tunings separate mainly in the thetadot0 direction and in transient cost (see robustness_sweep.png), not in the from-rest angle limit.
