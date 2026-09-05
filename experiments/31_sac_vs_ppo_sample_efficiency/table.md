# Experiment 31 - SAC vs PPO: sample efficiency on pendulum swing-up

pendulum swing-up, 60000 env-step budget, greedy eval every 4000.

'steps to threshold' = env steps to first reach return >= -966 (the hybrid's return minus 150).

| method | final eval return | steps to threshold |
| --- | --- | --- |
| SAC | -363.6 | 8000 |
| PPO | -1339.9 | n/a |
| Energy + LQR hybrid | -815.8 | 0 (no training) |

