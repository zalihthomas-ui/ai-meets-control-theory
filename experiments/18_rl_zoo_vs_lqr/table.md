# Experiment 18 - RL zoo vs LQR (cart-pole balance [FULL])

| agent | greedy return | held /200 | env steps | train s | params |
| --- | --- | --- | --- | --- | --- |
| LQR (analytic) | -0.3 | 200.0 | 0 | 0.0 | 4 |
| Tabular Q-learning | -55.5 | 92.0 | 300,000 | 7.1 | 354,375 |
| DQN (scratch) | -2.6 | 200.0 | 90,177 | 68.7 | 4,805 |
| PPO (scratch) | -0.3 | 200.0 | 240,000 | 72.6 | 4,545 |
| PPO (Stable-Baselines3) | -44.9 | 52.5 | 120,000 | 129.6 | 9,091 |
