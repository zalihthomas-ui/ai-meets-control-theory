# Live drone-vs-wind sandbox

An interactive real-time planar quadrotor (Crazyflie 2.0) holding a hover point
while **you** drive the wind. Switch controllers on the fly and watch which ones
droop, which ring, and which reject a steady gust outright.

```bash
python experiments/live_drone/live.py      # or:  python -m aimct live
```

Needs a desktop matplotlib backend (Tk ships with Python on Windows). Headless
smoke check: `python -m aimct live --headless`.

## Controls

| input | effect |
| --- | --- |
| **steady wind** slider | constant horizontal wind force [N] |
| **gust <<** / **gust >>** buttons | ~0.3 s hard gust |
| **← / →** arrow keys | gust left / right |
| **↑ / ↓** arrow keys | vertical gust |
| **mouse drag** on the sky | shove the drone toward the drag |
| **radio buttons** | switch controller live |
| **reset** / **R** | recentre the drone |
| **space** | clear the wind |

## Controllers

| name | behaviour |
| --- | --- |
| **LQR (stiff)** | Bryson-scaled full-state feedback. Fast, but a *steady* wind leaves a standing position offset (finite DC gain from disturbance to position) and a hard gust can ring against the thrust limit. |
| **LQR + integral (wind-adaptive)** | the same feedback plus an integral of position error, designed as one augmented LQR. Drives the steady offset to **zero** — the drone leans into a constant wind and sits exactly on target. The closest thing here to an "adaptive" hold. |
| **LQR (soft)** | low-gain feedback. Gentle and never saturates, but the largest steady droop. |

Headless numbers under a 0.03 N steady wind + a 0.05 N gust:

```
LQR (stiff)          steady-state error  153.8 mm
LQR + integral        steady-state error   10.4 mm
LQR (soft)           steady-state error  109.5 mm
```

The integral term is the whole story: constant disturbance rejection needs an
integrator (or an adaptive/observer-based feed-forward — Phase B). A pure
proportional state feedback, however well tuned, cannot null a steady wind.
