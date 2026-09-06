# Performance-regression bench

Times aimct's hot paths and fails CI if one gets *algorithmically* slower.

```bash
python benchmarks/perf/bench.py                # run, print a table
python benchmarks/perf/bench.py --check        # compare to baseline.json, exit 1 on a regression
python benchmarks/perf/bench.py --update       # (re)write baseline.json from this run
python benchmarks/perf/bench.py --only care_n16 ilqr_iter_cartpole
```

## What's measured

| case | routine |
| --- | --- |
| `care_n8` / `care_n16` | `solve_care` (the CARE Schur solve) at n = 8 / 16 |
| `qp_dense_n30` | `solve_qp` — dense active-set QP, 30 vars + box |
| `mpc_update_cartpole_N20` | one `LinearMPC.update()` (condense + QP) |
| `ilqr_iter_cartpole` | one `iLQR.solve(max_iter=1)` backward/forward sweep |
| `simulate_100_steps_quadrotor` | 100 RK4 steps of `PlanarQuadrotor` via `simulate` |
| `kf_step_n12` | one `DiscreteKalmanFilter` predict + update, n = 12 |
| `hinf_mixsyn_small` | one `mixsyn` H∞ synthesis (γ-bisection) |

Each is warmed up, then the **median** of `--reps` timed reps is recorded.

## The gate

`--check` fails when any `now / baseline` ratio exceeds `--threshold`
(default **2.0×**). It is deliberately loose — shared CI runners have large
timing variance, and the goal is to catch a routine that changed complexity
class, not a 10 % wobble.

`baseline.json` is **environment-specific** (CPU, OS, Python). The committed
copy is refreshed by the `perf` workflow on every push to `main` (it runs
`--update` and commits the new numbers back with `[skip ci]`), so pull
requests are always checked against a baseline measured on the same CI
image. A locally-run `--check` against a CI-measured baseline (or vice
versa) will show large ratios — that's expected; regenerate with `--update`
for local work.

## Adding a case

Add a `case_<name>()` returning a zero-arg callable to `CASES` in
`bench.py`, then `python benchmarks/perf/bench.py --only <name> --update`
… actually run the full `--update` so every number is from one session.
