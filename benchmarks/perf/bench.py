#!/usr/bin/env python
r"""Performance-regression bench for aimct's hot paths.

Times the handful of routines that dominate a real run -- the Riccati solve,
a dense QP, one MPC step, one iLQR iteration, an integration step, a Kalman
update, an H-infinity synthesis -- across a couple of problem sizes, and
compares the medians against a committed baseline.

    python benchmarks/perf/bench.py                 # run, print a table
    python benchmarks/perf/bench.py --json out.json # ... and write results
    python benchmarks/perf/bench.py --check         # compare to baseline.json, exit 1 on a regression
    python benchmarks/perf/bench.py --update        # rewrite baseline.json from this run

The regression gate (`--check`) is deliberately loose (default 2.0x) -- shared
CI runners have large timing variance; the point is to catch a routine that
got *algorithmically* slower, not a 10% wobble. Tighten with --threshold.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline.json"


# --------------------------------------------------------------------- cases


def _rand_lti(n, m, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.4, size=(n, n)) - 0.6 * np.eye(n)  # bias toward stable
    B = rng.normal(size=(n, m))
    return A, B


def case_care_n8():
    from aimct.controllers import solve_care

    A, B = _rand_lti(8, 2, 0)
    Q, R = np.eye(8), np.eye(2)
    return lambda: solve_care(A, B, Q, R)


def case_care_n16():
    from aimct.controllers import solve_care

    A, B = _rand_lti(16, 4, 1)
    Q, R = np.eye(16), np.eye(4)
    return lambda: solve_care(A, B, Q, R)


def case_qp_dense_n30():
    from aimct.controllers.mpc import solve_qp

    rng = np.random.default_rng(2)
    M = rng.normal(size=(30, 30))
    H = M @ M.T + np.eye(30)
    g = rng.normal(size=30)
    lb, ub = -np.ones(30), np.ones(30)
    return lambda: solve_qp(H, g, lb=lb, ub=ub)


def case_mpc_update_cartpole_N20():
    from aimct.controllers import LinearMPC
    from aimct.systems import CartPole

    cp = CartPole()
    A, B = cp.linearize()
    mpc = LinearMPC(A, B, Q=np.diag([10.0, 1, 10, 1]), R=np.array([[0.1]]), N=20,
                    u_bounds=(-20.0, 20.0))
    x = np.array([0.0, 0.0, 0.3, 0.0])
    return lambda: mpc.update(x, 0.02)


def case_ilqr_iter_cartpole():
    from aimct.controllers import iLQR
    from aimct.systems import CartPole

    cp = CartPole()
    opt = iLQR.from_system(cp, dt=0.02, horizon=50,
                           Q=np.diag([1.0, 1, 1, 1]), R=np.array([[0.01]]),
                           x_ref=np.array([0.0, 0.0, np.pi, 0.0]))
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    return lambda: opt.solve(x0, max_iter=1)


def case_simulate_100_steps_quadrotor():
    from aimct.simulate import simulate
    from aimct.systems import PlanarQuadrotor

    q = PlanarQuadrotor()
    ug = float(np.ravel(getattr(q, "u_eq", [q.m * q.g / 2.0]))[0]) if hasattr(q, "m") else 0.0
    ctrl = lambda y, dt: np.array([ug, ug])
    x0 = np.zeros(q.n_states)
    return lambda: simulate(q, ctrl, x0=x0, dt=0.01, t_final=1.0)


def case_kf_step_n12():
    from aimct.estimation import DiscreteKalmanFilter

    A, B = _rand_lti(12, 2, 3)
    Ad = np.eye(12) + 0.01 * A
    C = np.eye(4, 12)
    kf = DiscreteKalmanFilter(Ad, 0.01 * B, C, 1e-3 * np.eye(12), 1e-2 * np.eye(4), dt=0.01)
    y = np.zeros(4)
    u = np.zeros(2)

    def step():
        kf.predict(u)
        kf.update(y)

    return step


def case_hinf_mixsyn_small():
    from aimct.controllers import StateSpace, mixsyn, weight_S, weight_T

    G = StateSpace.from_tf([12.0], [1.0, 4.0, 3.0])       # 12/((s+1)(s+3))
    W_S = weight_S(wb=1.0, A=1e-2, M=2.0)
    W_KS = StateSpace.gain([[1e-1]])                       # control penalty -> D12 full rank (DGKF A2)
    W_T = weight_T(wb=20.0, A=0.1, M=2.0)
    return lambda: mixsyn(G, W_S, W_KS, W_T, tol=1e-4)


CASES = {
    "care_n8": case_care_n8,
    "care_n16": case_care_n16,
    "qp_dense_n30": case_qp_dense_n30,
    "mpc_update_cartpole_N20": case_mpc_update_cartpole_N20,
    "ilqr_iter_cartpole": case_ilqr_iter_cartpole,
    "simulate_100_steps_quadrotor": case_simulate_100_steps_quadrotor,
    "kf_step_n12": case_kf_step_n12,
    "hinf_mixsyn_small": case_hinf_mixsyn_small,
}


# --------------------------------------------------------------------- driver


def time_case(build, *, reps: int, warmup: int) -> float:
    """Median wall time [ms] of one call, over `reps` timed reps."""
    fn = build()
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def run_all(reps: int, warmup: int, only: list[str] | None) -> dict:
    names = only or list(CASES)
    results = {}
    for name in names:
        if name not in CASES:
            print(f"  ! unknown case {name!r}", file=sys.stderr)
            continue
        ms = time_case(CASES[name], reps=reps, warmup=warmup)
        results[name] = round(ms, 4)
        print(f"  {name:32s} {ms:9.3f} ms")
    return results


def check(results: dict, threshold: float) -> int:
    if not BASELINE.exists():
        print(f"no baseline at {BASELINE} -- run with --update first", file=sys.stderr)
        return 2
    base = json.loads(BASELINE.read_text())["results"]
    worst = 1.0
    regressions = []
    print(f"\n  {'case':32s} {'baseline':>10s} {'now':>10s} {'ratio':>8s}")
    for name, now in results.items():
        b = base.get(name)
        if b is None:
            print(f"  {name:32s} {'(new)':>10s} {now:10.3f}")
            continue
        ratio = now / b if b else float("inf")
        worst = max(worst, ratio)
        flag = "  <-- REGRESSION" if ratio > threshold else ""
        print(f"  {name:32s} {b:10.3f} {now:10.3f} {ratio:8.2f}x{flag}")
        if ratio > threshold:
            regressions.append((name, ratio))
    print(f"\n  worst ratio {worst:.2f}x  (gate {threshold:.2f}x)")
    if regressions:
        print("  FAIL: " + ", ".join(f"{n} {r:.2f}x" for n, r in regressions))
        return 1
    print("  OK")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, help="write results JSON here")
    p.add_argument("--check", action="store_true", help="compare to baseline.json, exit 1 on regression")
    p.add_argument("--update", action="store_true", help="(re)write baseline.json from this run")
    p.add_argument("--threshold", type=float, default=2.0, help="max now/baseline ratio before --check fails")
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--only", nargs="*", help="run only these cases")
    args = p.parse_args(argv)

    print(f"aimct perf bench  |  python {platform.python_version()}  {platform.system()} {platform.machine()}")
    payload = {
        "meta": {
            "python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
            "reps": args.reps,
        },
        "results": run_all(args.reps, args.warmup, args.only),
    }

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {args.json}")
    if args.update:
        BASELINE.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {BASELINE}")
    if args.check:
        return check(payload["results"], args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
