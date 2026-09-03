r"""Intelligent Control Challenge (ICC) - evaluation engine.

Runs a submitted controller against a *hidden* plant under a fixed, seeded
disturbance / noise / reference profile and scores it on the five dimensions of
``docs/references/challenge-spec.md`` (performance, effort, safety, robustness,
compute), then the composite score of spec eq. 1.

The submission implements :class:`ChallengeController` and never sees the plant
matrices - only ``state_dim``, ``action_dim``, ``action_limit``, ``dt`` and the
observations.

    class MyController(ChallengeController):
        def compute_action(self, obs, t): ...

    ch = Challenge("track1-msd")
    result = ch.evaluate(MyController, seed=0)
    print(result.report())

The scoring rubric (:mod:`aimct.benchmarks.challenge_scoring`) and the Track-3/4
plant wrappers (:mod:`aimct.benchmarks.challenge_wrappers`) are authored
separately; this module composes them.  A run that never reaches / holds the
target is ``FAILED`` with score 0 regardless of its effort numbers.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..controllers import LQR
from ..simulate import Trajectory, simulate
from ..systems import CartPole, DCMotor, MassSpringDamper, Pendulum
from ..systems.base import DynamicalSystem
from .metrics import (
    control_energy,
    itae,
    peak_overshoot,
    rmse,
    saturation_duty_cycle,
    settling_time,
    slew_rate,
)

# famo's authoritative scoring rubric + Track-3/4 plant wrappers
from .challenge_scoring import WEIGHTS, robust_degradation, score_run  # noqa: F401
from .challenge_wrappers import ImpulseInjector, perturbed_system


__all__ = ["ChallengeController", "Challenge", "ChallengeResult", "TRACKS"]


# ---------------------------------------------------------------- submission API

class ChallengeController(ABC):
    """Standard controller interface for the Intelligent Control Challenge.

    The constructor is handed a ``spec`` dict (``state_dim``, ``action_dim``,
    ``action_limit``, ``dt``, ``target_state``, ``t_final``) - never the plant.
    """

    def __init__(self, spec: dict) -> None:
        self.state_dim = int(spec["state_dim"])
        self.action_dim = int(spec["action_dim"])
        self.action_limit = np.asarray(spec["action_limit"], dtype=float)
        self.dt = float(spec["dt"])
        self.spec = spec

    @abstractmethod
    def reset(self, target_state: np.ndarray) -> None:
        """Clear integrators / estimators / policy state for a new episode."""

    @abstractmethod
    def compute_action(self, observation: np.ndarray, t: float) -> np.ndarray:
        """Return ``u(t)`` (length ``action_dim``), inside ``+/- action_limit``."""

    def observe_feedback(self, next_obs, reward_or_cost, done, info) -> None:
        """Optional online-adaptation hook, called after each environment step."""

    def get_diagnostics(self) -> dict:
        """Optional telemetry (estimated params, Lyapunov value, QP iters ...)."""
        return {}


class _Adapter:
    """Bridge a :class:`ChallengeController` to the ``update(measurement, dt)``
    loop of :func:`aimct.simulate.simulate`, timing each ``compute_action``."""

    def __init__(self, cc: ChallengeController, target: np.ndarray,
                 action_limit: np.ndarray) -> None:
        self.cc = cc
        self.target = target
        self.action_limit = action_limit
        self.latencies: list[float] = []
        self._t = 0.0

    def reset(self) -> None:
        self.cc.reset(self.target)
        self.latencies = []
        self._t = 0.0

    def update(self, measurement, dt):
        t0 = time.perf_counter()
        u = np.atleast_1d(np.asarray(
            self.cc.compute_action(np.asarray(measurement, dtype=float), self._t),
            dtype=float))
        self.latencies.append(time.perf_counter() - t0)
        self._t += dt
        return np.clip(u, -self.action_limit, self.action_limit)


# ----------------------------------------------------------------- track config

@dataclass
class TrackSpec:
    key: str
    title: str
    system_factory: Callable[[], DynamicalSystem]
    x0: np.ndarray
    target: np.ndarray
    t_final: float
    dt: float
    action_limit: np.ndarray
    safe_lo: np.ndarray                 # hard state box: violation => DQ
    safe_hi: np.ndarray
    output_index: int                   # state channel scored for tracking
    noise_sigma: np.ndarray             # per-observation Gaussian sensor noise
    reference: Callable[[float], float] | None = None   # None => regulate to target[output_index]
    n_nominal: int = 10
    n_robust: int = 50
    impulse_b: float = 0.0             # Laplace scale for disturbance suite (0 => skip)
    impulse_rate_hz: float = 0.5
    success_tol: float = 0.05          # |output - target| over the last 10 % => "did the task"
    angular_output: bool = False       # wrap the tracked-output error to (-pi, pi]


def _msd_track() -> TrackSpec:
    return TrackSpec(
        key="track1-msd", title="Track 1 - precision regulation (mass-spring-damper)",
        system_factory=lambda: MassSpringDamper(m=1.0, c=0.4, k=1.0),
        x0=np.zeros(2), target=np.array([1.0, 0.0]), t_final=8.0, dt=2e-3,
        action_limit=np.array([25.0]),
        safe_lo=np.array([-2.0, -10.0]), safe_hi=np.array([3.0, 10.0]),
        output_index=0, noise_sigma=np.array([2e-3, 2e-3]),
        impulse_b=1.5,
    )


def _dcmotor_track() -> TrackSpec:
    m = DCMotor().reduced()                                 # 2-state [angle, speed]
    return TrackSpec(
        key="track1-dcmotor", title="Track 1 - DC-motor position control",
        system_factory=lambda: DCMotor().reduced(),
        x0=np.zeros(2), target=np.array([np.pi / 2, 0.0]), t_final=3.0, dt=1e-3,
        action_limit=np.array([float(getattr(m, "v_max", 12.0))]),
        safe_lo=np.array([-np.pi, -60.0]), safe_hi=np.array([2.0 * np.pi, 60.0]),
        output_index=0, noise_sigma=np.array([1e-3, 5e-3]),
        n_robust=30, impulse_b=0.0,
    )


def _cartpole_swingup_track() -> TrackSpec:
    return TrackSpec(
        key="track2-cartpole", title="Track 2 - cart-pole swing-up & balance",
        system_factory=lambda: CartPole(),
        x0=np.array([0.0, 0.0, np.pi, 0.0]), target=np.zeros(4),
        t_final=10.0, dt=2e-3, action_limit=np.array([20.0]),
        safe_lo=np.array([-2.4, -np.inf, -np.inf, -12.0]),
        safe_hi=np.array([2.4, np.inf, np.inf, 12.0]),
        output_index=2, noise_sigma=np.array([1e-3, 1e-3, 1e-3, 1e-3]),
        success_tol=0.2, angular_output=True,
        n_robust=30, impulse_b=2.0,
    )


def _pendulum_swingup_track() -> TrackSpec:
    return TrackSpec(
        key="track2-pendulum", title="Track 2 - pendulum swing-up & balance",
        system_factory=lambda: Pendulum(),
        x0=np.array([0.0, 0.0]), target=np.array([np.pi, 0.0]),
        t_final=8.0, dt=2e-3, action_limit=np.array([5.0]),
        safe_lo=np.array([-np.inf, -12.0]), safe_hi=np.array([np.inf, 12.0]),
        output_index=0, noise_sigma=np.array([1e-3, 1e-3]),
        success_tol=0.2, angular_output=True,
        n_robust=30, impulse_b=1.0,
    )


TRACKS: dict[str, Callable[[], TrackSpec]] = {
    "track1-msd": _msd_track,
    "track2-pendulum": _pendulum_swingup_track,
    "track2-cartpole": _cartpole_swingup_track,
    # "track1-dcmotor": _dcmotor_track  -- pending: DCMotor's B is badly scaled
    #   (~1433 gain), the plain-LQR baseline needs Bryson scaling to reach the
    #   target within the horizon.  _dcmotor_track() is kept for that follow-up.
}


# --------------------------------------------------------------------- result

@dataclass
class ChallengeResult:
    track: str
    status: str                        # PASS | DQ_SAFETY | FAILED
    composite: float
    performance: dict
    effort: dict
    safety: dict
    robustness: dict
    compute: dict
    baseline: dict
    per_seed: list[dict] = field(default_factory=list)
    _sample_traj: Trajectory | None = field(default=None, repr=False)
    _sample_ref: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------- reporting

    def report(self) -> str:
        p, e, s = self.performance, self.effort, self.safety
        rows = [
            f"# ICC result - {self.track}",
            "",
            f"**Status: {self.status}    Composite: {self.composite:.1f} / 100**",
            "",
            "| dimension | metric | value | baseline |",
            "| :-- | :-- | --: | --: |",
            f"| performance | ITAE | {p['itae']:.4g} | {self.baseline['itae']:.4g} |",
            f"| performance | RMSE | {p['rmse']:.4g} | {self.baseline.get('rmse', float('nan')):.4g} |",
            f"| performance | settling s | {p['settling_time']:.3g} | - |",
            f"| performance | overshoot % | {p['peak_overshoot_pct']:.3g} | - |",
            f"| effort | energy | {e['energy']:.4g} | {self.baseline['energy']:.4g} |",
            f"| effort | slew | {e['slew']:.4g} | {self.baseline['slew']:.4g} |",
            f"| effort | saturation % | {e['saturation_pct']:.3g} | - |",
            f"| safety | violation penalty | {s['violation_penalty']:.4g} | - |",
            f"| safety | worst norm-margin | {s['worst_margin']:.3f} | - |",
            f"| robustness | S_robust | {self.robustness['s_robust']:.3f} | - |",
            f"| compute | mean latency ms | {self.compute['mean_latency_ms']:.3f} | - |",
            f"| compute | max latency ms | {self.compute['max_latency_ms']:.3f} | - |",
        ]
        return "\n".join(rows) + "\n"

    def leaderboard_row(self, name: str) -> str:
        return (f"| {name} | {self.composite:.1f} | {self.status} | "
                f"{self.performance['itae']:.3g} | {self.effort['energy']:.3g} | "
                f"{self.effort['slew']:.3g} | {self.robustness['s_robust']:.2f} | "
                f"{self.compute['mean_latency_ms']:.2f} |")

    def to_dict(self) -> dict:
        return {
            "track": self.track, "status": self.status, "composite": self.composite,
            "performance": self.performance, "effort": self.effort,
            "safety": self.safety, "robustness": self.robustness,
            "compute": self.compute, "baseline": self.baseline,
            "per_seed": self.per_seed,
        }

    def save(self, outdir: str | Path) -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        written = {}
        (outdir / "results.json").write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8", newline="\n")
        written["json"] = outdir / "results.json"
        (outdir / "summary_table.md").write_text(self.report(), encoding="utf-8", newline="\n")
        written["md"] = outdir / "summary_table.md"
        if self._sample_traj is not None:
            written["svg"] = self._plot(outdir / "trajectory_plots.svg")
        return written

    def _plot(self, path: Path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ..plot_style import set_aimct_style

        set_aimct_style()
        tr, ref = self._sample_traj, self._sample_ref
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        oi = self.safety["output_index"]
        ax[0, 0].plot(tr.t, tr.x[:, oi], label="output")
        if ref is not None:
            ax[0, 0].plot(tr.t, ref, "--", color="#555", label="reference")
        ax[0, 0].set_title("(a) tracked output"); ax[0, 0].legend(fontsize=8)
        err = tr.x[:, oi] - (ref if ref is not None else tr.x[:, oi] * 0)
        ax[0, 1].plot(tr.t, err); ax[0, 1].set_title("(b) tracking error")
        ax[1, 0].plot(tr.t, tr.u[:, 0]); ax[1, 0].set_title("(c) control")
        ax[1, 1].plot(tr.x[:, 0], tr.x[:, min(1, tr.x.shape[1] - 1)])
        ax[1, 1].set_title("(d) phase portrait")
        for a in ax.ravel():
            a.set_xlabel("t [s]")
        fig.suptitle(f"ICC {self.track} - sample episode", fontweight="bold")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path


# --------------------------------------------------------------------- engine

class Challenge:
    """Hidden-plant evaluation engine for one ICC track."""

    def __init__(self, track: str) -> None:
        if track not in TRACKS:
            raise KeyError(f"unknown track {track!r}; choices: {sorted(TRACKS)}")
        self.track_key = track
        self.ts: TrackSpec = TRACKS[track]()

    # -- what the submission sees ----------------------------------------

    @property
    def spec(self) -> dict:
        sys = self.ts.system_factory()
        return {
            "state_dim": int(sys.n_states),
            "action_dim": int(sys.n_inputs),
            "action_limit": self.ts.action_limit.tolist(),
            "dt": self.ts.dt,
            "target_state": self.ts.target.tolist(),
            "t_final": self.ts.t_final,
        }

    # -- one episode --------------------------------------------------

    def _reference_array(self, t_grid: np.ndarray) -> np.ndarray | None:
        if self.ts.reference is None:
            return None
        return np.array([float(self.ts.reference(tt)) for tt in t_grid])

    def _run_episode(self, controller_factory, system, *, noise_seed,
                     disturbance=None):
        ts = self.ts
        rng = np.random.default_rng(noise_seed)
        n_steps = int(round(ts.t_final / ts.dt))
        noise = rng.normal(0.0, 1.0, size=(n_steps + 2, ts.system_factory().n_states))
        noise = noise * ts.noise_sigma

        def measure(t, x, u):
            k = min(int(round(t / ts.dt)), len(noise) - 1)
            return x + noise[k]

        cc = controller_factory(self.spec)
        adapter = _Adapter(cc, ts.target, ts.action_limit)
        traj = simulate(system, adapter, x0=ts.x0, dt=ts.dt, t_final=ts.t_final,
                        u_bounds=(-ts.action_limit[0], ts.action_limit[0]),
                        measurement_fn=measure, input_disturbance=disturbance)
        return traj, np.array(adapter.latencies)

    def _episode_metrics(self, traj: Trajectory, ref: np.ndarray | None) -> dict:
        ts = self.ts
        t, x, u = traj.t, traj.x, traj.u
        y = x[:, ts.output_index]
        target = float(ref[-1]) if ref is not None else float(ts.target[ts.output_index])
        r = ref if ref is not None else np.full_like(t, target)
        u1 = u[:, 0]
        ulim = float(ts.action_limit[0])

        # safety: normalised distance outside the box, summed-squared (spec 3.3.1)
        finite = np.isfinite(ts.safe_lo) & np.isfinite(ts.safe_hi)
        half = np.ones(x.shape[1])
        mid = np.zeros(x.shape[1])
        half[finite] = 0.5 * (ts.safe_hi[finite] - ts.safe_lo[finite])
        mid[finite] = 0.5 * (ts.safe_hi[finite] + ts.safe_lo[finite])
        norm = np.abs((x[:, finite] - mid[finite]) / np.where(
            half[finite] == 0, 1.0, half[finite]))
        over = np.clip(norm - 1.0, 0.0, None)
        violation_penalty = float(np.sum(over ** 2))
        worst_margin = float(norm.max()) if norm.size else 0.0
        hard_fail = bool(traj.diverged or worst_margin > 1.0)

        # did the controller actually reach & hold the target? (mean |err| over
        # the last 10 % of the episode, wrapped for an angular output)
        tail = y[max(1, int(0.9 * len(y))):] - target
        if ts.angular_output:
            tail = _wrap(tail)
        final_err = float(np.mean(np.abs(tail)))

        return {
            "itae": float(itae(t, y, r)),
            "rmse": float(rmse(t, y, r)),
            "settling_time": float(settling_time(t, y, target)),
            "peak_overshoot_pct": float(peak_overshoot(t, y, target)),
            "energy": float(control_energy(t, u1)),
            "slew": float(slew_rate(t, u1)),
            "saturation_pct": float(saturation_duty_cycle(t, u1, ulim)),
            "violation_penalty": violation_penalty,
            "worst_margin": worst_margin,
            "final_err": final_err,
            "hard_fail": hard_fail,
        }

    # -- the reference baseline (spec 4: canonical controller) --------------

    def _baseline_factory(self):
        """The reference controller the composite score is normalised against.
        Track 1: LQR on the plant linearisation.  Track 2: energy-shaping
        swing-up + an LQR catch (a bare upright-LQR cannot swing up, so it is
        not a meaningful baseline there)."""
        ts = self.ts
        sys = ts.system_factory()

        if ts.key.startswith("track1"):
            A, B = sys.linearize()
            K = LQR(A, B, np.eye(A.shape[0]), np.eye(B.shape[1]) * 0.1).K
            Bpinv = np.linalg.pinv(B)

            class _T1(ChallengeController):
                def reset(self, target_state):
                    self._x_ref = np.asarray(target_state, float)
                    self._u_ff = -(Bpinv @ (A @ self._x_ref))   # setpoint feed-forward
                def compute_action(self, obs, t):
                    return self._u_ff - K @ (np.asarray(obs, float) - self._x_ref)
            return _T1

        A, B = sys.linearize()                              # about upright
        K = LQR(A, B, np.diag([10.0, 1.0] if sys.n_states == 2
                              else [1.0, 1.0, 10.0, 1.0]),
                np.array([[0.1]])).K
        m = getattr(sys, "m", getattr(sys, "mp", 1.0))     # pendulum m / cart-pole pole mass
        L = getattr(sys, "l", getattr(sys, "L", 1.0))
        g = sys.g
        J = m * L * L * (4.0 / 3.0 if sys.n_states == 4 else 1.0)
        oi = ts.output_index
        upright = float(ts.target[oi])
        k_e = 2.0

        class _T2(ChallengeController):
            def reset(self, target_state):
                pass
            def compute_action(self, obs, t):
                x = np.asarray(obs, float)
                th, om = x[oi], x[oi + 1]
                err = _wrap(th - upright)                   # angle from upright
                if abs(err) < 0.4 and abs(om) < 3.0:        # LQR catch
                    xw = x.copy()
                    xw[oi] = upright + err
                    return -(K @ (xw - np.asarray(ts.target, float)))
                E = 0.5 * J * om**2 + m * g * L * (np.cos(err) - 1.0)
                return np.array([-k_e * (np.sign(om) or 1.0) * E])
        return _T2

    # -- full evaluation --------------------------------------------

    def evaluate(self, controller_factory, *, seed: int = 0,
                 quick: bool = False) -> ChallengeResult:
        """Full ICC evaluation.  ``quick=True`` runs 3 nominal / 5 robust seeds
        for a fast sanity pass instead of the spec's 10 / 50."""
        ts = self.ts
        n_nominal = 3 if quick else ts.n_nominal
        n_robust = 5 if quick else ts.n_robust
        t_grid = np.arange(int(round(ts.t_final / ts.dt)) + 1) * ts.dt
        ref = self._reference_array(t_grid)

        # 1. nominal suite
        nominal, sample_traj = [], None
        for s in range(n_nominal):
            traj, lat = self._run_episode(controller_factory, ts.system_factory(),
                                          noise_seed=seed + s)
            m = self._episode_metrics(traj, ref)
            m["mean_latency_ms"] = float(np.mean(lat) * 1e3)
            m["max_latency_ms"] = float(np.max(lat) * 1e3)
            nominal.append(m)
            if s == 0:
                sample_traj = traj
        nom = {k: float(np.mean([m[k] for m in nominal]))
               for k in nominal[0] if k != "hard_fail"}
        safety_ok = not any(m["hard_fail"] for m in nominal)

        # 2. robustness sweep (+/-30% params)
        rob_costs = []
        for s in range(n_robust):
            sys = perturbed_system(ts.system_factory, seed + 10_000 + s, 0.30)
            traj, _ = self._run_episode(controller_factory, sys, noise_seed=seed + s)
            rm = self._episode_metrics(traj, ref)
            rob_costs.append(rm["itae"] if not rm["hard_fail"] else 1e9)
        s_robust = robust_degradation(nom["itae"], float(np.mean(rob_costs)))

        # 3. disturbance suite (Laplace impulses)
        dist = {}
        if ts.impulse_b > 0:
            d = ImpulseInjector(seed + 20_000, b_scale=ts.impulse_b,
                                rate_hz=ts.impulse_rate_hz, t_final=ts.t_final)
            traj, _ = self._run_episode(controller_factory, ts.system_factory(),
                                        noise_seed=seed, disturbance=d)
            dm = self._episode_metrics(traj, ref)
            dist = {"itae": dm["itae"], "rmse": dm["rmse"],
                    "hard_fail": dm["hard_fail"]}
            if dm["hard_fail"]:
                safety_ok = False

        # 4. baseline
        b_nominal = []
        for s in range(n_nominal):
            traj, _ = self._run_episode(self._baseline_factory(), ts.system_factory(),
                                        noise_seed=seed + s)
            b_nominal.append(self._episode_metrics(traj, ref))
        baseline = {k: float(np.mean([m[k] for m in b_nominal]))
                    for k in ("itae", "energy", "slew", "rmse")}

        # 5. composite score. A run that never reaches / holds the target is not
        # a valid entry, whatever its effort numbers (a do-nothing controller
        # spends no energy and would otherwise score middling).
        task_ok = nom["final_err"] < ts.success_tol
        dq_reasons = [] if task_ok else ["no_convergence"]
        scored = score_run(nom, baseline, robust=s_robust, safety_ok=safety_ok,
                           dq_reasons=dq_reasons)
        if not safety_ok:
            status = "DQ_SAFETY"
            scored["composite"] = 0.0
        elif not task_ok:
            status = "FAILED"
            scored["composite"] = 0.0
        else:
            status = scored["status"]

        return ChallengeResult(
            track=ts.title, status=status, composite=scored["composite"],
            performance={**{k: nom[k] for k in
                            ("itae", "rmse", "settling_time", "peak_overshoot_pct")},
                         "final_err": nom["final_err"], "task_ok": task_ok},
            effort={"energy": nom["energy"], "slew": nom["slew"],
                    "saturation_pct": nom["saturation_pct"]},
            safety={"violation_penalty": nom["violation_penalty"],
                    "worst_margin": nom["worst_margin"],
                    "output_index": ts.output_index,
                    "disturbance": dist},
            robustness={"s_robust": s_robust,
                        "j_nominal": nom["itae"],
                        "j_perturbed_mean": float(np.mean(rob_costs))},
            compute={"mean_latency_ms": nom["mean_latency_ms"],
                     "max_latency_ms": nom["max_latency_ms"],
                     "deadline_ms": 0.5 * ts.dt * 1e3},
            baseline=baseline,
            per_seed=[{k: v for k, v in m.items() if k != "hard_fail"}
                      for m in nominal],
            _sample_traj=sample_traj,
            _sample_ref=ref,
        )


def _wrap(a):
    return np.pi - (np.pi - np.asarray(a, dtype=float)) % (2.0 * np.pi)
