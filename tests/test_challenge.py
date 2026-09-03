"""Tests for :mod:`aimct.benchmarks.challenge` - the ICC evaluation engine.

These check the *engine* contract (hidden plant, the 5 metric dimensions, the
FAILED / DQ_SAFETY gates, reproducibility, artifacts).  The exact composite
value is set by ``challenge_scoring`` and is not asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.benchmarks.challenge import Challenge, ChallengeController, TRACKS
from aimct.controllers import LQR
from aimct.systems import MassSpringDamper

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


class _Zero(ChallengeController):
    def reset(self, target):
        pass

    def compute_action(self, obs, t):
        return np.zeros(self.action_dim)


class _MsdSetpointLQR(ChallengeController):
    """A competent Track-1 entry: builds its own model, LQR + set-point
    feed-forward."""

    def __init__(self, spec):
        super().__init__(spec)
        A, B = MassSpringDamper(m=1.0, c=0.4, k=1.0).linearize()
        self.K = LQR(A, B, np.eye(2), np.array([[0.1]])).K
        self._A, self._Bpinv = A, np.linalg.pinv(B)

    def reset(self, target):
        self._xr = np.asarray(target, float)
        self._uff = -(self._Bpinv @ (self._A @ self._xr))

    def compute_action(self, obs, t):
        return self._uff - self.K @ (np.asarray(obs, float) - self._xr)


class _Runaway(ChallengeController):
    """Shoves the actuator to its positive limit forever -> on the cart-pole the
    cart runs off the rail (safe-box breach)."""

    def reset(self, target):
        pass

    def compute_action(self, obs, t):
        return np.full(self.action_dim, 1e6)         # clipped to +action_limit


# --------------------------------------------------------------------- basics

def test_unknown_track_raises():
    with pytest.raises(KeyError):
        Challenge("no-such-track")


def test_spec_hides_the_plant():
    spec = Challenge("track1-msd").spec
    assert set(spec) == {"state_dim", "action_dim", "action_limit", "dt",
                         "target_state", "t_final"}
    assert "A" not in spec and "B" not in spec
    assert spec["state_dim"] == 2 and spec["action_dim"] == 1


@pytest.mark.parametrize("track", list(TRACKS))
def test_every_track_evaluates_and_fills_all_five_dimensions(track):
    res = Challenge(track).evaluate(_Zero, seed=0, quick=True)
    for dim in ("performance", "effort", "safety", "robustness", "compute"):
        assert isinstance(getattr(res, dim), dict) and getattr(res, dim)
    assert res.status in ("PASS", "FAILED", "DQ_SAFETY")
    assert 0.0 <= res.composite <= 100.0
    assert "itae" in res.baseline and res.baseline["itae"] > 0


# ------------------------------------------------------------------ the gates

def test_do_nothing_controller_is_failed_with_zero_score():
    res = Challenge("track1-msd").evaluate(_Zero, seed=0, quick=True)
    assert res.status == "FAILED"
    assert res.composite == 0.0
    assert res.performance["final_err"] > 0.5          # never left the origin


def test_unsafe_controller_is_disqualified():
    res = Challenge("track2-cartpole").evaluate(_Runaway, seed=0, quick=True)
    assert res.status == "DQ_SAFETY"
    assert res.composite == 0.0
    assert res.safety["worst_margin"] > 1.0          # left the safe box


def test_a_competent_entry_passes_and_holds_the_target():
    res = Challenge("track1-msd").evaluate(_MsdSetpointLQR, seed=0, quick=True)
    assert res.status == "PASS"
    assert res.performance["final_err"] < 0.05        # reached & held the set-point
    assert res.performance["task_ok"] is True
    assert res.robustness["s_robust"] >= 0.0
    assert res.compute["mean_latency_ms"] >= 0.0


# ----------------------------------------------------------- reproducibility

def test_evaluation_is_reproducible_for_a_fixed_seed():
    ch = Challenge("track1-msd")
    a = ch.evaluate(_MsdSetpointLQR, seed=7, quick=True)
    b = ch.evaluate(_MsdSetpointLQR, seed=7, quick=True)
    assert a.composite == pytest.approx(b.composite)
    assert a.performance["itae"] == pytest.approx(b.performance["itae"])
    assert a.robustness["s_robust"] == pytest.approx(b.robustness["s_robust"])


# --------------------------------------------------------------- adapter / io

def test_actions_are_clipped_to_the_declared_limit():
    ch = Challenge("track2-pendulum")
    lim = ch.ts.action_limit[0]
    res = ch.evaluate(_Runaway, seed=0, quick=True)
    # the 1e6 command must have been clipped: no NaN, sim didn't explode from it
    assert res._sample_traj is not None
    assert np.max(np.abs(res._sample_traj.u)) <= lim + 1e-9


def test_track3_robustness_runner():
    res = Challenge("track1-msd").evaluate_robust(_MsdSetpointLQR, seed=0, quick=True)
    assert "Track 3" in res.track
    assert res.status in ("PASS", "FAILED", "DQ_SAFETY")
    assert 0.0 <= res.composite <= 100.0
    assert 0.0 <= res.robustness["s_robust"] <= 1.0
    # a do-nothing entry still fails the stressed task
    assert Challenge("track1-msd").evaluate_robust(_Zero, seed=0, quick=True).status \
        == "FAILED"


def test_track4_blackbox_runner_and_budget():
    ch = Challenge("track1-msd")
    n_steps = int(round(ch.ts.t_final / ch.ts.dt))

    ok = ch.evaluate_blackbox(_MsdSetpointLQR, seed=0, budget=n_steps + 100)
    assert "Track 4" in ok.track and ok.safety["budget"] == n_steps + 100
    assert ok.safety["steps_used"] <= n_steps + 1
    assert ok.status == "PASS" and ok.performance["final_err"] < 0.05

    tight = ch.evaluate_blackbox(_MsdSetpointLQR, seed=0, budget=n_steps // 2)
    assert tight.status == "DQ_SAFETY"          # ran out of interaction budget
    assert tight.composite == 0.0


def test_report_and_save(tmp_path):
    res = Challenge("track1-msd").evaluate(_MsdSetpointLQR, seed=0, quick=True)
    md = res.report()
    assert md.startswith("# ICC result") and "Composite:" in md
    assert res.leaderboard_row("entry").startswith("| entry |")

    written = res.save(tmp_path)
    for key in ("json", "md", "svg"):
        assert written[key].exists() and written[key].stat().st_size > 0
    import json
    loaded = json.loads((tmp_path / "results.json").read_text())
    assert set(loaded) >= {"track", "status", "composite", "performance",
                           "effort", "safety", "robustness", "compute", "baseline"}
