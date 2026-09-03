"""Tests for :mod:`aimct.controllers.swingup` - energy-shaping swing-up and the
hysteresis swing-up <-> LQR hybrid.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers import EnergyShapingSwingUp, HybridSwingUpLQR, LQR, wrap_angle
from aimct.systems import CartPole
from aimct.simulate import simulate

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")  # divergent probes


@pytest.fixture
def cp():
    return CartPole()


# ------------------------------------------------------------------ wrap_angle

@pytest.mark.parametrize("a, want", [
    (0.0, 0.0), (np.pi, np.pi), (-np.pi, np.pi),
    (2 * np.pi + 0.1, 0.1), (3 * np.pi, np.pi), (-0.1 - 2 * np.pi, -0.1),
])
def test_wrap_angle(a, want):
    assert wrap_angle(a) == pytest.approx(want, abs=1e-9)


# ---------------------------------------------------------------- energy model

def test_pendulum_energy_reference_points(cp):
    su = EnergyShapingSwingUp(cp)
    mgl = cp.mp * cp.g * cp.l
    assert su.pendulum_energy([0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert su.pendulum_energy([0.0, 0.0, np.pi, 0.0]) == pytest.approx(-2.0 * mgl)
    # pure kinetic at upright
    I_eff = cp.mp * cp.l**2 * 4.0 / 3.0
    assert su.pendulum_energy([0.0, 0.0, 0.0, 3.0]) == pytest.approx(0.5 * I_eff * 9.0)


def test_energy_pumps_monotonically_from_hanging(cp):
    su = EnergyShapingSwingUp(cp, k_energy=8.0, u_max=20.0)
    traj = simulate(cp, su, x0=np.array([0.0, 0.0, np.pi, 0.0]),
                    dt=1e-3, t_final=10.0, u_bounds=(-20.0, 20.0))
    E = np.array([su.pendulum_energy(x) for x in traj.x])

    running_max = np.maximum.accumulate(E)
    assert np.all(np.diff(running_max) >= -1e-9)          # envelope never drops
    assert E[0] == pytest.approx(-2.0 * cp.mp * cp.g * cp.l)
    assert running_max[len(E) // 2] < running_max[-1] + 1e-9
    assert running_max[-1] > -0.05                        # reaches the separatrix
    assert E.max() < 0.5                                  # does not blow past it


def test_swingup_force_clamped_to_u_max(cp):
    su = EnergyShapingSwingUp(cp, k_energy=30.0, u_max=12.0)
    traj = simulate(cp, su, x0=np.array([0.0, 0.0, np.pi, 0.0]),
                    dt=1e-3, t_final=6.0)
    assert np.max(np.abs(traj.u)) <= 12.0 + 1e-9


def test_higher_pump_gain_reaches_upright_sooner(cp):
    def time_to_capture(k):
        su = EnergyShapingSwingUp(cp, k_energy=k, u_max=20.0)
        traj = simulate(cp, su, x0=np.array([0.0, 0.0, np.pi, 0.0]),
                        dt=2e-3, t_final=14.0, u_bounds=(-20.0, 20.0))
        thw = np.array([wrap_angle(a) for a in traj.x[:, 2]])
        near = np.where((np.abs(thw) < 0.35) & (np.abs(traj.x[:, 3]) < 1.5))[0]
        return traj.t[near[0]] if near.size else np.inf

    assert time_to_capture(12.0) < time_to_capture(6.0) < np.inf


# --------------------------------------------------------------------- hybrid

def _hybrid(cp, **kw):
    A, B = cp.linearize()
    lqr = LQR(A, B, np.diag([1.0, 1.0, 10.0, 1.0]), [[0.1]])
    su = EnergyShapingSwingUp(cp, k_energy=10.0, u_max=20.0)
    return HybridSwingUpLQR(su, lqr, **kw)


def test_hybrid_swings_up_and_balances_from_hanging(cp):
    hyb = _hybrid(cp)
    traj = simulate(cp, hyb, x0=np.array([0.0, 0.0, np.pi, 0.0]),
                    dt=2e-3, t_final=10.0, u_bounds=(-20.0, 20.0))

    assert hyb.mode_log[0] == "swingup"
    assert "balance" in hyb.mode_log
    assert len(hyb.switch_steps) >= 1

    assert abs(wrap_angle(traj.x[-1, 2])) < 0.05
    assert abs(traj.x[-1, 3]) < 0.2
    assert abs(traj.x[-1, 0]) < 0.5
    assert np.max(np.abs(traj.u)) <= 20.0 + 1e-9
    assert not traj.diverged


def test_hybrid_starts_balancing_immediately_when_already_upright(cp):
    hyb = _hybrid(cp)
    traj = simulate(cp, hyb, x0=np.array([0.0, 0.0, 0.1, 0.0]),
                    dt=2e-3, t_final=5.0)
    assert hyb.mode_log[0] == "balance"
    assert "swingup" not in hyb.mode_log            # never loses it
    assert abs(wrap_angle(traj.x[-1, 2])) < 1e-2


def test_hybrid_hysteresis_capture_and_release():
    hyb = _hybrid(CartPole(), capture_angle=0.3, capture_rate=1.0, release_angle=0.6)

    hyb.update(np.array([0.0, 0.0, np.pi, 0.0]), 0.01)
    assert hyb.mode == "swingup"
    # inside capture window -> switch to balance
    hyb.update(np.array([0.0, 0.0, 0.2, 0.5]), 0.01)
    assert hyb.mode == "balance"
    # angle between capture and release -> stays in balance (hysteresis)
    hyb.update(np.array([0.0, 0.0, 0.45, 0.0]), 0.01)
    assert hyb.mode == "balance"
    # past release -> back to swing-up
    hyb.update(np.array([0.0, 0.0, 0.8, 0.0]), 0.01)
    assert hyb.mode == "swingup"


def test_hybrid_feeds_lqr_a_wrapped_angle():
    cp = CartPole()
    A, B = cp.linearize()
    lqr = LQR(A, B, np.diag([1.0, 1.0, 10.0, 1.0]), [[0.1]])
    hyb = HybridSwingUpLQR(EnergyShapingSwingUp(cp), lqr,
                           capture_angle=0.35, capture_rate=2.0)
    # a pole that came round the long way: theta = 2*pi + 0.1  (upright, +0.1 rad)
    state = np.array([0.0, 0.0, 2.0 * np.pi + 0.1, 0.3])
    hyb.mode = "balance"
    u_hyb = hyb.update(state, 0.01)
    u_ref = lqr.update(np.array([0.0, 0.0, 0.1, 0.3]), 0.01)
    assert u_hyb == pytest.approx(u_ref)


def test_hybrid_rejects_non_hysteretic_thresholds(cp):
    with pytest.raises(ValueError, match="hysteresis"):
        _hybrid(cp, capture_angle=0.5, release_angle=0.5)


def test_hybrid_reset_clears_mode_and_log(cp):
    hyb = _hybrid(cp)
    simulate(cp, hyb, x0=np.array([0.0, 0.0, np.pi, 0.0]), dt=5e-3, t_final=4.0,
             u_bounds=(-20.0, 20.0))
    assert hyb.mode_log
    hyb.reset()
    assert hyb.mode == "swingup"
    assert hyb.mode_log == []
    assert hyb.switch_steps == []
