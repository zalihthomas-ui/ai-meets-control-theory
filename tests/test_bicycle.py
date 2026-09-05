"""BicycleVehicle (dynamic single-track, lateral tire forces) model checks."""

import numpy as np
import pytest

from aimct.controllers import LQR
from aimct.controllers.state_feedback import is_controllable
from aimct.simulate import simulate
from aimct.systems import BicycleVehicle


def test_shapes_and_bad_tire_model_rejected():
    veh = BicycleVehicle()
    xd = veh.dynamics(0.0, np.array([0, 0, 0, 20.0, 0, 0]), np.array([0.05, 0.0]))
    assert xd.shape == (6,)
    with pytest.raises(ValueError):
        BicycleVehicle(tire_model="magic")


def test_zero_input_at_cruise_holds_a_straight_line():
    veh = BicycleVehicle()
    x0 = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
    traj = simulate(veh, lambda y, dt: np.zeros(2), x0=x0, dt=0.01, t_final=3.0)
    assert not traj.diverged
    assert np.allclose(traj.x[-1, [1, 2, 4, 5]], 0.0, atol=1e-9)   # Y, psi, vy, r ~ 0
    assert traj.x[-1, 0] == pytest.approx(60.0, abs=1e-6)          # X = vx * t


def test_constant_small_steering_settles_to_a_steady_turn():
    veh = BicycleVehicle()
    x0 = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
    delta = np.radians(3.0)
    traj = simulate(veh, lambda y, dt: np.array([delta, 0.0]), x0=x0, dt=0.01, t_final=8.0)
    assert not traj.diverged
    r_final = traj.x[-1, 5]
    r_mid = traj.x[len(traj.t) // 2, 5]
    assert r_final == pytest.approx(r_mid, rel=0.05)               # yaw rate has settled
    assert r_final > 0.0                                           # left steer -> left turn


def test_linear_tire_force_is_proportional_and_symmetric():
    veh = BicycleVehicle(tire_model="linear", Caf=80_000.0)
    a = np.array([-0.05, 0.0, 0.05])
    Fy = veh.tire_force(a, veh.Fzf)
    assert np.allclose(Fy, -80_000.0 * a)


def test_pacejka_tire_saturates_and_eventually_falls():
    veh = BicycleVehicle(tire_model="pacejka")
    alpha = np.linspace(0.0, 0.6, 60)
    Fy = np.abs(veh.tire_force(alpha, veh.Fzf))
    assert Fy[0] == pytest.approx(0.0, abs=1.0)                    # zero slip -> zero force
    peak = np.argmax(Fy)
    assert 0 < peak < len(alpha) - 1                                # an interior peak exists
    assert Fy[-1] < Fy[peak]                                        # falls past the peak
    assert np.all(Fy <= veh.mu * veh.Fzf + 1e-6)                    # never exceeds mu*Fz


def test_linearize_default_equilibrium_is_controllable_and_stable_laterally():
    veh = BicycleVehicle()
    A, B = veh.linearize()
    assert A.shape == (6, 6) and B.shape == (6, 2)
    assert is_controllable(A, B)
    poles = np.linalg.eigvals(A)
    # X, Y, psi, vx are free kinematic integrators at a straight cruise (Re=0);
    # (vy, r) is the true lateral-dynamics pair and must be stable
    n_zero = np.sum(np.abs(poles.real) < 1e-6)
    assert n_zero == 4
    lateral = poles[np.abs(poles.real) > 1e-6]
    assert lateral.size == 2 and np.all(lateral.real < 0)


def test_lqr_about_cruise_regulates_a_lateral_offset():
    veh = BicycleVehicle()
    x_eq = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
    A, B = veh.linearize(x_eq)
    # X gets a tiny (not zero) weight: it is a free integrator at this
    # equilibrium, and CARE needs (A, sqrt(Q)) detectable to have a solution
    K = LQR(A, B, np.diag([0.01, 20.0, 5.0, 1.0, 1.0, 1.0]), np.diag([50.0, 1.0])).K

    class LaneKeep:
        def reset(self):
            pass

        def update(self, x, dt):
            return -K @ (np.asarray(x) - x_eq)

    x0 = np.array([0.0, 1.0, 0.1, 20.0, 0.0, 0.0])     # 1 m lateral offset, 0.1 rad heading
    traj = simulate(veh, LaneKeep(), x0=x0, dt=0.01, t_final=6.0)
    assert not traj.diverged
    assert abs(traj.x[-1, 1]) < 0.05                    # pulled the lateral offset in
    assert abs(traj.x[-1, 2]) < 0.02


def test_high_speed_hard_steer_stresses_linear_vs_pacejka_differently():
    """At the friction limit a linear tire keeps generating force (unphysical);
    Pacejka saturates and loses grip - the two should visibly diverge."""
    x0 = np.array([0.0, 0.0, 0.0, 25.0, 0.0, 0.0])
    hard_delta = np.radians(15.0)

    lin = BicycleVehicle(tire_model="linear")
    tr_lin = simulate(lin, lambda y, dt: np.array([hard_delta, 0.0]), x0=x0, dt=0.005, t_final=2.0)

    pac = BicycleVehicle(tire_model="pacejka")
    tr_pac = simulate(pac, lambda y, dt: np.array([hard_delta, 0.0]), x0=x0, dt=0.005, t_final=2.0)

    assert not tr_lin.diverged and not tr_pac.diverged
    # the saturating tire yields a smaller peak lateral force response (bounded by
    # mu*Fz) than the unbounded linear model at the same aggressive input
    assert np.max(np.abs(tr_pac.x[:, 5])) < np.max(np.abs(tr_lin.x[:, 5]))
