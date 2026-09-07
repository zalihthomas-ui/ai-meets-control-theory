"""Tests for :class:`aimct.estimation.ParticleFilter` (PF)."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.estimation import (
    DiscreteKalmanFilter,
    PF,
    ParticleFilter,
    systematic_resample,
)
from aimct.systems import MassSpringDamper, TwoTank


# -------------------------------------------------- alias and initialization

def test_particle_filter_alias_and_initialization():
    assert PF is ParticleFilter
    Q = np.eye(2) * 1e-4
    R = np.eye(1) * 1e-3
    pf = ParticleFilter(
        lambda x, u: x,
        lambda x: np.array([x[0]]),
        Q,
        R,
        n_particles=200,
        dt=0.01,
        seed=42,
    )
    assert pf.n == 2
    assert pf.p == 1
    assert pf.n_particles == 200
    assert pf.particles.shape == (200, 2)
    assert pf.weights.shape == (200,)
    assert pf.ess == pytest.approx(200.0)
    assert repr(pf).startswith("ParticleFilter")


# -------------------------------------------------- linear-Gaussian matches Kalman filter

def test_particle_filter_matches_kalman_filter_linear_gaussian():
    """With N=2000 particles, PF tracks a linear-Gaussian system within a small error margin of KF."""
    msd = MassSpringDamper()
    A, B = msd.linearize()
    C = np.array([[1.0, 0.0]])
    W = np.diag([1e-4, 1e-4])
    V = np.array([[1e-3]])
    dt = 0.02

    dkf = DiscreteKalmanFilter(A, B, C, W, V, dt, x_hat0=np.array([0.5, -0.2]))
    F, Gu, Qd, Rd = dkf.F, dkf.Gu, dkf.Qd, dkf.Rd

    f_map = lambda x, u: F @ x + Gu @ u
    h_map = lambda x: C @ x

    pf = ParticleFilter(
        f_map,
        h_map,
        Qd,
        Rd,
        n_particles=2000,
        dt=dt,
        discrete=True,
        x0=np.array([0.5, -0.2]),
        spread=np.eye(2) * 0.05,
        seed=123,
    )

    rng = np.random.default_rng(42)
    x = np.array([0.5, -0.2])

    kf_errors = []
    pf_errors = []

    for k in range(50):
        u = np.array([0.4 * np.sin(0.1 * k)])
        x = F @ x + Gu @ u + rng.multivariate_normal(np.zeros(2), Qd)
        y = C @ x + rng.normal(0.0, np.sqrt(Rd[0, 0]), size=1)

        x_kf = dkf.step(y, u)
        x_pf = pf.step(y, u)

        kf_errors.append(np.linalg.norm(x_kf - x))
        pf_errors.append(np.linalg.norm(x_pf - x))

    rmse_kf = np.mean(kf_errors)
    rmse_pf = np.mean(pf_errors)

    assert rmse_pf < 0.20
    assert abs(rmse_pf - rmse_kf) < 0.08


# -------------------------------------------------- bimodal initialization collapses to true mode

def test_particle_filter_bimodal_initialization():
    """A bimodal particle distribution initialized around +5 and -5 collapses to the true mode."""
    dt = 0.05
    f = lambda x, u: 0.95 * x
    h = lambda x: np.array([x[0] ** 2])  # Quadratic measurement creates ambiguity for sign
    Q = np.array([[1e-4]])
    R = np.array([[1e-3]])

    # Initialize half particles at +5, half at -5
    n_p = 1000
    p1 = np.random.default_rng(1).normal(5.0, 0.2, size=(n_p // 2, 1))
    p2 = np.random.default_rng(2).normal(-5.0, 0.2, size=(n_p // 2, 1))
    init_particles = np.vstack([p1, p2])

    # Distinct sign-revealing linear measurement
    h_linear = lambda x: np.array([x[0]])
    pf = ParticleFilter(
        f,
        h_linear,
        Q,
        R,
        n_particles=n_p,
        dt=dt,
        discrete=True,
        particles=init_particles,
        seed=42,
    )

    # True state is positive (+5.0 decaying)
    x_true = 5.0
    for _ in range(15):
        x_true = 0.95 * x_true
        y = np.array([x_true]) + np.random.default_rng(3).normal(0.0, 0.02, size=1)
        x_est = pf.step(y)

    # Particles should have collapsed onto the positive mode
    assert x_est[0] > 1.0
    assert np.all(pf.particles > 0.0)
    assert abs(x_est[0] - x_true) < 0.15


# -------------------------------------------------- systematic resampling & ESS boundedness

def test_systematic_resampling_and_ess():
    n_p = 300
    weights = np.zeros(n_p)
    weights[0] = 0.95
    weights[1:] = 0.05 / (n_p - 1)

    particles = np.linspace(0, 10, n_p).reshape(n_p, 1)
    rng = np.random.default_rng(0)

    new_p, new_w = systematic_resample(particles, weights, rng)
    assert new_p.shape == (n_p, 1)
    assert new_w.shape == (n_p,)
    assert np.allclose(new_w, 1.0 / n_p)
    # The dominant particle at index 0 should be replicated heavily
    replicated_count = np.sum(np.isclose(new_p, particles[0]))
    assert replicated_count > n_p * 0.85

    # Check adaptive resampling in PF
    pf = ParticleFilter(
        lambda x, u: x,
        lambda x: x,
        np.eye(1) * 1e-4,
        np.eye(1) * 1e-3,
        n_particles=n_p,
        dt=0.01,
        resample_thresh=0.5,
        seed=7,
    )
    for k in range(20):
        y = np.array([1.0 + 0.1 * k])
        pf.step(y)
        # ESS must remain above or equal to threshold / reset
        assert pf.ess >= 0.45 * n_p


# -------------------------------------------------- deterministic reproducibility under seed

def test_particle_filter_deterministic_with_seed():
    """Two identical PF instances with the same seed generate bitwise identical trajectories."""
    f = lambda x, u: x + 0.01 * u
    h = lambda x: np.array([np.sin(x[0]), np.cos(x[0])])
    Q = np.eye(1) * 1e-3
    R = np.eye(2) * 1e-2

    pf1 = ParticleFilter(f, h, Q, R, n_particles=500, dt=0.02, discrete=True, seed=999)
    pf2 = ParticleFilter(f, h, Q, R, n_particles=500, dt=0.02, discrete=True, seed=999)

    for k in range(25):
        y = np.array([np.sin(0.1 * k), np.cos(0.1 * k)])
        u = np.array([0.5])
        x1 = pf1.step(y, u)
        x2 = pf2.step(y, u)
        assert np.array_equal(x1, x2)
        assert np.array_equal(pf1.particles, pf2.particles)
        assert np.array_equal(pf1.weights, pf2.weights)


# -------------------------------------------------- from_system & reset

def test_particle_filter_from_system_and_reset():
    tank = TwoTank()
    dt = 0.05
    Q = np.diag([1e-5, 1e-5])
    R = np.diag([1e-4, 1e-4])

    pf = ParticleFilter.from_system(
        tank,
        Q=Q,
        R=R,
        n_particles=300,
        dt=dt,
        x0=[0.10, 0.05],
        seed=10,
    )
    assert pf.n == 2
    assert pf.p == 2
    assert pf.particles.shape == (300, 2)

    # Step forward
    x_hat = pf.step(np.array([0.10, 0.05]), np.array([4.0]))
    assert len(x_hat) == 2

    # Reset
    pf.reset(x0=np.array([0.20, 0.15]))
    assert np.allclose(np.mean(pf.particles, axis=0), [0.20, 0.15], atol=0.15)
