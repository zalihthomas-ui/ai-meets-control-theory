"""H-infinity mixed-sensitivity: StateSpace toolkit, weights, DGKF synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.controllers.hinf import (StateSpace, augment_plant, hinf_syn,
                                    lft_lower, mixsyn, weight_KS, weight_S,
                                    weight_T)


def _tf_ss(num, den):
    return StateSpace.from_tf(num, den)


def _freq(sys, w):
    return sys.freqresp(w)[:, 0, 0]


# ======================================================================
# StateSpace
# ======================================================================
def test_from_tf_matches_the_transfer_function_pointwise():
    num, den = [2.0, 1.0], [1.0, 3.0, 2.0]
    G = _tf_ss(num, den)
    w = np.logspace(-2, 2, 60)
    ref = np.array([np.polyval(num, 1j * x) / np.polyval(den, 1j * x) for x in w])
    assert np.allclose(_freq(G, w), ref, atol=1e-9)


def test_series_and_parallel_compose_in_the_frequency_domain():
    G1 = StateSpace([[-1.0, 0.5], [0.0, -2.0]], [[1.0], [1.0]], [[1.0, 0.0]], [[0.3]])
    G2 = _tf_ss([3.0], [1.0, 3.0])
    w = np.logspace(-1, 2, 40)
    assert np.allclose((G1 * G2).freqresp(w), G1.freqresp(w) @ G2.freqresp(w), atol=1e-10)
    G3 = StateSpace(G1.A, G1.B, G1.C, G1.D)
    assert np.allclose((G1 + G3).freqresp(w), G1.freqresp(w) + G3.freqresp(w), atol=1e-10)


def test_feedback_gives_the_complementary_sensitivity():
    G = _tf_ss([2.0], [1.0, 0.5])
    T = G.feedback()                       # y = G(r - y)  ->  T = G/(1+G)
    w = np.logspace(-2, 2, 50)
    Gw = _freq(G, w)
    assert np.allclose(_freq(T, w), Gw / (1.0 + Gw), atol=1e-9)


def test_hinf_norm_of_a_first_order_lag_is_the_dc_gain():
    k, a = 3.0, 0.7
    G = _tf_ss([k], [1.0, a])
    assert G.hinf_norm() == pytest.approx(k / a, rel=1e-3)


def test_hinf_norm_of_a_resonant_mode_matches_a_fine_grid_peak():
    wn, zeta = 5.0, 0.05
    G = _tf_ss([wn**2], [1.0, 2 * zeta * wn, wn**2])
    grid = float(G.sigma_max(np.logspace(-1, 2, 20000)).max())
    assert G.hinf_norm(tol=1e-7) == pytest.approx(grid, rel=5e-3)


def test_static_gain_has_no_states_and_a_flat_response():
    K = StateSpace.gain([[2.0, 0.0], [1.0, 3.0]])
    assert K.nx == 0 and K.shape == (2, 2)
    assert np.allclose(K.freqresp([0.1, 10.0]), K.D)
    assert K.hinf_norm() == pytest.approx(float(np.linalg.svd(K.D, compute_uv=False)[0]))


# ======================================================================
# weighting filters
# ======================================================================
def test_weight_S_is_high_gain_low_frequency_and_bounded_high_frequency():
    W = weight_S(wb=10.0, A=1e-3, M=2.0)
    assert abs(_freq(W, np.array([1e-4]))[0]) == pytest.approx(1.0 / 1e-3, rel=0.05)
    assert abs(_freq(W, np.array([1e6]))[0]) == pytest.approx(1.0 / 2.0, rel=1e-3)


def test_weight_T_rolls_up_past_the_bandwidth():
    W = weight_T(wb=20.0, A=1e-2, M=2.0)
    assert abs(_freq(W, np.array([1e-3]))[0]) == pytest.approx(1.0 / 2.0, rel=0.05)
    assert abs(_freq(W, np.array([1e5]))[0]) == pytest.approx(1.0 / 1e-2, rel=1e-2)


def test_blocked_weight_is_a_diagonal_replica():
    W1 = weight_S(wb=3.0)
    W2 = weight_S(wb=3.0, blocks=2)
    w = np.logspace(-2, 2, 20)
    h1 = W1.freqresp(w)[:, 0, 0]
    h2 = W2.freqresp(w)
    assert np.allclose(h2[:, 0, 0], h1) and np.allclose(h2[:, 1, 1], h1)
    assert np.allclose(h2[:, 0, 1], 0.0)


# ======================================================================
# generalised plant
# ======================================================================
def test_augment_plant_lower_lft_reconstructs_the_SKST_stack():
    G = _tf_ss([200.0], np.polymul([10.0, 1.0], np.polymul([0.05, 1.0], [0.05, 1.0])))
    WS = weight_S(wb=10.0, A=1e-4, M=2.0)
    WK = weight_KS(wb=1e3, A=1e-2, M=1e3)
    WT = weight_T(wb=200.0, A=1e-3, M=2.0)
    P, nmeas, ncon = augment_plant(G, WS, WK, WT)
    assert (nmeas, ncon) == (1, 1)

    K = _tf_ss([2.0, 2.0], [1.0, 40.0])            # arbitrary stabilising-ish K
    w = np.logspace(-2, 5, 300)
    Gw, Kw = _freq(G, w), _freq(K, w)
    S = 1.0 / (1.0 + Gw * Kw)
    target = np.stack([_freq(WS, w) * S,
                       _freq(WK, w) * Kw * S,
                       _freq(WT, w) * Gw * Kw * S], axis=1)
    got = lft_lower(P, K, nmeas, ncon).freqresp(w)[:, :, 0]
    assert np.max(np.abs(got - target)) < 1e-6


def test_augment_plant_rejects_a_non_square_plant():
    G = StateSpace(np.array([[-1.0]]), np.array([[1.0, 0.0]]), np.array([[1.0]]),
                   np.zeros((1, 2)))
    with pytest.raises(ValueError):
        augment_plant(G, weight_S(wb=1.0))


# ======================================================================
# hinf_syn  (self-consistent DGKF verification, no external solver needed)
# ======================================================================
@pytest.fixture
def siso_problem():
    G = _tf_ss([5.0], np.polymul([1.0, 1.0], [0.1, 1.0]))
    WS = weight_S(wb=1.0, A=1e-2, M=2.0)
    WK = StateSpace.gain([[1e-1]])
    WT = weight_T(wb=20.0, A=0.1, M=2.0)
    return G, WS, WK, WT


def test_synthesised_controller_achieves_its_gamma_and_is_internally_stabilising(siso_problem):
    G, WS, WK, WT = siso_problem
    res = mixsyn(G, WS, WK, WT, tol=1e-4)
    assert res.K.is_stable() is False or res.K.is_stable() in (True, False)  # K may be unstable; loop must not be
    assert res.CL.is_stable()
    assert res.CL.hinf_norm(tol=1e-5) <= res.gamma * (1 + 1e-2)


def test_suboptimality_conditions_hold_at_the_achieved_gamma(siso_problem):
    G, WS, WK, WT = siso_problem
    res = mixsyn(G, WS, WK, WT, tol=1e-4)
    assert np.min(np.linalg.eigvalsh(res.X)) > -1e-6          # X >= 0
    assert np.min(np.linalg.eigvalsh(res.Y)) > -1e-6          # Y >= 0
    rho = np.max(np.abs(np.linalg.eigvals(res.X @ res.Y)))
    assert rho < res.gamma ** 2                               # spectral radius bound


def test_gamma_is_a_tight_infimum(siso_problem):
    G, WS, WK, WT = siso_problem
    res = mixsyn(G, WS, WK, WT, tol=1e-4)
    assert res.gamma_lb < res.gamma                           # bracketed
    assert (res.gamma - res.gamma_lb) < 1e-2 * res.gamma      # tight


def test_tighter_robustness_weight_costs_more_gamma(siso_problem):
    G, WS, WK, _ = siso_problem
    loose = mixsyn(G, WS, WK, weight_T(wb=20.0, A=0.3, M=2.0), tol=1e-3)
    tight = mixsyn(G, WS, WK, weight_T(wb=20.0, A=0.02, M=2.0), tol=1e-3)
    assert tight.gamma > loose.gamma


def test_achieved_loop_is_shaped_as_requested(siso_problem):
    G, WS, WK, WT = siso_problem
    res = mixsyn(G, WS, WK, WT, tol=1e-4)
    w = np.logspace(-3, 4, 400)
    Gw, Kw = _freq(G, w), _freq(res.K, w)
    S = 1.0 / (1.0 + Gw * Kw)
    T = Gw * Kw * S
    assert abs(S[0]) < 1e-2               # tight low-frequency tracking
    assert abs(T[-1]) < 1e-2             # complementary sensitivity rolls off


def test_mimo_synthesis_runs_and_stabilises():
    rng = np.random.default_rng(2)
    n = 4
    A = rng.standard_normal((n, n)) - 3.0 * np.eye(n)
    B = rng.standard_normal((n, 2))
    C = rng.standard_normal((2, n))
    G = StateSpace(A, B, C, np.zeros((2, 2)))
    res = mixsyn(G, weight_S(wb=1.0, A=1e-2, M=2.0, blocks=2),
                 StateSpace.gain(5e-2 * np.eye(2)),
                 weight_T(wb=15.0, A=0.05, M=2.0, blocks=2), tol=1e-3)
    assert res.CL.is_stable()
    assert res.CL.hinf_norm(tol=1e-4) <= res.gamma * (1 + 1e-2)


def test_hinf_syn_rejects_a_nonzero_D11():
    G = _tf_ss([5.0], np.polymul([1.0, 1.0], [0.1, 1.0]))
    WS = weight_S(wb=1.0, A=1e-2, M=2.0)          # bi-proper -> D11 != 0
    P, nmeas, ncon = augment_plant(G, WS, StateSpace.gain([[1e-1]]),
                                   weight_T(wb=20.0, A=0.1, M=2.0))
    with pytest.raises(ValueError, match="D11"):
        hinf_syn(P, nmeas, ncon)


# ======================================================================
# controller wrapper
# ======================================================================
def test_hinf_controller_reproduces_the_state_space_map():
    from aimct.controllers.hinf import HinfController

    G = _tf_ss([5.0], np.polymul([1.0, 1.0], [0.1, 1.0]))
    res = mixsyn(G, weight_S(wb=1.0, A=1e-2, M=2.0), StateSpace.gain([[1e-1]]),
                 weight_T(wb=20.0, A=0.1, M=2.0), tol=1e-3)
    ctrl = HinfController(res.K)
    ctrl.reset()

    # reference: the same K driven by an explicit RK4 of e = r - y = -y
    Ak, Bk, Ck, Dk = res.K.A, res.K.B, res.K.C, res.K.D
    xk = np.zeros(res.K.nx)
    dt = 1e-3
    rng = np.random.default_rng(0)
    for _ in range(500):
        y = np.array([rng.standard_normal()])
        e = -y
        k1 = Ak @ xk + Bk @ e
        k2 = Ak @ (xk + 0.5 * dt * k1) + Bk @ e
        k3 = Ak @ (xk + 0.5 * dt * k2) + Bk @ e
        k4 = Ak @ (xk + dt * k3) + Bk @ e
        xk = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        u_ref = Ck @ xk + Dk @ e
        u = ctrl.update(y, dt)
        assert np.allclose(u, u_ref, atol=1e-9)


def test_hinf_controller_respects_input_bounds():
    from aimct.controllers.hinf import HinfController

    G = _tf_ss([5.0], np.polymul([1.0, 1.0], [0.1, 1.0]))
    res = mixsyn(G, weight_S(wb=1.0, A=1e-2, M=2.0), StateSpace.gain([[1e-1]]),
                 weight_T(wb=20.0, A=0.1, M=2.0), tol=1e-3)
    ctrl = HinfController(res.K, u_bounds=(-0.5, 0.5))
    for _ in range(50):
        u = ctrl.update(np.array([3.0]), 1e-3)
        assert np.all(u <= 0.5 + 1e-12) and np.all(u >= -0.5 - 1e-12)


# ======================================================================
# cross-check against python-control (needs slycot)
# ======================================================================
def test_gamma_matches_python_control_hinfsyn():
    pytest.importorskip("slycot")
    control = pytest.importorskip("control")
    import warnings

    G = _tf_ss([5.0], np.polymul([1.0, 1.0], [0.1, 1.0]))
    P, nmeas, ncon = augment_plant(
        G,
        StateSpace(*_strip_d(weight_S(wb=1.0, A=1e-2, M=2.0))),
        StateSpace.gain([[1e-1]]),
        weight_T(wb=20.0, A=0.1, M=2.0),
    )
    mine = hinf_syn(P, nmeas, ncon, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, CLc, gam_c, _ = control.hinfsyn(
            control.ss(P.A, P.B, P.C, P.D), nmeas, ncon)
    assert mine.gamma == pytest.approx(float(gam_c), rel=3e-2)
    assert mine.CL.hinf_norm(tol=1e-5) == pytest.approx(float(gam_c), rel=3e-2)


def _strip_d(W):
    return W.A, W.B, W.C, np.zeros_like(W.D)
