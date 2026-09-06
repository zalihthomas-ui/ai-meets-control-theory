"""Structured singular value (mu): bounds, frequency sweeps, D-K iteration."""

from __future__ import annotations

import numpy as np
import pytest

from aimct.robust import (BlockStructure, dk_iterate, mu,
                          robust_performance_margin, robust_stability_margin)


# ======================================================================
# BlockStructure
# ======================================================================
def test_block_structure_parse_forms():
    assert BlockStructure.parse(3).blocks == ((3, "F"),)
    assert BlockStructure.parse([(1, "R"), (2, "C")]).blocks == ((1, "R"), (2, "C"))
    S = BlockStructure([(2, "F"), (1, "C")])
    assert S.n == 3 and BlockStructure.parse(S) is S


def test_block_structure_rejects_bad_kind():
    with pytest.raises(ValueError):
        BlockStructure([(2, "X")])


# ======================================================================
# mu of a single matrix
# ======================================================================
def test_single_full_block_mu_equals_sigma_max():
    rng = np.random.default_rng(0)
    for n in (1, 3, 5):
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        r = mu(M, n)
        sm = float(np.linalg.svd(M, compute_uv=False)[0])
        assert r["lower_bound"] == pytest.approx(sm, rel=1e-9)
        assert r["upper_bound"] == pytest.approx(sm, rel=1e-9)


def test_bounds_bracket_rho_and_sigma_for_every_structure():
    rng = np.random.default_rng(1)
    structures = [
        [(1, "C")] * 4,
        [(2, "F"), (2, "F")],
        [(1, "R"), (1, "R"), (2, "F")],
        [(1, "C"), (3, "F")],
    ]
    for spec in structures:
        S = BlockStructure(tuple(spec))
        M = (rng.standard_normal((S.n, S.n))
             + 1j * rng.standard_normal((S.n, S.n))) * 0.4
        r = mu(M, S)
        assert r["rho"] - 1e-6 <= r["lower_bound"]
        assert r["lower_bound"] <= r["upper_bound"] + 1e-6
        assert r["upper_bound"] <= r["sigma_max"] + 1e-6


def test_rank_one_scalar_block_upper_bound_is_exact():
    # M = u v^T with n complex scalar blocks:  mu = sum_k |u_k v_k|
    rng = np.random.default_rng(2)
    u = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    v = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    M = np.outer(u, v)
    r = mu(M, BlockStructure([(1, "C")] * 4))
    exact = float(np.sum(np.abs(u * v)))
    assert r["upper_bound"] == pytest.approx(exact, rel=1e-4)
    assert r["lower_bound"] <= exact + 1e-6
    assert r["lower_bound"] >= 0.9 * exact          # power iteration gets close


@pytest.mark.slow
def test_two_real_blocks_match_a_brute_force_search():
    M = np.array([[0.6, 0.5], [-0.3, 0.4]])
    r = mu(M, BlockStructure([(1, "R"), (1, "R")]))
    best = np.inf
    g = np.linspace(-6.0, 6.0, 801)
    for d1 in g:
        for d2 in g:
            if abs(np.linalg.det(np.eye(2) - M @ np.diag([d1, d2]))) < 2e-3:
                best = min(best, max(abs(d1), abs(d2)))
    mu_bf = 1.0 / best
    assert r["lower_bound"] == pytest.approx(mu_bf, abs=2e-2)
    assert r["upper_bound"] == pytest.approx(mu_bf, abs=5e-2)


def test_mu_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mu(np.zeros((3, 3)), BlockStructure([(2, "F")]))
    with pytest.raises(ValueError):
        mu(np.zeros((2, 3)), 2)


# ======================================================================
# frequency sweeps
# ======================================================================
def _Wm(w):
    s = 1j * w
    return 0.2 * (s + 1.0) / (0.05 * s + 1.0)          # 0.2 low, 4 high


def _T(w):
    s = 1j * w
    return 4.0 / (s * s + 2 * 0.7 * 2 * s + 4.0)       # bw ~2


def test_siso_rs_margin_equals_the_peak_of_the_scalar_transfer():
    grid = np.logspace(-2, 3, 400)

    def M(w):
        return np.array([[-_Wm(w) * _T(w)]], dtype=complex)

    rs = robust_stability_margin(M, 1, grid)
    peak = max(abs(_Wm(w) * _T(w)) for w in grid)
    assert rs["peak_upper"] == pytest.approx(peak, rel=1e-6)
    assert rs["margin"] == pytest.approx(1.0 / peak, rel=1e-6)
    assert rs["robust"] is True


@pytest.mark.slow
def test_structured_rs_margin_matches_a_brute_force_delta_search():
    # small 2-block M(w); RS margin = 1 / max_w mu(M(jw))
    def M(w):
        s = 1j * w
        return np.array([[0.5 / (0.1 * s + 1), 0.2 / (s + 2)],
                         [0.15 / (s + 1), -_Wm(w) * _T(w)]], dtype=complex)

    S = BlockStructure([(1, "R"), (1, "C")])
    grid = np.logspace(-2, 2, 120)
    rs = robust_stability_margin(M, S, grid)

    # brute force at each frequency: smallest max(|d_R|, |d_C|) with
    # det(I - M diag(d_R, d_C)) = 0,  d_R real, d_C complex on a coarse grid
    def mu_bf(Mw):
        best = np.inf
        for dr in np.linspace(-4, 4, 121):
            for mag in np.linspace(0.05, 4, 60):
                for ph in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                    dc = mag * np.exp(1j * ph)
                    if abs(np.linalg.det(np.eye(2) - Mw @ np.diag([dr, dc]))) < 3e-3:
                        best = min(best, max(abs(dr), mag))
        return 1.0 / best if np.isfinite(best) else 0.0

    i = int(np.argmax(rs["mu_upper"]))
    ref = mu_bf(M(grid[i]))
    assert rs["mu_lower"][i] <= rs["peak_upper"] + 1e-6
    assert ref <= rs["peak_upper"] + 5e-2                 # bf can't beat the upper bound
    assert rs["peak_lower"] >= 0.8 * ref                  # lower bound tracks bf


def test_robust_performance_is_rs_of_the_augmented_structure():
    grid = np.logspace(-2, 2, 80)

    def N(w):
        s = 1j * w
        return np.array([[-_Wm(w) * _T(w), 0.4 / (s + 1)],
                         [0.5 / (s + 2), 0.6 / (s * s + s + 1)]], dtype=complex)

    rp = robust_performance_margin(N, [(1, "C")], (1, 1), grid)
    # same numbers as calling robust_stability_margin with the augmented structure
    rs = robust_stability_margin(N, BlockStructure([(1, "C"), (1, "F")]), grid)
    assert rp["peak_upper"] == pytest.approx(rs["peak_upper"], rel=1e-9)


def test_robust_performance_requires_square_perf_block():
    with pytest.raises(ValueError):
        robust_performance_margin(lambda w: np.zeros((3, 3)), [(1, "C")],
                                  (2, 1), np.array([1.0]))


# ======================================================================
# D-K iteration
# ======================================================================
def test_dk_iteration_does_not_worsen_the_rs_margin():
    # toy: a "plant" scalar d scales an uncertainty channel; resynthesise
    # returns a matrix builder whose peak mu decreases as d grows, up to a point
    grid = np.logspace(-1, 1, 60)

    def resynthesise(d):
        return float(d[0])                     # the "controller" is just a scalar

    def mu_matrix(K, w):
        s = 1j * w
        base = 0.9 / (0.5 * s + 1)
        return np.array([[base / (1.0 + 0.6 * (K - 1.0))]], dtype=complex)

    K, hist = dk_iterate(resynthesise, mu_matrix, [(1, "C")], grid, iterations=4)
    assert len(hist) >= 1
    assert hist[-1]["peak_upper"] <= hist[0]["peak_upper"] + 1e-6
