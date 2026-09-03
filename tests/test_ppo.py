"""From-scratch PPO: GAE, clipped surrogate, and learning on cart-pole."""

import numpy as np
import pytest

from aimct.rl import make
from aimct.rl.ppo import PPO


def test_gae_matches_a_hand_computed_short_case():
    p = PPO.__new__(PPO)
    p.gamma, p.lam = 1.0, 1.0                    # -> advantage = plain return - V
    R = np.array([1.0, 1.0, 1.0])
    D = np.array([0.0, 0.0, 1.0])
    V = np.array([0.0, 0.0, 0.0])
    adv, ret = p._gae(R, D, V, last_v=0.0)
    assert np.allclose(ret, [3.0, 2.0, 1.0])
    assert np.allclose(adv, [3.0, 2.0, 1.0])


def test_gae_bootstraps_on_non_terminal_last_step():
    p = PPO.__new__(PPO)
    p.gamma, p.lam = 0.9, 1.0
    R = np.array([0.0, 0.0])
    D = np.array([0.0, 0.0])                     # not done -> bootstrap last_v
    V = np.array([0.0, 0.0])
    adv, ret = p._gae(R, D, V, last_v=10.0)
    assert ret[-1] == pytest_approx(9.0)        # 0 + 0.9 * 10
    assert ret[0] == pytest_approx(0.9 * 9.0)


def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) <= tol
    return _A()


@pytest.mark.slow
def test_ppo_learns_a_balancing_policy_on_cartpole():
    env = make("cartpole-balance", max_steps=200)
    p = PPO(env, hidden=(64, 64), seed=0, rollout_steps=2000, epochs=10,
            minibatch=64, lr_pi=3e-4, lr_v=1e-3, ent_coef=0.0)
    p.train(iterations=40)
    ev = p.evaluate(episodes=10, seed=11)
    # the deterministic (mean) policy should hold the pole for essentially the
    # whole horizon (an untrained one tips over in ~50-60 steps)
    assert np.mean([len(o) for o in ev["obs"]]) > 170
    assert ev["mean_return"] > -45.0


def test_ppo_clipped_weight_zeroes_the_binding_clip_side():
    """w = adv*ratio where the unclipped surrogate is the min, else 0."""
    adv = np.array([2.0, 2.0, -2.0, -2.0])
    ratio = np.array([1.5, 0.5, 1.5, 0.5])       # clip band 0.8..1.2
    clip = 0.2
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1 - clip, 1 + clip) * adv
    w = np.where(surr1 <= surr2, adv * ratio, 0.0)
    # adv>0, ratio>1+clip -> clipped is the min -> w = 0
    assert w[0] == 0.0
    # adv>0, ratio<1 -> unclipped is the min -> w != 0
    assert w[1] != 0.0
    # adv<0, ratio>1+clip -> unclipped (more negative) is the min -> w != 0
    assert w[2] != 0.0
    # adv<0, ratio<1-clip -> clipped is the min -> w = 0
    assert w[3] == 0.0
