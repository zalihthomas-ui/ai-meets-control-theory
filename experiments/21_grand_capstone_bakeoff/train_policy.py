"""Train (and reload) the RL entry for the Experiment-21 grand bake-off:
a from-scratch NumPy PPO policy that flies the planar quadrotor around the
figure-8 of Experiments 14 / 20.

The policy is the *learning* entry of the bake-off.  Pure from-scratch PPO
cannot bootstrap on this task - the quadrotor's feather-light pitch axis makes
every early rollout diverge inside ~20 steps, before the critic learns anything
- so the net is first behaviour-cloned from the LQR+flatness expert (Exp 20),
then PPO fine-tunes it against the task's scalar reward.  It is still expected
to track worse than the expert it was cloned from; the point of the capstone is
whether *combining* it with a safety shield closes that gap.

Usage
-----
    python experiments/21_grand_capstone_bakeoff/train_policy.py          # quick (~40 iters)
    AIMCT_EXP_FULL=1 python experiments/21_grand_capstone_bakeoff/train_policy.py   # full

Writes ``quad_ppo_policy.npz`` next to this file.  Consume it from the bake-off
with :func:`load_quad_policy`::

    from experiments.__thispath__ import load_quad_policy   # or import by path
    pol = load_quad_policy()
    u = pol.action(quad_state, t)          # (state, time) -> 2 rotor thrusts
    # or as a Controller:  pol.reset(); u = pol.update(measurement, dt)
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aimct.controllers import LQR
from aimct.rl import PPO, figure8_obs, figure8_reference, make
from aimct.rl.policy_gradient import GaussianPolicy
from aimct.systems import PlanarQuadrotor

HERE = Path(__file__).parent
POLICY_PATH = HERE / "quad_ppo_policy.npz"
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"

# The quadrotor input matrix is viciously scaled: the differential-thrust ->
# pitch-accel gain (2 l / Iyy ~ 6600) dwarfs the collective -> heave gain
# (1 / m ~ 36).  Isotropic exploration in raw per-rotor thrust therefore pitches
# the vehicle over within a couple of steps and PPO never sees a flyable
# trajectory.  So the *policy* acts in a conditioned
# ``[collective_dev, differential_dev]`` space about hover, with the differential
# channel deliberately throttled; the map back to two rotor thrusts (what the
# task and the bake-off want) lives in one place: :func:`_to_thrusts`.
_COLL_SCALE = 0.6                 # * u_hover_mean  -> +/-60% collective authority
_DIFF_SCALE = 0.0015            # N, per-rotor differential.  The figure-8 needs
                                 # ~1e-5 N of it; keep the channel on a very short
                                 # leash so even a rail-to-rail command can't ramp
                                 # the feather-light pitch axis past recovery
                                 # inside an episode.


def _to_thrusts(a, quad: PlanarQuadrotor) -> np.ndarray:
    """``[coll, diff]`` in roughly [-1, 1] -> two rotor thrusts in [0, tmax]."""
    a = np.asarray(a, float).ravel()
    uh = float(np.mean(quad.u_hover))
    coll = a[0] * _COLL_SCALE * uh
    diff = a[1] * _DIFF_SCALE
    return np.clip(uh + np.array([coll + diff, coll - diff]), 0.0, quad.thrust_max)


def _to_policy_action(u, quad: PlanarQuadrotor) -> np.ndarray:
    """Inverse of :func:`_to_thrusts`: two rotor thrusts -> ``[coll, diff]``."""
    u = np.asarray(u, float).ravel()
    uh = float(np.mean(quad.u_hover))
    coll = (0.5 * (u[0] + u[1]) - uh) / (_COLL_SCALE * uh)
    diff = 0.5 * (u[0] - u[1]) / _DIFF_SCALE
    return np.clip([coll, diff], -1.0, 1.0)


def _lqr_ff_expert(quad: PlanarQuadrotor):
    """LQR-about-hover + differential-flatness feed-forward - the same expert
    that flies Experiment 20.  Used only to behaviour-clone a sane starting
    policy; PPO then fine-tunes against the scalar reward."""
    A, B = quad.linearize()
    K = LQR(A, B,
            np.diag(1.0 / np.array([.1, .1, .2, .5, .5, 3.]) ** 2),
            np.diag(1.0 / np.array([.15, .15]) ** 2)).K

    def act(state, t):
        xr, ur = figure8_reference(t, quad)
        return np.clip(ur - K @ (np.asarray(state, float) - xr),
                       0.0, quad.thrust_max)

    return act


def _behaviour_clone_dataset(quad: PlanarQuadrotor, n_steps: int, seed: int):
    """Roll the LQR+FF expert on the task (with small action noise so the state
    distribution is not a single trajectory) and record
    ``(scaled_obs, [coll, diff])`` pairs."""
    env = make("quad-figure8-track")
    expert = _lqr_ff_expert(quad)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    env.reset(seed=int(rng.integers(1 << 31)))
    t = 0.0
    for _ in range(n_steps):
        s = env.state
        u_star = expert(s, t)
        X.append(figure8_obs(s, t, quad))
        Y.append(_to_policy_action(u_star, quad))
        u_take = np.clip(u_star + rng.normal(0.0, [3e-3, 4e-4], 2),
                         0.0, quad.thrust_max)
        _, _, term, trunc, _ = env.step(u_take)
        t += env.dt
        if term or trunc:
            env.reset(seed=int(rng.integers(1 << 31)))
            t = 0.0
    return np.asarray(X), np.asarray(Y)


class _HoverActionWrapper(gym.Wrapper):
    """Expose a well-conditioned 2-D ``[coll, diff]`` action; convert to rotor
    thrusts for the wrapped :class:`quad-figure8-track` env."""

    def __init__(self, env):
        super().__init__(env)
        self._quad = PlanarQuadrotor()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        return self.env.step(_to_thrusts(action, self._quad))


# --------------------------------------------------------------------- train

def train_quad_policy(*, seed: int = 0, full: bool | None = None):
    """Behaviour-clone the LQR+FF expert into the policy net, then fine-tune with
    PPO against the task's scalar reward.  Pure from-scratch PPO cannot bootstrap
    a value function on this feather-light attitude axis (episodes collapse to
    ~20 steps before the critic learns anything); the clone gives it a flyable
    starting policy, and RL takes it from there.  Returns ``(ppo, result)``."""
    full = FULL if full is None else full
    quad = PlanarQuadrotor()
    env = _HoverActionWrapper(make("quad-figure8-track"))
    ppo = PPO(
        env,
        hidden=(64, 64),
        seed=seed,
        gamma=0.99,
        lam=0.95,
        clip=0.15,
        lr_pi=1e-4,
        lr_v=8e-4,
        epochs=6,
        minibatch=256,
        rollout_steps=8192 if full else 4096,
        ent_coef=0.0,
        target_kl=0.015,
    )

    # 1) behaviour clone the LQR+FF expert
    n_bc = 60_000 if full else 25_000
    Xbc, Ybc = _behaviour_clone_dataset(quad, n_bc, seed)
    ppo.pi.net.b[-1][:] = 0.0
    ppo.pi.net.fit(Xbc, Ybc, epochs=300 if full else 200,
                   batch_size=256, lr=3e-3, seed=seed, verbose=True)
    ppo.pi.log_std[:] = np.log(0.06)          # tight cloud around the cloned mean
    bc_snapshot = ([W.copy() for W in ppo.pi.net.W],
                   [b.copy() for b in ppo.pi.net.b],
                   ppo.pi.log_std.copy())
    bc_rms = tracking_rms(QuadFigure8Policy(ppo.pi), episodes=10)["rms_pos_err_m"]

    # 2) light PPO fine-tune against the scalar reward (keeps the clone honest;
    #    a heavier touch on this axis just destabilises it).  Keep whichever of
    #    the cloned / fine-tuned policy actually tracks better.
    result = ppo.train(iterations=150 if full else 40, verbose=True)
    ft_rms = tracking_rms(QuadFigure8Policy(ppo.pi), episodes=10)["rms_pos_err_m"]
    if ft_rms > bc_rms * 1.02:
        Ws, bs, ls = bc_snapshot
        ppo.pi.net.W, ppo.pi.net.b, ppo.pi.log_std = Ws, bs, ls
        print(f"  fine-tune worse ({ft_rms*1e3:.1f} vs {bc_rms*1e3:.1f} mm) "
              f"-> keeping the behaviour-cloned policy")
    return ppo, result


# ----------------------------------------------------------------- persistence

def save_policy(policy: GaussianPolicy, path: Path | str = POLICY_PATH, *,
                meta: dict | None = None) -> Path:
    """Serialise a :class:`GaussianPolicy` (MLP weights + log-std + action box)."""
    path = Path(path)
    blob: dict[str, np.ndarray] = {
        "log_std": np.asarray(policy.log_std, float),
        "act_low": np.asarray(policy.act_low, float),
        "act_high": np.asarray(policy.act_high, float),
        "n_layers": np.array(len(policy.net.W)),
    }
    for i, (W, b) in enumerate(zip(policy.net.W, policy.net.b)):
        blob[f"W{i}"] = np.asarray(W, float)
        blob[f"b{i}"] = np.asarray(b, float)
    for k, v in (meta or {}).items():
        blob[f"meta_{k}"] = np.asarray(v)
    np.savez(path, **blob)
    return path


def _load_gaussian_policy(path: Path | str = POLICY_PATH) -> GaussianPolicy:
    d = np.load(Path(path), allow_pickle=False)
    n = int(d["n_layers"])
    Ws = [d[f"W{i}"] for i in range(n)]
    bs = [d[f"b{i}"] for i in range(n)]
    obs_dim, act_dim = Ws[0].shape[0], Ws[-1].shape[1]
    hidden = tuple(W.shape[1] for W in Ws[:-1])
    pi = GaussianPolicy(obs_dim, act_dim, hidden=hidden,
                        act_low=d["act_low"], act_high=d["act_high"])
    pi.net.W, pi.net.b = [W.copy() for W in Ws], [b.copy() for b in bs]
    pi.log_std = d["log_std"].copy()
    return pi


class QuadFigure8Policy:
    """A trained PPO policy wrapped for the bake-off: turns a raw quadrotor
    state + time into two rotor thrusts, doing the reference-error / phase-clock
    observation construction *and* the ``[coll, diff]`` -> thrust map internally
    so callers never touch the task obs or the action conditioning."""

    obs_dim = 8

    def __init__(self, policy: GaussianPolicy, quad: PlanarQuadrotor | None = None):
        self.policy = policy
        self.quad = quad or PlanarQuadrotor()
        self._t = 0.0

    # -- (state, time) interface --------------------------------------------
    def observe(self, state, t: float) -> np.ndarray:
        return figure8_obs(state, t, self.quad)

    def action(self, state, t: float) -> np.ndarray:
        raw = self.policy.greedy(self.observe(state, t))        # [coll, diff]
        return _to_thrusts(raw, self.quad)                      # 2 rotor thrusts

    # -- Controller interface (self-clocked, for aimct.simulate) -----------
    def reset(self) -> None:
        self._t = 0.0

    def update(self, measurement, dt: float) -> np.ndarray:
        u = self.action(measurement, self._t)
        self._t += dt
        return u


def load_quad_policy(path: Path | str = POLICY_PATH) -> QuadFigure8Policy:
    """Load the committed PPO policy as a ready-to-fly :class:`QuadFigure8Policy`."""
    return QuadFigure8Policy(_load_gaussian_policy(path))


# ------------------------------------------------------------------- evaluate

def tracking_rms(policy: QuadFigure8Policy, *, seed: int = 123,
                 episodes: int = 20) -> dict[str, float]:
    """Greedy roll-outs on the task; reports position-tracking RMS (m),
    survival fraction and mean reward."""
    env = make("quad-figure8-track")
    quad = PlanarQuadrotor()
    rng = np.random.default_rng(seed)
    sq_err, n_pt, survived, ret_sum, n_ep_steps = 0.0, 0, 0, 0.0, 0
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(1 << 31)))
        t, done = 0.0, False
        steps = 0
        while not done:
            u = policy.action(env.state, t)
            _, r, term, trunc, _ = env.step(u)
            t += env.dt
            steps += 1
            ret_sum += r
            xr, _ = figure8_reference(t, quad)
            sq_err += float(np.sum((env.state[:2] - xr[:2]) ** 2))
            n_pt += 1
            done = term or trunc
        survived += int(not term)
        n_ep_steps += steps
    return {
        "rms_pos_err_m": float(np.sqrt(sq_err / max(n_pt, 1))),
        "survival_frac": survived / episodes,
        "mean_reward": ret_sum / max(n_pt, 1),
        "mean_ep_len": n_ep_steps / episodes,
    }


# ----------------------------------------------------------------------- main

def main() -> None:
    ppo, result = train_quad_policy(seed=0)
    wrapped = QuadFigure8Policy(ppo.pi)
    stats = tracking_rms(wrapped, episodes=20)

    print("\n=== quad-figure8 PPO ===")
    print(f"iterations           : {len(result.returns)}"
          f"  ({'FULL' if FULL else 'quick'})")
    print(f"final mean return    : {result.returns[-1]:.3f}")
    print(f"tracking RMS (pos)   : {stats['rms_pos_err_m'] * 1e3:.1f} mm")
    print(f"survival fraction    : {stats['survival_frac']:.2f}")
    print(f"mean episode length  : {stats['mean_ep_len']:.0f} / 600")
    print(f"mean reward / step   : {stats['mean_reward']:.4f}")

    save_policy(ppo.pi, POLICY_PATH, meta={
        "rms_pos_err_m": stats["rms_pos_err_m"],
        "survival_frac": stats["survival_frac"],
        "iters": len(result.returns),
        "full": int(FULL),
    })
    print(f"\nsaved -> {POLICY_PATH.name}")

    # round-trip check
    reloaded = load_quad_policy(POLICY_PATH)
    a = ppo.pi.greedy(np.zeros(8))
    b = reloaded.policy.greedy(np.zeros(8))
    assert np.allclose(a, b), "save/load round-trip mismatch"
    print("round-trip OK")


if __name__ == "__main__":
    main()
