"""Train (and reload) the RL entry for Experiment 27: a from-scratch PPO
policy that steers the dynamic bicycle model through the double lane change.

Self-contained (not registered in aimct.rl.env.TASKS) - the lane-change
geometry is this experiment's, not a reusable reference like the quad
figure-8. Builds an aimct.rl.env.ControlEnv around BicycleVehicle directly.

Usage
-----
    python experiments/27_bicycle_double_lane_change/train_policy.py
    AIMCT_EXP_FULL=1 python experiments/27_bicycle_double_lane_change/train_policy.py

Writes ``bicycle_ppo_policy.npz`` next to this file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aimct.controllers import wrap_angle
from aimct.rl import PPO
from aimct.rl.env import ControlEnv
from aimct.rl.policy_gradient import GaussianPolicy
from aimct.systems import BicycleVehicle

HERE = Path(__file__).parent
POLICY_PATH = HERE / "bicycle_ppo_policy.npz"
FULL = os.environ.get("AIMCT_EXP_FULL") == "1"


_OBS_SCALE = np.array([2.0, 0.3, 5.0, 2.0, 1.0])   # Y_err, psi_err, vx_err, vy, r


def _obs(x, ref_at_X, vx0, obs_scale=_OBS_SCALE):
    X, Y, psi, vx, vy, r = x
    Yref, dYref = ref_at_X(X)
    psi_ref = np.arctan(dYref)
    raw = np.array([Y - Yref, wrap_angle(psi - psi_ref), vx - vx0, vy, r])
    return raw / obs_scale


# --------------------------------------------------------------- the task env

def _make_env(veh, ref_at_X, vx0, dt, t_final, u_bounds):
    w_y, w_psi, w_delta, w_ax = 8.0, 2.0, 0.5, 0.1

    def obs_fn(x):
        return _obs(x, ref_at_X, vx0)

    def reward_fn(x, u, x_next):
        X, Y, psi, vx, vy, r = x_next
        Yref, dYref = ref_at_X(X)
        psi_ref = np.arctan(dYref)
        e, psi_e = Y - Yref, wrap_angle(psi - psi_ref)
        return -(w_y * e ** 2 + w_psi * psi_e ** 2
                 + w_delta * u[0] ** 2 + w_ax * u[1] ** 2)

    def terminated(x):
        X, Y, psi, vx, vy, r = x
        Yref, _ = ref_at_X(X)
        return bool(abs(Y - Yref) > 5.0 or vx < 3.0)

    return ControlEnv(
        veh, dt=dt, max_steps=int(round(t_final / dt)), action_bounds=u_bounds,
        obs_fn=obs_fn, reward_fn=reward_fn, terminated_fn=terminated,
        x0=np.array([0.0, 0.0, 0.0, vx0, 0.0, 0.0]),
        reset_noise=np.array([0.0, 0.1, 0.02, 0.2, 0.05, 0.02]),
    )


def _lqr_lane_change_expert(veh, x_eq, vx0):
    """The Experiment-27 LQR entry, as a bare (state) -> (action) function -
    the behaviour-cloning expert for the RL policy."""
    from aimct.controllers import LQR

    A, B = veh.linearize(x_eq)
    K = LQR(A, B, np.diag([0.01, 30.0, 8.0, 1.0, 1.0, 1.0]), np.diag([80.0, 1.0])).K

    def act(x, ref_at_X):
        X, Y, psi, vx, vy, r = x
        Yref, dYref = ref_at_X(X)
        psi_ref = np.arctan(dYref)
        xr = np.array([X, Yref, psi_ref, vx0, 0.0, 0.0])
        u = -K @ (x - xr)
        u[1] = 0.5 * (vx0 - vx)             # same speed-hold every entry uses
        return u

    return act


def _behaviour_clone_dataset(veh, ref_at_X, vx0, dt, t_final, u_bounds, obs_scale,
                             n_steps, seed):
    expert = _lqr_lane_change_expert(veh, np.array([0.0, 0.0, 0.0, vx0, 0.0, 0.0]), vx0)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    x0 = np.array([0.0, 0.0, 0.0, vx0, 0.0, 0.0])
    x = x0.copy()
    t = 0.0
    for _ in range(n_steps):
        u_star = expert(x, ref_at_X)
        X.append(_obs(x, ref_at_X, vx0, obs_scale))
        Y.append(np.clip(u_star, u_bounds[0], u_bounds[1]))
        noise = rng.normal(0.0, [0.01, 0.05], 2)
        u_take = np.clip(u_star + noise, u_bounds[0], u_bounds[1])
        from aimct.simulate import rk4_step
        x = rk4_step(veh.dynamics, t, x, u_take, dt)
        t += dt
        if t > t_final or abs(x[1] - ref_at_X(x[0])[0]) > 5.0:
            x, t = x0.copy(), 0.0
    return np.asarray(X), np.asarray(Y)


def train_bicycle_policy(veh, ref_at_X, vx0, dt, t_final, u_bounds, *, seed: int = 0):
    """Behaviour-clone the LQR expert, then a light PPO fine-tune (kept only if
    it does not regress tracking - see the quad policy in Experiment 21/C9.3
    for why: from-scratch on-policy PPO plateaus badly on this task too)."""
    obs_scale = np.array([2.0, 0.3, 5.0, 2.0, 1.0])
    env = _make_env(veh, ref_at_X, vx0, dt, t_final, u_bounds)
    ppo = PPO(env, hidden=(64, 64), seed=seed, gamma=0.99, lam=0.95, clip=0.15,
             lr_pi=1e-4, lr_v=8e-4, epochs=6, minibatch=128,
             rollout_steps=4096 if FULL else 2048, ent_coef=0.0, target_kl=0.015)

    n_bc = 40_000 if FULL else 15_000
    Xbc, Ybc = _behaviour_clone_dataset(veh, ref_at_X, vx0, dt, t_final, u_bounds,
                                        obs_scale, n_bc, seed)
    ppo.pi.net.b[-1][:] = 0.0
    ppo.pi.net.fit(Xbc, Ybc, epochs=250 if FULL else 150, batch_size=256,
                   lr=3e-3, seed=seed, verbose=True)
    ppo.pi.log_std[:] = np.log([0.02, 0.1])
    bc_rms = _rms(ppo, env, ref_at_X, vx0, obs_scale, dt, t_final, u_bounds)

    bc_snapshot = ([W.copy() for W in ppo.pi.net.W], [b.copy() for b in ppo.pi.net.b],
                  ppo.pi.log_std.copy())
    result = ppo.train(iterations=60 if FULL else 20, verbose=True)
    ft_rms = _rms(ppo, env, ref_at_X, vx0, obs_scale, dt, t_final, u_bounds)
    if ft_rms > bc_rms * 1.02:
        ppo.pi.net.W, ppo.pi.net.b, ppo.pi.log_std = (*bc_snapshot[:2], bc_snapshot[2])
        print(f"  fine-tune worse ({ft_rms:.3f} vs {bc_rms:.3f}) -> keeping the clone")
    return ppo, result


def _rms(ppo, env, ref_at_X, vx0, obs_scale, dt, t_final, u_bounds):
    """Greedy rollout RMS lateral error, for the keep-BC-or-fine-tune check."""
    x = np.array([0.0, 0.0, 0.0, vx0, 0.0, 0.0])
    sq, n = 0.0, 0
    t = 0.0
    from aimct.simulate import rk4_step
    while t < t_final:
        o = _obs(x, ref_at_X, vx0, obs_scale)
        u = np.clip(ppo.pi.greedy(o), u_bounds[0], u_bounds[1])
        x = rk4_step(env.system.dynamics, t, x, u, dt)
        t += dt
        Yref, _ = ref_at_X(x[0])
        sq += (x[1] - Yref) ** 2
        n += 1
        if abs(x[1] - Yref) > 20.0:
            break
    return float(np.sqrt(sq / max(n, 1)))


class BicyclePolicy:
    """Wraps a trained GaussianPolicy for the [delta, ax] action, doing the
    observation construction internally."""

    name = "RL (PPO)"

    def __init__(self, policy: GaussianPolicy, ref_at_X, vx0):
        self.policy, self.ref_at_X, self.vx0 = policy, ref_at_X, vx0

    def reset(self):
        pass

    def _obs(self, x):
        return _obs(x, self.ref_at_X, self.vx0)

    def update(self, x, dt):
        return np.asarray(self.policy.greedy(self._obs(np.asarray(x, float))), float)


def save_policy(policy: GaussianPolicy, path=POLICY_PATH) -> Path:
    blob = {"log_std": policy.log_std, "act_low": policy.act_low,
           "act_high": policy.act_high, "n_layers": np.array(len(policy.net.W))}
    for i, (W, b) in enumerate(zip(policy.net.W, policy.net.b)):
        blob[f"W{i}"], blob[f"b{i}"] = W, b
    np.savez(path, **blob)
    return path


def _load_gaussian_policy(path=POLICY_PATH) -> GaussianPolicy:
    d = np.load(Path(path))
    n = int(d["n_layers"])
    Ws = [d[f"W{i}"] for i in range(n)]
    bs = [d[f"b{i}"] for i in range(n)]
    hidden = tuple(W.shape[1] for W in Ws[:-1])
    pi = GaussianPolicy(Ws[0].shape[0], Ws[-1].shape[1], hidden=hidden,
                        act_low=d["act_low"], act_high=d["act_high"])
    pi.net.W, pi.net.b = [W.copy() for W in Ws], [b.copy() for b in bs]
    pi.log_std = d["log_std"].copy()
    return pi


def load_or_train_policy(veh, ref_at_X, x0, u_bounds, *, full=False, retrain=False):
    """Load the committed policy if present; otherwise train (and save) one."""
    if POLICY_PATH.exists() and not retrain:
        return BicyclePolicy(_load_gaussian_policy(), ref_at_X, float(x0[3]))
    ppo, _ = train_bicycle_policy(veh, ref_at_X, float(x0[3]), 0.01, 8.0, u_bounds)
    save_policy(ppo.pi)
    return BicyclePolicy(ppo.pi, ref_at_X, float(x0[3]))


if __name__ == "__main__":
    import run as exp

    ppo, result = train_bicycle_policy(exp.veh, exp.ref_at_X, exp.VX0, exp.DT,
                                       exp.T_FINAL, exp.U_BOUNDS)
    pol = BicyclePolicy(ppo.pi, exp.ref_at_X, exp.VX0)
    m = exp.run_one(pol)
    print(f"\nfinal mean return: {result.returns[-1]:.3f}")
    print(f"rms_err_mm={m['rms_err_mm']:.1f}  max_err_mm={m['max_err_mm']:.1f}  "
         f"diverged={m['diverged']}")
    save_policy(ppo.pi)
    print(f"saved -> {POLICY_PATH.name}")
