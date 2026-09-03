r"""From-scratch tabular Q-learning for :class:`aimct.rl.ControlEnv`.

Continuous observations and the continuous scalar action are binned by
:class:`Discretizer`; :class:`QLearning` keeps a dense ``Q`` table and does the
one-line update

.. math::

    Q(s,a) \leftarrow Q(s,a) + \alpha\,
        \big[\, r + \gamma \max_{a'} Q(s',a')\,[\text{if not terminal}] - Q(s,a)\,\big],

with :math:`\varepsilon`-greedy exploration decayed each episode.  Everything is
plain NumPy and inspectable: the table is ``agent.Q``, the greedy policy is
``agent.greedy_policy()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Discretizer", "QLearning", "train", "evaluate"]


class Discretizer:
    """Uniform binning of a Box observation + a scalar action.

    Parameters
    ----------
    obs_low, obs_high:
        Observation box (finite); values are clipped into it.
    obs_bins:
        Bins per observation dimension (int -> same for all).
    action_low, action_high, n_actions:
        The scalar action grid ``np.linspace(action_low, action_high, n_actions)``.
    """

    def __init__(self, obs_low, obs_high, obs_bins, action_low, action_high, n_actions):
        self.low = np.asarray(obs_low, dtype=float)
        self.high = np.asarray(obs_high, dtype=float)
        d = self.low.size
        bins = ([int(obs_bins)] * d if np.isscalar(obs_bins)
                else [int(b) for b in obs_bins])
        if len(bins) != d:
            raise ValueError("obs_bins length must match the observation dimension")
        self.bins = bins
        # interior edges for np.digitize -> indices 0 .. bins[i]-1
        self.edges = [np.linspace(self.low[i], self.high[i], bins[i] + 1)[1:-1]
                      for i in range(d)]
        self.n_states = int(np.prod(bins))
        self.actions = np.linspace(action_low, action_high, int(n_actions))
        self.n_actions = int(n_actions)

    def encode(self, obs) -> int:
        obs = np.clip(np.asarray(obs, dtype=float), self.low, self.high)
        idx = [int(np.digitize(obs[i], self.edges[i])) for i in range(obs.size)]
        return int(np.ravel_multi_index(idx, self.bins))

    def action(self, a_idx: int) -> np.ndarray:
        return np.array([self.actions[int(a_idx)]])

    def state_index(self, s: int):
        """Inverse of :meth:`encode` at the bin level (multi-index tuple)."""
        return np.unravel_index(int(s), self.bins)


class QLearning:
    """Tabular Q-learning with epsilon-greedy exploration.

    ``lambda_ > 0`` turns on accumulating eligibility traces (naive Q(lambda));
    traces are kept sparse (only visited state-action pairs) so the per-step cost
    stays O(episode length) rather than O(table size). This sharply speeds up
    credit assignment on temporally extended tasks such as pendulum swing-up.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        lambda_: float = 0.0,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.999,
        optimistic_init: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.Q = np.full((self.n_states, self.n_actions), float(optimistic_init))
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)
        self._e: dict[tuple[int, int], float] = {}

    def reset_traces(self) -> None:
        self._e.clear()

    def act(self, s: int, *, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        row = self.Q[s]
        best = np.flatnonzero(row == np.nanmax(row))
        if best.size == 0:                     # all-NaN row -> uniform
            return int(self.rng.integers(self.n_actions))
        return int(self.rng.choice(best))

    def update(self, s: int, a: int, r: float, s_next: int, terminal: bool) -> float:
        target = r + (0.0 if terminal else self.gamma * self.Q[s_next].max())
        td = target - self.Q[s, a]
        if self.lambda_ > 0.0:
            self._e[(s, a)] = 1.0                            # replacing trace
            step = self.alpha * td
            decay = self.gamma * self.lambda_
            for key, ev in list(self._e.items()):
                self.Q[key] += step * ev
                nv = ev * decay
                if nv < 1e-3 or terminal:
                    del self._e[key]
                else:
                    self._e[key] = nv
        else:
            self.Q[s, a] += self.alpha * td
        return float(td)

    def decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        """Best action index per state (shape ``(n_states,)``)."""
        return self.Q.argmax(axis=1)


@dataclass
class TrainResult:
    returns: np.ndarray            # per-episode total reward
    td_errors: np.ndarray          # per-episode mean |TD error|
    agent: QLearning
    discretizer: Discretizer


def train(
    env,
    agent: QLearning,
    disc: Discretizer,
    *,
    episodes: int,
    seed: int = 0,
    callback=None,
) -> TrainResult:
    """Run ``episodes`` of Q-learning. Episode ``k`` resets with ``seed + k`` so a
    run is reproducible; bootstrapping is cut on ``terminated`` only (a
    ``truncated`` episode still bootstraps off the final state)."""
    returns = np.zeros(episodes)
    td_hist = np.zeros(episodes)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        agent.reset_traces()
        s = disc.encode(obs)
        done = False
        total = 0.0
        tds = []
        while not done:
            a = agent.act(s)
            obs, r, terminated, truncated, _ = env.step(disc.action(a))
            s_next = disc.encode(obs)
            tds.append(abs(agent.update(s, a, r, s_next, terminated)))
            s = s_next
            total += r
            done = terminated or truncated
        agent.decay()
        returns[ep] = total
        td_hist[ep] = np.mean(tds) if tds else 0.0
        if callback is not None:
            callback(ep, total, agent)
    return TrainResult(returns, td_hist, agent, disc)


def evaluate(
    env,
    agent: QLearning,
    disc: Discretizer,
    *,
    episodes: int = 5,
    seed: int = 10_000,
    greedy: bool = True,
) -> dict:
    """Roll the (greedy) policy for a few episodes; return mean return and one
    full state/action trajectory."""
    rets = np.zeros(episodes)
    traj_states: list[np.ndarray] = []
    traj_actions: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        s = disc.encode(obs)
        done = False
        total = 0.0
        while not done:
            a = agent.act(s, greedy=greedy)
            if ep == 0:
                traj_states.append(env.state)
                traj_actions.append(float(disc.actions[a]))
            obs, r, terminated, truncated, _ = env.step(disc.action(a))
            s = disc.encode(obs)
            total += r
            done = terminated or truncated
        rets[ep] = total
    return {
        "returns": rets,
        "mean_return": float(rets.mean()),
        "states": np.array(traj_states),
        "actions": np.array(traj_actions),
    }


if __name__ == "__main__":  # pragma: no cover - manual demo
    from .env import make, wrap_to_pi

    env = make("pendulum-swingup")
    disc = Discretizer([-1, -1, -10], [1, 1, 10], [15, 15, 25], -4.0, 4.0, 11)
    agent = QLearning(disc.n_states, disc.n_actions, alpha=0.2, gamma=0.99,
                      epsilon_decay=0.999, seed=0)
    res = train(env, agent, disc, episodes=4000, seed=0)
    ev = evaluate(env, agent, disc)
    err = np.abs(wrap_to_pi(ev["states"][:, 0] - np.pi))
    print(f"return: start {res.returns[:50].mean():.1f} -> end "
          f"{res.returns[-50:].mean():.1f}  greedy {ev['mean_return']:.1f}")
    print(f"greedy episode: min |angle-to-upright| {err.min():.3f} rad, "
          f"final {err[-1]:.3f} rad")
