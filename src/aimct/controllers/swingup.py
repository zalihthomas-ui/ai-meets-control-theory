r"""Energy-based cart-pole swing-up and a hybrid swing-up <-> LQR controller.

Swing-up (Spong energy shaping)
------------------------------

The pendulum's mechanical energy, measured against the upright separatrix
(:math:`E = 0` upright, :math:`E = -2 m g l` hanging):

.. math::

    E(\theta, \dot\theta) = \tfrac12 (I + m l^2)\,\dot\theta^2
        + m g l (\cos\theta - 1),
    \qquad
    \dot E = -\,m l\,\ddot x\,\cos\theta\,\dot\theta .

Choosing the cart acceleration

.. math::

    \ddot x_{\text{des}} = k_E\,E\,\operatorname{sign}(\dot\theta\cos\theta)
        - k_x\,x - k_{\dot x}\,\dot x

gives :math:`\dot E = m l\,k_E |E|\,|\dot\theta\cos\theta| \ge 0` while
:math:`E < 0`, so the energy is pumped monotonically (in envelope) toward the
separatrix; the last two terms keep the cart near the rail centre.  The desired
acceleration is realised as a motor force by partial feedback linearisation
(eliminating :math:`\ddot\theta` from the two coupled rigid-body equations):

.. math::

    F = \Big[(M+m) - \frac{m^2 l^2 \cos^2\theta}{I+m l^2}\Big]\ddot x_{\text{des}}
        - m l\,\dot\theta^2 \sin\theta
        + \frac{m^2 g l^2 \sin\theta\cos\theta}{I+m l^2}.

(The sign of the gravity term differs from
``docs/references/swingup-and-basin.md`` §1.2, which is written for a
downward-zero angle; this form is consistent with ``aimct.systems.CartPole``,
where :math:`\theta = 0` is upright and gravity is destabilising.)

Hybrid handoff
--------------

:class:`HybridSwingUpLQR` runs the swing-up until the state enters a capture
window near upright, hands off to an LQR balance law, and falls back to swing-up
if the pole is knocked past a (wider) release angle -- a hysteresis switch that
avoids chattering at the boundary.  The active mode is recorded every step for
plotting.
"""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, Controller

__all__ = ["EnergyShapingSwingUp", "HybridSwingUpLQR", "wrap_angle"]


def wrap_angle(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]`` (hanging maps to ``+pi``)."""
    return np.pi - (np.pi - a) % (2.0 * np.pi)


class EnergyShapingSwingUp(Controller):
    """Spong energy-shaping swing-up for :class:`aimct.systems.CartPole`.

    Parameters
    ----------
    cartpole:
        The :class:`~aimct.systems.CartPole` instance (its ``mc, mp, l, g`` set
        the model; a uniform rod, ``I = m l**2 / 3``, is assumed to match
        ``CartPole``'s ``4/3`` term).
    k_energy:
        Energy-pumping gain (larger = faster swing-up, more force).
    k_cart, k_cart_rate:
        Cart centring proportional / derivative gains.
    u_max:
        Optional force clamp applied to the command.
    """

    def __init__(
        self,
        cartpole,
        *,
        k_energy: float = 10.0,
        k_cart: float = 2.0,
        k_cart_rate: float = 1.5,
        u_max: float | None = None,
    ) -> None:
        self.mc = float(cartpole.mc)
        self.mp = float(cartpole.mp)
        self.l = float(cartpole.l)
        self.g = float(cartpole.g)
        self.I_eff = self.mp * self.l**2 * 4.0 / 3.0   # I + m l**2, with I = m l**2 / 3
        self.k_energy = float(k_energy)
        self.k_cart = float(k_cart)
        self.k_cart_rate = float(k_cart_rate)
        self.u_max = None if u_max is None else float(u_max)
        self.reset()

    def reset(self) -> None:
        self.output: float = 0.0

    # -- physics ------------------------------------------------------------

    def pendulum_energy(self, state: ArrayLike) -> float:
        """Mechanical energy relative to the upright separatrix (``0`` upright,
        ``-2 m g l`` hanging)."""
        _, _, th, thd = np.asarray(state, dtype=float)
        return float(
            0.5 * self.I_eff * thd**2 + self.mp * self.g * self.l * (np.cos(th) - 1.0)
        )

    def desired_cart_accel(self, state: ArrayLike) -> float:
        pos, vel, th, thd = np.asarray(state, dtype=float)
        s = np.sign(thd * np.cos(th))
        if s == 0.0:                       # exact hanging rest: pick a direction
            s = 1.0
        return (self.k_energy * self.pendulum_energy(state) * s
                - self.k_cart * pos - self.k_cart_rate * vel)

    def _force_for_accel(self, a_des: float, th: float, thd: float) -> float:
        s, c = np.sin(th), np.cos(th)
        m, l, Ie = self.mp, self.l, self.I_eff
        coef = (self.mc + m) - (m**2 * l**2 * c**2) / Ie
        return coef * a_des - m * l * thd**2 * s + (m**2 * self.g * l**2 * s * c) / Ie

    # -- step -------------------------------------------------------------

    def update(self, measurement: ArrayLike, dt: float) -> float:
        x = np.atleast_1d(np.asarray(measurement, dtype=float))
        F = self._force_for_accel(self.desired_cart_accel(x), x[2], x[3])
        if self.u_max is not None:
            F = float(np.clip(F, -self.u_max, self.u_max))
        self.output = float(F)
        return self.output


class HybridSwingUpLQR(Controller):
    """Swing-up until captured near upright, then LQR balance, with a hysteresis
    switch back to swing-up if the pole is lost.

    The LQR must regulate to the upright equilibrium (reference angle 0); the
    hybrid feeds it a wrapped angle so a pole that came round the long way is
    still seen as a small error.

    Parameters
    ----------
    swingup, balance:
        The two sub-controllers (any ``Controller``); ``balance`` is typically an
        :class:`~aimct.controllers.LQR`.
    capture_angle, capture_rate:
        Enter LQR when ``|wrap(theta)| <= capture_angle`` and
        ``|theta_dot| <= capture_rate``.
    release_angle:
        Fall back to swing-up when ``|wrap(theta)| > release_angle``
        (``> capture_angle`` for hysteresis).
    """

    def __init__(
        self,
        swingup: Controller,
        balance: Controller,
        *,
        capture_angle: float = 0.35,
        capture_rate: float = 1.5,
        release_angle: float = 0.60,
    ) -> None:
        if release_angle <= capture_angle:
            raise ValueError("release_angle must exceed capture_angle (hysteresis)")
        self.swingup = swingup
        self.balance = balance
        self.capture_angle = float(capture_angle)
        self.capture_rate = float(capture_rate)
        self.release_angle = float(release_angle)
        self.reset()

    def reset(self) -> None:
        for c in (self.swingup, self.balance):
            if hasattr(c, "reset"):
                c.reset()
        self.mode = "swingup"
        self.mode_log: list[str] = []
        self.output: ArrayLike = 0.0

    def _select_mode(self, th_wrapped: float, thd: float) -> None:
        if self.mode == "swingup":
            if abs(th_wrapped) <= self.capture_angle and abs(thd) <= self.capture_rate:
                self.mode = "balance"
        elif abs(th_wrapped) > self.release_angle:
            self.mode = "swingup"

    def update(self, measurement: ArrayLike, dt: float) -> ArrayLike:
        x = np.atleast_1d(np.asarray(measurement, dtype=float)).astype(float)
        th_wrapped = wrap_angle(x[2])
        self._select_mode(th_wrapped, x[3])
        self.mode_log.append(self.mode)

        if self.mode == "balance":
            x_bal = x.copy()
            x_bal[2] = th_wrapped
            u = self.balance.update(x_bal, dt)
        else:
            u = self.swingup.update(x, dt)
        self.output = u
        return u

    # -- introspection --------------------------------------------------

    @property
    def switch_steps(self) -> list[int]:
        """Indices into ``mode_log`` where the active mode changed."""
        return [i for i in range(1, len(self.mode_log))
                if self.mode_log[i] != self.mode_log[i - 1]]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"HybridSwingUpLQR(mode={self.mode!r}, "
                f"capture={self.capture_angle}, release={self.release_angle})")
