"""AI + control hybrids.

``shield`` : :class:`ShieldedController` - run a learned / RL policy while a
classical fallback keeps the state inside a safe set, with intervention logging.
"""

from .shield import ShieldedController, barrier_predicate, box_predicate

__all__ = ["ShieldedController", "box_predicate", "barrier_predicate"]
