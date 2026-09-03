r"""Observability of a linear pair :math:`(A, C)`.

Dual to controllability: :math:`(A, C)` is observable iff :math:`(A^\top, C^\top)`
is controllable. The state can be reconstructed from the output history iff

.. math::

    \mathcal{O} = \begin{bmatrix} C \\ CA \\ \vdots \\ CA^{n-1} \end{bmatrix}
"""

from __future__ import annotations

import numpy as np


def observability_matrix(A, C) -> np.ndarray:
    r"""Return :math:`\mathcal{O} = [\,C;\ CA;\ \dots;\ CA^{n-1}\,]`, shape ``(p n, n)``."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if C.shape[1] != n:
        raise ValueError("C must have n columns")
    blocks = [C]
    for _ in range(n - 1):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def observability_rank(A, C) -> int:
    return int(np.linalg.matrix_rank(observability_matrix(A, C)))


def is_observable(A, C) -> bool:
    """True iff ``rank(O) == n`` (full state reconstructable from the output)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return observability_rank(A, C) == A.shape[0]
