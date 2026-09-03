r"""A small multilayer perceptron, forward + backprop from scratch in NumPy.

No autograd, no torch. One hidden-nonlinearity choice (``tanh`` or ``relu``), an
MSE head, and a built-in Adam loop. This is the "implement it from scratch first"
reference; :mod:`aimct.ml` also offers a torch path for anything larger.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MLP"]


def _act(name: str):
    if name == "tanh":
        return (np.tanh, lambda a: 1.0 - a**2)          # f, f'(in terms of output a)
    if name == "relu":
        return (lambda z: np.maximum(z, 0.0), lambda a: (a > 0.0).astype(a.dtype))
    raise ValueError(f"unknown activation {name!r}")


class MLP:
    """Fully-connected net ``sizes[0] -> ... -> sizes[-1]`` with a linear output.

    Parameters
    ----------
    sizes : e.g. ``[6, 64, 64, 4]`` (input dim ... output dim).
    activation : ``"tanh"`` (default) or ``"relu"`` on the hidden layers.
    seed : RNG seed for the weight init.
    """

    def __init__(self, sizes, activation: str = "tanh", seed: int = 0) -> None:
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output size")
        self.sizes = list(sizes)
        self.activation = activation
        self._f, self._df = _act(activation)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for nin, nout in zip(self.sizes[:-1], self.sizes[1:]):
            scale = np.sqrt(2.0 / nin) if activation == "relu" else np.sqrt(1.0 / nin)
            self.W.append(rng.standard_normal((nin, nout)) * scale)
            self.b.append(np.zeros(nout))
        self._adam = None

    # ---------------------------------------------------------------- forward

    def forward(self, X, *, cache: bool = False):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        a = X
        acts = [a]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = z if i == len(self.W) - 1 else self._f(z)
            acts.append(a)
        return (a, acts) if cache else a

    __call__ = forward

    # ---------------------------------------------------------------- backward

    def backprop(self, acts, delta):
        """Backpropagate an arbitrary output-layer gradient ``delta`` (shape
        ``(B, out)`` = dL/d(output)) through the cached forward ``acts``.
        Returns ``(gW, gb)``. This is the generic hook the MSE head and the
        policy-gradient head both use."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._df(acts[i])
        return gW, gb

    def _loss_grad(self, X, Y):
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        yhat, acts = self.forward(X, cache=True)
        n = X.shape[0] if np.ndim(X) > 1 else 1
        resid = yhat - Y
        loss = float(np.mean(np.sum(resid**2, axis=1)))
        gW, gb = self.backprop(acts, (2.0 / n) * resid)
        return loss, gW, gb

    # ---------------------------------------------------------------- fit

    def fit(self, X, Y, *, epochs: int = 400, batch_size: int = 128,
            lr: float = 1e-2, seed: int = 0, verbose: bool = False):
        """Adam training. Returns the per-epoch mean loss history."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        rng = np.random.default_rng(seed)
        b1, b2, eps = 0.9, 0.999, 1e-8
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(x) for x in self.b]
        vb = [np.zeros_like(x) for x in self.b]
        t = 0
        hist = []
        n = X.shape[0]
        for ep in range(epochs):
            idx = rng.permutation(n)
            losses = []
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                loss, gW, gb = self._loss_grad(X[bi], Y[bi])
                losses.append(loss)
                t += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
                    vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                    vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                    mh = mW[i] / (1 - b1**t); vh = vW[i] / (1 - b2**t)
                    self.W[i] -= lr * mh / (np.sqrt(vh) + eps)
                    mhb = mb[i] / (1 - b1**t); vhb = vb[i] / (1 - b2**t)
                    self.b[i] -= lr * mhb / (np.sqrt(vhb) + eps)
            hist.append(float(np.mean(losses)))
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  loss {hist[-1]:.3e}")
        return hist

    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)
