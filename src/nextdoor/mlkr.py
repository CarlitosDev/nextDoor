"""Diagonal Metric Learning for Kernel Regression (MLKR).

Weinberger & Tesauro (AISTATS 2007) pose the objective the paper's Eq. (10)
describes -- minimise the leave-one-out Nadaraya-Watson regression error over a
Mahalanobis metric -- and solve it by gradient descent. The paper deemed this
intractable in 2018 and substituted the biased NNLS proxy; on per-SKU problems
(tens to a few hundred rows) it converges in milliseconds, so here it is.

We learn a *diagonal* metric to preserve the per-feature interpretability of
the original method: distances are d2_ij = sum_f v_f (x_if - x_jf)^2 with
v_f = a_f^2 >= 0, kernel k_ij = exp(-d2_ij), and the loss is the sum of squared
leave-one-out errors. Because MLKR optimises prediction error directly, it does
not assert the bidirectional identity |dy|^2 = m.v that corrupts the NNLS
solution (accidentally-similar outcomes, dropped cross terms, noise floor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

_EPS = 1e-12


@dataclass
class MLKRResult:
    """Result of a diagonal MLKR fit."""

    weights: FloatArray  # v = a^2, the diagonal metric
    final_loss: float
    loss_history: list[float]


def _loo_loss_and_grad(
    a: FloatArray,
    sq_diffs: FloatArray,  # (n, n, p) pairwise squared feature differences
    z: FloatArray,
) -> tuple[float, FloatArray]:
    """Leave-one-out NW loss and gradient wrt a (v = a^2)."""
    v = a**2
    d2 = sq_diffs @ v  # (n, n)
    np.fill_diagonal(d2, np.inf)  # exclude self from the LOO prediction
    # subtract row-min for numerical stability of the softmax-like weights
    d2_min = np.min(d2, axis=1, keepdims=True)
    k = np.exp(-(d2 - d2_min))
    np.fill_diagonal(k, 0.0)
    k_sum = k.sum(axis=1, keepdims=True) + _EPS
    w = k / k_sum  # (n, n), rows sum to 1
    z_hat = w @ z
    err = z_hat - z  # (n,)
    loss = float(err @ err)

    # dyhat_i/dd2_ij = -w_ij (z_j - z_hat_i), d2_ij = sum_f v_f sq_ijf, hence
    # dL/dv_f = -2 sum_ij err_i w_ij (z_j - z_hat_i) sq_diffs_ijf
    s = (err[:, None] * w) * (z[None, :] - z_hat[:, None])  # (n, n)
    grad_v = -2.0 * np.einsum("ij,ijf->f", s, sq_diffs)
    grad_a = 2.0 * a * grad_v
    return loss, grad_a


def fit_diagonal_mlkr(
    x: FloatArray,
    z: FloatArray,
    n_iters: int = 300,
    lr: float = 0.05,
    max_rows: int = 400,
    tol: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> MLKRResult:
    """Fit a non-negative diagonal metric by gradient descent on the LOO loss.

    Args:
        x: Feature matrix (already scaled), shape (n, p).
        z: Targets (already transformed), shape (n,).
        n_iters: Maximum gradient steps.
        lr: Initial learning rate (adaptive: halved on a worsening step,
            gently increased on improving ones).
        max_rows: Memory/time cap; larger problems are subsampled (the
            pairwise tensor is O(n^2 p)).
        tol: Relative improvement tolerance for early stopping.
        rng: Random generator for the subsample.

    Returns:
        MLKRResult with the learned diagonal weights v = a^2.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n, p = x.shape
    if n > max_rows:
        idx = rng.choice(n, max_rows, replace=False)
        x, z = x[idx], z[idx]
        n = max_rows

    # standardise z for conditioning; v is scale-free for the downstream
    # forecaster (normalised weights), so this does not change predictions
    z_std = z.std()
    z_work = (z - z.mean()) / (z_std + _EPS)

    sq_diffs = (x[:, None, :] - x[None, :, :]) ** 2  # (n, n, p)

    # init: uniform weights scaled so the mean pairwise distance is ~1
    mean_d2 = float(sq_diffs.sum(axis=2).mean()) / max(p, 1)
    a = np.full(p, 1.0 / np.sqrt(max(p * mean_d2, _EPS)))

    loss, grad = _loo_loss_and_grad(a, sq_diffs, z_work)
    history = [loss]
    best_a, best_loss = a.copy(), loss

    for _ in range(n_iters):
        step = lr * grad / (np.linalg.norm(grad) + _EPS)
        a_new = a - step
        new_loss, new_grad = _loo_loss_and_grad(a_new, sq_diffs, z_work)
        if new_loss <= loss:
            if loss - new_loss < tol * max(loss, _EPS):
                a, loss = a_new, new_loss
                history.append(loss)
                break
            a, loss, grad = a_new, new_loss, new_grad
            lr *= 1.05
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
        history.append(loss)
        if loss < best_loss:
            best_a, best_loss = a.copy(), loss

    return MLKRResult(weights=best_a**2, final_loss=best_loss, loss_history=history)
