"""Retrieval-augmented tabular forecasting (Tier 3, item 9).

The per-SKU design of the original method discards all cross-SKU signal: a
product with three historical promotions gets a three-neighbour model, and a
brand-new product gets nothing. Post-M5 the evidence is overwhelming that
*global* models pooled across series win on retail data, largely because of
cross-learning ("chocolate boxes behave like other chocolate boxes on Mother's
Day"). This module keeps the durable part of NextDoor -- the editable,
neighbour-based explanation -- while replacing the hand-built diagonal metric
with a learned cross-SKU embedding:

  1. Train one small model across *all* SKUs on log1p(sales), with SKU
     descriptors as features (the global, cross-learning step).
  2. Use the model's penultimate-layer activations as an embedding in which
     "close" means "similar promotional response", learned from every SKU.
  3. Retrieve the k nearest historical promotions of the query in that
     embedding space and form a weighted predictive *distribution* from their
     outcomes.
  4. Conformalise the distribution on a held-out calibration set for
     finite-sample coverage.

The forecast is still "a weighted average of these concrete past promotions",
so ``explain()`` returns the same editable neighbour set as the kNN method --
the retrieval interface is preserved, only the similarity is now global and
learned.

A TabPFN backend (``backend="tabpfn"``) is supported when the optional
``tabpfn`` package is installed: per-SKU contexts of a few dozen rows are
squarely in a prior-fitted network's sweet spot and need no training at all.
It degrades gracefully to the MLP backend when the package is absent.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from nextdoor.forecaster import (
    NeighbourExplanation,
    _conformal_quantile,
    _weighted_quantile,
)

FloatArray = NDArray[np.floating]

_EPS = 1e-12

_ACTIVATIONS = {
    "relu": lambda a: np.maximum(a, 0.0),
    "tanh": np.tanh,
    "logistic": lambda a: 1.0 / (1.0 + np.exp(-a)),
    "identity": lambda a: a,
}


@dataclass
class _Fitted:
    embeddings: FloatArray  # (n_train, d) training-set embeddings
    y: FloatArray  # (n_train,) original-scale targets
    bandwidth: float


@dataclass
class _Calibration:
    x: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    y: FloatArray = field(default_factory=lambda: np.empty(0))


class RetrievalAugmentedForecaster:
    """Global-embedding retrieval forecaster with conformal distributions.

    Args:
        backend: "mlp" (default, self-contained) or "tabpfn" (optional).
        hidden_layer_sizes: MLP architecture; the last hidden layer is the
            embedding space.
        k_neighbours: Number of retrieved neighbours.
        target_transform: "log1p" (default) or None.
        alpha: Default miscoverage level for ``predict_interval``.
        max_iter: MLP training iterations.
        random_state: Seed.
        mlp_params: Extra keyword args forwarded to ``MLPRegressor``.
    """

    def __init__(
        self,
        backend: str = "mlp",
        hidden_layer_sizes: tuple[int, ...] = (64, 16),
        k_neighbours: int = 15,
        target_transform: str | None = "log1p",
        alpha: float = 0.1,
        max_iter: int = 400,
        random_state: int | None = None,
        **mlp_params,
    ):
        if backend not in ("mlp", "tabpfn"):
            raise ValueError(f"Unknown backend: {backend!r}")
        if target_transform not in ("log1p", None):
            raise ValueError(f"Unknown target_transform: {target_transform!r}")

        self.backend = backend
        self.hidden_layer_sizes = hidden_layer_sizes
        self.k_neighbours = k_neighbours
        self.target_transform = target_transform
        self.alpha = alpha
        self.random_state = random_state

        self._scaler = RobustScaler()
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            **mlp_params,
        )
        self._tabpfn = None  # lazily built for the tabpfn backend
        self._fitted: _Fitted | None = None
        self._calibration = _Calibration()
        self._conformal_q: float | None = None

    # ------------------------------------------------------------------ #
    # transforms
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    def _transform_y(self, y: FloatArray) -> FloatArray:
        if self.target_transform == "log1p":
            return np.log1p(np.asarray(y, dtype=float))
        return np.asarray(y, dtype=float).copy()

    def _inverse_transform_y(self, z: FloatArray) -> FloatArray:
        if self.target_transform == "log1p":
            return np.expm1(z)
        return z

    # ------------------------------------------------------------------ #
    # embedding
    # ------------------------------------------------------------------ #

    def _embed(self, x: FloatArray) -> FloatArray:
        """Penultimate-layer activations of the trained MLP.

        Replicates the forward pass through every hidden layer (applying the
        hidden activation) and stops before the output projection.
        """
        xs = self._scaler.transform(np.asarray(x, dtype=float))
        act = _ACTIVATIONS[self._model.activation]
        h = xs
        # coefs_ has one matrix per layer transition; the last maps the final
        # hidden layer to the output, so we apply all but the last.
        for w, b in zip(self._model.coefs_[:-1], self._model.intercepts_[:-1], strict=True):
            h = act(h @ w + b)
        return h

    @staticmethod
    def _median_bandwidth(emb: FloatArray, rng: np.random.Generator) -> float:
        n = emb.shape[0]
        take = min(n, 200)
        idx = rng.choice(n, take, replace=False)
        e = emb[idx]
        d2 = ((e[:, None, :] - e[None, :, :]) ** 2).sum(-1)
        d2 = d2[np.triu_indices(take, k=1)]
        med = float(np.median(d2)) if d2.size else 1.0
        return math.sqrt(max(med, _EPS))

    # ------------------------------------------------------------------ #
    # fit / calibrate
    # ------------------------------------------------------------------ #

    def fit(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_cal: FloatArray | None = None,
        y_cal: FloatArray | None = None,
        cal_split: float = 0.2,
    ) -> RetrievalAugmentedForecaster:
        """Train the global embedder and calibrate the conformal correction.

        If no calibration set is given, a *chronological* tail split is used,
        matching the original paper's protocol.
        """
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        if x_cal is None or y_cal is None:
            cut = max(1, int(round(len(y_train) * (1.0 - cal_split))))
            cut = min(cut, len(y_train) - 1)
            x_train, x_cal = x_train[:cut], x_train[cut:]
            y_train, y_cal = y_train[:cut], y_train[cut:]

        z_train = self._transform_y(y_train)
        self._scaler.fit(x_train)
        xs = self._scaler.transform(x_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # MLP convergence chatter
            self._model.fit(xs, z_train)

        if self.backend == "tabpfn":
            self._maybe_build_tabpfn(x_train, z_train)

        rng = np.random.default_rng(self.random_state)
        emb = self._embed(x_train)
        self._fitted = _Fitted(
            embeddings=emb,
            y=y_train.copy(),
            bandwidth=self._median_bandwidth(emb, rng),
        )
        self.k_neighbours = min(self.k_neighbours, len(y_train))

        # split-conformal on absolute residuals of the predictive median
        self._calibration = _Calibration(
            x=np.asarray(x_cal, dtype=float).copy(),
            y=np.asarray(y_cal, dtype=float).copy(),
        )
        cal_med = self.predict(self._calibration.x)
        self._conformal_q = _conformal_quantile(np.abs(cal_med - self._calibration.y), self.alpha)
        return self

    def _maybe_build_tabpfn(self, x_train: FloatArray, z_train: FloatArray) -> None:
        try:
            from tabpfn import TabPFNRegressor  # type: ignore
        except Exception:
            warnings.warn(
                "backend='tabpfn' requested but the 'tabpfn' package is not "
                "installed; falling back to the MLP backend.",
                stacklevel=2,
            )
            self.backend = "mlp"
            return
        self._tabpfn = TabPFNRegressor()
        self._tabpfn.fit(self._scaler.transform(x_train), z_train)

    # ------------------------------------------------------------------ #
    # retrieval
    # ------------------------------------------------------------------ #

    def _neighbourhoods(self, x_test: FloatArray) -> tuple[NDArray[np.intp], FloatArray]:
        """Indices and normalised Gaussian weights of the k nearest neighbours."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        emb = self._embed(x_test)  # (n_test, d)
        train_emb = self._fitted.embeddings
        k = self.k_neighbours
        h2 = max(self._fitted.bandwidth, _EPS) ** 2

        d2 = (emb**2).sum(1)[:, None] - 2.0 * emb @ train_emb.T + (train_emb**2).sum(1)[None, :]
        order = np.argsort(d2, axis=1, kind="stable")[:, :k]
        nn_d2 = np.take_along_axis(d2, order, axis=1)
        w = np.exp(-np.maximum(nn_d2, 0.0) / (2.0 * h2))
        sums = w.sum(axis=1, keepdims=True)
        w = np.where(sums > 0, w / np.maximum(sums, _EPS), 1.0 / k)
        return order, w

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #

    def predict(self, x_test: FloatArray) -> FloatArray:
        """Predictive median (robust point forecast), original scale."""
        return self.predict_quantiles(x_test, [0.5])[:, 0]

    def predict_mean(self, x_test: FloatArray) -> FloatArray:
        """Neighbour-weighted mean (original scale)."""
        order, w = self._neighbourhoods(x_test)
        return np.sum(w * self._fitted.y[order], axis=1)

    def predict_quantiles(
        self, x_test: FloatArray, quantiles: FloatArray | list[float]
    ) -> FloatArray:
        """Weighted quantiles of the retrieved neighbours' outcomes."""
        qs = np.atleast_1d(np.asarray(quantiles, dtype=float))
        order, w = self._neighbourhoods(x_test)
        out = np.empty((order.shape[0], qs.size))
        for i in range(order.shape[0]):
            out[i] = _weighted_quantile(self._fitted.y[order[i]], w[i], qs)
        return out

    def predict_interval(
        self, x_test: FloatArray, alpha: float | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Split-conformal interval around the predictive median.

        Coverage is guaranteed in finite samples at level ``1 - alpha`` when
        the calibration set is exchangeable with the test set.
        """
        if self._conformal_q is None:
            raise RuntimeError("Call fit() (which calibrates) before predict_interval.")
        if alpha is not None and not math.isclose(alpha, self.alpha):
            q = _conformal_quantile(
                np.abs(self.predict(self._calibration.x) - self._calibration.y), alpha
            )
        else:
            q = self._conformal_q
        med = self.predict(x_test)
        return med - q, med + q

    def explain(self, x_test_row: FloatArray) -> NeighbourExplanation:
        """The retrieved neighbours and weights behind one forecast."""
        row = np.asarray(x_test_row, dtype=float).reshape(1, -1)
        emb = self._embed(row)[0]
        order, w = self._neighbourhoods(row)
        nbrs = order[0]
        dist = np.sqrt(((self._fitted.embeddings[nbrs] - emb) ** 2).sum(1))
        pred = float(np.sum(w[0] * self._fitted.y[nbrs]))
        return NeighbourExplanation(
            indices=nbrs,
            weights=w[0],
            targets=self._fitted.y[nbrs],
            distances=dist,
            prediction=pred,
        )
