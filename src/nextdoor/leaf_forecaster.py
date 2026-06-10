"""Leaf-space similarity forecasting: tree-ensemble proximities + neighbours.

The durable insight of the NextDoor method is the *interface*: a forecast
expressed as a weighted set of concrete past promotions that a human can edit.
Nothing about that interface requires the similarity to come from a diagonal
NNLS metric. Here the similarity is the leaf co-occurrence proximity of a
random forest (Lin & Jeon 2006: a forest *is* an adaptive nearest-neighbour
method), which buys:

  * feature interactions and automatic feature selection (no dropped
    cross-terms, no collinearity smearing),
  * robustness to outliers via the log1p target option,
  * the ability to train *globally across SKUs* (pass SKU descriptors as
    features) while still explaining each forecast through retrieved
    neighbours -- the cross-learning the per-SKU design leaves on the table,

while the forecast remains "weighted average of similar past promotions".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from nextdoor.forecaster import NeighbourExplanation, _weighted_quantile

FloatArray = NDArray[np.floating]


@dataclass
class _Fitted:
    x_leaves: NDArray[np.int64]  # (n_train, n_trees)
    y: FloatArray
    z: FloatArray


class LeafSimilarityForecaster:
    """kNN forecaster whose metric is random-forest leaf co-occurrence.

    Args:
        n_estimators: Number of trees.
        k_neighbours: Number of neighbours used in the weighted average.
        min_samples_leaf: Leaf size; larger values give smoother proximities.
        target_transform: "log1p" (default) or None. Controls the scale the
            *forest* is fitted on (robust splits under fat tails).
        average_scale: "original" (default, unbiased in units) or
            "transformed" for the neighbour-weighted point forecast.
        random_state: Forest seed.
        rf_params: Extra keyword arguments for RandomForestRegressor.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        k_neighbours: int = 15,
        min_samples_leaf: int = 3,
        target_transform: str | None = "log1p",
        average_scale: str = "original",
        random_state: int | None = None,
        **rf_params,
    ):
        if average_scale not in ("original", "transformed"):
            raise ValueError(f"Unknown average_scale: {average_scale!r}")
        self.k_neighbours = k_neighbours
        self.target_transform = target_transform
        self.average_scale = average_scale
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            **rf_params,
        )
        self._fitted: _Fitted | None = None

    # ------------------------------------------------------------------ #

    def _transform_y(self, y: FloatArray) -> FloatArray:
        if self.target_transform == "log1p":
            return np.log1p(np.asarray(y, dtype=float))
        return np.asarray(y, dtype=float).copy()

    def _inverse_transform_y(self, z: FloatArray) -> FloatArray:
        if self.target_transform == "log1p":
            return np.expm1(z)
        return z

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    def fit(self, x_train: FloatArray, y_train: FloatArray) -> LeafSimilarityForecaster:
        """Fit the forest on (optionally log-transformed) targets."""
        x = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train, dtype=float).copy()
        z = self._transform_y(y)
        self.model.fit(x, z)
        self._fitted = _Fitted(x_leaves=self.model.apply(x), y=y, z=z)
        self.k_neighbours = min(self.k_neighbours, len(y))
        return self

    def _proximities(self, x_test: FloatArray) -> FloatArray:
        """Fraction of trees in which test and train rows share a leaf.

        Returns an (n_test, n_train) matrix in [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        test_leaves = self.model.apply(np.asarray(x_test, dtype=float))
        n_trees = test_leaves.shape[1]
        train_leaves = self._fitted.x_leaves
        prox = np.zeros((test_leaves.shape[0], train_leaves.shape[0]))
        for t in range(n_trees):
            prox += test_leaves[:, t][:, None] == train_leaves[:, t][None, :]
        return prox / n_trees

    def _neighbourhoods(self, x_test: FloatArray) -> tuple[NDArray[np.intp], FloatArray]:
        prox = self._proximities(x_test)
        k = self.k_neighbours
        order = np.argsort(-prox, axis=1, kind="stable")[:, :k]
        weights = np.take_along_axis(prox, order, axis=1)
        sums = weights.sum(axis=1, keepdims=True)
        flat = sums[:, 0] <= 0
        weights = np.where(sums > 0, weights / np.maximum(sums, 1e-12), 1.0 / k)
        if flat.any():
            weights[flat] = 1.0 / k
        return order, weights

    # ------------------------------------------------------------------ #

    def predict(self, x_test: FloatArray) -> FloatArray:
        """Neighbour-weighted point predictions (original scale)."""
        order, weights = self._neighbourhoods(x_test)
        if self.average_scale == "transformed":
            z_pred = np.sum(weights * self._fitted.z[order], axis=1)
            return self._inverse_transform_y(z_pred)
        return np.sum(weights * self._fitted.y[order], axis=1)

    def predict_quantiles(
        self, x_test: FloatArray, quantiles: FloatArray | list[float]
    ) -> FloatArray:
        """Weighted quantiles of the neighbour outcomes (original scale)."""
        qs = np.atleast_1d(np.asarray(quantiles, dtype=float))
        order, weights = self._neighbourhoods(x_test)
        out = np.empty((order.shape[0], qs.size))
        for i in range(order.shape[0]):
            out[i] = _weighted_quantile(self._fitted.y[order[i]], weights[i], qs)
        return out

    def explain(self, x_test_row: FloatArray) -> NeighbourExplanation:
        """The neighbours and weights behind a single forecast."""
        row = np.asarray(x_test_row, dtype=float).reshape(1, -1)
        prox = self._proximities(row)[0]
        order, weights = self._neighbourhoods(row)
        return NeighbourExplanation(
            indices=order[0],
            weights=weights[0],
            targets=self._fitted.y[order[0]],
            distances=1.0 - prox[order[0]],  # 1 - leaf co-occurrence proximity
            prediction=float(self.predict(row)[0]),
        )
