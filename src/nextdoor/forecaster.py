"""Next Door Forecaster - kNN with learned feature metric for promotional sales.

This is the revised implementation described in ``fable_thought_about_this.md``.
It keeps the public API of v2 (train_nn / cv_neighbours / predict /
train_and_validate / fit_ensemble) while fixing the defects of the original
method and adding probabilistic output:

Bug fixes vs. the 2019 implementation
  * k is selected by mean *absolute* validation error (the original minimised
    the absolute value of the mean signed error, i.e. picked the k where
    errors cancel). The legacy behaviour is available as ``k_selection="bias"``
    so the impact can be measured.
  * No in-place mutation: callers' arrays are never modified, and calling
    ``predict`` twice with the same array returns the same result.
  * The default validation split in ``train_and_validate`` is chronological
    (tail split), matching the paper's protocol, instead of a random split.
  * The k sweep includes ``k_neighbours`` itself (the original stopped one
    short).

Statistical fixes
  * ``target_transform="log1p"`` learns the metric (and averages neighbours)
    on log(1+y), removing the (dy)^4 outlier sensitivity of the NNLS loss.
  * ``fit_intercept=True`` adds an intercept column to the NNLS system so the
    irreducible noise floor 2*sigma^2 of squared target differences is not
    absorbed into the feature weights. The intercept is also added to the
    predicted squared distances, which removes the pole of the inverse-distance
    kernel at zero distance.
  * ``pair_weighting=True`` down-weights pairs with small target difference:
    "far in outcome implies far in features" is sound, the converse is not,
    so accidentally-similar pairs should not be allowed to zero out genuinely
    relevant features.
  * ``scaler="robust"`` uses median/IQR scaling so a single outlying value of
    one feature cannot silently shrink that feature's effective range.
  * ``whiten=True`` decorrelates the features before learning the diagonal
    metric, addressing the dropped-cross-terms problem under collinearity.
  * ``kernel="gaussian"`` replaces the pole-at-zero 1/d kernel with a Gaussian
    kernel whose bandwidth is selected on the validation set.
  * ``time_decay_half_life`` applies exponential recency decay to neighbour
    weights.
  * ``metric_learner`` selects the engine: ``"nnls"`` (paper), ``"mlkr"``
    (diagonal metric learning for kernel regression, gradient-based) or
    ``"uniform"`` (no learning; the ablation baseline the paper was missing).

Probabilistic output
  * ``predict_quantiles`` returns weighted quantiles of the neighbour
    outcomes (a nonparametric predictive distribution, for newsvendor-style
    decisions).
  * ``predict_interval`` returns split-conformal (``method="absolute"``) or
    conformalised-quantile (``method="cqr"``) intervals calibrated on the
    validation set, with finite-sample coverage guarantees.
  * ``explain`` returns the retrieved neighbours and their weights -- the
    interpretability story of the paper, now as a first-class API.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import nnls
from sklearn import preprocessing

from nextdoor.mlkr import fit_diagonal_mlkr

FloatArray = NDArray[np.floating]

_EPS = 1e-12


@dataclass
class ForecastMetrics:
    """Container for forecast error metrics."""

    mae: float
    mse: float
    mean_error: float
    mape: float
    residuals: FloatArray


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""

    predictions: FloatArray
    predictions_std: FloatArray
    predictions_min: FloatArray | None = None
    predictions_max: FloatArray | None = None
    num_neighbours: float | None = None
    features: FloatArray | None = None


@dataclass
class NeighbourExplanation:
    """The neighbours behind one forecast: the editable explanation."""

    indices: NDArray[np.intp]
    weights: FloatArray
    targets: FloatArray
    distances: FloatArray
    prediction: float

    def as_rows(self) -> list[dict[str, float]]:
        return [
            {
                "train_index": int(i),
                "weight": float(w),
                "target": float(t),
                "distance": float(d),
            }
            for i, w, t, d in zip(
                self.indices, self.weights, self.targets, self.distances, strict=True
            )
        ]


@dataclass
class _Calibration:
    """Validation-set artefacts kept for conformal calibration."""

    x_raw: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    y: FloatArray = field(default_factory=lambda: np.empty(0))
    t: FloatArray | None = None


def _weighted_quantile(
    values: FloatArray, weights: FloatArray, quantiles: FloatArray
) -> FloatArray:
    """Weighted empirical quantiles (Hazen-type plotting positions)."""
    sorter = np.argsort(values)
    v = np.asarray(values, dtype=float)[sorter]
    w = np.asarray(weights, dtype=float)[sorter]
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        w = np.ones_like(w)
        total = w.sum()
    cw = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(np.asarray(quantiles, dtype=float), cw, v)


def _conformal_quantile(scores: FloatArray, alpha: float) -> float:
    """Finite-sample conformal quantile with the (n+1) correction."""
    n = len(scores)
    if n == 0:
        raise RuntimeError("No calibration data: run cv_neighbours/train_and_validate first.")
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


class NextDoorForecaster:
    """Forecasting engine based on kNN with a learned feature metric.

    Args:
        training_split: Fraction of training rows used as anchors when building
            the NNLS pair system.
        k_neighbours: Maximum number of neighbours to consider.
        max_weight_scale: Cap for infinite weights (legacy inverse kernel only).
        num_nnls_batches: Number of batches for NNLS (0 = no batching).
        kernel: "gaussian" (default) or "inverse" (the paper's 1/d kernel).
        target_transform: "auto" (default), "log1p" or None. The *metric* is
            learned on the transformed scale (under log1p this removes the
            (dy)^4 outlier sensitivity for multiplicative-noise sales, but it
            distorts additively-noised targets); quantiles are always computed
            on the original scale. "auto" fits both in ``train_and_validate``
            and keeps the one with the lower validation MAE.
        average_scale: Scale on which neighbour outcomes are averaged for the
            point forecast. "original" (default) keeps the forecast unbiased
            in units; "transformed" averages on the transformed scale
            (geometric-mean-like under log1p, biased low on volume metrics
            but more robust to a single outlying neighbour).
        scaler: "robust" (default) or "minmax" (the paper's choice).
        fit_intercept: Add a non-negative intercept to the NNLS system
            (estimates the noise floor 2*sigma^2; reported as ``noise_floor_``).
        pair_weighting: Down-weight small-|dy| pairs in the NNLS system
            ("far in outcome implies far in features" is sound; the converse
            is not). Off by default: on the paper's own surrogate generator it
            slightly hurts, because under heavy additive noise the large-|dy|
            pairs it emphasises are the noise-dominated ones.
        whiten: Decorrelate features (PCA whitening) before metric learning.
        metric_learner: "nnls" (default), "mlkr" or "uniform".
        k_selection: "mae" (default) or "bias" (legacy bug, for ablations).
        time_decay_half_life: Optional recency half-life. Times are taken from
            the optional ``t_*`` arguments, defaulting to row order.
        random_state: Seed for the anchor subsampling.

    Attributes:
        feat_weight: Learned feature weights (diagonal metric), in the
            (possibly whitened) metric space.
        noise_floor_: Learned NNLS intercept (estimate of 2*sigma^2 on the
            transformed target scale), 0.0 when disabled.
        bandwidth_: Gaussian kernel bandwidth (validated in cv_neighbours).
        k_neighbours: Number of neighbours after cross-validation.
        val_set_error: Validation-set signed forecast errors (original scale).
        val_set_forecast: Validation-set predictions (original scale).
    """

    def __init__(
        self,
        training_split: float = 0.25,
        k_neighbours: int = 15,
        max_weight_scale: float = 15.0,
        num_nnls_batches: int = 0,
        *,
        kernel: str = "gaussian",
        target_transform: str | None = "auto",
        average_scale: str = "original",
        scaler: str = "robust",
        fit_intercept: bool = True,
        pair_weighting: bool = False,
        whiten: bool = False,
        metric_learner: str = "nnls",
        k_selection: str = "mae",
        time_decay_half_life: float | None = None,
        random_state: int | None = None,
    ):
        if kernel not in ("gaussian", "inverse"):
            raise ValueError(f"Unknown kernel: {kernel!r}")
        if target_transform not in ("auto", "log1p", None):
            raise ValueError(f"Unknown target_transform: {target_transform!r}")
        if average_scale not in ("original", "transformed"):
            raise ValueError(f"Unknown average_scale: {average_scale!r}")
        if scaler not in ("robust", "minmax"):
            raise ValueError(f"Unknown scaler: {scaler!r}")
        if metric_learner not in ("nnls", "mlkr", "uniform"):
            raise ValueError(f"Unknown metric_learner: {metric_learner!r}")
        if k_selection not in ("mae", "bias"):
            raise ValueError(f"Unknown k_selection: {k_selection!r}")

        self.training_split = training_split
        self.k_neighbours = k_neighbours
        self.max_weight_scale = max_weight_scale
        self.num_nnls_batches = num_nnls_batches
        self.kernel = kernel
        self.target_transform = target_transform
        self.average_scale = average_scale
        self.scaler_kind = scaler
        self.fit_intercept = fit_intercept
        self.pair_weighting = pair_weighting
        self.whiten = whiten
        self.metric_learner = metric_learner
        self.k_selection = k_selection
        self.time_decay_half_life = time_decay_half_life

        self._uuid = str(uuid.uuid4())
        self._created_at = datetime.now()
        self._rng = np.random.default_rng(random_state)

        if scaler == "robust":
            self._scaler = preprocessing.RobustScaler()
        else:
            self._scaler = preprocessing.MinMaxScaler()
        self._whiten_matrix: FloatArray | None = None
        self._whiten_mean: FloatArray | None = None

        self._X_train: FloatArray | None = None  # metric space
        self._Y_train: FloatArray | None = None  # original scale
        self._Z_train: FloatArray | None = None  # transformed scale
        self._T_train: FloatArray | None = None

        self.feat_weight: FloatArray | None = None
        self.noise_floor_: float = 0.0
        self.rnorm: float | None = None
        self.bandwidth_: float | None = None
        self.features_batches: list[FloatArray] = []

        self.val_set_error: FloatArray = np.array([])
        self.val_set_forecast: FloatArray = np.array([])
        self._calibration = _Calibration()

    # ------------------------------------------------------------------ #
    # fitted state / transforms
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Check if the forecaster has been trained."""
        return self._X_train is not None and self.feat_weight is not None

    @property
    def feature_importances_(self) -> FloatArray:
        """Learned importances mapped back to the original feature space.

        Without whitening this is just ``feat_weight`` normalised to sum to 1
        (the raw scale of v is not identifiable: scaling v rescales all
        distances uniformly and leaves the normalised weighted mean
        unchanged). With whitening the diagonal weights of the whitened
        coordinates are mapped back through the squared loadings, ignoring
        cross terms, so treat the result as approximate.
        """
        if self.feat_weight is None:
            raise RuntimeError("Model must be trained first")
        if self._whiten_matrix is None:
            v = self.feat_weight
        else:
            v = (self._whiten_matrix**2) @ self.feat_weight
        total = v.sum()
        return v / total if total > 0 else v

    def _transform_y(self, y: FloatArray) -> FloatArray:
        if self.target_transform == "auto":
            # proper selection happens in train_and_validate; when train_nn is
            # called directly there is no validation set to select with
            warnings.warn(
                "target_transform='auto' needs a validation set; using 'log1p'. "
                "Call train_and_validate, or set the transform explicitly.",
                stacklevel=2,
            )
            self.target_transform = "log1p"
        if self.target_transform == "log1p":
            if np.any(y <= -1.0):
                warnings.warn(
                    "target_transform='log1p' requires y > -1; falling back to identity.",
                    stacklevel=2,
                )
                self.target_transform = None
                return np.asarray(y, dtype=float).copy()
            return np.log1p(np.asarray(y, dtype=float))
        return np.asarray(y, dtype=float).copy()

    def _inverse_transform_y(self, z: FloatArray) -> FloatArray:
        if self.target_transform == "log1p":
            return np.expm1(z)
        return z

    def _fit_feature_space(self, x: FloatArray) -> FloatArray:
        """Fit the scaler (and optional whitening) and return transformed copy."""
        xs = self._scaler.fit_transform(np.asarray(x, dtype=float))
        if self.whiten:
            self._whiten_mean = xs.mean(axis=0)
            xc = xs - self._whiten_mean
            cov = np.cov(xc, rowvar=False)
            cov = np.atleast_2d(cov)
            evals, evecs = np.linalg.eigh(cov)
            keep = evals > max(evals.max(), 1.0) * 1e-9
            self._whiten_matrix = evecs[:, keep] / np.sqrt(evals[keep])
            xs = xc @ self._whiten_matrix
        return xs

    def _to_feature_space(self, x: FloatArray) -> FloatArray:
        """Transform new data into the metric space. Never mutates the input."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        try:
            xs = self._scaler.transform(x)
        except Exception as exc:  # NotFittedError and friends
            raise RuntimeError("Scaler not fitted. Train the model first.") from exc
        if self.whiten:
            xs = (xs - self._whiten_mean) @ self._whiten_matrix
        return xs

    # ------------------------------------------------------------------ #
    # metric learning
    # ------------------------------------------------------------------ #

    def _pair_system(self, x: FloatArray, z: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Build the (squared feature diff, squared target diff) pair system."""
        num_records = x.shape[0]
        n_anchors = max(1, round(num_records * self.training_split))
        anchors = self._rng.choice(num_records, n_anchors, replace=False)

        m_list = []
        e_list = []
        mask = np.ones(num_records, dtype=bool)
        for idx in anchors:
            mask[idx] = False
            m_list.append((x[mask] - x[idx]) ** 2)
            e_list.append((z[mask] - z[idx]) ** 2)
            mask[idx] = True
        return np.concatenate(m_list, axis=0), np.concatenate(e_list, axis=0)

    def _solve_nnls(
        self, x: FloatArray, z: FloatArray, lambda_reg: float
    ) -> tuple[FloatArray, float, float]:
        """Solve the (weighted, intercepted, regularised) NNLS system."""
        m, e = self._pair_system(x, z)
        num_features = x.shape[1]

        if self.pair_weighting:
            # Rows with large |dz| carry the sound implication ("far in outcome
            # => far in features"); rows with tiny |dz| are accidental matches.
            row_w = np.sqrt(e + np.median(e) + _EPS)
            row_w /= row_w.mean()
            m = m * row_w[:, None]
            e = e * row_w

        if self.fit_intercept:
            m = np.hstack([m, np.ones((m.shape[0], 1))])

        if lambda_reg > 0.0:
            reg = lambda_reg * np.eye(num_features)
            if self.fit_intercept:
                # do not shrink the intercept
                reg = np.hstack([reg, np.zeros((num_features, 1))])
            m = np.vstack([m, reg])
            e = np.append(e, np.zeros(num_features))

        coeffs, rnorm = nnls(m, e)
        if self.fit_intercept:
            return coeffs[:-1], float(coeffs[-1]), float(rnorm)
        return coeffs, 0.0, float(rnorm)

    def _learn_metric(self, x: FloatArray, z: FloatArray, lambda_reg: float) -> None:
        if self.metric_learner == "uniform":
            self.feat_weight = np.ones(x.shape[1])
            self.noise_floor_ = 0.0
            self.rnorm = None
        elif self.metric_learner == "mlkr":
            result = fit_diagonal_mlkr(x, z, rng=self._rng)
            self.feat_weight = result.weights
            self.noise_floor_ = 0.0
            self.rnorm = result.final_loss
        elif self.num_nnls_batches == 0:
            self.feat_weight, self.noise_floor_, self.rnorm = self._solve_nnls(x, z, lambda_reg)
        else:
            idx = self._rng.permutation(x.shape[0])
            all_v, all_b, all_r = [], [], []
            for part in np.array_split(idx, self.num_nnls_batches):
                v, b, r = self._solve_nnls(x[part], z[part], lambda_reg)
                all_v.append(v)
                all_b.append(b)
                all_r.append(r)
            self.features_batches = all_v
            self.feat_weight = np.mean(all_v, axis=0)
            self.noise_floor_ = float(np.mean(all_b))
            self.rnorm = float(np.mean(all_r))

        if not np.any(self.feat_weight > 0):
            # degenerate solution: fall back to uniform so distances exist
            self.feat_weight = np.ones(x.shape[1])

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def train_nn(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        lambda_reg: float = 0.0,
        t_train: FloatArray | None = None,
    ) -> None:
        """Train the nearest neighbour model.

        Args:
            x_train: Training features (not modified).
            y_train: Training targets (not modified).
            lambda_reg: L2 regularisation parameter for the NNLS system.
            t_train: Optional timestamps (any monotone numeric unit) used for
                recency decay; defaults to row order.
        """
        x = self._fit_feature_space(x_train)
        y = np.asarray(y_train, dtype=float).copy()
        z = self._transform_y(y)

        self._learn_metric(x, z, lambda_reg)

        self._X_train = x
        self._Y_train = y
        self._Z_train = z
        if t_train is not None:
            self._T_train = np.asarray(t_train, dtype=float).copy()
        else:
            self._T_train = np.arange(len(y), dtype=float)
        self.k_neighbours = min(len(y), self.k_neighbours)
        self.bandwidth_ = self._default_bandwidth()

    def _default_bandwidth(self) -> float:
        """Median-heuristic bandwidth from a sample of training distances."""
        n = self._X_train.shape[0]
        take = min(n, 200)
        idx = self._rng.choice(n, take, replace=False)
        xs = self._X_train[idx]
        d2 = ((xs[:, None, :] - xs[None, :, :]) ** 2 @ self.feat_weight) + self.noise_floor_
        d2 = d2[np.triu_indices(take, k=1)]
        med = float(np.median(d2)) if d2.size else 1.0
        return math.sqrt(max(med, _EPS))

    # ------------------------------------------------------------------ #
    # distances / weights
    # ------------------------------------------------------------------ #

    def _squared_distances(self, x_row: FloatArray) -> FloatArray:
        return (self._X_train - x_row) ** 2 @ self.feat_weight + self.noise_floor_

    def _decay_multiplier(self) -> FloatArray:
        if self.time_decay_half_life is None:
            return np.ones(len(self._Y_train))
        age = self._T_train.max() - self._T_train
        return np.power(0.5, age / float(self.time_decay_half_life))

    def _kernel_weights(self, d2: FloatArray, bandwidth: float) -> FloatArray:
        if self.kernel == "gaussian":
            h2 = max(bandwidth, _EPS) ** 2
            return np.exp(-d2 / (2.0 * h2))
        # legacy inverse-distance kernel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            w = np.power(d2, -0.5)
        idx_inf = ~np.isfinite(w)
        if idx_inf.any():
            finite = w[~idx_inf]
            max_w = float(np.max(np.append(finite, 1.0)))
            w[idx_inf] = self.max_weight_scale * max_w
        return w

    def _neighbourhood(
        self, x_row: FloatArray, bandwidth: float | None = None
    ) -> tuple[NDArray[np.intp], FloatArray, FloatArray]:
        """Indices (by proximity), combined weights and squared distances."""
        d2 = self._squared_distances(x_row)
        order = np.argsort(d2, kind="stable")
        h = self.bandwidth_ if bandwidth is None else bandwidth
        w = self._kernel_weights(d2[order], h) * self._decay_multiplier()[order]
        return order, w, d2[order]

    def calculate_weights(
        self, x_test: FloatArray
    ) -> tuple[
        FloatArray,
        FloatArray,
        NDArray[np.intp],
        FloatArray,
    ]:
        """Calculate neighbour weights for a single test sample.

        The sample is expected in the *original* feature space; it is
        transformed internally and never modified.

        Returns:
            Tuple of (weights ordered by proximity, targets in the same order,
            ordering indices, weights in training-row order).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before calculating weights")
        row = self._to_feature_space(x_test)[0]
        order, w_sorted, _ = self._neighbourhood(row)
        unsorted = np.empty_like(w_sorted)
        unsorted[order] = w_sorted
        return w_sorted, self._Y_train[order], order, unsorted

    @staticmethod
    def _normalise(w: FloatArray) -> FloatArray:
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full_like(w, 1.0 / len(w))
        return w / s

    def _averaging_targets(self) -> FloatArray:
        """Targets on the scale used for the neighbour-weighted point forecast."""
        if self.average_scale == "transformed":
            return self._Z_train
        return self._Y_train

    def _from_averaging_scale(self, pred: FloatArray) -> FloatArray:
        if self.average_scale == "transformed":
            return self._inverse_transform_y(pred)
        return pred

    # ------------------------------------------------------------------ #
    # validation: joint (k, bandwidth) selection + conformal calibration
    # ------------------------------------------------------------------ #

    def _bandwidth_grid(self) -> list[float]:
        if self.kernel != "gaussian":
            return [self.bandwidth_ or 1.0]
        base = self.bandwidth_ or 1.0
        return [base * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)]

    def cv_neighbours(
        self,
        x_val: FloatArray,
        y_val: FloatArray,
        t_val: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Select k (and the kernel bandwidth) on a validation set.

        Also stores the validation set for conformal calibration.

        Args:
            x_val: Validation features (not modified).
            y_val: Validation targets (not modified).
            t_val: Optional timestamps (unused in selection, stored for
                completeness).

        Returns:
            Tuple of (predictions, signed errors) at the selected (k, h),
            on the original target scale.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before cross-validation")

        y_val = np.asarray(y_val, dtype=float)
        xs = self._to_feature_space(x_val)
        n_val = xs.shape[0]
        k_max = self.k_neighbours
        grid = self._bandwidth_grid()

        # errors[h_idx, k_idx, val_idx]
        targets = self._averaging_targets()
        y_hat = np.zeros((len(grid), k_max, n_val))
        for i, row in enumerate(xs):
            d2 = self._squared_distances(row)
            order = np.argsort(d2, kind="stable")
            decay = self._decay_multiplier()[order]
            a_sorted = targets[order]
            for hi, h in enumerate(grid):
                w = self._kernel_weights(d2[order], h) * decay
                cum_wa = np.cumsum(w * a_sorted)
                cum_w = np.cumsum(w)
                for k in range(1, k_max + 1):
                    if cum_w[k - 1] > 0:
                        pred = cum_wa[k - 1] / cum_w[k - 1]
                    else:
                        pred = a_sorted[:k].mean()
                    y_hat[hi, k - 1, i] = pred
        y_hat = self._from_averaging_scale(y_hat)
        errors = y_hat - y_val[None, None, :]

        if self.k_selection == "mae":
            criterion = np.mean(np.abs(errors), axis=2)
        else:  # "bias": the original implementation's (buggy) criterion
            criterion = np.abs(np.mean(errors, axis=2))

        hi, ki = np.unravel_index(np.argmin(criterion), criterion.shape)
        self.bandwidth_ = grid[hi]
        self.k_neighbours = int(ki + 1)
        self.val_set_forecast = y_hat[hi, ki]
        self.val_set_error = errors[hi, ki]

        self._calibration = _Calibration(
            x_raw=np.asarray(x_val, dtype=float).copy(),
            y=y_val.copy(),
            t=None if t_val is None else np.asarray(t_val, dtype=float).copy(),
        )
        return self.val_set_forecast, self.val_set_error

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #

    def predict(self, x_test: FloatArray) -> FloatArray:
        """Point predictions for test data (input is not modified)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        xs = self._to_feature_space(x_test)
        k = self.k_neighbours
        targets = self._averaging_targets()
        out = np.empty(xs.shape[0])
        for i, row in enumerate(xs):
            order, w, _ = self._neighbourhood(row)
            nw = self._normalise(w[:k])
            out[i] = nw @ targets[order[:k]]
        return self._from_averaging_scale(out)

    def predict_quantiles(
        self, x_test: FloatArray, quantiles: FloatArray | list[float]
    ) -> FloatArray:
        """Weighted quantiles of the neighbour outcomes (original scale).

        The k weighted neighbours form a nonparametric predictive
        distribution; this returns its quantiles, which is what a
        newsvendor-style ordering decision needs.

        Returns an array of shape ``(len(x_test), len(quantiles))``.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        qs = np.atleast_1d(np.asarray(quantiles, dtype=float))
        xs = self._to_feature_space(x_test)
        k = self.k_neighbours
        out = np.empty((xs.shape[0], qs.size))
        for i, row in enumerate(xs):
            order, w, _ = self._neighbourhood(row)
            out[i] = _weighted_quantile(self._Y_train[order[:k]], self._normalise(w[:k]), qs)
        return out

    def predict_interval(
        self,
        x_test: FloatArray,
        alpha: float = 0.1,
        method: str = "cqr",
    ) -> tuple[FloatArray, FloatArray]:
        """Conformal prediction intervals with finite-sample coverage.

        Requires ``cv_neighbours`` (or ``train_and_validate``) to have been
        run: the validation set doubles as the conformal calibration set.

        Args:
            x_test: Test features.
            alpha: Miscoverage level (0.1 = 90% intervals).
            method: "cqr" (conformalised quantile regression, adaptive width)
                or "absolute" (split conformal on absolute residuals,
                constant width).

        Returns:
            Tuple (lower, upper) arrays.
        """
        if self._calibration.y.size == 0:
            raise RuntimeError("No calibration set: run cv_neighbours/train_and_validate first.")
        if method == "absolute":
            q = _conformal_quantile(np.abs(self.val_set_error), alpha)
            y_hat = self.predict(x_test)
            return y_hat - q, y_hat + q
        if method != "cqr":
            raise ValueError(f"Unknown interval method: {method!r}")

        q_pair = [alpha / 2.0, 1.0 - alpha / 2.0]
        cal_q = self.predict_quantiles(self._calibration.x_raw, q_pair)
        scores = np.maximum(cal_q[:, 0] - self._calibration.y, self._calibration.y - cal_q[:, 1])
        q = _conformal_quantile(scores, alpha)
        test_q = self.predict_quantiles(x_test, q_pair)
        return test_q[:, 0] - q, test_q[:, 1] + q

    def explain(self, x_test_row: FloatArray) -> NeighbourExplanation:
        """Return the neighbours and weights behind a single forecast."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        row = self._to_feature_space(x_test_row)[0]
        order, w, d2 = self._neighbourhood(row)
        k = self.k_neighbours
        nw = self._normalise(w[:k])
        pred = nw @ self._averaging_targets()[order[:k]]
        return NeighbourExplanation(
            indices=order[:k],
            weights=nw,
            targets=self._Y_train[order[:k]],
            distances=np.sqrt(np.maximum(d2[:k], 0.0)),
            prediction=float(self._from_averaging_scale(np.asarray(pred))),
        )

    # ------------------------------------------------------------------ #
    # train + validate
    # ------------------------------------------------------------------ #

    def train_and_validate(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray | None = None,
        y_val: FloatArray | None = None,
        lambda_reg: float = 0.0,
        val_split: float = 0.2,
        t_train: FloatArray | None = None,
        t_val: FloatArray | None = None,
    ) -> None:
        """Train and validate the model.

        If no validation set is given, a *chronological* tail split is used
        (the paper's protocol): the last ``val_split`` fraction of the rows is
        held out. A random split would leak future information into the
        selection of k and the conformal calibration.

        With ``target_transform="auto"`` the model is fitted once per
        candidate transform and the one with the lower validation MAE wins
        (the right scale depends on whether the noise is multiplicative or
        additive, which is an empirical property of the data).
        """
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        if x_val is None or y_val is None:
            cut = max(1, int(round(len(y_train) * (1.0 - val_split))))
            cut = min(cut, len(y_train) - 1)
            x_train, x_val = x_train[:cut], x_train[cut:]
            y_train, y_val = y_train[:cut], y_train[cut:]
            if t_train is not None:
                t_train, t_val = t_train[:cut], t_train[cut:]

        if self.target_transform == "auto":
            candidates = []
            for transform in ("log1p", None):
                trial = copy.deepcopy(self)
                trial.target_transform = transform
                trial.train_nn(x_train, y_train, lambda_reg=lambda_reg, t_train=t_train)
                trial.cv_neighbours(x_val, y_val, t_val=t_val)
                candidates.append((float(np.mean(np.abs(trial.val_set_error))), trial))
            best = min(candidates, key=lambda c: c[0])[1]
            self.__dict__.update(best.__dict__)
            return

        self.train_nn(x_train, y_train, lambda_reg=lambda_reg, t_train=t_train)
        self.cv_neighbours(x_val, y_val, t_val=t_val)

    # ------------------------------------------------------------------ #
    # metrics / ensembles (API preserved from v2)
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_forecast_metrics(y_true: FloatArray, y_pred: FloatArray) -> ForecastMetrics:
        """Calculate forecast error metrics."""
        residuals = y_pred - y_true
        abs_residuals = np.abs(residuals)
        return ForecastMetrics(
            mae=abs_residuals.mean(),
            mse=np.power(residuals, 2).mean(),
            mean_error=residuals.mean(),
            mape=100 * (abs_residuals / np.abs(y_true)).mean(),
            residuals=residuals,
        )

    @staticmethod
    def _single_forecast(
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray,
        y_val: FloatArray,
        x_test: FloatArray,
        lambda_reg: float = 0.0,
        training_split: float = 0.5,
        num_nnls_batches: int = 0,
        seed: int | None = None,
        forecaster_kwargs: dict | None = None,
    ) -> tuple[FloatArray, int, FloatArray]:
        """Single forecaster training and prediction (for parallel execution)."""
        forecaster = NextDoorForecaster(
            training_split=training_split,
            num_nnls_batches=num_nnls_batches,
            random_state=seed,
            **(forecaster_kwargs or {}),
        )
        forecaster.train_and_validate(
            x_train, y_train, x_val=x_val, y_val=y_val, lambda_reg=lambda_reg
        )
        predictions = forecaster.predict(x_test)
        return predictions, forecaster.k_neighbours, forecaster.feat_weight

    @classmethod
    def fit_ensemble(
        cls,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray,
        y_val: FloatArray,
        x_test: FloatArray,
        num_forecasters: int = 100,
        lambda_reg: float = 0.0,
        training_split: float = 0.5,
        num_nnls_batches: int = 0,
        n_jobs: int = -1,
        random_state: int | None = None,
        **forecaster_kwargs,
    ) -> EnsemblePrediction:
        """Fit an ensemble of forecasters and make predictions.

        The ensemble spread (``predictions_std``) measures procedure variance
        (anchor subsampling), *not* predictive uncertainty; use
        ``predict_interval`` on a single forecaster for calibrated intervals.
        """
        query_start = time.time()
        seeds = np.random.default_rng(random_state).integers(0, 2**31, num_forecasters)

        results = Parallel(n_jobs=n_jobs)(
            delayed(cls._single_forecast)(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                lambda_reg,
                training_split,
                num_nnls_batches,
                int(seed),
                forecaster_kwargs,
            )
            for seed in seeds
        )

        query_elapsed = time.time() - query_start
        print(f"...prediction with {num_forecasters} forecasters done in {query_elapsed:.2f} sec!")

        all_predictions = np.array([r[0] for r in results])
        all_k_neighbours = np.array([r[1] for r in results])
        all_feat_weight = np.array([r[2] for r in results])

        return EnsemblePrediction(
            predictions=np.mean(all_predictions, axis=0),
            predictions_std=np.std(all_predictions, axis=0),
            predictions_min=np.min(all_predictions, axis=0),
            predictions_max=np.max(all_predictions, axis=0),
            num_neighbours=all_k_neighbours.mean(),
            features=all_feat_weight.mean(axis=0),
        )

    @classmethod
    def train_ensemble(
        cls,
        x_train: FloatArray,
        y_train: FloatArray,
        x_val: FloatArray,
        y_val: FloatArray,
        num_forecasters: int = 100,
        lambda_reg: float = 0.0,
        training_split: float = 0.5,
        num_nnls_batches: int = 0,
        n_jobs: int = -1,
        random_state: int | None = None,
        **forecaster_kwargs,
    ) -> list[NextDoorForecaster]:
        """Train an ensemble of forecasters."""
        query_start = time.time()
        seeds = np.random.default_rng(random_state).integers(0, 2**31, num_forecasters)

        def train_single(seed: int) -> NextDoorForecaster:
            forecaster = cls(
                training_split=training_split,
                num_nnls_batches=num_nnls_batches,
                random_state=seed,
                **forecaster_kwargs,
            )
            forecaster.train_and_validate(
                x_train, y_train, x_val=x_val, y_val=y_val, lambda_reg=lambda_reg
            )
            return forecaster

        ensemble = Parallel(n_jobs=n_jobs)(delayed(train_single)(int(seed)) for seed in seeds)

        query_elapsed = time.time() - query_start
        print(f"...training {num_forecasters} forecasters done in {query_elapsed:.2f} sec!")
        return ensemble

    @staticmethod
    def predict_with_ensemble(
        ensemble: list[NextDoorForecaster],
        x_test: FloatArray,
    ) -> EnsemblePrediction:
        """Make predictions using a trained ensemble."""
        all_predictions = np.array([frc.predict(x_test) for frc in ensemble])
        return EnsemblePrediction(
            predictions=np.mean(all_predictions, axis=0),
            predictions_std=np.std(all_predictions, axis=0),
        )

    # ------------------------------------------------------------------ #
    # misc utilities (API preserved from v2)
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_vector(vector: FloatArray, max_val: float) -> FloatArray:
        """Normalize vector from 0 to max_val."""
        min_val = vector.min()
        range_val = vector.max() - min_val
        if range_val == 0.0:
            return np.full_like(vector, max_val)
        return max_val * (vector - min_val) / range_val

    @staticmethod
    def get_basic_stats(vector: FloatArray) -> dict[str, float]:
        """Get basic statistics of a vector."""
        return {
            "max": float(vector.max()),
            "min": float(vector.min()),
            "mean": float(vector.mean()),
        }
