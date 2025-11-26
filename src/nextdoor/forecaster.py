"""Next Door Forecaster - kNN with feature learning for promotional sales forecasting."""

from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import nnls
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split


@dataclass
class ForecastMetrics:
    """Container for forecast error metrics."""

    mae: float
    mse: float
    mean_error: float
    mape: float
    residuals: NDArray[np.floating]


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""

    predictions: NDArray[np.floating]
    predictions_std: NDArray[np.floating]
    predictions_min: NDArray[np.floating] | None = None
    predictions_max: NDArray[np.floating] | None = None
    num_neighbours: float | None = None
    features: NDArray[np.floating] | None = None


class NextDoorForecaster:
    """Forecasting engine based on kNN with feature learning.

    This forecaster uses Non-Negative Least Squares (NNLS) to learn feature
    weights and k-Nearest Neighbours for predictions.

    Args:
        training_split: Fraction of training data to use for NNLS optimization
        k_neighbours: Maximum number of neighbours to consider
        max_weight_scale: Scale factor for infinite weights
        num_nnls_batches: Number of batches for NNLS (0 = no batching)

    Attributes:
        feat_weight: Learned feature weights from NNLS
        k_neighbours: Optimal number of neighbours after cross-validation
        val_set_error: Validation set forecast errors
        val_set_forecast: Validation set predictions
    """

    def __init__(
        self,
        training_split: float = 0.25,
        k_neighbours: int = 15,
        max_weight_scale: float = 15.0,
        num_nnls_batches: int = 0,
    ):
        self.training_split = training_split
        self.k_neighbours = k_neighbours
        self.max_weight_scale = max_weight_scale
        self.num_nnls_batches = num_nnls_batches

        self._uuid = str(uuid.uuid4())
        self._created_at = datetime.now()

        self._scaler = preprocessing.MinMaxScaler(copy=False)
        self._X_train: NDArray[np.floating] | None = None
        self._Y_train: NDArray[np.floating] | None = None

        self.feat_weight: NDArray[np.floating] | None = None
        self.rnorm: float | None = None
        self.features_batches: list[NDArray[np.floating]] = []

        self.val_set_error: NDArray[np.floating] = np.array([])
        self.val_set_forecast: NDArray[np.floating] = np.array([])

    @property
    def is_fitted(self) -> bool:
        """Check if the forecaster has been trained."""
        return self._X_train is not None and self.feat_weight is not None

    def _scale_x_train(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Normalize training data between 0 and 1."""
        self._scaler = self._scaler.fit(x)
        self._scaler.transform(x)
        return x

    def _scale_x_test(self, x_test: NDArray[np.floating]) -> NDArray[np.floating]:
        """Normalize test data using training parameters."""
        try:
            self._scaler.transform(x_test)
        except NotFittedError:
            raise RuntimeError("Scaler not fitted. Train the model first.") from None
        return x_test

    def _solve_nnls_training(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        lambda_reg: float,
    ) -> None:
        """Solve NNLS problem for feature weight learning."""
        num_records = x.shape[0]
        num_features = x.shape[1]
        test_size = round(num_records * self.training_split)

        idx_test = np.random.choice(num_records, test_size, replace=False)

        m_list = []
        e_list = []
        current_promo = np.zeros(num_records, dtype=bool)
        remaining_promos = np.ones(num_records, dtype=bool)

        for k in range(test_size):
            idx = idx_test[k]
            current_promo[idx] = True
            remaining_promos[idx] = False
            m_list.append(np.power(x[remaining_promos] - x[current_promo], 2))
            e_list.append(np.power(y[remaining_promos] - y[current_promo], 2))
            current_promo[idx] = False
            remaining_promos[idx] = True

        m = np.concatenate(m_list, axis=0).copy()
        e = np.concatenate(e_list, axis=0).copy()

        if lambda_reg > 0.0:
            m_prime = lambda_reg * np.eye(num_features)
            e_prime = np.zeros(num_features)
            m = np.append(m, m_prime, axis=0)
            e = np.append(e, e_prime, axis=0)

        feat_weight, rnorm = nnls(m, e)
        self.feat_weight = feat_weight
        self.rnorm = rnorm

    def _solve_nnls_training_batches(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        lambda_reg: float,
    ) -> None:
        """Solve NNLS with data batching for large datasets."""
        all_num_records = x.shape[0]
        idx = np.arange(all_num_records)
        np.random.shuffle(idx)
        idx_step = np.split(idx, self.num_nnls_batches)

        all_feat_weight = []
        all_rnorm = []

        for this_idx in idx_step:
            this_x = x[this_idx]
            this_y = y[this_idx]

            num_records = this_x.shape[0]
            num_features = this_x.shape[1]
            test_size = round(num_records * self.training_split)

            idx_test = np.random.choice(num_records, test_size, replace=False)

            m_list = []
            e_list = []
            current_promo = np.zeros(num_records, dtype=bool)
            remaining_promos = np.ones(num_records, dtype=bool)

            for k in range(test_size):
                idx = idx_test[k]
                current_promo[idx] = True
                remaining_promos[idx] = False
                m_list.append(
                    np.power(this_x[remaining_promos] - this_x[current_promo], 2)
                )
                e_list.append(
                    np.power(this_y[remaining_promos] - this_y[current_promo], 2)
                )
                current_promo[idx] = False
                remaining_promos[idx] = True

            m = np.concatenate(m_list, axis=0).copy()
            e = np.concatenate(e_list, axis=0).copy()

            if lambda_reg > 0.0:
                m_prime = lambda_reg * np.eye(num_features)
                e_prime = np.zeros(num_features)
                m = np.append(m, m_prime, axis=0)
                e = np.append(e, e_prime, axis=0)

            feat_weight, rnorm = nnls(m, e)
            all_feat_weight.append(feat_weight)
            all_rnorm.append(rnorm)

        self.features_batches = all_feat_weight
        self.feat_weight = np.mean(all_feat_weight, axis=0)
        self.rnorm = np.mean(all_rnorm)

    def train_nn(
        self,
        x_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        lambda_reg: float = 0.0,
    ) -> None:
        """Train the nearest neighbour model.

        Args:
            x_train: Training features
            y_train: Training targets
            lambda_reg: L2 regularization parameter
        """
        self._scale_x_train(x_train)

        if self.num_nnls_batches == 0:
            self._solve_nnls_training(x_train, y_train, lambda_reg)
        else:
            self._solve_nnls_training_batches(x_train, y_train, lambda_reg)

        self._X_train = x_train
        self._Y_train = y_train
        self.k_neighbours = min(len(y_train), self.k_neighbours)

    def cv_neighbours(
        self,
        x_val: NDArray[np.floating],
        y_val: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Cross-validate optimal number of neighbours.

        Args:
            x_val: Validation features
            y_val: Validation targets

        Returns:
            Tuple of (predictions, errors) for optimal k
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before cross-validation")

        self._scale_x_test(x_val)
        y_hat = np.zeros((y_val.shape[0], self.k_neighbours - 1))
        frc_error = np.zeros((y_val.shape[0], self.k_neighbours - 1))

        for idx, cpromo in enumerate(x_val):
            current_weights_sorted, y_train_sorted, _, _ = self.calculate_weights(
                cpromo
            )
            for k in range(1, self.k_neighbours):
                normalised_weights = current_weights_sorted[:k] / np.sum(
                    current_weights_sorted[:k]
                )
                current_frc = normalised_weights.dot(y_train_sorted[:k])
                y_hat[idx, k - 1] = current_frc
                frc_error[idx, k - 1] = current_frc - y_val[idx]

        frc_error_mu = np.mean(frc_error, axis=0)
        frc_abs_error = np.abs(frc_error_mu)

        self.k_neighbours = 1 + np.argmin(frc_abs_error)
        self.val_set_error = frc_error[:, self.k_neighbours - 1]
        self.val_set_forecast = y_hat[:, self.k_neighbours - 1]

        return self.val_set_forecast, self.val_set_error

    def calculate_weights(self, x_test: NDArray[np.floating]) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.intp],
        NDArray[np.floating],
    ]:
        """Calculate weights for a test sample.

        Args:
            x_test: Single test sample

        Returns:
            Tuple of (sorted_weights, sorted_targets, sorted_indices, weights)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before calculating weights")

        m_test = np.power(self._X_train - x_test, 2)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            current_weights = np.power(m_test.dot(self.feat_weight), -0.5)

        idx_inf = np.isinf(current_weights)
        max_weight = np.max(np.append(current_weights[~idx_inf], 1.0))
        current_weights[idx_inf] = self.max_weight_scale * max_weight

        idx_sorted = np.argsort(current_weights)[::-1]

        current_weights_sorted = current_weights[idx_sorted]
        y_train_sorted = self._Y_train[idx_sorted]

        return current_weights_sorted, y_train_sorted, idx_sorted, current_weights

    def predict(self, x_test: NDArray[np.floating]) -> NDArray[np.floating]:
        """Make predictions for test data.

        Args:
            x_test: Test features

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")

        self._scale_x_test(x_test)
        y_hat = []

        for cpromo in x_test:
            current_weights_sorted, y_train_sorted, _, _ = self.calculate_weights(
                cpromo
            )

            normalised_weights = current_weights_sorted[: self.k_neighbours] / np.sum(
                current_weights_sorted[: self.k_neighbours]
            )
            current_frc = normalised_weights.dot(y_train_sorted[: self.k_neighbours])
            y_hat.append(current_frc)

        return np.array(y_hat)

    def train_and_validate(
        self,
        x_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        x_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
        lambda_reg: float = 0.0,
        val_split: float = 0.2,
    ) -> None:
        """Train and validate the model.

        Args:
            x_train: Training features
            y_train: Training targets
            x_val: Validation features (optional)
            y_val: Validation targets (optional)
            lambda_reg: L2 regularization parameter
            val_split: Validation split if x_val/y_val not provided
        """
        if x_val is None or y_val is None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=val_split, random_state=42
            )

        self.train_nn(x_train.copy(), y_train.copy(), lambda_reg=lambda_reg)
        self.cv_neighbours(x_val.copy(), y_val.copy())

    @staticmethod
    def get_forecast_metrics(
        y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> ForecastMetrics:
        """Calculate forecast error metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            ForecastMetrics object with MAE, MSE, mean error, MAPE, and residuals
        """
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
        x_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        x_val: NDArray[np.floating],
        y_val: NDArray[np.floating],
        x_test: NDArray[np.floating],
        lambda_reg: float = 0.0,
        training_split: float = 0.5,
        num_nnls_batches: int = 0,
    ) -> tuple[NDArray[np.floating], int, NDArray[np.floating]]:
        """Single forecaster training and prediction (for parallel execution)."""
        forecaster = NextDoorForecaster(
            training_split=training_split, num_nnls_batches=num_nnls_batches
        )
        forecaster.train_and_validate(
            x_train.copy(),
            y_train.copy(),
            x_val=x_val.copy(),
            y_val=y_val.copy(),
            lambda_reg=lambda_reg,
        )
        predictions = forecaster.predict(x_test)
        return predictions, forecaster.k_neighbours, forecaster.feat_weight

    @classmethod
    def fit_ensemble(
        cls,
        x_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        x_val: NDArray[np.floating],
        y_val: NDArray[np.floating],
        x_test: NDArray[np.floating],
        num_forecasters: int = 100,
        lambda_reg: float = 0.0,
        training_split: float = 0.5,
        num_nnls_batches: int = 0,
        n_jobs: int = -1,
    ) -> EnsemblePrediction:
        """Fit ensemble of forecasters and make predictions.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            num_forecasters: Number of forecasters in ensemble
            lambda_reg: L2 regularization parameter
            training_split: Training split for NNLS
            num_nnls_batches: Number of NNLS batches
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            EnsemblePrediction with aggregated results
        """
        query_start = time.time()

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
            )
            for _ in range(num_forecasters)
        )

        query_elapsed = time.time() - query_start
        print(
            f"...prediction with {num_forecasters} forecasters done in {query_elapsed:.2f} sec!"
        )

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
        x_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        x_val: NDArray[np.floating],
        y_val: NDArray[np.floating],
        num_forecasters: int = 100,
        lambda_reg: float = 0.0,
        training_split: float = 0.5,
        num_nnls_batches: int = 0,
        n_jobs: int = -1,
    ) -> list[NextDoorForecaster]:
        """Train ensemble of forecasters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            num_forecasters: Number of forecasters in ensemble
            lambda_reg: L2 regularization parameter
            training_split: Training split for NNLS
            num_nnls_batches: Number of NNLS batches
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            List of trained forecasters
        """
        query_start = time.time()

        def train_single():
            forecaster = cls(
                training_split=training_split, num_nnls_batches=num_nnls_batches
            )
            forecaster.train_and_validate(
                x_train.copy(),
                y_train.copy(),
                x_val=x_val.copy(),
                y_val=y_val.copy(),
                lambda_reg=lambda_reg,
            )
            return forecaster

        ensemble = Parallel(n_jobs=n_jobs)(
            delayed(train_single)() for _ in range(num_forecasters)
        )

        query_elapsed = time.time() - query_start
        print(
            f"...training {num_forecasters} forecasters done in {query_elapsed:.2f} sec!"
        )

        return ensemble

    @staticmethod
    def predict_with_ensemble(
        ensemble: list[NextDoorForecaster],
        x_test: NDArray[np.floating],
    ) -> EnsemblePrediction:
        """Make predictions using a trained ensemble.

        Args:
            ensemble: List of trained forecasters
            x_test: Test features

        Returns:
            EnsemblePrediction with aggregated results
        """
        all_predictions = np.array([frc.predict(x_test.copy()) for frc in ensemble])

        return EnsemblePrediction(
            predictions=np.mean(all_predictions, axis=0),
            predictions_std=np.std(all_predictions, axis=0),
        )

    @staticmethod
    def normalize_vector(
        vector: NDArray[np.floating], max_val: float
    ) -> NDArray[np.floating]:
        """Normalize vector from 0 to max_val.

        Args:
            vector: Input vector
            max_val: Maximum value for normalization

        Returns:
            Normalized vector
        """
        min_val = vector.min()
        max_val_data = vector.max()
        range_val = max_val_data - min_val

        if range_val == 0.0:
            return np.full_like(vector, max_val)

        return max_val * (vector - min_val) / range_val

    @staticmethod
    def get_basic_stats(vector: NDArray[np.floating]) -> dict[str, float]:
        """Get basic statistics of a vector.

        Args:
            vector: Input vector

        Returns:
            Dictionary with max, min, and mean values
        """
        return {
            "max": float(vector.max()),
            "min": float(vector.min()),
            "mean": float(vector.mean()),
        }
