#!/usr/bin/env python3
"""Example usage of the refactored NextDoorForecaster.

This script demonstrates the main features of the NextDoorForecaster class.
"""

import numpy as np

from nextdoor import NextDoorForecaster

print("=" * 70)
print("NextDoor Forecaster v2.0 - Example Usage")
print("=" * 70)

# Generate synthetic data
print("\n1. Generating synthetic data...")
np.random.seed(42)

n_samples = 1000
n_features = 8

X = np.random.randn(n_samples, n_features)
# Create a simple linear relationship with some noise
y = (
    X[:, 0] * 2.5
    + X[:, 1] * 1.8
    + X[:, 2] * -1.2
    + X[:, 3] * 0.5
    + np.random.randn(n_samples) * 0.3
)

# Split data
train_size = int(n_samples * 0.8)
val_size = int(n_samples * 0.1)
test_size = n_samples - train_size - val_size
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size : train_size + val_size]
y_val = y[train_size : train_size + val_size]
X_test = X[train_size + val_size : train_size + val_size + test_size]
y_test = y[train_size + val_size : train_size + val_size + test_size]

print(f"   Training samples: {len(y_train)}")
print(f"   Validation samples: {len(y_val)}")
print(f"   Test samples: {len(y_test)}")
print(f"   Features: {n_features}")

# Example 1: Single Forecaster
print("\n2. Training single forecaster...")
forecaster = NextDoorForecaster(
    training_split=0.3, k_neighbours=10, max_weight_scale=15.0
)

forecaster.train_and_validate(X_train, y_train, X_val, y_val, lambda_reg=0.1)

print("   ✓ Model trained successfully")
print(f"   ✓ Optimal k neighbours: {forecaster.k_neighbours}")
print(f"   ✓ Feature weights shape: {forecaster.feat_weight.shape}")
print(f"   ✓ Top 3 feature weights: {np.sort(forecaster.feat_weight)[-3:]}")

# Make predictions
print("\n3. Making predictions...")
predictions = forecaster.predict(X_test)
print(f"   ✓ Predictions shape: {predictions.shape}")

# Calculate metrics
metrics = NextDoorForecaster.get_forecast_metrics(y_test, predictions)
print("\n4. Forecast metrics (single forecaster):")
print(f"   MAE:  {metrics.mae:.4f}")
print(f"   MSE:  {metrics.mse:.4f}")
print(f"   RMSE: {np.sqrt(metrics.mse):.4f}")
print(f"   MAPE: {metrics.mape:.2f}%")
print(f"   Mean Error: {metrics.mean_error:.4f}")

# Example 2: Ensemble Forecasting
print("\n5. Training ensemble forecaster (20 models)...")
num_forecasters = 25
ensemble_result = NextDoorForecaster.fit_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    num_forecasters=num_forecasters,
    lambda_reg=0.1,
    training_split=0.3,
    n_jobs=-1,  # Use all CPU cores
)

print(f"   ✓ Ensemble predictions shape: {ensemble_result.predictions.shape}")
print(f"   ✓ Average optimal k: {ensemble_result.num_neighbours:.1f}")

# Calculate ensemble metrics
ensemble_metrics = NextDoorForecaster.get_forecast_metrics(
    y_test, ensemble_result.predictions
)

print("\n6. Forecast metrics (ensemble):")
print(f"   MAE:  {ensemble_metrics.mae:.4f}")
print(f"   MSE:  {ensemble_metrics.mse:.4f}")
print(f"   RMSE: {np.sqrt(ensemble_metrics.mse):.4f}")
print(f"   MAPE: {ensemble_metrics.mape:.2f}%")
print(f"   Mean prediction std: {ensemble_result.predictions_std.mean():.4f}")

# Compare single vs ensemble
print("\n7. Single vs Ensemble comparison:")
improvement_mae = (metrics.mae - ensemble_metrics.mae) / metrics.mae * 100
improvement_rmse = (
    (np.sqrt(metrics.mse) - np.sqrt(ensemble_metrics.mse)) / np.sqrt(metrics.mse) * 100
)

print(f"   MAE improvement:  {improvement_mae:+.2f}%")
print(f"   RMSE improvement: {improvement_rmse:+.2f}%")

# Example 3: Train ensemble for later use
print("\n8. Training ensemble for later use...")
trained_ensemble = NextDoorForecaster.train_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    num_forecasters=num_forecasters,
    lambda_reg=0.1,
    n_jobs=-1,
)

print(f"   ✓ Trained {len(trained_ensemble)} forecasters")

# Use the trained ensemble
ensemble_pred = NextDoorForecaster.predict_with_ensemble(trained_ensemble, X_test)
print(f"   ✓ Predictions from saved ensemble: {ensemble_pred.predictions.shape}")

# Show prediction confidence intervals
print("\n9. Prediction confidence intervals (first 5 test samples):")
print(f"   {'Index':<8} {'Actual':>10} {'Predicted':>10} {'Std':>10} {'95% CI':>20}")
print(f"   {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")

for i in range(min(5, len(y_test))):
    actual = y_test[i]
    pred = ensemble_result.predictions[i]
    std = ensemble_result.predictions_std[i]
    ci_lower = pred - 1.96 * std
    ci_upper = pred + 1.96 * std

    print(
        f"   {i:<8} {actual:>10.4f} {pred:>10.4f} {std:>10.4f} "
        f"[{ci_lower:>6.4f}, {ci_upper:>6.4f}]"
    )

print("\n" + "=" * 70)
