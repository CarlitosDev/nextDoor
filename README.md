# nextDoor

Python implementation of the method "Forecasting Promotional Sales Within the Neighbourhood", publicly available [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8727882)

![Alt text](figs/Forecasting_as_a_service.png?raw=true "Summary of the implementation")


## Installation

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

pre-commit install

# Clone and install
git clone https://github.com/CarlitosDev/nextDoor
cd nextDoor
uv sync
```

### Using pip

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from nextdoor import NextDoorForecaster

# Prepare your data
X_train, y_train = ...  # Training features and targets
X_val, y_val = ...      # Validation features and targets
X_test = ...            # Test features

# Single forecaster
forecaster = NextDoorForecaster(training_split=0.25, k_neighbours=15)
forecaster.train_and_validate(X_train, y_train, X_val, y_val, lambda_reg=0.1)
predictions = forecaster.predict(X_test)

# Calculate metrics
metrics = NextDoorForecaster.get_forecast_metrics(y_test, predictions)
print(f"MAE: {metrics.mae:.4f}, MAPE: {metrics.mape:.2f}%")

# Ensemble forecasting (recommended for better accuracy)
result = NextDoorForecaster.fit_ensemble(
    X_train, y_train, X_val, y_val, X_test,
    num_forecasters=100,
    lambda_reg=0.1,
    n_jobs=-1  # Use all CPU cores
)

print(f"Predictions: {result.predictions}")
print(f"Std dev: {result.predictions_std}")
print(f"Optimal k: {result.num_neighbours}")
```

## Features

### NextDoorForecaster Class

The main forecasting engine with k-Nearest Neighbours and feature learning.

**Key Methods:**
- `train_nn()`: Train the nearest neighbour model
- `cv_neighbours()`: Cross-validate optimal number of neighbours
- `predict()`: Make predictions
- `train_and_validate()`: Combined training and validation
- `fit_ensemble()`: Fit ensemble of forecasters (static method)
- `train_ensemble()`: Train ensemble for later use (static method)
- `predict_with_ensemble()`: Predict using trained ensemble (static method)

**Parameters:**
- `training_split`: Fraction of training data for NNLS optimization (default: 0.25)
- `k_neighbours`: Maximum number of neighbours (default: 15)
- `max_weight_scale`: Scale factor for infinite weights (default: 15.0)
- `num_nnls_batches`: Number of batches for NNLS (default: 0, no batching)

### Return Types

The new version uses dataclasses for structured returns:

```python
# ForecastMetrics
metrics = NextDoorForecaster.get_forecast_metrics(y_true, y_pred)
metrics.mae        # Mean Absolute Error
metrics.mse        # Mean Squared Error
metrics.mape       # Mean Absolute Percentage Error
metrics.mean_error # Mean forecast error
metrics.residuals  # Forecast residuals

# EnsemblePrediction
result = NextDoorForecaster.fit_ensemble(...)
result.predictions      # Mean predictions
result.predictions_std  # Standard deviation of predictions
result.predictions_min  # Minimum predictions
result.predictions_max  # Maximum predictions
result.num_neighbours   # Average optimal k
result.features        # Average feature weights
```

## Examples

See the Jupyter notebooks for detailed examples:
* **Variable selection**: `example1_variable selection.ipynb`
* **Multicollinearity**: `example2_CollinearityAndEndogeneity.ipynb`


## Development

```bash
# Install development dependencies
uv sync

# Run tests
uv run python example_usage.py

# Format code (if you add ruff/black)
uv run ruff format src/

# Type checking (if you add mypy)
uv run mypy src/
```

## Project Structure

```
nextDoor/
├── src/
│   └── nextdoor/
│       ├── __init__.py
│       └── forecaster.py       # Main NextDoorForecaster class
├── deprecated/
│   ├── nextDoorForecaster.py   # Old version (v1)
│   └── nextDoorForecasterV2.py # Old version (v2)
├── data/                        # Example data
├── figs/                        # Figures
├── example1_variable selection.ipynb
├── example2_CollinearityAndEndogeneity.ipynb
├── pyproject.toml              # Modern Python project config
├── MIGRATION_GUIDE.md          # v1 to v2 migration guide
└── README.md                   # This file
```

## Algorithm Details

The method implements a k-Nearest Neighbours approach with learned feature weights using Non-Negative Least Squares (NNLS). Key aspects:

1. **Feature Weight Learning**: NNLS optimization to learn importance of each feature
2. **Distance-based Weighting**: Closer neighbours have higher influence
3. **Automatic k Selection**: Cross-validation to find optimal number of neighbours
4. **Ensemble Support**: Multiple forecasters for robust predictions
5. **L2 Regularization**: Optional regularization to prevent overfitting

## Citation

If you use this code in your research, please cite:

```bibtex
@article{aguilar2019forecasting,
  title={Forecasting Promotional Sales Within the Neighbourhood},
  author={Aguilar, Carlos and others},
  journal={IEEE},
  year={2019},
  url={https://ieeexplore.ieee.org/document/8727882}
}
```

## Contact

Any questions/remarks, feel free to drop me a line: carlos.aguilar.palacios@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.