"""nextDoor: Forecasting Promotional Sales Within the Neighbourhood.

A kNN + learned-metric forecasting engine. v3 implements the revisions
described in fable_thought_about_this.md: bug fixes (k-selection criterion,
in-place scaling, chronological splits), statistical fixes (log1p targets,
NNLS intercept, pair weighting, Gaussian kernel with validated bandwidth,
optional whitening and recency decay), probabilistic output (weighted
neighbour quantiles, conformal intervals) and alternative similarity engines
(diagonal MLKR; random-forest leaf proximities; a global cross-SKU
retrieval-augmented forecaster).

Reference:
    https://ieeexplore.ieee.org/document/8727882
"""

from nextdoor.forecaster import (
    EnsemblePrediction,
    ForecastMetrics,
    NeighbourExplanation,
    NextDoorForecaster,
)
from nextdoor.leaf_forecaster import LeafSimilarityForecaster
from nextdoor.mlkr import MLKRResult, fit_diagonal_mlkr
from nextdoor.retrieval import RetrievalAugmentedForecaster

__version__ = "3.0.0"
__all__ = [
    "EnsemblePrediction",
    "ForecastMetrics",
    "LeafSimilarityForecaster",
    "MLKRResult",
    "NeighbourExplanation",
    "NextDoorForecaster",
    "RetrievalAugmentedForecaster",
    "fit_diagonal_mlkr",
]
