"""Ablation study on the paper's own surrogate data generator.

Recreates the collinearity/endogeneity experiment of Section IV-A of the paper
(ready-to-eat vegetables box: avalanche-mapped discount, exponential baseline,
Rayleigh confounder, collinear copies) and runs the ablation the paper was
missing: every method variant on every scenario, so each fix's contribution is
measurable -- including the legacy k-selection bug, kept available as
``k_selection="bias"``.

Run with:  uv run python benchmarks/ablation.py [--seeds 5] [--n 500]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from nextdoor import LeafSimilarityForecaster, NextDoorForecaster

# --------------------------------------------------------------------- #
# the paper's data generator (Section IV-A)
# --------------------------------------------------------------------- #


def avalanche(x: np.ndarray) -> np.ndarray:
    """Eq. (20): large advertised discounts trigger avalanche sales."""
    out = x.copy()
    out = np.where((x >= 0.5) & (x < 0.8), 1000.0 + x, out)
    out = np.where((x >= 0.8) & (x < 0.9), 2500.0 + x, out)
    out = np.where(x >= 0.9, 3800.0 + x, out)
    return out


def generate(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    x1 = rng.uniform(0, 1, n)  # normalised discount
    x2 = rng.exponential(1.0, n)  # baseline sales driver
    z = rng.rayleigh(math.sqrt(2.0 / math.pi), n)  # confounder (store affluence)
    u1 = rng.uniform(0, 1, n)
    x3 = u1 + 0.15 * z  # noisy echo of the confounder
    x4 = 0.5 + 2.0 * x2  # collinear copy of x2
    c1 = rng.binomial(1, 0.5, n).astype(float)  # featured display
    c2 = (x2 > 0.85).astype(float)  # near-out-of-stock flag

    noise = rng.normal(46.5, math.sqrt(1.6e6), n)
    y = avalanche(x1) + 3100.0 * x2 + 10.0 * c1 + 200.0 * z + noise
    y = np.maximum(y, 1.0)  # the paper's datasets contain no non-positive sales
    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "z": z, "c1": c1, "c2": c2, "y": y}


SCENARIOS = {
    # the paper's three blocks (Table 1)
    "ideal": ["x1", "x2", "c1", "z"],
    "collinearity": ["x1", "x2", "x3", "x4", "z", "c1", "c2"],
    "endogeneity": ["x1", "x2", "x3", "c1"],
}


# --------------------------------------------------------------------- #
# metrics (paper's business metrics + MAE)
# --------------------------------------------------------------------- #


@dataclass
class Scores:
    mae: float
    mape: float
    w20p: float  # % of sales volume forecast within 20% relative error
    out50p: float  # % of sales volume forecast off by more than 50%


def score(y: np.ndarray, y_hat: np.ndarray) -> Scores:
    rel = np.abs(y - y_hat) / np.abs(y)
    vol = y.sum()
    return Scores(
        mae=float(np.mean(np.abs(y - y_hat))),
        mape=float(100 * rel.mean()),
        w20p=float(100 * y[rel <= 0.20].sum() / vol),
        out50p=float(100 * y[rel > 0.50].sum() / vol),
    )


# --------------------------------------------------------------------- #
# method variants
# --------------------------------------------------------------------- #

LEGACY = dict(
    kernel="inverse",
    target_transform=None,
    scaler="minmax",
    fit_intercept=False,
    pair_weighting=False,
    metric_learner="nnls",
    k_selection="bias",  # the original implementation's criterion
)

VARIANTS: dict[str, dict | str] = {
    "legacy (2019 paper)": LEGACY,
    "legacy + k-fix": {**LEGACY, "k_selection": "mae"},
    "uniform kNN (ablation)": {"metric_learner": "uniform"},
    "improved NNLS": {},  # defaults: auto transform, intercept, gaussian kernel, robust scaler
    "improved + whiten": {"whiten": True},
    "MLKR": {"metric_learner": "mlkr"},
    "leaf similarity (RF)": "leaf",
    "least squares": "ls",
}


def run_variant(
    name: str,
    spec: dict | str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[Scores, dict | None]:
    n = len(y)
    i_tr, i_va = int(0.6 * n), int(0.8 * n)
    x_tr, x_va, x_te = x[:i_tr], x[i_tr:i_va], x[i_va:]
    y_tr, y_va, y_te = y[:i_tr], y[i_tr:i_va], y[i_va:]

    if spec == "ls":
        model = LinearRegression().fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))
        return score(y_te, model.predict(x_te)), None
    if spec == "leaf":
        model = LeafSimilarityForecaster(random_state=seed).fit(
            np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va])
        )
        return score(y_te, model.predict(x_te)), None

    f = NextDoorForecaster(random_state=seed, **spec)
    f.train_and_validate(x_tr, y_tr, x_val=x_va, y_val=y_va, lambda_reg=0.1)
    scores = score(y_te, f.predict(x_te))

    interval = None
    if name == "improved NNLS":
        lo, hi = f.predict_interval(x_te, alpha=0.2, method="cqr")
        interval = {
            "coverage": float(np.mean((y_te >= lo) & (y_te <= hi))),
            "median_width": float(np.median(hi - lo)),
        }
    return scores, interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    for scenario, cols in SCENARIOS.items():
        print(f"\n=== Scenario: {scenario} (features: {', '.join(cols)}) ===")
        rows: dict[str, list[Scores]] = {name: [] for name in VARIANTS}
        intervals: list[dict] = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            data = generate(args.n, rng)
            x = np.column_stack([data[c] for c in cols])
            y = data["y"]
            for name, spec in VARIANTS.items():
                s, interval = run_variant(name, spec, x, y, seed)
                rows[name].append(s)
                if interval:
                    intervals.append(interval)

        header = f"{'method':<26}{'MAE':>9}{'MAPE%':>9}{'w20p%':>9}{'out50p%':>9}"
        print(header)
        print("-" * len(header))
        for name, ss in rows.items():
            mae = np.mean([s.mae for s in ss])
            mape = np.mean([s.mape for s in ss])
            w20 = np.mean([s.w20p for s in ss])
            o50 = np.mean([s.out50p for s in ss])
            print(f"{name:<26}{mae:>9.0f}{mape:>9.1f}{w20:>9.1f}{o50:>9.1f}")
        if intervals:
            cov = np.mean([i["coverage"] for i in intervals])
            width = np.mean([i["median_width"] for i in intervals])
            print(
                f"\n  CQR 80% intervals (improved NNLS): "
                f"empirical coverage {cov:.2f}, median width {width:.0f} units"
            )


if __name__ == "__main__":
    main()
