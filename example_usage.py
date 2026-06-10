#!/usr/bin/env python3
"""Example usage of the NextDoor forecasting engine (v3).

A guided tour of the package, from the core forecaster to the v3 additions:
probabilistic output, editable neighbour explanations, the alternative
similarity engines (MLKR, random-forest proximities), recency decay, and the
global cross-SKU retrieval forecaster for cold-start products.

Run with:  uv run python example_usage.py
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler

from nextdoor import (
    LeafSimilarityForecaster,
    NextDoorForecaster,
    RetrievalAugmentedForecaster,
    fit_diagonal_mlkr,
)


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------- #
# Synthetic promotional data
# --------------------------------------------------------------------- #
# Sales are driven by the discount (x0) and a baseline-demand feature (x1)
# only. x2 is irrelevant noise; x3 is a near-collinear copy of x1; x4 is a
# noisy proxy. A good metric should concentrate importance on x0 and x1.

header("NextDoor Forecaster v3 - Example Usage")

rng = np.random.default_rng(42)
n_samples, n_features = 1200, 5

x0 = rng.uniform(0, 1, n_samples)  # discount
x1 = rng.uniform(0, 1, n_samples)  # baseline demand
x2 = rng.uniform(0, 1, n_samples)  # irrelevant
x3 = x1 + 0.02 * rng.normal(size=n_samples)  # collinear with x1
x4 = 0.5 * x1 + 0.5 * rng.uniform(0, 1, n_samples)  # noisy proxy of x1
X = np.column_stack([x0, x1, x2, x3, x4])

sales = 500 + 3000 * x0 + 1500 * x1 + rng.normal(0, 120, n_samples)
y = np.maximum(sales, 1.0)

# chronological split (train -> validation -> test), the paper's protocol
i_tr, i_va = int(0.7 * n_samples), int(0.85 * n_samples)
X_train, y_train = X[:i_tr], y[:i_tr]
X_val, y_val = X[i_tr:i_va], y[i_tr:i_va]
X_test, y_test = X[i_va:], y[i_va:]

print("\n1. Synthetic promotional data")
print(f"   train / val / test: {len(y_train)} / {len(y_val)} / {len(y_test)}")
print(f"   features: {n_features} (x0=discount, x1=demand; x2 noise, x3 collinear, x4 proxy)")
print(f"   sales range: [{y.min():.0f}, {y.max():.0f}], mean {y.mean():.0f}")


# --------------------------------------------------------------------- #
# 2. Single forecaster with the v3 defaults
# --------------------------------------------------------------------- #
header("2. Single forecaster (v3 defaults)")

forecaster = NextDoorForecaster(k_neighbours=15, random_state=0)
forecaster.train_and_validate(X_train, y_train, X_val, y_val, lambda_reg=0.1)
preds = forecaster.predict(X_test)
metrics = NextDoorForecaster.get_forecast_metrics(y_test, preds)

print(
    f"   chosen target_transform : {forecaster.target_transform!r}  (auto-selected on validation)"
)
print(f"   chosen k neighbours      : {forecaster.k_neighbours}")
print(f"   Gaussian kernel bandwidth: {forecaster.bandwidth_:.4f}")
print(f"   learned noise-floor intercept  : {forecaster.noise_floor_:.2f}")
imp = forecaster.feature_importances_
print("   feature importances      :")
for name, w in zip(["x0", "x1", "x2", "x3", "x4"], imp, strict=True):
    bar = "#" * int(round(40 * w / imp.max()))
    print(f"       {name}: {w:5.3f} {bar}")
print("   -> x0 dominates and x2/x4 are correctly ~0; note the demand signal")
print("      smears between x1 and its collinear copy x3 (a known NNLS effect;")
print("      MLKR in section 6 disentangles it and recovers x1 directly).")
print(f"\n   MAE {metrics.mae:.1f} | RMSE {np.sqrt(metrics.mse):.1f} | MAPE {metrics.mape:.2f}%")


# --------------------------------------------------------------------- #
# 3. Probabilistic forecasting: quantiles + conformal intervals
# --------------------------------------------------------------------- #
header("3. Probabilistic output (newsvendor quantiles + conformal intervals)")

# Weighted quantiles of the neighbour outcomes = a predictive distribution.
# A 90% service level means ordering the 0.90 quantile of demand.
q = forecaster.predict_quantiles(X_test, [0.1, 0.5, 0.9])
print("   Predictive quantiles for the first 4 test promotions:")
print(f"   {'actual':>10}{'q10':>10}{'q50':>10}{'q90 (order)':>14}")
for i in range(4):
    print(f"   {y_test[i]:>10.0f}{q[i, 0]:>10.0f}{q[i, 1]:>10.0f}{q[i, 2]:>14.0f}")

# Split-conformal (CQR) intervals calibrated on the validation set.
lo, hi = forecaster.predict_interval(X_test, alpha=0.1, method="cqr")
coverage = np.mean((y_test >= lo) & (y_test <= hi))
print(f"\n   90% CQR intervals: empirical coverage on test = {coverage:.1%}")
print(f"   median interval width = {np.median(hi - lo):.0f} units")


# --------------------------------------------------------------------- #
# 4. Interpretability: the editable neighbour explanation
# --------------------------------------------------------------------- #
header("4. Explainable forecast: the neighbours behind one prediction")

expl = forecaster.explain(X_test[0])
print(f"   Forecast for test promotion 0: {expl.prediction:.0f} units")
print(f"   (actual: {y_test[0]:.0f})")
print(f"   Built from {len(expl.indices)} historical promotions:\n")
print(f"   {'train_idx':>10}{'weight':>10}{'past_sales':>12}{'distance':>10}")
for r in expl.as_rows()[:6]:
    print(
        f"   {r['train_index']:>10}{r['weight']:>10.3f}{r['target']:>12.0f}{r['distance']:>10.3f}"
    )
print("   ... a forecaster can drop or re-weight any of these rows.")


# --------------------------------------------------------------------- #
# 5. Alternative similarity engines (and the legacy bug, for contrast)
# --------------------------------------------------------------------- #
header("5. Similarity engines: NNLS vs MLKR vs uniform (and the k-selection bug)")


def mae_for(**kwargs) -> float:
    f = NextDoorForecaster(k_neighbours=15, random_state=0, **kwargs)
    f.train_and_validate(X_train, y_train, X_val, y_val, lambda_reg=0.1)
    return NextDoorForecaster.get_forecast_metrics(y_test, f.predict(X_test)).mae


print(f"   {'engine / setting':<38}{'test MAE':>10}")
print("   " + "-" * 48)
print(f"   {'NNLS metric (default)':<38}{mae_for():>10.1f}")
print(f"   {'diagonal MLKR (gradient-based)':<38}{mae_for(metric_learner='mlkr'):>10.1f}")
print(f"   {'uniform weights (no metric learning)':<38}{mae_for(metric_learner='uniform'):>10.1f}")
print(f"   {'legacy k-selection bug (k_selection=bias)':<38}{mae_for(k_selection='bias'):>10.1f}")
print("   -> learned metrics beat uniform kNN; the bias criterion is worse.")


# --------------------------------------------------------------------- #
# 6. MLKR engine internals (optional low-level API)
# --------------------------------------------------------------------- #
header("6. Low-level: fitting a diagonal metric directly with MLKR")

# Operates on scaled features and (here) log1p targets; v = a^2 >= 0.
Xs = RobustScaler().fit_transform(X_train)
res = fit_diagonal_mlkr(Xs, np.log1p(y_train), rng=np.random.default_rng(0))
print(
    f"   converged: loss {res.loss_history[0]:.3f} -> {res.final_loss:.3f}"
    f" in {len(res.loss_history)} steps"
)
v = res.weights / res.weights.sum()
print(f"   normalised diagonal metric: {np.round(v, 3)}")
print("   -> again concentrates on x0/x1 without asserting the biased identity.")


# --------------------------------------------------------------------- #
# 7. Recency decay under concept drift
# --------------------------------------------------------------------- #
header("7. Recency decay: tracking a drifting sales level")

# Features carry no signal; the level steps up halfway through the history.
n = 300
Xd = rng.uniform(0, 1, (n, 3))
yd = np.concatenate([np.full(n // 2, 800.0), np.full(n // 2, 1600.0)])
yd = yd + rng.normal(0, 20, n)
Xnew = rng.uniform(0, 1, (40, 3))

flat = NextDoorForecaster(random_state=0)
flat.train_and_validate(Xd, yd)
decayed = NextDoorForecaster(random_state=0, time_decay_half_life=15.0)
decayed.train_and_validate(Xd, yd)
print("   recent regime level ~ 1600")
print(f"   forecast without decay : {flat.predict(Xnew).mean():.0f}  (blends old + new)")
print(f"   forecast with decay    : {decayed.predict(Xnew).mean():.0f}  (tracks the recent level)")


# --------------------------------------------------------------------- #
# 8. Ensemble forecasting
# --------------------------------------------------------------------- #
header("8. Ensemble forecasting")

ens = NextDoorForecaster.fit_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    num_forecasters=25,
    lambda_reg=0.1,
    n_jobs=-1,
    random_state=0,
)
ens_metrics = NextDoorForecaster.get_forecast_metrics(y_test, ens.predictions)
print(f"   ensemble MAE {ens_metrics.mae:.1f} (single {metrics.mae:.1f})")
print(f"   average optimal k: {ens.num_neighbours:.1f}")
print(f"   mean procedure-variance std: {ens.predictions_std.mean():.1f}")
print("   note: ensemble std measures procedure variance, not predictive")
print("   uncertainty -- use predict_interval for calibrated intervals.")


# --------------------------------------------------------------------- #
# 9. Random-forest leaf-proximity similarity
# --------------------------------------------------------------------- #
header("9. LeafSimilarityForecaster (random-forest proximity)")

leaf = LeafSimilarityForecaster(n_estimators=200, k_neighbours=15, random_state=0)
leaf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
leaf_preds = leaf.predict(X_test)
leaf_mae = NextDoorForecaster.get_forecast_metrics(y_test, leaf_preds).mae
leaf_q = leaf.predict_quantiles(X_test[:1], [0.1, 0.5, 0.9])[0]
print(f"   test MAE: {leaf_mae:.1f}")
print(
    f"   same neighbour interface -- quantiles for test promo 0: "
    f"q10={leaf_q[0]:.0f}, q50={leaf_q[1]:.0f}, q90={leaf_q[2]:.0f}"
)
print(f"   explain() works identically: {len(leaf.explain(X_test[0]).indices)} neighbours returned")


# --------------------------------------------------------------------- #
# 10. Global retrieval forecaster: the cold-start case
# --------------------------------------------------------------------- #
header("10. RetrievalAugmentedForecaster (cross-SKU, for cold-start products)")

# A pooled multi-SKU catalogue. SKU level/slope are functions of observable
# category descriptors, so a brand-new SKU (no own history) can still be
# forecast by borrowing from similar SKUs.
n_skus, per_sku = 80, 8
desc = rng.uniform(0, 1, (n_skus, 3))
level = 300 + 1500 * desc[:, 0] + rng.normal(0, 50, n_skus)
slope = 200 + 700 * desc[:, 1] + rng.normal(0, 30, n_skus)
rows = []
for s in range(n_skus):
    d = rng.uniform(0, 1, per_sku)
    st = rng.uniform(0, 1, per_sku)
    sv = level[s] + slope[s] * d + 200 * st + rng.normal(0, 40, per_sku)
    for i in range(per_sku):
        rows.append((d[i], st[i], *desc[s], s, max(sv[i], 1.0)))
data = np.array(rows)
sku = data[:, 5].astype(int)

# hold out 20% of SKUs entirely (brand-new products)
new_skus = set(rng.choice(n_skus, n_skus // 5, replace=False))
is_new = np.array([s in new_skus for s in sku])
Xtr_g, ytr_g = data[~is_new][:, :5], data[~is_new][:, 6]  # promo feats + descriptors
Xte_g, yte_g = data[is_new][:, :5], data[is_new][:, 6]

retr = RetrievalAugmentedForecaster(random_state=0, max_iter=500).fit(Xtr_g, ytr_g)
retr_preds = retr.predict(Xte_g)
retr_mae = np.mean(np.abs(retr_preds - yte_g))
naive_mae = np.mean(np.abs(np.median(ytr_g) - yte_g))  # what per-SKU can do: nothing

lo_g, hi_g = retr.predict_interval(Xte_g, alpha=0.2)
cov_g = np.mean((yte_g >= lo_g) & (yte_g <= hi_g))

print(f"   {len(new_skus)} brand-new SKUs with NO own history in the test set")
print(f"   per-SKU method (forced to global median): MAE {naive_mae:.0f}")
print(
    f"   global retrieval via descriptors        : MAE {retr_mae:.0f}"
    f"  ({100 * (1 - retr_mae / naive_mae):.0f}% better)"
)
print(f"   80% conformal coverage on new SKUs: {cov_g:.0%}")

expl_g = retr.explain(Xte_g[0])
print(
    f"   still explainable: forecast {expl_g.prediction:.0f} from "
    f"{len(expl_g.indices)} retrieved promotions of similar SKUs"
)

print("\n" + "=" * 72)
print("Done. See fable_thought_about_this.md for the full rationale and benchmarks.")
print("=" * 72)
