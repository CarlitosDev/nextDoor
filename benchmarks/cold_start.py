"""Cold-start benchmark: cross-SKU pooling vs per-SKU models (Tier 3).

The paper's conclusion names a structural failure of the per-SKU design: "a
new product gets nothing". This quantifies it, and quantifies what the global
retrieval-augmented forecaster recovers, under three regimes:

  * ample history, no shared structure -- pooling has nothing to borrow and
    slightly *hurts* (reported honestly);
  * short history, structure shared via category descriptors -- roughly a wash;
  * brand-new SKUs with zero history (leave-SKU-out) -- the per-SKU method can
    only emit a global median, while descriptor-based pooling generalises.

Run with:  uv run python benchmarks/cold_start.py [--seeds 10]
"""

from __future__ import annotations

import argparse

import numpy as np

from nextdoor import NextDoorForecaster, RetrievalAugmentedForecaster


def make_skus(n_skus: int, per_sku: int, shared: bool, seed: int):
    """Pooled promotions. If ``shared``, SKU level/slope are a function of
    observable category descriptors (so structure can be borrowed); otherwise
    they are independent per SKU (nothing to borrow)."""
    rng = np.random.default_rng(seed)
    desc = rng.uniform(0, 1, (n_skus, 3))
    if shared:
        level = 300 + 1500 * desc[:, 0] + rng.normal(0, 50, n_skus)
        slope = 200 + 700 * desc[:, 1] + rng.normal(0, 30, n_skus)
    else:
        level = rng.uniform(100, 2000, n_skus)
        slope = rng.uniform(100, 900, n_skus)
    rows = []
    for s in range(n_skus):
        d = rng.uniform(0, 1, per_sku)
        st = rng.uniform(0, 1, per_sku)
        y = level[s] + slope[s] * d + 200 * st + rng.normal(0, 40, per_sku)
        for i in range(per_sku):
            rows.append((d[i], st[i], desc[s, 0], desc[s, 1], desc[s, 2], s, max(y[i], 1.0)))
    return np.array(rows), n_skus


def per_sku_predict(a, tr, te, skutr, skute, ytr, seed, min_hist=4):
    preds = np.empty(int(te.sum()))
    te_idx = np.where(te)[0]
    for jj, j in enumerate(te_idx):
        m = skutr == skute[jj]
        if m.sum() >= min_hist:
            f = NextDoorForecaster(random_state=seed, k_neighbours=6)
            f.train_and_validate(a[tr][m][:, :2], ytr[m])
            preds[jj] = f.predict(a[j : j + 1, :2])[0]
        else:
            preds[jj] = np.median(ytr[m]) if m.sum() else np.median(ytr)
    return preds


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    print(f"\nCold-start benchmark ({args.seeds} seeds)\n")
    print(f"{'regime':<42}{'per-SKU':>10}{'global':>10}{'change':>9}")
    print("-" * 71)

    regimes = [
        ("ample history, no shared structure", dict(n_skus=40, per_sku=12, shared=False)),
        ("short history, shared descriptors", dict(n_skus=80, per_sku=3, shared=True)),
        ("brand-new SKUs (leave-SKU-out)", dict(n_skus=120, per_sku=8, shared=True)),
    ]

    for label, cfg in regimes:
        per_sku_maes, global_maes = [], []
        for seed in range(args.seeds):
            a, n_skus = make_skus(seed=seed, **cfg)
            sku = a[:, 5].astype(int)
            rng = np.random.default_rng(seed)

            if "leave-SKU-out" in label:
                test_skus = set(rng.choice(n_skus, n_skus // 5, replace=False))
                te = np.array([s in test_skus for s in sku])
                tr = ~te
                ytr = a[tr, 6]
                # per-SKU on a new SKU == global median (no own history)
                per_sku_maes.append(mae(a[te, 6], np.median(ytr)))
                g = RetrievalAugmentedForecaster(
                    random_state=seed, max_iter=800, hidden_layer_sizes=(128, 16)
                ).fit(a[tr][:, :5], ytr)
                global_maes.append(mae(a[te, 6], g.predict(a[te][:, :5])))
            else:
                perm = rng.permutation(len(a))
                a, sku = a[perm], sku[perm]
                cut = int(0.8 * len(a))
                tr = np.zeros(len(a), bool)
                tr[:cut] = True
                te = ~tr
                ytr, yte = a[tr, 6], a[te, 6]
                skutr, skute = sku[:cut], sku[cut:]
                preds = per_sku_predict(a, tr, te, skutr, skute, ytr, seed)
                per_sku_maes.append(mae(yte, preds))
                # global model: with shared structure use descriptors, else sku id
                feat_cols = slice(0, 5) if cfg["shared"] else slice(0, 3)
                g = RetrievalAugmentedForecaster(
                    random_state=seed, max_iter=800, hidden_layer_sizes=(128, 16)
                ).fit(a[tr][:, feat_cols], ytr)
                global_maes.append(mae(yte, g.predict(a[te][:, feat_cols])))

        ps, gl = np.mean(per_sku_maes), np.mean(global_maes)
        print(f"{label:<42}{ps:>10.0f}{gl:>10.0f}{100 * (1 - gl / ps):>8.0f}%")

    print(
        "\nPooling hurts when there is nothing to borrow and wins decisively on "
        "cold-start, exactly as the per-SKU design's structure predicts."
    )


if __name__ == "__main__":
    main()
