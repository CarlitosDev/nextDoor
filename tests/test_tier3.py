"""Tests for the Tier 3 retrieval-augmented forecaster and loop study."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nextdoor import RetrievalAugmentedForecaster


def make_pooled_data(n=400, n_skus=8, seed=0):
    """Cross-SKU promotional data: each SKU has its own level and slope."""
    rng = np.random.default_rng(seed)
    sku = rng.integers(0, n_skus, n)
    sku_level = rng.uniform(100, 2000, n_skus)
    sku_slope = rng.uniform(100, 900, n_skus)
    discount = rng.uniform(0, 1, n)
    stores = rng.uniform(0, 1, n)
    x = np.column_stack([discount, stores, sku.astype(float)])
    y = sku_level[sku] + sku_slope[sku] * discount + 200.0 * stores
    y = y + rng.normal(0, 40.0, n)
    return x, np.maximum(y, 1.0)


def fitted(seed=0, **kwargs):
    x, y = make_pooled_data(seed=seed)
    f = RetrievalAugmentedForecaster(random_state=seed, max_iter=300, **kwargs).fit(x, y)
    return f, x, y


class TestRetrievalForecaster:
    def test_beats_global_mean(self):
        x, y = make_pooled_data(seed=1)
        f = RetrievalAugmentedForecaster(random_state=1, max_iter=400)
        f.fit(x[:320], y[:320])
        preds = f.predict(x[320:])
        mae = np.mean(np.abs(preds - y[320:]))
        baseline = np.mean(np.abs(y[320:] - y[:320].mean()))
        assert mae < baseline

    def test_does_not_mutate_inputs(self):
        x, y = make_pooled_data()
        xc, yc = x.copy(), y.copy()
        RetrievalAugmentedForecaster(random_state=0, max_iter=200).fit(x, y)
        np.testing.assert_array_equal(x, xc)
        np.testing.assert_array_equal(y, yc)

    def test_embedding_shape_matches_last_hidden_layer(self):
        f, x, _ = fitted(hidden_layer_sizes=(32, 8))
        emb = f._embed(x[:5])
        assert emb.shape == (5, 8)

    def test_quantiles_monotone(self):
        f, x, _ = fitted()
        q = f.predict_quantiles(x[:10], [0.1, 0.5, 0.9])
        assert np.all(np.diff(q, axis=1) >= 0)

    def test_conformal_coverage(self):
        x, y = make_pooled_data(n=700, seed=5)
        f = RetrievalAugmentedForecaster(random_state=5, max_iter=400)
        f.fit(x[:560], y[:560])
        lo, hi = f.predict_interval(x[560:], alpha=0.2)
        covered = np.mean((y[560:] >= lo) & (y[560:] <= hi))
        assert covered >= 0.7
        assert np.all(hi >= lo)

    def test_explain_matches_mean_forecast(self):
        f, x, _ = fitted()
        expl = f.explain(x[7])
        assert expl.weights.sum() == pytest.approx(1.0)
        assert expl.prediction == pytest.approx(f.predict_mean(x[[7]])[0], rel=1e-9)
        assert len(expl.indices) == f.k_neighbours

    def test_tabpfn_backend_degrades_gracefully(self):
        # tabpfn is not a dependency; requesting it must warn and fall back.
        x, y = make_pooled_data(n=200, seed=2)
        f = RetrievalAugmentedForecaster(backend="tabpfn", random_state=2, max_iter=200)
        if importlib.util.find_spec("tabpfn") is None:
            with pytest.warns(UserWarning, match="tabpfn"):
                f.fit(x, y)
            assert f.backend == "mlp"
        else:
            f.fit(x, y)
        assert np.all(np.isfinite(f.predict(x[:5])))


class TestForecasterInTheLoop:
    def test_editing_beats_accept_reject_at_high_skill(self):
        """The study's thesis, run on a few seeds: a skilled forecaster editing
        neighbours ends up at least as accurate as one who can only veto."""
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "forecaster_in_the_loop.py"
        spec = importlib.util.spec_from_file_location("fitl", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fitl"] = mod
        spec.loader.exec_module(mod)

        skills = [0.0, 1.0]
        edit, ar = [], []
        for seed in range(4):
            res = mod.run_once(n=500, skills=skills, seed=seed)
            edit.append(res["edit_mae"])
            ar.append(res["ar_mae"])
        edit_mae = np.mean(edit, axis=0)
        ar_mae = np.mean(ar, axis=0)
        # at full skill, editing should not be worse than accept/reject
        assert edit_mae[-1] <= ar_mae[-1] + 1e-6
        # editing at full skill should improve over zero skill
        assert edit_mae[-1] <= edit_mae[0] + 1e-6
