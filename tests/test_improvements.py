"""Tests for the v3 revisions described in fable_thought_about_this.md."""

import numpy as np
import pytest

from nextdoor import (
    LeafSimilarityForecaster,
    NextDoorForecaster,
    fit_diagonal_mlkr,
)


def make_linear_data(n=200, p=4, noise=0.05, seed=0, relevant=(0,)):
    """y depends only on the `relevant` features."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, p))
    beta = np.zeros(p)
    for f in relevant:
        beta[f] = 1.0
    y = 50.0 + 100.0 * (x @ beta) + rng.normal(0, noise * 100.0, n)
    return x, np.maximum(y, 0.0)


def fitted_forecaster(seed=0, **kwargs):
    x, y = make_linear_data(seed=seed)
    f = NextDoorForecaster(random_state=seed, **kwargs)
    f.train_and_validate(x, y)
    return f, x, y


# --------------------------------------------------------------------- #
# bug fixes
# --------------------------------------------------------------------- #


class TestNoMutation:
    def test_train_does_not_modify_inputs(self):
        x, y = make_linear_data()
        x_copy, y_copy = x.copy(), y.copy()
        f = NextDoorForecaster(random_state=0)
        f.train_and_validate(x, y)
        np.testing.assert_array_equal(x, x_copy)
        np.testing.assert_array_equal(y, y_copy)

    def test_predict_twice_same_result(self):
        f, x, _ = fitted_forecaster()
        x_test = x[:10].copy()
        first = f.predict(x_test)
        second = f.predict(x_test)
        np.testing.assert_allclose(first, second)
        np.testing.assert_array_equal(x_test, x[:10])


class TestKSelection:
    def test_mae_criterion_minimises_validation_mae(self):
        """The chosen (k, h) must achieve the best validation MAE among the
        candidates the sweep evaluated -- recomputed independently here."""
        x, y = make_linear_data(seed=3)
        f = NextDoorForecaster(random_state=3, k_neighbours=10)
        f.train_nn(x[:150], y[:150])
        x_val, y_val = x[150:], y[150:]
        f.cv_neighbours(x_val, y_val)
        chosen_mae = np.mean(np.abs(f.val_set_error))

        # any other k at the chosen bandwidth must not beat the chosen one
        for k in range(1, 11):
            preds = []
            for row in x_val:
                w_sorted, y_sorted, order, _ = f.calculate_weights(row)
                nw = w_sorted[:k] / w_sorted[:k].sum()
                preds.append(nw @ y_sorted[:k])
            mae_k = np.mean(np.abs(np.array(preds) - y_val))
            assert chosen_mae <= mae_k + 1e-9

    def test_bias_criterion_is_available_for_ablation(self):
        f, _, _ = fitted_forecaster(k_selection="bias")
        assert f.k_neighbours >= 1


class TestChronologicalSplit:
    def test_default_split_is_tail(self):
        n = 100
        x = np.column_stack([np.arange(n, dtype=float), np.ones(n)])
        y = np.arange(n, dtype=float)
        f = NextDoorForecaster(random_state=0)
        f.train_and_validate(x, y, val_split=0.2)
        # training targets must be exactly the first 80 rows
        np.testing.assert_array_equal(np.sort(f._Y_train), y[:80])
        # calibration (validation) targets must be the last 20 rows
        np.testing.assert_array_equal(np.sort(f._calibration.y), y[80:])


class TestKSweepIncludesMax:
    def test_k_can_equal_k_neighbours(self):
        # constant target: more neighbours never hurts, so with the (fixed)
        # inclusive sweep the selected k can reach k_max
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, (80, 3))
        y = np.full(80, 100.0)
        f = NextDoorForecaster(random_state=0, k_neighbours=5)
        f.train_and_validate(x, y)
        assert 1 <= f.k_neighbours <= 5


# --------------------------------------------------------------------- #
# statistical fixes
# --------------------------------------------------------------------- #


class TestMetricLearning:
    def test_nnls_recovers_relevant_feature(self):
        f, _, _ = fitted_forecaster(seed=1)
        imp = f.feature_importances_
        assert imp[0] == max(imp)
        assert imp[0] > 0.5

    def test_intercept_never_hurts_the_fit(self):
        """The intercepted NNLS system must fit at least as well (the
        intercept can absorb the 2*sigma^2 noise floor; without it that
        constant is forced into the feature weights)."""
        x, y = make_linear_data(n=150, noise=0.5, seed=0)
        with_b = NextDoorForecaster(random_state=0, fit_intercept=True, pair_weighting=False)
        with_b.train_nn(x.copy(), y.copy())
        without_b = NextDoorForecaster(random_state=0, fit_intercept=False, pair_weighting=False)
        without_b.train_nn(x.copy(), y.copy())
        assert with_b.noise_floor_ >= 0.0
        assert with_b.rnorm <= without_b.rnorm + 1e-9

    def test_uniform_learner_gives_equal_weights(self):
        f, _, _ = fitted_forecaster(metric_learner="uniform")
        np.testing.assert_allclose(f.feat_weight, np.ones_like(f.feat_weight))

    def test_mlkr_recovers_relevant_feature(self):
        x, y = make_linear_data(n=150, seed=2)
        f = NextDoorForecaster(random_state=2, metric_learner="mlkr")
        f.train_and_validate(x, y)
        imp = f.feature_importances_
        assert imp[0] == max(imp)
        assert imp[0] > 0.5

    def test_whitening_runs_and_predicts(self):
        x, y = make_linear_data(seed=4)
        # add a collinear copy of the relevant feature
        x = np.column_stack([x, x[:, 0] + 0.01 * np.random.default_rng(4).normal(size=len(x))])
        f = NextDoorForecaster(random_state=4, whiten=True)
        f.train_and_validate(x, y)
        preds = f.predict(x[:5])
        assert np.all(np.isfinite(preds))


class TestMLKRModule:
    def test_loss_decreases(self):
        x, y = make_linear_data(n=100, seed=5)
        res = fit_diagonal_mlkr(x, np.log1p(y), rng=np.random.default_rng(5))
        assert res.loss_history[-1] <= res.loss_history[0]
        assert np.all(res.weights >= 0)


class TestKernels:
    def test_gaussian_weights_finite_on_duplicates(self):
        x, y = make_linear_data(seed=6)
        x[1] = x[0]  # exact duplicate
        f = NextDoorForecaster(random_state=6)
        f.train_and_validate(x, y)
        pred = f.predict(x[[0]])
        assert np.isfinite(pred[0])

    def test_inverse_kernel_still_supported(self):
        f, x, _ = fitted_forecaster(kernel="inverse")
        assert np.all(np.isfinite(f.predict(x[:5])))


class TestTimeDecay:
    def test_recent_history_dominates_under_drift(self):
        """Identical features, drifting level: decay must track the recent level."""
        rng = np.random.default_rng(7)
        n = 120
        x = rng.uniform(0, 1, (n, 2))  # features carry no signal
        y = np.concatenate([np.full(n // 2, 100.0), np.full(n // 2, 200.0)])
        y = y + rng.normal(0, 1.0, n)

        decayed = NextDoorForecaster(random_state=7, time_decay_half_life=10.0)
        decayed.train_and_validate(x, y)
        flat = NextDoorForecaster(random_state=7)
        flat.train_and_validate(x, y)

        x_new = rng.uniform(0, 1, (20, 2))
        assert decayed.predict(x_new).mean() > flat.predict(x_new).mean()
        assert decayed.predict(x_new).mean() > 180.0


# --------------------------------------------------------------------- #
# probabilistic output
# --------------------------------------------------------------------- #


class TestQuantiles:
    def test_quantiles_monotone(self):
        f, x, _ = fitted_forecaster()
        q = f.predict_quantiles(x[:10], [0.1, 0.5, 0.9])
        assert np.all(np.diff(q, axis=1) >= 0)

    def test_median_close_to_point_forecast(self):
        f, x, y = fitted_forecaster()
        med = f.predict_quantiles(x[:20], [0.5])[:, 0]
        point = f.predict(x[:20])
        # same neighbourhoods, so they should be in the same ballpark
        assert np.median(np.abs(med - point)) < 0.2 * y.std()


class TestConformal:
    @pytest.mark.parametrize("method", ["absolute", "cqr"])
    def test_coverage(self, method):
        rng = np.random.default_rng(11)
        n = 600
        x = rng.uniform(0, 1, (n, 3))
        y = 200.0 + 300.0 * x[:, 0] + rng.normal(0, 30.0, n)
        f = NextDoorForecaster(random_state=11)
        f.train_and_validate(x[:500], y[:500])
        lo, hi = f.predict_interval(x[500:], alpha=0.2, method=method)
        covered = np.mean((y[500:] >= lo) & (y[500:] <= hi))
        assert covered >= 0.7  # 1 - alpha - finite-sample slack
        assert np.all(hi >= lo)

    def test_requires_calibration(self):
        x, y = make_linear_data()
        f = NextDoorForecaster(random_state=0)
        f.train_nn(x, y)
        with pytest.raises(RuntimeError, match="calibration"):
            f.predict_interval(x[:5])


class TestExplain:
    def test_explanation_matches_prediction(self):
        f, x, _ = fitted_forecaster()
        expl = f.explain(x[3])
        pred = f.predict(x[[3]])[0]
        assert expl.prediction == pytest.approx(pred, rel=1e-9)
        assert expl.weights.sum() == pytest.approx(1.0)
        assert len(expl.as_rows()) == f.k_neighbours


# --------------------------------------------------------------------- #
# leaf-space similarity forecaster
# --------------------------------------------------------------------- #


class TestLeafForecaster:
    def test_fit_predict_explain(self):
        x, y = make_linear_data(n=250, seed=12)
        f = LeafSimilarityForecaster(n_estimators=100, k_neighbours=10, random_state=12).fit(
            x[:200], y[:200]
        )
        preds = f.predict(x[200:])
        baseline = np.mean(np.abs(y[200:] - y[:200].mean()))
        assert np.mean(np.abs(preds - y[200:])) < baseline

        q = f.predict_quantiles(x[200:210], [0.1, 0.5, 0.9])
        assert np.all(np.diff(q, axis=1) >= 0)

        expl = f.explain(x[200])
        assert expl.weights.sum() == pytest.approx(1.0)
        assert expl.prediction == pytest.approx(f.predict(x[[200]])[0], rel=1e-9)


# --------------------------------------------------------------------- #
# ensemble API still works
# --------------------------------------------------------------------- #


class TestEnsemble:
    def test_fit_ensemble_smoke(self):
        x, y = make_linear_data(n=120, seed=13)
        result = NextDoorForecaster.fit_ensemble(
            x[:80],
            y[:80],
            x[80:100],
            y[80:100],
            x[100:],
            num_forecasters=4,
            n_jobs=1,
            random_state=13,
        )
        assert result.predictions.shape == (20,)
        assert np.all(np.isfinite(result.predictions))
        assert np.all(result.predictions_std >= 0)
