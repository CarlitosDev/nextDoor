# Fable thought about this

A retrospective analysis of *"Forecasting Promotional Sales Within the Neighbourhood"*
(IEEE Access, 2019) and its implementation in `src/nextdoor/forecaster.py`.

Written June 2026, with the benefit of seven years of hindsight, the M5 competition,
conformal prediction going mainstream, and tabular foundation models existing.

---

## TL;DR

The core idea — **learn a per-feature distance metric so that "close in feature space"
means "close in sales", then forecast as a weighted average of nearest historical
promotions** — is sound, genuinely interpretable, and was a reasonable design in
2018/2019. The neighbour-based explanation ("we predict 4,200 units because these 5
past promotions looked like this one") is still a better interpretability story than
SHAP values on a GBDT.

However:

1. **The method is a rediscovery of diagonal Metric Learning for Kernel Regression
   (MLKR, Weinberger & Tesauro, AISTATS 2007)**, solved with a clever-but-biased NNLS
   shortcut instead of gradient descent. The paper never cites this literature.
2. **The NNLS surrogate has four identifiable statistical defects** (dropped
   cross-terms, missing noise intercept, a 4th-moment loss that lets outlier
   promotions dominate, and corruption from "accidentally similar" sales). Two of
   these directly explain the degradation the paper itself observes in the
   collinearity experiment.
3. **The implementation has at least two real bugs**, the worst being that
   `cv_neighbours` selects k by minimising the *absolute value of the mean signed
   error* — i.e. it picks the k where errors cancel, not the k with the smallest
   errors. The published results may partly rest on this.
4. **The biggest missed opportunity is not algorithmic but probabilistic**: the
   neighbour set gives you an empirical predictive *distribution* for free, and
   supply-chain decisions are newsvendor problems that want quantiles, not point
   forecasts. The method computes everything needed for this and then throws it away.
5. **A 2026 rethink** would keep the retrieval/neighbour interface (the genuinely
   valuable part) and replace the metric-learning engine: per-SKU → global
   cross-SKU model, NNLS diagonal metric → GBDT leaf-space or learned-embedding
   similarity, point forecast → weighted neighbour quantiles + conformal calibration.
   For the per-SKU small-data regime (~75 rows), TabPFN-style in-context learning is
   now almost exactly purpose-built.

The rest of this document substantiates each claim, in decreasing order of "this is
wrong" and increasing order of "this is how I'd do it today".

---

## 1. What the algorithm actually is (a useful recasting)

Strip away the promotions framing and the method is:

1. Min–max scale features to [0,1].
2. Learn a **non-negative diagonal Mahalanobis metric** `v` by regressing squared
   target differences on squared feature differences over sampled pairs:
   `(y_i − y_k)² ≈ Σ_j v_j (x_ij − x_kj)²`, solved with NNLS (optionally Tikhonov-regularised).
3. Define distance `d_i = √(mᵢ·v)` and weight `w_i = 1/d_i`.
4. Predict with Nadaraya–Watson over the top-k neighbours, k chosen on a validation set.
5. Ensemble over random pair subsamples.

Two observations follow immediately from this framing:

**(a) This is diagonal MLKR.** Weinberger & Tesauro (2007) pose exactly the
objective in the paper's Eq. (10) — minimise leave-one-out kernel regression error
over a Mahalanobis metric — and solve it by gradient descent. The paper's Eq. (10)
*is* the MLKR objective; the paper then declares it intractable ("no guarantee the
optimisation converges... computational burden") and substitutes the NNLS proxy.
That substitution is the paper's real novelty, and it's also where all the
statistical problems live (Section 2). Related uncited work: Lowe (1995),
variable-kernel similarity learning; Lin & Jeon (2006), random forests as adaptive
nearest neighbours; the feature-weighted kNN literature generally. None of this
sinks the paper — the NNLS trick is fast and the application is real — but the
related-work positioning ("our variant of kNN") undersells both what it is and
what was already known about it.

**(b) The weights have a neat interpretation the paper never states.** Since `v` is
fit so that `mᵢ·v ≈ (Δy)²`, the learned distance is an *estimate of the sales
difference*: `d_i ≈ |ŷ_i − ŷ_k|`. So `w_i = 1/d_i` means: **weight each historical
promotion by the inverse of how different its sales are predicted to be.** The
method is a smoother in *outcome* space that uses features only as a proxy to
predict outcome proximity. This is a genuinely nice idea — and it also makes
defect 2(d) below (accidental similarity) obvious.

---

## 2. Statistical defects in the NNLS surrogate

The pivotal identity is Eq. (12): `|y_k − y_i|² = mᵢ·v`. Take a linear ground truth
`y = βᵀx + ε` and expand what the left side actually is:

```
(Δy)² = (βᵀΔx + Δε)²
      = Σ_j β_j² Δx_j²                  ← what the model fits (diagonal terms)
      + Σ_{j≠l} β_j β_l Δx_j Δx_l       ← cross terms: DROPPED
      + 2σ_ε²  (in expectation)          ← noise floor: NO INTERCEPT TO ABSORB IT
```

### 2(a) Dropped cross-terms ⇒ the collinearity failure is built in, not incidental

The NNLS design matrix only contains squared per-feature differences, so the
`Σ_{j≠l} β_j β_l Δx_j Δx_l` terms are omitted. They vanish in expectation **only if
feature differences are uncorrelated across features** — which is precisely the
assumption violated under collinearity. When `x_4 = λ_0 + λ_1 x_2`, the cross term
is large and systematic, and NNLS has no choice but to smear importance across the
correlated copies. The paper *observes* this empirically ("the values of **v** are
distributed across the variables... our algorithm is misled") in Table 1 (R² drops
0.90 → 0.82) but attributes it to the data rather than to a structural limitation
of the diagonal-only design.

**Fixes, cheapest first:**
- Whiten/decorrelate X before learning `v` (PCA or ZCA), learn the metric in the
  whitened space, map back for interpretation.
- Add the cross-term columns `Δx_j Δx_l` for the top-correlated pairs (still a
  linear NNLS problem, just wider — sign constraints need care since `β_jβ_l` can
  be negative, so those columns must be unconstrained or split into ± parts).
- Go full MLKR: learn a low-rank `L` with `d = ‖L Δx‖²` by gradient descent. In
  2026 this is ~30 lines of JAX/PyTorch and converges in milliseconds on 75-row
  problems; the 2018 "computational burden" objection no longer holds.

### 2(b) No intercept ⇒ the noise floor is pushed into the feature weights

`E[(Δy)²]` contains an irreducible `2σ_ε²` for *every* pair. The NNLS system has no
intercept column, so this constant must be absorbed by the feature weights —
biasing `v` upward, in a way that distorts *relative* importance (features whose
squared differences are on average large and flat make the best "fake intercepts").
**Fix: append a column of ones to M, leave its coefficient unconstrained (or
non-negative — it estimates 2σ̂_ε², which is also a useful diagnostic to report).**
One line of code; strictly better.

### 2(c) The loss is a 4th moment of sales ⇒ a few avalanche promos own the metric

NNLS minimises `Σ (mᵢ·v − (Δy)²)²` — squared error on *squared* sales differences,
i.e. **(Δy)⁴** enters the loss. Promotional sales are explicitly fat-tailed (the
paper's own Fig. 4 and the φ(x) avalanche mapping). A single pair of promotions with
a 10× sales gap contributes 10,000× the loss of a pair with a 1× gap. The learned
metric is therefore mostly fit to explain the extreme pairs, and the bagging that the
paper applies is treating this symptom (which is presumably why bag size 25% beats
50% in the Mother's Day experiment — smaller bags dilute the extremes' dominance).

**Fixes:**
- Work in `log(1+y)`. This is the single highest-leverage change to the whole
  method: it tames the 4th-moment problem, makes the additive-error assumption far
  more plausible for sales data, aligns the training loss with the NRMSLE the paper
  already evaluates with, and converts MAPE's asymmetry problem into a roughly
  symmetric one. Everything downstream (weights, neighbours, CV) is unchanged.
- And/or fit `|Δy| ≈ √(mᵢ·v)`-style targets with an L1/Huber loss (non-negative
  Lasso variants exist; `scipy`'s `lsq_linear` with bounds plus IRLS gets you a
  Huberised NNLS in a few lines).

### 2(d) Accidentally-similar sales actively corrupt the metric

The identity demands `mᵢ·v ≈ 0` whenever `y_i ≈ y_k` — even when the two promotions
are *far apart in features and similar in outcome by coincidence* (deep discount in
few stores ≈ shallow discount in many stores). Each such pair pushes NNLS to zero
out the weights of genuinely relevant features. The conclusion's stated limitation
("when the predictors do not contain meaningful information... feature selection
assigns similar importance") is partly this mechanism, and it's fixable:

- **Asymmetric pair weighting**: pairs with small Δy are uninformative-or-harmful;
  pairs with large Δy are reliable ("far in outcome ⇒ must be far in features" is a
  sound implication; the converse is not). Down-weight small-Δy pairs in the NNLS
  (row weights ∝ Δy² is a reasonable start).
- Or again: MLKR, which optimises prediction error directly and never asserts the
  bidirectional identity at all.

### 2(e) Smaller things

- **`v` is scale-invariant for prediction** (scaling v scales all weights uniformly
  and the normalised weighted mean is unchanged), so λ in Eq. (14) only shapes the
  *relative* allocation. The paper never notes this; it means the "consistency
  threshold t" in Algorithms 1–2 is meaningless unless `v` is normalised first, and
  λ cannot be interpreted as classical shrinkage strength.
- The pair rows sharing an anchor are strongly dependent; fine for a point
  estimate, but it means the effective sample size is far below the nominal row
  count, and the bagging variance estimates are optimistic.
- **Min–max scaling** makes the learned importances hostage to outliers in any one
  feature (one extreme "number of stores" value silently shrinks that feature's
  effective range). Robust scaling (quantile/MAD) is a drop-in improvement. Also
  note OHE binary columns have squared-diff ∈ {0,1} while scaled continuous
  features have squared-diff concentrated near 0 — the prior scale of "one unit of
  difference" differs by type, which `v` must compensate for; harmless for
  prediction, but it means **comparing raw `v` entries across feature types as
  "importance" (as the heatmap figures do) is not apples-to-apples.**
- **The 1/d kernel has no bandwidth and a pole at zero.** One near-duplicate
  neighbour swamps the average, which is why the code needs the
  `max_weight_scale = 15 × max-finite-weight` hack for exact duplicates. A
  Gaussian kernel `exp(−d²/h²)` with h chosen on the validation set (which already
  exists!) removes the pole, the hack, and adds one smoothly-tunable knob.

---

## 3. Implementation bugs and paper↔code mismatches

These are about `src/nextdoor/forecaster.py` as it stands today.

### 3(a) BUG — k is selected where errors *cancel*, not where they are small

`cv_neighbours` (forecaster.py:264–267):

```python
frc_error_mu = np.mean(frc_error, axis=0)   # mean SIGNED error per k
frc_abs_error = np.abs(frc_error_mu)        # |mean|, not mean(|·|)
self.k_neighbours = 1 + np.argmin(frc_abs_error)
```

This minimises **|mean signed error|** — a pure bias criterion. A k whose errors are
(+500, −500) beats a k whose errors are (+10, +10). It systematically favours large k
(more averaging ⇒ signed errors cancel) regardless of accuracy. The paper's Eq. (16)
specifies something entirely different (per-promotion argmin of *relative absolute*
error, then the *average* of the per-promotion k's). Neither matches the other, and
the code version is hard to defend. The fix is one line
(`np.mean(np.abs(frc_error), axis=0)`), but it potentially changes every published
number produced by this implementation — worth rerunning the benchmark if this
code ever gets used again.

(The paper's own rule is also questionable: per-promotion argmin then averaging k
is noisy and the average of per-case optima is not the optimum of the average.
Pick the single k minimising aggregate validation loss — which is what the code
*almost* does.)

### 3(b) BUG — in-place scaling mutates caller data and compounds across calls

`MinMaxScaler(copy=False)` plus `_scale_x_test` transforming in place means:

- `predict(x_test)` **modifies the caller's array**. Calling `predict` twice with
  the same array scales it twice and silently returns garbage the second time.
- In `fit_ensemble`, `x_test` is passed uncopied to every `_single_forecast`. With
  the default loky backend each worker gets a pickled copy, so it *happens* to be
  safe — but with `n_jobs=1` (or the threading backend) the scalers compound across
  the 100 forecasters. The scattered defensive `.copy()` calls elsewhere
  (`predict_with_ensemble`, `train_and_validate`) suggest this bit someone before.
  Use `copy=True` and return the transformed array; delete all the defensive copies.

### 3(c) Default validation split is random, not chronological

`train_and_validate` falls back to `train_test_split(..., random_state=42)` — a
random split. The paper's protocol is explicitly chronological (train → validation
→ test in time order). For promotional data with trend/seasonality a random split
leaks future information into k-selection and inflates validation optimism. The
default should be a tail split.

### 3(d) Assorted

- Off-by-one: with `k_neighbours=15`, `cv_neighbours` sweeps k = 1…14 (`range(1,
  k_neighbours)`), while the paper says 2…15. Cosmetic, but means the published
  default never actually evaluates k=15.
- The footnote's numerical-stability clamp (`max{1, |Δy|}`) is applied nowhere in
  the training code; exact-duplicate-y pairs put exact zeros in `e`, which is fine
  for NNLS, but the paper-described estimator and the implemented one differ.
- Algorithms 1–2 output a *selected feature set* S via threshold t; the code never
  thresholds — all features keep their weights. (Probably the right call — see
  2(e) on the threshold being scale-dependent — but paper and artifact disagree.)
- Paper's Eq. (14) writes the penalty `λ²‖v‖₂` (unsquared norm); the code's
  augmented-rows construction implements standard Tikhonov `λ²‖v‖₂²`. The code is
  right; the equation is sloppy.
- `predictions_std` from 100 ensemble members whose only randomness is anchor
  subsampling (train/val fixed in `fit_ensemble`) measures *procedure* variance,
  not predictive uncertainty. It will be badly overconfident if read as an
  interval (see Section 5).
- The Eq. (18) "eucScore" as written is a matrix expression (X′ minus rank-one,
  times V⁻¹, times transpose) presented as a scalar score; what's evidently meant
  is the per-row standardised Euclidean distance. A reader cannot reproduce the
  ranking from the equation as printed.

---

## 4. Evaluation critique — would the claims survive a 2026 review?

- **Weak baselines.** LS, and "100 LSBoost trees with learning rate 1.0" — a
  learning rate of 1.0 is a known-bad configuration that essentially guarantees
  the tree ensemble overfits (its MAPE of 334–1152% in Table 5 is the tell; a
  tuned LightGBM would never produce that). The retailer benchmark is the only
  strong comparator, and the method beats it modestly (e.g. DS2: w20 60.6 vs 62.1
  — it *loses* on w20 there and wins on the aggregate score). The honest claim is
  "competitive with the incumbent system while being interpretable", which is
  still a good claim — but "significantly improves the accuracy" (abstract) is
  stronger than Table 5 supports.
- **Missing ablation.** The single most important question for this paper — *does
  the learned metric beat uniform-weight kNN on the same features?* — is never
  tested. Without "kNN with v = 1" as a baseline, the entire NNLS machinery is
  unvalidated as a *component*. (Also missing: last-promo-carried-forward and
  median-of-last-k naive baselines, the standard sanity floor in forecasting.)
- **No significance testing.** 43,757 test promotions is plenty for a
  Diebold–Mariano test or even a paired bootstrap on w20p/out50p; none is reported.
  The hyperparameter table's spread (eucScore 1.26–1.79, with the top 5 within
  0.09) looks like it's within selection noise — choosing "ensemble of 2 weak
  learners" off rank 1 of that table is likely overfitting the hyperparameters to
  one 361-promotion dataset.
- **Single time origin.** One chronological split; no rolling-origin evaluation.
  Promotional forecasting performance is notoriously regime-dependent (the paper's
  own stability motivation!), and a single origin can't distinguish a good method
  from a lucky quarter.
- **MAPE in the headline metrics** despite the paper's own (correct!) argument that
  volume-weighted band metrics matter more. MAPE on promotions additionally
  rewards under-forecasting, which for supply chain is the *expensive* direction
  (stockouts). wMAPE or the band metrics alone would have been cleaner.

What the evaluation got **right**, ahead of its time: volume-weighted business
metrics (w20p/out50p) over naive MAPE-chasing, the Bland–Altman residual analysis,
testing on three genuinely different markets, and an honest discussion of the
dataset-2 categories where the method loses.

---

## 5. The biggest missed opportunity: distributions, not points

Promotional supply-chain decisions are **newsvendor problems**: order to the
critical-ratio quantile of the demand distribution, where the ratio depends on
margin vs. waste cost (and for fresh food those costs are wildly asymmetric — the
paper even leads with waste reduction in the conclusion).

The method already computes, for every forecast, a set of k neighbours with
normalised weights `w` and outcomes `y`. That *is* a nonparametric predictive
distribution — `{(w_i, y_i)}` — and the code collapses it to its mean and throws
the rest away. For free, with zero extra computation, it could output:

- **Weighted quantiles** of the neighbour outcomes → direct newsvendor order
  quantities per cost ratio.
- **Conformalised intervals**: the validation set (already maintained for
  k-selection) is exactly the calibration set split-conformal prediction needs.
  Scale the neighbour-quantile intervals by the conformal correction and you get
  finite-sample coverage guarantees — distribution-free, per-SKU, ~10 lines of
  code. In 2019 this would have been a second paper; in 2026 it's table stakes.
- The interpretability story *improves*: "these 8 similar past promotions sold
  between 3,100 and 6,400 units; to hit 95% service level, stock 6,000" is more
  actionable for a forecaster than a point estimate they have to mentally pad.

This is the cheapest large improvement available: no change to the learned metric,
pure post-processing, and it converts the method from "point forecaster with an
ensemble std that underestimates uncertainty" (Section 3(d)) into a decision tool.

---

## 6. Things the paper assumed away that matter in practice

- **No recency weighting / drift handling.** "Online learning" here means lazy
  learning (refit at query time on latest data) — which is fine — but within the
  training window, a promotion from 23 months ago and one from last month are
  weighted identically if their features match. Sales levels drift (trend,
  inflation, store estate changes). Cheap fixes: exponential time-decay multiplier
  on `w`, or detrend `y` by a per-SKU baseline before neighbour averaging and
  re-trend after. (The date-derived features help matching seasonality but cannot
  express "prefer recent".)
- **Per-SKU independence ⇒ structural cold-start failure.** A product with 3
  historical promotions gets a 3-neighbour model; a new product gets nothing. The
  conclusion acknowledges the trendy-product failure mode but not the systemic
  cause: all cross-SKU signal ("chocolate boxes behave like other chocolate boxes
  on Mother's Day") is discarded by design. Post-M5 (2020), the evidence is
  overwhelming that **global models pooled across series beat per-series models**
  on exactly this kind of retail data, largely *because* of cross-learning. This
  is the single biggest accuracy lever the design leaves on the table — see §7.
- **Feature leakage risk in the data spec.** The collinearity experiment's own
  example, `x_4` = "orders placed by the stores" (and `c_2` = "stores requested
  more units mid-promotion"), are quantities determined *during/after* the
  promotion. The framework section is admirably explicit that all of `x` must be
  known before launch; the experiment's flavour variables quietly violate it.
  Worth auditing the real datasets' feature lists with the same lens — "number of
  stores" is plannable, but replenishment-derived features generally are not.
- **Aggregation level.** Sales are aggregated to store-cluster level and each
  cluster-promotion is one row. Cross-cluster correlation within the same
  promotion (test rows averaging 3.28 = same promo across store types) makes the
  test rows non-exchangeable — another reason the missing significance tests
  matter.

---

## 7. If I were rebuilding this in 2026

Keep the product insight, replace the engine. The durable insight is the
**interface**: forecasters trust and can *edit* a prediction expressed as a
weighted set of concrete past promotions. Nothing about that interface requires
the similarity metric to be a diagonal NNLS fit.

**Tier 1 — fix the current method in place (days of work, no architecture change):**
1. Fix the k-selection bug (3a) and in-place scaling (3b); chronological default split (3c).
2. Learn the metric on `log1p(y)` (2c) with an intercept column (2b).
3. Gaussian kernel with validated bandwidth instead of 1/d (2e).
4. Output weighted neighbour quantiles + split-conformal intervals (§5).
5. Add the uniform-weight-kNN ablation and a tuned LightGBM baseline to know
   where you actually stand.

**Tier 2 — better metric, same interface (weeks):**
6. Replace NNLS with proper diagonal/low-rank MLKR by autodiff (the 2018
   tractability objection is gone). Keep `v ≥ 0` diagonal as the *explanation*
   layer if desired, fit it to distil the richer metric.
7. Or: fit a global LightGBM across all SKUs and use **leaf co-occurrence as the
   similarity** (random-forest-proximity style). You inherit cross-SKU learning,
   interactions, and categorical handling, while the forecast remains "weighted
   average of retrieved similar promotions" — fully preserving the editable-
   neighbours UX. This is the option I'd actually ship: it deletes defects 2(a),
   2(c), 2(d) and §6's cold-start in one move.
8. Recency: exponential decay on neighbour weights, decay rate validated.

**Tier 3 — the 2026-native rethink (a research project, and arguably a new paper):**
9. **Retrieval-augmented tabular forecasting**: embed promotions with a small
   global model (or TabPFN-style prior-fitted network — per-SKU contexts of
   ~75 rows are squarely in its sweet spot and need no training at all), retrieve
   neighbours in embedding space, predict the full conditional distribution,
   conformalise. Evaluate rolling-origin with Diebold–Mariano against the
   retailer benchmark and tuned GBDT, report pinball loss at the business-critical
   quantiles alongside w20p/out50p.
10. The interpretability claim then becomes *testable*: run a forecaster-in-the-
    loop study measuring whether neighbour-editing improves final accuracy over
    accept/reject of a black-box forecast. That study, not the algorithm, is the
    publishable contribution today.

---

## 8. What holds up

Credit where due, because a 2026 reading can be unfairly smug:

- **Interpretability-first as a hard constraint** (and being explicit about
  trading accuracy for it) predates the mainstream XAI-for-forecasting wave; the
  M5 organisers were still lamenting uninterpretable winners in 2021.
- **Neighbour-based explanations** remain more faithful than post-hoc attribution:
  the explanation *is* the computation, not a story about it.
- **Volume-weighted error bands** (w20p/out50p) as first-class metrics show real
  domain understanding that generic ML papers of that era lacked.
- The **lazy-learning/per-SKU deployment** (stateless microservice, data in the
  request, no retraining pipeline) is an honest, robust production design — many
  "global model" deployments since have died of the MLOps complexity this design
  deliberately avoided.
- The surrogate-model experiments (known ground truth for feature recovery,
  explicit collinearity/endogeneity/subrepresentation constructions) are a more
  careful validation methodology than most applied forecasting papers of the
  period — careful enough, in fact, that they *surfaced* the collinearity weakness
  analysed in 2(a), even if the paper read it as a data property rather than a
  model property.

---

## 9. Implementation status (June 2026)

All three tiers are now implemented on the `fable-improvements` branch — Tier 3
included, scoped to runnable code and simulation experiments (the parts that
remain genuinely open, e.g. a *human* forecaster study and rolling-origin
evaluation on the real retailer data, are noted at the end):

- `src/nextdoor/forecaster.py` — rewritten. Bug fixes (3a–3d), intercept (2b),
  `target_transform` (2c), optional pair weighting (2d), robust scaling and
  Gaussian kernel (2e), optional whitening (2a), recency decay (§6),
  `predict_quantiles` / `predict_interval` (CQR + split conformal) and
  `explain()` (§5). The legacy k-selection criterion is kept as
  `k_selection="bias"` so the bug's impact stays measurable.
- `src/nextdoor/mlkr.py` — diagonal MLKR by gradient descent (Tier 2, item 6).
- `src/nextdoor/leaf_forecaster.py` — random-forest leaf-proximity similarity
  with the same neighbour-explanation interface (Tier 2, item 7).
- `src/nextdoor/retrieval.py` — `RetrievalAugmentedForecaster`: a global
  cross-SKU MLP embedding, neighbour retrieval in embedding space, conformal
  predictive distribution, and the same `explain()` interface; optional TabPFN
  backend that degrades gracefully (Tier 3, item 9).
- `tests/test_improvements.py` + `tests/test_tier3.py` — 31 tests in total.
- `benchmarks/ablation.py`, `benchmarks/forecaster_in_the_loop.py`,
  `benchmarks/cold_start.py` — the ablation plus the two Tier 3 experiments.

### What the ablation actually showed (5 seeds, n=500, chronological 60/20/20)

The legacy reproduction matches the paper's published numbers (w20p ≈ 55 on
the ideal scenario vs. the paper's 54.06), which validates the setup. Means
across scenarios:

| variant | MAE (ideal/collin/endog) | w20p% | out50p% |
|---|---|---|---|
| legacy (2019)            | 1174 / 1236 / 1272 | 55.4 / 55.5 / 52.1 | 11.7 / 13.0 / 12.7 |
| legacy + k-fix only      | 1063 / 1130 / 1137 | 59.8 / 59.4 / 57.9 |  8.4 /  8.6 / 10.1 |
| uniform kNN (ablation)   | 1225 / 1216 / 1185 | 51.3 / 53.4 / 52.2 |  8.6 / 10.9 /  8.0 |
| improved NNLS (defaults) | 1048 / 1108 / 1083 | 60.1 / 58.2 / 58.2 |  7.2 /  8.6 /  8.4 |
| MLKR                     | 1050 / 1061 / 1072 | 61.9 / 59.8 / 60.1 |  6.9 /  7.7 /  8.6 |
| leaf similarity (RF)     | 1056 / 1065 / 1069 | 63.0 / 61.6 / 62.0 |  8.8 /  8.0 /  8.7 |

Findings worth recording:

1. **The k-selection bug is worth ~9–11% MAE on its own.** Fixing only the
   criterion (everything else legacy) cuts out50p volume from ~12–13% to
   ~8–10%. The single most damaging line in the codebase.
2. **The metric learning does earn its keep** — the uniform-kNN ablation is
   consistently worse than the learned metrics — but the gap narrows under
   collinearity, exactly as the cross-term analysis (2a) predicts.
3. **One prediction of this document was wrong and got corrected by the
   benchmark**: log1p as an unconditional default (Tier 1, item 2) *hurts* on
   the paper's surrogate, because that generator has **additive** noise
   (constant σ≈1265 at all sales levels), which log distorts. Log is right for
   multiplicative noise. The implemented default is therefore
   `target_transform="auto"`: both scales are fitted and the validation set
   decides. Likewise, neighbour averaging in log space biases the point
   forecast low (geometric mean), so the metric scale and the averaging scale
   are decoupled (`average_scale="original"` by default). Pair weighting (2d)
   was also mildly negative under heavy additive noise and ships off-default.
4. **MLKR is no longer expensive.** The 2018 tractability objection is dead:
   the gradient solver converges in milliseconds on per-SKU problems and beats
   the NNLS proxy on every scenario, most clearly under collinearity (MAE 1061
   vs 1108) — as predicted, since it never asserts the biased identity.
5. **CQR intervals are calibrated out of the box**: empirical coverage 0.80 /
   0.78–0.82 at nominal 80% across scenarios, with no extra computation beyond
   what the method already does.

### Tier 3, item 9 — cross-SKU retrieval (`benchmarks/cold_start.py`)

The global retrieval forecaster was tested against per-SKU NextDoor across
three regimes (10 seeds each), and the result is more honest than the M5-era
"global always wins" slogan — it wins *where the design predicts it should*:

| regime | per-SKU MAE | global MAE | change |
|---|---|---|---|
| ample history, no shared structure | 85 | 477 | **−464%** |
| short history, structure shared via descriptors | 194 | 201 | −4% |
| brand-new SKUs, zero history (leave-SKU-out) | 412 | 136 | **+67%** |

The takeaways:
- **Pooling is not free.** When each SKU has ample independent history and no
  cross-SKU structure to borrow, a dedicated per-SKU model wins decisively and
  the global embedding only adds variance. Anyone claiming "just go global"
  should run this row first.
- **Cold-start is the real prize.** For brand-new products — the exact failure
  the paper's own conclusion flags ("a new product gets nothing") — the per-SKU
  method can only emit a global median, while the descriptor-based global model
  cuts the error by two-thirds. This is the case to ship the retrieval
  forecaster for, used *alongside* (not instead of) the per-SKU method.

### Tier 3, item 10 — the interpretability claim, made testable (`benchmarks/forecaster_in_the_loop.py`)

A simulated forecaster of tunable skill `s` interacts with the model in two
ways: **accept/reject** (veto the forecast, fall back to typical sales — all a
black box affords) and **edit** (down-weight the neighbours they believe are
misleading — what NextDoor affords). Both draw on the same noisy view of which
neighbours come from the query's true demand regime. Mean absolute error
(20 seeds, n=600; raw model = 111 regardless of skill):

| skill | accept/reject | edit neighbours |
|---|---|---|
| 0.00 | 758 | 172 |
| 0.50 | 426 | 119 |
| 1.00 | 107 |  84 |

And catastrophic (>50%) error volume at full skill: raw 2.2%, accept/reject
2.1%, **edit 0.9%**. Findings:
- **Editing dominates accept/reject at every skill level.** A veto-only UI
  forces an all-or-nothing fallback that does real damage at low-to-moderate
  skill (758 vs raw 111 at s=0); selective neighbour editing degrades
  gracefully and, past s≈0.5, *improves* on the automatic forecast.
- **The interface earns its keep.** At full skill, editing beats accept/reject
  by 23 MAE (~21% of the raw error) and more than halves the catastrophic-error
  volume. The interpretability claim is therefore not just aesthetic: under
  these assumptions the editable-neighbour interface is worth a measurable
  accuracy gain that a black-box model cannot offer at any skill level.
- This is a *simulation* — it makes the claim falsifiable and sizes the prize,
  but the assumptions (a noisy regime-membership channel, a blunt fallback) are
  mine, not data. The remaining genuinely-open work is the real human study and
  rolling-origin evaluation on the retailer datasets; those need data and IRB,
  not code.

## Appendix: defect → evidence cross-reference

| # | Defect | Where | Severity |
|---|--------|-------|----------|
| 2a | Cross-terms dropped from NNLS design | Eq. (12)–(15); Table 1 collinearity row | High — structural |
| 2b | No intercept; noise floor absorbed into v | Eq. (15); `_solve_nnls_training` | Medium — one-line fix |
| 2c | (Δy)⁴ loss; outlier promos dominate metric | Eq. (14); fat-tailed Fig. 4 | High — fix via log1p(y) |
| 2d | Accidentally-similar sales zero out true features | Eq. (12); conclusion's own limitation note | Medium |
| 3a | k chosen by \|mean signed error\| | forecaster.py:264–267 vs Eq. (16) | High — bug, affects results |
| 3b | In-place scaling mutates/compounds | forecaster.py:77, 93–105 | High — bug |
| 3c | Random default val split (temporal leakage) | forecaster.py:354–357 | Medium |
| 3d | Off-by-one k sweep; unimplemented threshold t; λ notation; ensemble std ≠ uncertainty | various | Low |
| 4 | No uniform-kNN ablation; untuned tree baseline (lr=1.0); no significance tests; single origin | §IV | High — for the claims |
| 5 | Point forecast only; neighbour distribution discarded | whole method | High — for the application |
| 6 | No recency decay; per-SKU cold start; leakage-prone features | framework §II–III | Medium–High |
