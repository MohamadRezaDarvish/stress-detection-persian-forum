# Model card — Hybrid Persian Forum Stress Fusion

## Status

**Candidate for a limited human-in-the-loop risk-monitoring overlay; not a clinical or autonomous decision system.**

The selected model passed all four validation recall constraints. On the one-time locked
test evaluation, Low, High, and Very-High recall passed, but Moderate recall was
41.79% instead of the required
50%. The project therefore met **3 of 4 internal recall objectives**, with Moderate as the improvement area. The internal gate and deployment framing are separate: this result can support controlled human-in-the-loop monitoring evaluation, but it is not clinical validation or production approval.

No additional threshold or model tuning should be performed against this test set.

## Selected architecture

- Candidate: `hybrid_catboost_rmse_depth3`
- Fusion family: shallow CatBoost regression
- Inputs:
  - Member 1 leakage-safe scalar prediction
  - Member 2 leakage-safe scalar prediction
  - Member 1 predefined metadata and count features
  - base-model mean and disagreement features
- Continuous output clipped to 1–10
- Clinical categories calibrated with validation thresholds:
  - Low/Moderate: `3.095439`
  - Moderate/High: `4.905161`
  - High/Very High: `6.462135`

## Validation

| Metric | Value |
|---|---:|
| MAE | 0.8348 |
| RMSE | 1.1336 |
| Pearson | 0.8577 |
| Accuracy | 0.7566 |
| Macro F1 | 0.6717 |
| Very-High PR-AUC | 0.7777 |

| Class | Recall | Precision | Requirement |
|---|---:|---:|---:|
| Low | 84.51% | 91.94% | 75% |
| Moderate | 58.21% | 38.24% | 50% |
| High | 52.38% | 62.26% | 50% |
| Very High | 76.00% | 79.17% | 75% |

## Locked test

| Metric | Value |
|---|---:|
| MAE | 0.9174 |
| RMSE | 1.2459 |
| Pearson | 0.8333 |
| Accuracy | 0.7528 |
| Macro F1 | 0.6284 |
| Very-High PR-AUC | 0.6427 |

| Class | Correct / support | Recall | Precision | Requirement | Gate |
|---|---:|---:|---:|---:|---|
| Low | 260 / 297 | 87.54% | 91.55% | 75% | Pass |
| Moderate | 28 / 67 | 41.79% | 35.44% | 50% | **Fail** |
| High | 33 / 63 | 52.38% | 56.90% | 50% | Pass |
| Very High | 20 / 26 | 76.92% | 62.50% | 75% | Pass |

Moderate required at least 34 correct predictions; the locked test obtained 28.

## Explainability

Global importance and validation SHAP summaries are included. The dominant signals are
Member 2's text prediction and the mean of the two base predictions. Member 1's prediction,
emoji counts, signal ratios, sub-category, reply status, and post length provide smaller
complementary contributions.

## Limitations

- The evaluation is safety-enriched and does not estimate natural website prevalence.
- The Very-High validation/test sample sizes are only 25 and 26.
- Continuous predictions systematically understate many High and Very-High labels; the
  classification thresholds compensate by being lower than the true-score boundaries.
- Member 1's original lexical extractor is still external.
- Member 2's five fold checkpoints are external and must be preserved.
- This model must not be presented as a clinical diagnostic tool.

## Next legitimate improvement

Because the test has been opened, improve the model using nested cross-validation and a
new untouched confirmation set. Target active learning toward posts near the
Low/Moderate and Moderate/High boundaries, high base-model disagreement, and errors that
are currently mapped from Moderate to Low or High.
