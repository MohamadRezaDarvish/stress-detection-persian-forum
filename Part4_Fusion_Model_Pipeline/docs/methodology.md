# Fusion methodology

## Leakage-safe inputs

Training uses Member 1 and Member 2 out-of-fold predictions. Validation and test use the
mean predictions from each member's five fold models. This prevents the fusion model from
learning from in-sample base-model predictions.

The hybrid fusion also receives Member 1's predefined metadata and count features. Raw
post content is not modeled by the fusion CatBoost model; text enters through Member 2's
ParsBERT prediction.

## Candidate search

The project compares:

- Member 1 only
- Member 2 only
- fixed weighted blends
- linear and Ridge scalar stackers
- monotone scalar CatBoost stackers
- shallow hybrid CatBoost regressors with RMSE or MAE loss

Every candidate is fitted on the 4,226 training rows and evaluated on the 452 official
dual-labeled validation rows.

## Threshold calibration

True clinical labels are authoritative and right-closed:

- Low: score <= 3
- Moderate: 3 < score <= 5
- High: 5 < score <= 7
- Very High: score > 7

Prediction thresholds are searched independently. Candidate selection first requires
validation recall of at least 75% / 50% / 50% / 75%, then maximizes Very-High precision,
macro F1, High precision, and finally minimizes MAE.

## Test discipline

The test set is loaded only by `evaluate_locked_test.py`, after the selected model and
thresholds have been written. The package contains the resulting one-time locked
evaluation.
