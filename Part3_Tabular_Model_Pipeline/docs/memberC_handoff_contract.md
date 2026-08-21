# Member C handoff contract

## Prediction files

### `member1_oof_predictions.csv`
- 4,226 train rows.
- `unique_post_id`: string alignment key.
- `fold`: supplied held-out fold.
- `true_stress`: training target.
- `prediction`: one prediction from a model that did not train on the row.
- `prediction_std_optional`: blank because each OOF row has one held-out model prediction.

### `member1_validation_predictions.csv`
- 452 validation rows.
- Prediction is the arithmetic mean of five fold models.
- `prediction_std_optional` is the population standard deviation across the five models.

### `member1_test_predictions.csv`
- 453 locked-test rows.
- Prediction is the arithmetic mean of five fold models.
- Test labels are not used for model or threshold selection.

## Engineered feature files

`member1_train_engineered_features.csv.gz`, `member1_validation_engineered_features.csv.gz`, and `member1_test_engineered_features.csv.gz` preserve the exact 67-feature order and categorical values used by CatBoost.

## Range and missing policy

Predictions are clipped to [1, 10]. Numeric missing values remain NaN for native CatBoost handling. Missing categoricals use `__MISSING__`. Signature `-1` values mean absent signature and become zero with `has_signature=0`.
