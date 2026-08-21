# Member C Handoff Contract

This contract describes exactly what Member 2 delivers to Member C for the final fusion
model, file by file, column by column.

## Handoff files

| File | Rows | Type |
|------|-----:|------|
| `member2_oof_predictions_deterministic.csv` | 4,226 | OOF (train) |
| `member2_validation_predictions_fold_ensemble.csv` | 452 | validation ensemble |
| `member2_test_predictions_fold_ensemble.csv` | 453 | test ensemble |

Canonical location: `outputs/predictions/`. A copy is also staged in `data/handoff/`.

## Column descriptions

### OOF file (`member2_oof_predictions_deterministic.csv`)

| Column | Description |
|--------|-------------|
| `unique_post_id` | join key (matches frozen manifest) |
| `fold` | the held-out fold (0–4) whose model produced this prediction |
| `true_stress` | `final_stress` label (train row) |
| `prediction` | deterministic OOF prediction from the held-out fold model |

### Validation / Test ensemble files

| Column | Description |
|--------|-------------|
| `unique_post_id` | join key |
| `model_role` | `validation` or `test` |
| `true_stress` | `final_stress` label |
| `prediction` | **mean of the 5 fold models' deterministic predictions** |
| `prediction_std_fold_models` | std across the 5 fold models (uncertainty between models) |
| `fold_model_0` … `fold_model_4` | each fold model's individual deterministic prediction |

## Generation method

- **OOF:** each train row is predicted by the model trained on the other 4 folds
  (its assigned held-out fold). Every train row appears exactly once.
- **Validation/Test:** all 5 fold models predict each row; final = mean, std =
  between-model std. Deterministic (`model.eval()`, no MC dropout).
- The `prediction` column of the ensemble files is exactly the mean of `fold_model_0..4`
  (verified in tests).

## ID requirements

- Every row keyed by `unique_post_id`, matching the frozen manifest exactly.
- OOF: all 4,226 train IDs present, each exactly once; no validation/test/embargo IDs.
- Validation: exactly the 452 validation IDs. Test: exactly the 453 test IDs.
- Order does not matter — alignment is by `unique_post_id`.

## Prediction range

- Predictions are **not clipped** to [1,10] (matches production). They typically lie in
  ~[0.2, 8.6] for OOF and ~[0.6, 8.3] for holdouts. If Member C requires strict [1,10],
  apply `np.clip(pred, 1, 10)`.

## Missing-value policy

- No missing IDs, no duplicate IDs, no NaN predictions. Any violation fails the notebook's
  verification step (§11) and `tests/test_prediction_contract.py`.

## Uncertainty semantics

| Column | Meaning |
|--------|---------|
| `prediction_std_fold_models` | disagreement **between the 5 fold models** |
| (MC files, not in this deterministic set) | `prediction_std_mc_dropout` = std over MC-dropout passes of one model |

These are different uncertainty sources and are kept in separate files/columns.

## How Member C should use it

1. Join `unique_post_id` to Member 1's 45-feature table.
2. Use `prediction` from the OOF file as the training target-adjacent feature for the
   stacker; use validation/test ensemble `prediction` for evaluation.
3. Optionally use `prediction_std_fold_models` as an uncertainty feature.
4. Do **not** use `true_stress` as an input feature (it is the label).
5. Embeddings (CLS 768-d) are available in `outputs_final/`/`deliverables/` as an
   explicitly-labeled exploratory artifact; the primary stacker uses the scalar
   prediction.
