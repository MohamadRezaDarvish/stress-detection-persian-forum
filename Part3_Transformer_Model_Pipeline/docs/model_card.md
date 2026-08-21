# Model Card — ParsBERT Continuous Stress Regression

## Model

| Field | Value |
|-------|-------|
| Base checkpoint | `HooshvareLab/bert-fa-zwnj-base` |
| Architecture | ParsBERT v3 (BERT-based, ZWNJ-aware) → `BertForSequenceClassification(num_labels=1, problem_type="regression")` |
| Task | Single-output continuous regression of `stress` (≈1–10) from Persian forum text |
| Output | 1 continuous value (predicted stress) |
| Embeddings | CLS token last hidden state (768-d) |

## Intended use

Predict the continuous stress level of Persian pregnancy-forum posts from text alone.
Output is consumed by Member C's fusion model (stacked with Member 1's 45 tabular
features) and, at the end, binned into clinical categories (Low / Moderate / High /
Very High) using the authoritative right-closed thresholds (≤3, 3–5, 5–7, >7).

## Training data

- Final accepted split: **4,226 train / 452 validation / 453 test / 484 embargo**.
- Embargo rows were never used.
- Only `content` (text) and `final_stress` (target) were used; `training_sample_weight`
  used in the loss.

## Training procedure

- **5 fold models**, each trained on the other 4 folds (fold_epochs = 5).
- Loss: `weighted_asymmetric_mse` — `training_sample_weight * under_penalty * error²`,
  with `under_prediction_penalty = 1.75` applied when `pred < true` (under-prediction is
  penalized more: false reassurance is more dangerous than a false alarm).
- AdamW lr 2e-5, wd 0.01, batch 16, linear warmup (10%) → linear decay, grad clip 1.0,
  seed 42, max_length 384.
- Text normalization: `hazm.Normalizer`.

## Evaluation results (test, locked)

| Metric | Value |
|--------|------:|
| MAE | 0.969 |
| RMSE | 1.309 |
| Pearson | 0.825 |
| Accuracy (4 bins) | 0.729 |

Validation: MAE 0.825, Pearson 0.860, accuracy 0.741. OOF (train): MAE 0.985, Pearson 0.813.

Per-class recall (test): Low 0.84, Moderate 0.45, High 0.59, Very High 0.58 (full detail in
`outputs/metrics/test_metrics.json` and the notebook).

## Known limitations

- **Very High** stress is the hardest class (only 25 validation / 26 test samples) and is
  systematically under-predicted.
- The model over-predicts more than it under-predicts overall (intended safety bias).
- Text-only: no tabular features; fusion with Member 1 is expected to improve tail recall.
- Requires the fine-tuned fold checkpoints; base ParsBERT alone is NOT sufficient for
  inference.

## Considerations

- The asymmetric loss encodes a clinical safety preference (false reassurance worse than
  false alarm); downstream clinical use should account for the resulting calibration shift.
