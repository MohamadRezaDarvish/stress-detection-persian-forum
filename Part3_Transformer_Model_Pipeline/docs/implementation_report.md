# Implementation Report — Member 2

## Objective

Fine-tune a Persian transformer to predict the continuous `stress` score of forum posts
from raw Persian text only, and hand OOF + fold-ensemble predictions to Member C for the
final fusion model.

## Environment

- Compute: Modal cloud GPU, **A10G** (T4 verified for connectivity).
- Stack: PyTorch, Hugging Face Transformers, `hazm`, pandas/numpy, scikit-learn/scipy.
- All training runs as a remote Modal GPU function; artifacts persist on a Modal Volume
  and are reproducible locally.

## Data

Final frozen split (from Member C): **4,226 train / 452 validation / 453 test /
484 embargo**. Only `content` (input), `final_stress` (target), `training_sample_weight`
(loss weight), `unique_post_id` (key), and `clinical_class` (evaluation) are used.

## Text preprocessing

- `hazm.Normalizer` applied to `content`: normalizes ي/ي, ك/ك, ZWNJ (نیمفاصله),
  whitespace, punctuation.
- Tokenizer: `HooshvareLab/bert-fa-zwnj-base`; `max_length=384` (99th percentile = 262,
  only 9 rows exceed 384); truncation + padding to max_length.
- No aggressive cleaning (no stopword removal, no punctuation/emoji stripping).

## Model & training

- Base: `HooshvareLab/bert-fa-zwnj-base` (ParsBERT v3, ZWNJ-aware), regression head
  `Linear(768→1)`, `num_labels=1`.
- **5 fold models**, each trained on the other 4 folds (fold_epochs=5).
- Loss: `weighted_asymmetric_mse` = `training_sample_weight * under_penalty * error²`,
  `under_penalty=1.75` for under-prediction (error<0).
- AdamW lr 2e-5, wd 0.01, batch 16, linear warmup 10% → linear decay, grad clip 1.0,
  seed 42.
- Deterministic inference (`eval()` mode, no MC dropout) for the delivered predictions.

## Key decisions

| Decision | Reason |
|----------|--------|
| Regression, not classification | `stress` is a genuine continuous multi-rater average; fusion needs a number |
| Weighted + asymmetric MSE | right-skew (imbalance) + clinical safety preference |
| 5 grouped folds (supplied) | prevents author/thread/content leakage; no new split |
| OOF + fold-ensemble, not full-model holdouts | avoids reusing validation labels through the full model's early stopping; gives the stacker a consistent fold-model family |
| Deterministic (no MC) for delivery | matches the default production inference path |
| Not clipped to [1,10] | matches production; explicit optional clipping provided |

## Results (test, locked)

- MAE **0.969**, RMSE **1.309**, Pearson **0.825**, accuracy (4 bins) **0.729**.
- Validation: MAE 0.825, Pearson 0.860, accuracy 0.741.
- OOF (train): MAE 0.985, Pearson 0.813.
- Acceptance-floor gaps: Very-High recall is below the 75% target (small support class,
  safety under-prediction). Low/Moderate/High meet their floors.

## Explainability

- Not used for feature selection. Post-hoc only: error analysis by stress band + gradient
  token attribution on examples. See `outputs/explainability/`.

## Reproducibility

- The notebook (`notebooks/Member2_Full_Reproduction.ipynb`) runs end-to-end:
  validation → preparation → folds → model load/retrain → OOF → val/test ensembles →
  evaluation → explainability → handoff export → artifact verification → tests.
- `FULL_RETRAIN=True` retrains the 5 fold models; `FULL_RETRAIN=False` loads the supplied
  checkpoints. Prediction backend: modal / cpu / load.
- All 16 tests pass. Artifact checksums are in `outputs/logs/artifact_checksums.json`.

## Limitations

- Very-High stress is under-predicted and has tiny support.
- Text-only (no tabular signal); fusion with Member 1 is expected to improve tail recall.
- Inference requires the fine-tuned fold checkpoints (not just base ParsBERT).
