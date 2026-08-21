# Member 2 Workflow

This document describes Member 2's complete workflow in two stages — before active
learning and after active learning — and is the written companion to the diagrams in
this folder.

**Two workflow diagrams:**

- `workflow_pre_active_learning.png` — the first modeling stage.
- `workflow_final.png` — the final, post-active-learning production stage.

---

## 1. Role in the first modeling stage (before active learning)

**Stage:** an initial labeled text dataset is provided before any active-learning round.

- **Initial labeled text dataset received:**
  - `train.csv` — 1,561 rows
  - `val.csv` — 335 rows
  - `test.csv` — 335 rows
  - Only three columns were used for modeling: `content` (input), `stress` (target),
    `unique_post_id` (join key). All metadata columns (45 tabular features, `stress_proxy`,
    `anxiety`, `depression`) were ignored.

- **Persian text preprocessing performed:**
  - `hazm.Normalizer` — normalizes ي/ي, ك/ك (Arabic-to-Persian), ZWNJ / half-space
    (نیمفاصله), extra whitespace, and punctuation variants.
  - Applied before tokenization to every text.

- **Transformer model used:**
  - `HooshvareLab/bert-fa-zwnj-base` — **ParsBERT v3** (BERT-based, ZWNJ-aware).
  - Configured as `BertForSequenceClassification(num_labels=1, problem_type="regression")`.

- **How the model was fine-tuned for stress regression:**
  - `max_length = 256`, batch 16, AdamW lr 2e-5, wd 0.01, up to 6 epochs.
  - Custom loss: 10-bin inverse-frequency **weighted MSE** × **asymmetric under-prediction
    penalty 1.75** (under-predicting stress is penalized more).
  - Early stopping on validation MAE (patience 2); best checkpoint at epoch 2.
  - Result (test): **MAE 1.260, RMSE 1.527, Pearson 0.746**.

- **How predictions contributed to the first-level fusion model:**
  - Continuous `stress` predictions were produced for every post and handed to Member 3,
    who concatenated Member 2's CLS embeddings (768-d) with Member 1's 45 tabular
    features in the first-level fusion network.

- **How that fusion model supported active-learning candidate selection:**
  - The first-level fusion model (Member 1 + Member 2) was used to score unlabeled posts;
    low-confidence / high-uncertainty posts were selected as candidates for new human
    annotation; those new annotations updated the labeled dataset for the next round.

**Pre-active-learning diagram** (`workflow_pre_active_learning.png`):
```
Initial labeled text (1,561/335/335)
→ hazm.Normalizer
→ ParsBERT tokenizer (max_length=256)
→ fine-tuned ParsBERT regression
→ continuous stress prediction
→ first-level fusion with Member 1
→ active-learning candidate selection → new annotations → updated dataset (loop)
```

---

## 2. Role after active learning

**Stage:** the team froze one official, leakage-safe split for the final fusion system.

- **Final handoff dataset received from Member C:**
  - `member2_handoff.csv` (content, labels, weights, roles)
  - `modeling_manifest_v2.csv` (IDs, groups, folds, hashes, flags)

- **Exact train / validation / test / embargo roles:**

  | Role | Rows |
  |------|-----:|
  | train | 4,226 |
  | validation | 452 (official dual-labeled) |
  | test | 453 (official dual-labeled) |
  | embargo | 484 (**never used**) |

- **How grouped folds were used:**
  - 5 pre-assigned folds (`oof_fold` 0–4) were used exactly as supplied. Grouping prevents
    the same author / thread / duplicated content from crossing folds or evaluation splits.
  - No new random split was created.

- **How many Transformer fold models were trained:**
  - **5** fold models (`models/fold_0` … `models/fold_4`), each fine-tuned on the other 4
    folds (fold_epochs = 5).

- **How OOF predictions were generated for the training set:**
  - For each fold, the model trained on the other 4 folds predicts its held-out fold.
  - Every train row receives exactly **one** held-out (out-of-fold) prediction.
  - Deterministic (`model.eval()`, no MC dropout).
  - File: `member2_oof_predictions_deterministic.csv` (4,226 rows).

- **How validation and test predictions were generated using the fold ensemble:**
  - For every validation/test row, all **5** fold models predict deterministically.
  - Final prediction = **mean across the 5 fold models**; `prediction_std_fold_models` =
    std across the 5 models; per-model columns `fold_model_0..4` are also saved.
  - Test was evaluated **once**, after settings were fixed, and was never tuned on.

**Final diagram** (`workflow_final.png`):
```
Accepted handoff data (4,226/452/453/484)
→ hazm.Normalizer
→ ParsBERT tokenizer (max_length=384)
→ grouped fold assignment (5 folds)
→ fine-tune 5 ParsBERT fold models
→ OOF predictions for train
→ 5-fold ensemble predictions for validation/test
→ continuous Member 2 stress prediction
→ handoff to Member C fusion
```

---

## 3. Persian text preprocessing (exact)

| Item | Value |
|------|-------|
| Input text column | `content` |
| Normalization | `hazm.Normalizer` (ي/ك variants, ZWNJ / نیمفاصله, whitespace, punctuation) |
| Empty text | filled with `""` before normalization |
| URLs / emojis / punctuation | **not removed** — kept as-is (the model handles them via subwords; no cleaning step removed them) |
| Tokenizer | `HooshvareLab/bert-fa-zwnj-base` tokenizer (WordPiece) |
| Max token length | **384** (covers ~99.8% of train rows; 99th pct = 262, max = 1071) |
| Truncation / padding | truncation=True, padding="max_length" (to 384) |
| Deliberately avoided | removing stopwords, stemming, or discarding punctuation/emoji — no aggressive cleaning that could lose signal |

---

## 4. Transformer model (exact)

| Setting | Value |
|---------|-------|
| Checkpoint | `HooshvareLab/bert-fa-zwnj-base` |
| Type | **ParsBERT v3** (BERT-based, ZWNJ-aware) |
| Regression head | `BertForSequenceClassification` → `classifier` Linear(hidden=768 → 1) |
| Output values | **1** (continuous stress prediction) |
| Regression target | `final_stress` (≈1–10, continuous) |
| Loss | `weighted_asymmetric_mse`: `training_sample_weight * under_penalty * error²`, `under_penalty=1.75` when error<0 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs (fold models) | 5 per fold |
| Epochs (full model, exploratory only) | up to 6 (not used in final ensemble) |
| Weight decay | 0.01 |
| Warmup / scheduler | linear warmup (10%) → linear decay |
| Gradient clipping | clip_grad_norm = 1.0 |
| Gradient accumulation | none (accumulation steps = 1) |
| Random seed | 42 |
| Mixed precision | none |
| Device | CUDA (Modal A10G) in production; CPU allowed for inference |

---

## 5. Fold training and predictions

- **How each fold model was trained:** on the 4 folds not including its held-out fold,
  with the same weighted asymmetric loss, 5 epochs, AdamW 2e-5.
- **How the held-out fold was predicted:** the trained fold model runs in `eval()` mode on
  the held-out fold rows (deterministic).
- **How the five OOF files were combined:** the 5 per-fold held-out prediction blocks are
  concatenated into one file sorted by `unique_post_id` (each train row appears exactly once).
- **How validation/test predictions were averaged across fold models:** all 5 fold models
  predict each row; the mean is the final prediction, std is `prediction_std_fold_models`.
- **Full-train model:** a full-train model (all 4,226 rows) was trained during development
  and used for the earlier MC-dropout holdout files, but the **final deterministic
  deliverables use the 5-fold ensemble only** (so the official validation labels are not
  reused through the full model's early-stopping checkpoint).
- **Clipping:** the final predictions are **not clipped** to [1,10] (matches the production
  deliverables). Clipping is available in `predict.py` as an explicit optional `--clip_min/--clip_max`.

---

## 6. Prediction outputs

| Purpose | File | Rows | Columns |
|---------|------|-----:|---------|
| OOF (train) | `outputs/predictions/member2_oof_predictions_deterministic.csv` | 4,226 | `unique_post_id, fold, true_stress, prediction` |
| Validation | `outputs/predictions/member2_validation_predictions_fold_ensemble.csv` | 452 | `unique_post_id, model_role, true_stress, prediction, prediction_std_fold_models, fold_model_0..4` |
| Test | `outputs/predictions/member2_test_predictions_fold_ensemble.csv` | 453 | same as validation |

- **Checkpoint / model directories:** `models/fold_0` … `models/fold_4` (each contains
  `config.json`, `model.safetensors`, `tokenizer/`).
- **unique_post_id alignment:** preserved end-to-end; every row is keyed by
  `unique_post_id` and verified against the frozen manifest in the notebook (§2, §11) and
  in `tests/`.

---

## 7. Handoff to Member C

- **Continuous prediction vs embeddings:** Member C received the **continuous prediction**
  (primary scalar stacker) and the **CLS embeddings** were exported as an explicitly-labeled
  exploratory artifact. The primary fusion uses the **prediction** (OOF / fold-ensemble).
- **Why the final fusion used the prediction rather than the full embedding vector:**
  the OOF predictions are stable, leakage-safe scalar features that directly express the
  text model's stress estimate; raw 768-d embeddings would need a consistent per-fold
  representation protocol and are higher-dimensional (overfitting risk in the fusion stage).
- **To predict a completely new post in production:** load `models/fold_0..4` (or the
  full-train `model_checkpoint`), run `hazm.Normalizer` + tokenizer (max_length=384),
  run the 5 fold models in eval mode, and average the predictions. This is implemented in
  `predict.py` (see README). If fold checkpoints are absent, inference is not supported
  (the notebook fails clearly rather than silently using base ParsBERT).

---

## 8. Explainability

- **Methods actually used:** no SHAP / Integrated Gradients / attention was used in the
  production pipeline. Explainability in this project is:
  1. **Model-level error analysis** — residuals by stress band, under/over-prediction rates.
  2. **Gradient-based token attribution** on a few example posts (interpretation only).
- **Stage:** produced **after training**, for interpretation/reporting only — it did not
  guide training or feature selection.
- **Important findings:**
  - The model is systematically biased toward **over-prediction** (~60–67% of cases),
    which is the intended safety profile.
  - **Very High** stress is the hardest class (small support: 25 validation / 26 test) and
    is frequently under-predicted (false reassurance risk).
  - Example token attribution highlights emotionally-loaded words (e.g., Persian words for
    "can't sleep", "have", "stress") — consistent with the model's reliance on lexical cues.

---

## Metrics (final, from `outputs/metrics/`)

| Split | MAE | RMSE | Pearson | Spearman | Accuracy | Macro F1 |
|-------|----:|-----:|--------:|---------:|---------:|---------:|
| OOF (train, 4,226) | 0.985 | — | 0.813 | — | — | — |
| Validation (452) | 0.825 | 1.139 | 0.860 | — | 0.741 | — |
| Test (453) | 0.969 | 1.309 | 0.825 | — | 0.729 | — |

Full per-class precision/recall/F1, support, and confusion matrices are in the notebook
and in `outputs/metrics/*.json`.
