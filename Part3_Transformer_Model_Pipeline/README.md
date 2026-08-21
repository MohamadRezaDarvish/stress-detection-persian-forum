# Member 2 — Final Reproducible Project: Persian Text → Continuous Stress Regression

Member 2 (NLP Expert) fine-tunes **ParsBERT** on raw Persian forum text to predict the
continuous `stress` score and hands OOF + fold-ensemble predictions to Member C for the
final fusion model.

This is a **complete, reproducible project**: run the single main notebook from start to
finish to reproduce every artifact.

---

## 1. Role in the project

- **Text-only:** model input is `content`; target is `final_stress`; `unique_post_id`
  is the join key. All tabular/metadata columns (Member 1's 45 features, `stress_proxy`,
  `anxiety`, `depression`) are excluded.
- Deliverable: continuous predictions (OOF for train, fold-ensemble for validation/test)
  and CLS embeddings for Member C's fusion model.

## 2. Pre-active-learning workflow

Initial labeled split (1,561/335/335) → `hazm.Normalizer` → ParsBERT tokenizer
(max_length=256) → fine-tuned ParsBERT regression → continuous predictions → first-level
fusion with Member 1 → active-learning candidate selection → new annotations → updated
dataset. See `docs/workflow_pre_active_learning.png`.

## 3. Final post-active-learning workflow

Frozen split (4,226/452/453/484) → `hazm.Normalizer` → tokenizer (max_length=384) →
grouped fold assignment (5 supplied folds) → fine-tune 5 fold models → OOF predictions for
train → 5-fold ensemble predictions for validation/test → handoff to Member C.
See `docs/workflow_final.png`.

## 4. Directory structure

```
Member2_Final_Reproducible_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/Member2_Full_Reproduction.ipynb   <- MAIN ENTRY POINT
├── configs/project_config.json, model_config.json, paths.example.json
├── src/*.py                                      <- reusable pipeline modules
├── inputs/                                       <- required input CSVs
├── data/processed, data/folds, data/handoff      <- generated
├── models/fold_0..4, model_manifest.json         <- fine-tuned fold checkpoints
├── outputs/predictions, metrics, figures, explainability, logs
├── docs/*.md, workflow_*.png                     <- documentation + diagrams
└── tests/*.py                                    <- contract tests
```

## 5. Required input files (`inputs/`)

- `member2_handoff.csv`
- `modeling_manifest_v2.csv`
- `oof_predictions_template.csv`
- `holdout_predictions_template.csv`

See `docs/data_contract.md` for columns and validation.

## 6. Installation

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## 7. Main notebook execution

Open `notebooks/Member2_Full_Reproduction.ipynb` and run all cells (Kernel → Restart &
Run All).

- `FULL_RETRAIN = False` (default): loads the fine-tuned fold checkpoints from `models/`.
- `FULL_RETRAIN = True`: retrains all 5 fold models (requires GPU).
- `PREDICT_BACKEND`:
  - `"modal"` — runs predictions on Modal A10G (production environment; needs Modal auth).
  - `"cpu"` — runs locally on CPU (slow: ~1 hour for all predictions).
  - `"load"` — loads already-generated predictions from `data/handoff/`.
  - Default: auto (modal → load → cpu).

## 8. Expected execution time

- `FULL_RETRAIN=False` + `backend="load"`: **~1–2 minutes** (no GPU needed).
- `FULL_RETRAIN=False` + `backend="modal"`: **~4–8 minutes** (GPU, incl. model load).
- `FULL_RETRAIN=True` + GPU: **~1–2 hours** (5 × 5-epoch ParsBERT fold trainings).

## 9. CPU/GPU and memory requirements

- CPU-only is fine for `FULL_RETRAIN=False` with `backend="load"` (predictions are
  pre-generated in `data/handoff/`).
- `backend="modal"` requires Modal credentials + A10G quota.
- `FULL_RETRAIN=True` requires a GPU with ≥16 GB VRAM (A10G); ~32 GB RAM recommended.
- Each fold checkpoint is ~474 MB (5 × ~474 MB = ~2.4 GB on disk in `models/`).

## 10. Expected generated outputs

- `outputs/predictions/member2_oof_predictions_deterministic.csv` (4,226 rows)
- `outputs/predictions/member2_validation_predictions_fold_ensemble.csv` (452)
- `outputs/predictions/member2_test_predictions_fold_ensemble.csv` (453)
- `outputs/metrics/{oof,validation,test}_metrics.json`
- `outputs/figures/*.png` (confusion matrices, scatter plots)
- `outputs/explainability/*.json`
- `outputs/logs/artifact_checksums.json`
- `data/processed/processed_manifest.json`

## 11. How Member C should use the handoff files

Read `docs/memberC_handoff_contract.md`. Join everything by `unique_post_id`; use
`prediction` from the three CSVs as the text-model feature for the stacker;
`prediction_std_fold_models` is an optional uncertainty feature.

## 12. Limitations

- Very-High stress is under-predicted (small support; safety bias).
- Text-only; tabular fusion (Member 1) is expected to improve tail recall.
- Deterministic predictions are not clipped to [1,10] (matches production; optional
  clipping available in `predict.py`).
- Embeddings are an exploratory artifact (OOF embeddings come from per-fold models).

## 13. Large files excluded from a ZIP

The fine-tuned fold checkpoints (`models/fold_0..4`, ~2.4 GB total) may be too large for
a distribution ZIP. See `models/model_manifest.json` for the exact manifest and placement
instructions.

## 14. How to obtain / place the excluded checkpoints

Option A — from this machine: copy `models/fold_0..4` as-is.
Option B — from the Modal volume `persian-stress-vol`:
```bash
modal volume get persian-stress-vol outputs_final/fold_checkpoints ./models
```
Option C — reproduce them: set `FULL_RETRAIN=True` in the notebook and run on a GPU
(writes into `models/`).

If the fold checkpoints are missing, the notebook **fails with a clear error** rather than
silently using the base ParsBERT model.

---

## New-post production inference

```bash
# deterministic single prediction
python predict.py --text "من خیلی استرس دارم و نمیتونم بخوابم"

# batch + embeddings
python predict.py --csv new_posts.csv --text_col content --id_col unique_post_id \
    --out new_predictions.csv --save_embeddings

# MC-dropout uncertainty
python predict.py --csv new_posts.csv --text_col content --out pred_mc.csv --mc_trials 8
```

`predict.py` requires `models/fold_0..4` (or a `model_checkpoint/` with the full model).
