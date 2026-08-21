# Member 1 — Final Reproducible Tabular Project

This folder reproduces the complete Member 1 workflow from the frozen Member C handoff through leakage-safe feature engineering, five-fold CatBoost training, OOF prediction generation, validation/test fold-ensemble prediction, evaluation, explainability, and fusion handoff export.

## Member 1 role

Member 1 models tabular behavioral, demographic, interaction, post-count, and signature-count signals. Raw `content` and `thread_title` are never passed to CatBoost as text. `stress_proxy` and all target/split/weight columns are forbidden model features.

## Pre-active-learning workflow

Earlier experiments used smaller train/validation/test CSV files and explored LightGBM/CatBoost, weighting, and threshold calibration. Those experiments are historical only and are not used by this final project.

## Final post-active-learning workflow

The final project uses only the frozen handoff and manifest:

- 4,226 train rows with supplied grouped folds 0–4;
- 452 official validation rows;
- 453 locked test rows;
- 484 embargo rows that are validated but never transformed, trained on, or predicted for.

Each fold model trains on four supplied folds and predicts the held-out fold. Validation and test predictions are the mean of all five fold models.

## Install on Windows with Python 3.12

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Primary execution entry point: `notebooks/Member1_Full_Reproduction.ipynb` (run all cells from top to bottom).

## Execution modes

The notebook contains:

```python
FULL_RETRAIN = True
```

- `True`: retrains and overwrites all five fold models and the optional full-train model.
- `False`: loads the included `.cbm` models and reproduces OOF/validation/test predictions, metrics, explainability, and handoff files.

No command-line scripts must be run manually. The notebook is the main entry point.

## Expected runtime and resources

- CPU: 4 logical threads configured; GPU is not required.
- Memory: approximately 2–4 GB.
- Full retrain: typically a few minutes on a modern laptop CPU.
- Load-and-reproduce mode: usually under two minutes.

## Required inputs

All five accepted small input files are included under `inputs/`. See `inputs/README.md` and `docs/data_contract.md`.

## Main generated outputs

- `data/handoff/member1_oof_predictions.csv`
- `data/handoff/member1_validation_predictions.csv`
- `data/handoff/member1_test_predictions.csv`
- engineered feature handoff tables for train/validation/test
- five fold `.cbm` models under `models/fold_0` ... `models/fold_4`
- metrics, confusion matrices, figures, CatBoost feature importance, and optional native CatBoost SHAP outputs

## Member C fusion use

Use `unique_post_id` as the only alignment key. Training fusion rows must consume `member1_oof_predictions.csv`; validation and test use the corresponding five-model ensemble files. Never replace OOF predictions with in-sample predictions.

## Future website-post inference

```bat
python -m src.predict_new_posts --input inputs\future_posts.csv --package-dir . --output outputs\predictions\future_member1_predictions.csv
```

The future input must already contain the upstream post/signature signal columns specified in `data/processed/member1_preprocessor_config.json`. The original positive/negative lexicon extractor was not included in the accepted handoff, so raw-text-only inference is not claimed.

## Limitations

- The tabular-only component does not meet every final-fusion internal recall floor.
- The supplied count/signal columns depend on an upstream extractor not included in the handoff.
- The project was generated in Python 3.13, while requirements are chosen for Python 3.12 compatibility; the prediction CLI has a stateless transformer fallback if joblib loading fails.
- CatBoost training can show tiny floating-point differences across hardware/thread scheduling, while ID/fold/prediction contracts remain reproducible.

## Large files excluded

No required fold model is excluded. All five comparatively small CatBoost `.cbm` files are included. Raw historical datasets and unrelated exploratory notebooks are excluded.
