# Part 4 — Fusion, calibration, and final evaluation

This project integrates the accepted outputs from Member 1 and Member 2 and completes:

- leakage-safe fusion data preparation,
- candidate stacker comparison,
- constrained threshold calibration,
- one-time locked test evaluation,
- global and local explainability support,
- deployable prediction CLI,
- temporal user-risk aggregation,
- active-learning acquisition scoring,
- fusion-level OOF generation,
- reproducibility tests and artifact validation.

## Selected model

`hybrid_catboost_rmse_depth3`

The primary model is a shallow hybrid CatBoost regressor using the two leakage-safe base
predictions plus Member 1's predefined tabular features.

Thresholds:

```text
Low / Moderate:      3.095439
Moderate / High:     4.905161
High / Very High:    6.462135
```

## Gate result

Validation passed all minimum recalls:

- Low: 84.51%
- Moderate: 58.21%
- High: 52.38%
- Very High: 76.00%

The locked test passed Low, High, and Very High, but Moderate recall was only
41.79%. Therefore **3 of 4 internal project recall targets were met**. Moderate remains
the main improvement area.

The separate deployment interpretation is **candidate for a limited human-in-the-loop
risk-monitoring overlay**. This is not production/clinical approval and not autonomous
diagnosis or emergency decision-making.

## Recommended execution — notebook

Execution environment: install the listed requirements in Jupyter/Colab, then open:

`notebooks/MemberC_Phases_4_8_Full_Reproduction.ipynb`

Run the notebook from top to bottom with **Run All**. It reproduces the fusion preparation,
candidate comparison, validation-only calibration, fusion OOF generation, the already-recorded
locked-test evaluation, explainability outputs, monitoring utilities, active-learning demonstration,
and report figures.

No shell or PowerShell launcher is required.

## Predict enriched new posts

The input needs the accepted Member 1/Member 2 fold-ensemble predictions and Member 1's
upstream feature columns:

```powershell
py -3.12 .\src\predict_fusion.py `
  --input .\templates\fusion_input_template.csv `
  --output .\outputs\new_fusion_predictions.csv `
  --project-root .
```

Add local SHAP explanations:

```powershell
py -3.12 .\src\predict_fusion.py `
  --input .\templates\fusion_input_template.csv `
  --output .\outputs\new_fusion_predictions.csv `
  --shap-output .\outputs\new_fusion_shap.csv `
  --project-root .
```

## Principal outputs

- `models/fusion_selected.cbm`
- `models/fusion_bundle.json`
- `outputs/candidate_leaderboard.csv`
- `outputs/validation_metrics.json`
- `outputs/test_metrics.json`
- `outputs/final_metrics.json`
- `outputs/fusion_oof_predictions.csv`
- `outputs/global_feature_importance.csv`
- `outputs/validation_shap_summary.csv`
- `docs/model_card.md`
- `docs/test_lock_report.md`

## External assets to retain

This package intentionally does not contain:

- Member 2's five fold checkpoints, approximately 2.5 GB.
- Member 2's full 500 MB checkpoint.
- Member 1's missing historical lexical-resource extractor.

Those assets are not needed to review or reproduce the fusion training from the included
prediction files. They are needed for fully end-to-end website inference from raw posts.

## Report and presentation figures

Generate the full visual package with:

```powershell
py -3.12 .\src\generate_report_figures.py --project-root .
```

Outputs are written to `outputs/report_figures` in both PNG and SVG formats. The package
includes row-normalized confusion matrices, score distributions with calibrated
thresholds, before/after active-learning MAE, class-colored scatter plots, recall and
precision/recall/F1 summaries, threshold trade-offs, explainability charts, error
diagnostics, candidate comparison and an end-to-end system diagram.

See `docs/report_figure_guide.md` for recommended captions and which figures belong in
the main report versus the appendix.

## Full executable Member C notebook

Open:

`notebooks/MemberC_Phases_4_8_Full_Reproduction.ipynb`

The notebook has been executed and contains saved outputs. It creates a fresh
`notebook_run/` directory when run from the beginning.
