# Member 1 workflow

1. **Resolve the project root and environment.** Folder names may change; the notebook
   locates the root through the configuration and `src/` directories.
2. **Validate the frozen inputs.** Check IDs, roles, targets, folds, groups, weights,
   schemas, and forbidden-feature rules before modeling.
3. **Quarantine embargo.** Embargo rows are retained for audit but are not transformed,
   fitted, or evaluated.
4. **Recreate the exact 67 structured features.** Apply deterministic missing-value,
   Jalali-date, cyclic-time, log, ratio, categorical, signature, and surface-count rules.
5. **Retain the supplied grouped folds.** No new random split is created; author, thread,
   duplicate-content, and component leakage barriers are checked.
6. **Train or load five CatBoost models.** Each model fits four folds using the supplied
   weights and a fixed 558-tree configuration.
7. **Generate OOF train predictions.** Each training row is predicted exactly once by the
   model that held its fold out.
8. **Generate validation/test ensembles.** Every holdout row is predicted by all five
   models; the mean is the delivered score and the standard deviation is disagreement.
9. **Export Member C handoffs.** Save scalar predictions and exact engineered feature
   matrices aligned by `unique_post_id`.
10. **Evaluate.** Report MAE/RMSE/correlation plus ordered-class precision/recall/F1,
    macro F1, confusion matrices, and Very-High PR-AUC.
11. **Explain after training.** Generate global CatBoost feature importance; optionally
    run native SHAP. Explanations do not select the frozen features.
12. **Verify and test.** Check models, files, row/ID coverage, prediction ranges, schemas,
    and reusable code invariants.
