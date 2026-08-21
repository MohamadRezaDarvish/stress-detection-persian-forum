# Model card — Member 1 structured CatBoost branch

**Model family:** five `CatBoostRegressor` fold models.

**Task:** continuous regression of `final_stress` from 67 ordered structured features.
Raw post text is not modeled.

**Frozen configuration:** 558 iterations, depth 8, learning rate 0.025, L2 leaf
regularization 10, random strength 0.3, weighted RMSE, CPU training.

**Training protocol:** each model trains on four supplied leakage-safe folds with
`training_sample_weight`; the held-out fold produces OOF predictions. Validation and test
predictions are arithmetic means of the five fold models.

**Why CatBoost:** native mixed numeric/categorical handling, native numerical missing
values, nonlinear interactions, and strong performance on small-to-medium tabular data.

**Accepted results:** OOF MAE 1.2663, validation MAE 1.2657, locked-test MAE 1.2889.

**Use:** complementary structured branch and input to Member C late fusion when all
upstream metadata/count columns are supplied.

**Not for:** stand-alone diagnosis, natural-prevalence estimation, or autonomous action.
The tabular model does not meet all internal final-project recall objectives.

**Limitations:** the original upstream positive/negative lexicon extractor is not included;
exact raw-post-only inference therefore requires the upstream signal-generation service.
