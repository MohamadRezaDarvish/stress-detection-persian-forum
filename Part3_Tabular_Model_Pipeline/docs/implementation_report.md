# Implementation report

The final Member 1 implementation replaces exploratory row-random splits with the frozen
Member C leakage-safe roles and grouped folds. A stateless feature engineer reconstructs
67 features in one fixed order from structured metadata and predefined count signals.
Raw `content`, `thread_title`, `stress_proxy`, targets, IDs, roles, folds, and weights are
forbidden model inputs.

Five weighted `CatBoostRegressor` models are trained. For fold *k*, the model fits the
other four folds and predicts only fold *k*, creating honest out-of-fold predictions for
all 4,226 training rows. Validation and test predictions are the mean of all five fold
models, with between-model standard deviation retained as a disagreement diagnostic.
These files are joined to Member C by `unique_post_id`.

CatBoost was selected because the feature matrix mixes numeric, categorical, missing,
skewed, and nonlinear signals. It handles categorical values and numeric NaNs natively
and can learn interactions without a large one-hot preprocessing pipeline. The frozen
model uses 558 trees, depth 8, learning rate 0.025, L2 regularization 10, random strength
0.3, RMSE loss, and the supplied `training_sample_weight`.

The accepted locked-test MAE is 1.2889. This tabular branch is weaker than the Transformer
and final fusion, but its structured predictions/features remain complementary. The
executed package generates averaged CatBoost global feature importance. Native CatBoost
SHAP is implemented as an optional post-hoc explanation and was disabled in the delivered
executed run; explanation was not used for formal feature selection.

The model is a research fusion component, not a stand-alone clinical diagnostic system.
The class recall floors are internal project objectives rather than medical standards.
