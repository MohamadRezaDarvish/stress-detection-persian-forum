# Data Contract — Member 2 inputs

## Required input files (`inputs/`)

| File | Required | Description |
|------|----------|-------------|
| `member2_handoff.csv` | yes | post content, labels, roles, folds, weights |
| `modeling_manifest_v2.csv` | yes | IDs, groups, hashes, split flags, fold assignments |
| `oof_predictions_template.csv` | yes (schema reference) | header-only template for OOF output |
| `holdout_predictions_template.csv` | yes (schema reference) | header-only template for holdout output |

## Key columns used

From `member2_handoff.csv`:
- `unique_post_id` — unique row identifier (join key)
- `content` — raw Persian text (only model input)
- `final_stress` — continuous target (≈1–10)
- `clinical_class` — authoritative class label (Low/Moderate/High/Very High)
- `model_role` — train / validation / test / embargo
- `oof_fold` — 5-fold assignment (0–4, train only)
- `training_sample_weight` — per-sample loss weight

From `modeling_manifest_v2.csv` (joined on `unique_post_id`):
- `group_id`, `author`, `thread_id`, `content_hash` — grouping/leakage checks
- `use_for_training`, `use_for_model_selection`, `use_for_final_test`, `exclude_from_modeling`

## Roles and sizes (frozen)

| Role | Rows |
|------|-----:|
| train | 4,226 |
| validation | 452 |
| test | 453 |
| embargo | 484 |

## Fold sizes (train)

| Fold | Rows |
|------|-----:|
| 0 | 844 |
| 1 | 844 |
| 2 | 848 |
| 3 | 847 |
| 4 | 843 |

## Validations enforced (see `src/validate_inputs.py` and `tests/`)

- required columns present
- `unique_post_id` present, non-null, unique
- `final_stress` present for labeled roles; in expected range [1, 10]
- `model_role` values are exactly {train, validation, test, embargo}
- train rows have a fold in {0..4}; non-train rows have no fold
- no ID overlap between train / validation / test / embargo
- `clinical_class` present; consistency with stress thresholds is **reported** but not
  enforced (the column is authoritative and contains minor boundary noise)

## Column policy (text-only)

Only `content` is used as model input. `stress` (`final_stress`) is the target.
**Explicitly ignored** for modeling: `anxiety`, `depression`, `stress_proxy`, all
`sig_*`/`post_*` counts, demographics, timestamps, thread metadata, and all other
tabular columns (Member 1's domain).
