# Input files

Place the following files in this folder. They are the frozen Member C handoff data.

| Filename | Expected | Size (approx.) | Required columns |
|----------|----------|----------------|------------------|
| `member2_handoff.csv` | yes | ~2.9 MB | `unique_post_id, final_stress, clinical_class, model_role, oof_fold, training_sample_weight, content, ...` |
| `modeling_manifest_v2.csv` | yes | ~1.9 MB | `unique_post_id, group_id, author, thread_id, content_hash, oof_fold, model_role, use_for_* , exclude_from_modeling, ...` |
| `oof_predictions_template.csv` | yes | 67 B | header: `unique_post_id,fold,true_stress,prediction,prediction_std_optional` |
| `holdout_predictions_template.csv` | yes | 73 B | header: `unique_post_id,model_role,true_stress,prediction,prediction_std_optional` |

The notebook validates these files before any training (see `src/validate_inputs.py` and
`tests/test_input_contract.py`). If a required file is missing, the notebook raises a
clear error and stops.

These files are excluded from version control and from any large-file ZIP instructions;
copy them from the shared drive / Member C handoff.
