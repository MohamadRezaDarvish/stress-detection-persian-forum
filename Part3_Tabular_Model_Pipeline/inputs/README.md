# Required inputs

Place these exact files in this directory:

- `member1_handoff.csv.gz` — accepted Member 1 handoff, about 1.2 MB compressed.
- `modeling_manifest_v2.csv` — frozen role/fold/group manifest, about 1.8 MB.
- `member1_feature_contract.json` — forbidden-feature contract.
- `oof_predictions_template.csv` — required OOF output schema.
- `holdout_predictions_template.csv` — required validation/test output schema.

The notebook stops with a clear error if any file or required column is missing. Do not edit IDs, roles, labels, groups, folds, or weights.
