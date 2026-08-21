# Deployment input contract

The primary hybrid model requires:

1. `member1_prediction`: mean prediction from Member 1's five fold models.
2. `member2_prediction`: deterministic mean prediction from Member 2's five fold models.
3. The raw/enriched metadata and predefined count fields required by
   `Member1FeatureEngineer`.

The project does not contain Member 2's 2.5 GB fold checkpoints or the missing historical
lexical extractor used to create some Member 1 signal counts. Preserve those external
assets.

For an already enriched row:

```powershell
py -3.12 .\src\predict_fusion.py `
  --input .\templates\fusion_input_template.csv `
  --output .\outputs\new_fusion_predictions.csv `
  --project-root .
```

The scalar fallback needs only the two base prediction columns:

```powershell
py -3.12 .\src\predict_fusion.py `
  --input .\my_base_predictions.csv `
  --output .\outputs\fallback_predictions.csv `
  --profile scalar `
  --project-root .
```
