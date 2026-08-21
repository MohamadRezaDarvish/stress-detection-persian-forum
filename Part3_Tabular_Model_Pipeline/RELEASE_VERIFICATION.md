# Release verification

- Main executed notebook: `notebooks/Member1_Full_Reproduction_executed.ipynb`
- Total notebook cells: 27
- Successfully executed code cells: 13/13
- Execution errors: 0
- Full retrain mode executed: yes
- Fold models included: 5
- Artifact verification: passed
- Lightweight tests: 3 passed
- Required large files excluded: none
- Explainability executed: CatBoost global feature importance. Native CatBoost SHAP is implemented and can be enabled with `MEMBER1_RUN_SHAP=true`; it was disabled in the delivered executed run to keep runtime practical.
- Not supplied by the accepted handoff: the original upstream positive/negative lexicon extractor.
