# Frozen data inputs

The included files are the accepted leakage-safe handoffs.

Do not replace validation/test inputs with predictions from a full-training base model.
The fusion distribution is defined as:

- Train: held-out OOF prediction
- Validation: mean of five fold models
- Test: locked mean of five fold models

The test files are included for exact project reproduction. They must not be used for
further model or threshold tuning.
