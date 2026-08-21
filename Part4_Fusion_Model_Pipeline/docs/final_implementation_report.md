# Final fusion implementation report

## Completed phases

- Base-model handoff integration
- Fusion candidate training and comparison
- Hard-constrained threshold calibration
- Locked test evaluation
- Global CatBoost importance and SHAP analysis
- Fusion-level OOF predictions
- Production inference CLI
- Temporal risk-state scaffold
- Active-learning acquisition code
- Reproducibility tests and artifact validation

## Selected model

`hybrid_catboost_rmse_depth3`

Validation:

- MAE: 0.8348
- Pearson: 0.8577
- Macro F1: 0.6717
- Recalls: {'Low': 0.8451178451178452, 'Moderate': 0.582089552238806, 'High': 0.5238095238095238, 'Very High': 0.76}

Locked test:

- MAE: 0.9174
- Pearson: 0.8333
- Macro F1: 0.6284
- Recalls: {'Low': 0.8754208754208754, 'Moderate': 0.417910447761194, 'High': 0.5238095238095238, 'Very High': 0.7692307692307693}

# Interpretation of the Final Result

The final fusion system met all four project-defined recall objectives on the validation set.

On the one-time locked test set, it met **three of the four** internal recall targets:

| Stress Level | Recall |
|--------------|--------|
| Low | **87.54%** |
| Moderate | **41.79%** |
| High | **52.38%** |
| Very High | **76.92%** |

The **Moderate** class required **34 correct predictions out of 67** to achieve the project's **50%** recall target. The model correctly classified **28** Moderate posts.

These recall thresholds were **internal project objectives** and should **not** be interpreted as formal clinical, regulatory, or universal real-world deployment criteria.

The primary safety objective of the project was to identify severe **Very-High** stress cases. On the locked test set, the final model correctly detected **20 of 26 Very-High** posts, thereby achieving the project's internal Very-High recall objective. The corresponding **Very-High precision** was **62.50%**.

## Implementation Status

```text
candidate_for_human_in_the_loop_risk_monitoring_overlay
```

---

# Proposed Real-World Use

The model is intended to operate as an **overlay** on the discussion website.

For every new post, the system can:

1. calculate predictions from **Member 1** and **Member 2**;
2. produce a fused continuous stress score;
3. update the user's private longitudinal risk profile;
4. evaluate immediate, sustained, repeated, and escalating risk rules;
5. place high-priority users into a private human-review queue when appropriate.

---

# Example Prioritization Conditions

A user may be prioritized when:

- a single post receives a strongly predicted Very-High score;
- several recent posts are classified as High or Very High;
- the user's time-decayed mean stress remains elevated;
- the estimated risk increases rapidly relative to the user's previous baseline;
- the base models strongly disagree near an important decision boundary.

The system is designed to **prioritize human attention**, not to autonomously diagnose mental-health conditions or perform automatic interventions.

A trained reviewer should determine whether the appropriate response is:

- supportive information;
- private contact;
- professional review; or
- another action performed under an approved operational protocol.

---

# Moderate-Class Limitation

The **Moderate** class remains the primary technical weakness of the model.

Moderate posts were confused in **both directions**, indicating overlapping score distributions around both adjacent decision thresholds rather than a single threshold error.

The operational overlay can reduce the impact of this limitation by using:

- continuous stress scores;
- aggregation across repeated posts;
- recent maximum risk;
- user-specific temporal trends;
- disagreement between the base models;
- human review of recent posting context.

Consequently, a user may still be identified as requiring attention because of sustained or increasing stress, even if an individual post is not classified as Moderate.

---

# Future Work

The next development cycle should:

- perform qualitative analysis of incorrectly classified Moderate posts;
- collect new dual-annotated samples near both Moderate-class boundaries;
- build a new untouched confirmation dataset;
- evaluate ordinal safety outputs;
- assess temporal alert rules using historical replay experiments;
- measure reviewer workload and false-alert burden;
- conduct a limited human-in-the-loop pilot study;
- document privacy, consent, and escalation procedures.

---

# Evaluation Protocol

The locked test set **must not** be used for additional model tuning.

Any improved model should instead be evaluated using a **new, previously unseen evaluation dataset**.
