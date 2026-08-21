# Part 2 — Label and leakage-safe data foundation

Part 2 sits between the website-scale scraping/cleaning work (Part 1) and the two base models (Part 3).

The primary executable notebook is:

`notebooks/MemberC_Phases_1_3_Full_Reproduction.ipynb`

Historical source/module names still use Phase 1, 2, 3 and 3.1 for artifact provenance.

## What Part 2 produces

- 5,617 canonical labels before enrichment;
- 5,615 safely enriched rows after excluding two unresolved ambiguous duplicate-ID matches;
- 3,622 author/thread/exact-content connected components;
- final model roles: 4,226 train / 452 validation / 453 test / 484 embargo;
- five grouped OOF folds: 844 / 844 / 848 / 847 / 843;
- frozen `training_sample_weight`;
- Member 1 and Member 2 handoff files.

Only strong dual-label sources define official validation/test ground truth. Weaker/calibrated sources can support training with provenance-aware weights.

## External raw-data option

The large cleaned website corpus is not required for the default compact reproduction because the accepted Phase 2 enrichment is included.

To re-stream enrichment from the raw cleaned corpus, provide the external CSV through the primary notebook's expected path/environment and rerun all cells.

Primary execution entry point: `notebooks/MemberC_Phases_1_3_Full_Reproduction.ipynb`; execution proceeds from top to bottom in Jupyter/Colab.

## Evaluation interpretation

The validation/test sets are safety-enriched and do not estimate natural Ninisite prevalence. Natural-stream alert precision and reviewer workload require a separate unselected replay/sample.

## Internal recall objectives used later in Part 4

- Low ≥ 75%
- Moderate ≥ 50%
- High ≥ 50%
- Very High ≥ 75%

These are project evaluation objectives, not clinical/regulatory standards.


## How to run

Open:

`notebooks/MemberC_Phases_1_3_Full_Reproduction.ipynb`

Execution mode: **Run All**. No `.ps1`, `.sh`, or standalone checksum manifest is required.
