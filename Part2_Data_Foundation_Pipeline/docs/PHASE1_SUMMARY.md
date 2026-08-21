# Phase 1 execution summary

Phase 1 was executed against the uploaded annotation files and the compact ID/stress index reconstructed from the historical fusion splits.

## Canonical data

- Original dual-labeled rows: **2,000**
- Original dual-labeled rows retained in the old fusion data: **1,607**
- Original dual-labeled rows removed by the old undersampling step: **393**
- Recovered trusted-Person-1-only high-selected rows: **622**
- Active-learning rows: **3,000**
  - Real dual labels: **1,000**
  - Person-2-only labels calibrated through the overlap: **2,000**
- IDs repeated across the original and active-learning rounds: **5**
- Final canonical unique posts after duplicate-ID resolution: **5,617**

The five repeated IDs are not duplicated in the canonical table. Their earlier dual labels were retained because they have stronger provenance; all repeat annotation events remain in `annotation_history.csv.gz`.

## Canonical class distribution

Using fixed true-class bins `<3`, `3–<5`, `5–<7`, and `>=7`:

- Low: **3,029**
- Moderate: **1,222**
- High: **974**
- Very High: **392**

## Active-learning label reconstruction

The historical 3,000-row targets were reproduced exactly.

For the 1,000 overlap rows:

`consensus = (1.7 * Person1 + Person2) / 2.7`

For the remaining 2,000 rows, the historical process is exactly equivalent to:

1. Fit OLS on the 1,000 overlap rows:
   `estimated_Person1 = 0.6459265243 * Person2 + 0.5037576412`
2. Apply the same weighted consensus:
   `final = (1.7 * estimated_Person1 + Person2) / 2.7`

This produces:

`final = 0.7770648486 * Person2 + 0.3171807371`

Five-fold cross-validation on the overlap gave MAE **0.3913** against the observed weighted consensus. Isotonic/ordinal lookup gave MAE **0.3978**, so the historical linear method is retained.

## Duplicate policy

- Duplicate IDs across rounds are resolved by label provenance, not by row order.
- Every annotation event remains in the history table.
- Exact duplicate text receives a shared `content_hash` and must stay within one split.
- Phase 2 never performs an unchecked many-to-many merge.


## High-confidence evaluation pool available after enrichment

Using only real dual annotations (`original_dual` and `active_dual`), the pool contains:

- Low: **1,971**
- Moderate: **445**
- High: **417**
- Very High: **167**

This is sufficient to build safety-focused validation/test sets without using the two single-rater groups as holdout ground truth.
