# Legacy notebook mapping

The clean project was reconstructed from the following historical notebooks:

- `ninisitescraper2.ipynb` — forum/thread/profile scraping, blocking recovery, checkpoints, duplicate accumulation lessons.
- `New Text Document.ipynb` — missing-metadata repair passes, especially `user_post_count`.
- `cleaner.ipynb` — core cleaning, education mapping, age extraction, safe IDs, signature features.
- `cleaner2.ipynb` — gender recovery/imputation and duplicate-ID investigation.
- `cleaner3.ipynb` — post linguistic features, stress-proxy bootstrapping, differential lexicon induction, sampling experiments, annotation templates.

Important historical caveats preserved here:

- temporary seed features are kept rather than overwritten;
- `is_starter` is not a proxy feature;
- the final lexicon selection used frequency ratios, not simple count subtraction;
- naive `unique_post_id` deduplication was found capable of deleting distinct rows, so conflict auditing is retained;
- the exact historical final 2,000-row sampling cell cannot be proven solely from notebook execution metadata.


## Neutral names for the historical gender artifacts

The original exploratory notebook used several informal/conflict-related
intermediate filenames. The final project does not use those names.

Historical artifacts are exposed with neutral names:

- `key_to_url.pkl`
  → `inputs/enrichment/gender_recovery_key_to_profile_url.csv`
- `gender_scrape_fresh.pkl`
  → `inputs/enrichment/gender_recovery_results.csv`

The old names may still appear inside archived legacy notebooks because those
notebooks are preserved as historical evidence, but no final pipeline component
depends on them.
