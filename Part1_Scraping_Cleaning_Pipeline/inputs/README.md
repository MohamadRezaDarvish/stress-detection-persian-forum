# Part 1 input contract

## Primary historical raw input

Expected path:

`inputs/combined_all.csv`

Approximate size: ~330 MB.

Expected schema includes:

`thread_id, thread_title, post_id, author, profile_url, posted_at, content, likes, reply_to, is_starter, user_post_count, gender, gender_code, join_date, user_age, education, status, children_count, signature, thread_url, category, sub_category`

This is the preferred input for `notebooks/02_Cleaning_Reproducible.ipynb`.

Processed outputs belong under `data/processed/`, not under `inputs/`.

## Live-scraping inputs

`inputs/forum_sections.csv` defines the forum/sub-category listing URLs used by the scraping notebook. Live scraping is disabled by default through `configs/project_config.json`.

When live scraping is enabled, resumable post/profile checkpoints are written under `data/raw/`.

## Historical enrichment inputs

Compact replay artifacts required for the historically faithful cleaning run are stored under:

`inputs/enrichment/`

These include:

- `category_recovery.csv` - frozen mapping for the 99,442 historical category/sub-category repairs;
- `historical_male_profile_cache.pkl` - reproducibility replay of the 1,437 historical male-profile results;
- `historical_male_profile_urls.csv` - corresponding profile URL set;
- `gender_recovery_key_to_profile_url.csv` and `gender_recovery_results.csv` - legacy recovery support files.

The three large historical category-backfill source CSVs are not required by the finalized reconstruction because their completed mapping is already frozen in `category_recovery.csv`.

## Optional repair file

Recovered `user_post_count` values, when needed for a raw scrape, use:

`data/raw/repairs/user_post_count_repair.csv`

## Data-flow summary

`inputs/combined_all.csv` -> Notebook 02 -> `data/processed/ninisite_cleaned_enriched.csv` -> Notebook 03 -> `data/processed/ninisite_cleaned_stress_proxy.csv`
