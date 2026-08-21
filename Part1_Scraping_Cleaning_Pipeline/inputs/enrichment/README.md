# Historical gender-recovery artifacts

These files are the cleaned, human-readable versions of the historical
`key_to_url.pkl` and `gender_scrape_fresh.pkl` artifacts.

They are used only to **replay the completed historical gender repair**.
Notebook 02 does not make network requests.

## Files

- `gender_recovery_key_to_profile_url.csv`
  - `10,826` cleaned-row join keys.
  - `2,418` unique profile URLs.
  - Maps `thread_id + author + posted_at` back to the original `profile_url`.

- `gender_recovery_results.csv`
  - `2,418` unique profile URLs.
  - Historical fresh re-scrape results:
    - male (`مرد`): 50
    - unresolved (`unknown`): 2,368
    - explicitly detected female (`زن`): 0

## Historical interpretation

During development, the first gender parser used the wrong HTML assumption.
After the cleaned intermediate data had already dropped `profile_url`, the team:

1. reconstructed a stable join key from `thread_id + author + posted_at`;
2. used the original `combined_all.csv` to recover the lost profile URLs;
3. re-scraped only the unresolved profiles;
4. explicitly recovered male profiles;
5. treated the still-unresolved profiles as female as a documented project
   imputation, because the pregnancy-forum population was overwhelmingly female.

The final reproducible cleaning notebook replays these cached artifacts and
records provenance. It does **not** scrape profiles again.

The improved scraping notebook now recognizes the known female and male marker
families directly, so this recovery procedure should not be needed in a fresh run.


## Category recovery

`category_recovery.csv` is derived from the three historical source CSVs that lacked
category columns. The label pair is determined by source-file identity exactly as in
`cleaner.ipynb`, and rows are keyed by `thread_id + author + posted_at`.

The large original source files are intentionally not bundled because this compact cache
is sufficient to reproduce the completed historical repair.
