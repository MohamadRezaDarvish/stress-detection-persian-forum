# Part 1 - Scraping and Cleaning Pipeline

Part 1 reconstructs the pre-modeling data pipeline from the historical project code in three notebook-driven stages:

1. **Scraping** - forum listings -> thread manifests -> paginated posts -> profile metadata, with retries, checkpoints, resumable execution, and recovery passes.
2. **Cleaning** - combine raw files, filter unusable/short posts, repair metadata, construct stable `unique_post_id` values, preserve raw post text, and generate structured/signature features.
3. **Stress proxy and initial sampling** - compute seed lexical signals, induce broader data-driven lexicons, reconstruct the refined stress proxy, and generate the initial annotation-sampling artifacts.

## Historical scraping status

`notebooks/01_Scraping_Reproducible.ipynb` is a cleaned reproducible reconstruction of the real scraping logic used during data collection. It documents listing discovery, thread pagination, post parsing, retry/backoff, checkpoint recovery, and profile enrichment.

The finalized reproduction does **not** perform a new live scrape. The reported dataset comes from the historical scraper runs. A fresh live scrape would create a different time snapshot and is not required to reproduce the reported cleaning/modeling results.

## Main execution path from the historical raw corpus

The historical combined scrape is expected at:

`inputs/combined_all.csv`

The normal reproduction sequence is:

1. `notebooks/02_Cleaning_Reproducible.ipynb`
2. `notebooks/03_Stress_Proxy_and_Label_Sampling.ipynb`

Notebook 02 writes:

`data/processed/ninisite_cleaned_enriched.csv`

Notebook 03 reads that file and writes the stress-proxy / lexicon / sampling outputs, including:

`data/processed/ninisite_cleaned_stress_proxy.csv`

The notebooks resolve the project root automatically and use relative paths.

## Scraping implementation

The scraper is intentionally rate-limited and resumable because the historical runs encountered temporary blocks, empty responses, and interrupted long jobs. Completed work is checkpointed so recovery can target only unresolved threads/profiles.

Important mechanisms include:

- section-specific thread discovery;
- pagination over every topic page;
- robust starter/reply parsing;
- retries with backoff;
- per-thread completion state;
- shared profile metadata cache;
- fallback deduplication when a site post ID is unavailable.

`run_live_scrape` remains disabled by default. The scraping code is retained for reproducibility and methodological inspection.

## Raw-content fidelity

Stored `content` is not globally Hazm-normalized during cleaning. Raw surface text is preserved after structural filtering so punctuation, emoji, and character-count features remain faithful to the historical implementation. Persian normalization is applied locally only in routines that require token normalization or stemming.

## Historical cleaning checkpoints

The historical combined corpus contained 414,994 rows. The reconstructed filtering sequence is:

- 414,994 raw rows;
- 413,180 nonempty-content rows;
- 362,026 rows with at least three raw whitespace-separated words;
- 362,017 final cleaned rows after removing nine rows with missing thread title.

Stable IDs use the site post ID when available, with numeric float-like IDs canonicalized (for example `412966185.0` -> `412966185`). Starter/fallback identifiers are constructed only when the site ID is unavailable.

## Category recovery

Three historical source files lacked `category` and `sub_category`. Their completed row-key mapping is frozen in:

`inputs/enrichment/category_recovery.csv`

Notebook 02 replays 99,442 historical category/sub-category repairs. Fresh scraping assigns section metadata upstream and does not require this historical repair.

## Gender recovery

The historical gender parser originally used an incorrect male-marker assumption. The corrected historical markers are retained in the scraping code, and the final cleaning reconstruction replays the historical recovery through compact caches under `inputs/enrichment/`.

The replay reproduces the historical 1,437 unique male-profile checkpoint and 6,535 final male post rows. Remaining unresolved profiles follow the documented historical female-imputation rule and are explicitly identified through provenance fields. The cache is a reproduction artifact, not a new scrape.

## Stress-proxy reconstruction

The proxy is an exploratory sampling signal, not a clinical label and not a feature of the final supervised stress model.

The reconstruction follows the consolidated historical logic:

- upper-quartile initial high-reference group;
- 162,000 lowest-scoring posts as the low-reference pool;
- minimum associated-side count of 10;
- frequency-ratio threshold greater than 2;
- manual filtering of neutral/noisy stems;
- seed counts retained as `post_neg_count_temp` / `post_pos_count_temp`;
- refined proxy combining seed lexical counts, induced lexical counts, emoji counts, and clipped punctuation counts.

The intended final three-band sampling configuration is 500 low / 600 middle / 900 high posts. The retained historical annotation files remain the provenance authority; regenerated sampling templates are reconstruction outputs.

## `is_starter` design choice

`is_starter` is retained for diagnostics but is not an input to the stress-proxy formula. This prevents the proxy from learning a shortcut such as “starter post = stressed” merely because replies are often shorter/supportive.

## Notebook documentation

Each major notebook block contains Markdown describing:

- input and output contracts;
- algorithmic logic;
- reasons for the chosen library/data structure;
- assumptions and failure modes;
- historical-fidelity checkpoints;
- connection to the next project stage.

Additional algorithm notes are available in `docs/ALGORITHM_GUIDE.md`.

## Demo behavior

The final project configuration keeps:

```json
"use_demo_if_input_missing": false
```

Missing real input therefore raises an explicit error rather than silently producing synthetic project results. Developer-only demo utilities remain in `src/` for smoke testing and are not used for the reported results.
