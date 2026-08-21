# Reconstructed early-project workflow

```text
Ninisite forum sections
  -> forum listing pages
  -> thread manifest
  -> multi-page thread scraping
  -> per-thread checkpoint
  -> repeated recovery passes for empty/blocked/missing threads
  -> profile checkpoint and metadata enrichment
  -> combined raw corpus

  -> remove empty / short / authorless rows
  -> Persian normalization
  -> age + education cleanup
  -> profile merge and gender handling
  -> safe duplicate audit
  -> guaranteed unique_post_id
  -> signature features
  -> cleaned enriched corpus

  -> post punctuation/emoji/word features
  -> small manually seeded positive/negative lexicons
  -> post_neg_count_temp / post_pos_count_temp
  -> initial weak stress proxy
  -> high-proxy vs low-proxy corpus regions
  -> Persian stem frequency counters
  -> >2x differential frequency-ratio vocabulary discovery
  -> manual neutral/noise filtering
  -> larger negative/positive lexicons
  -> final post_neg_count / post_pos_count
  -> refined stress proxy
  -> starter/reply audit ONLY (not proxy input)
  -> stress-enriched annotation sample
  -> annotation templates for stress/anxiety/depression
  -> first Member 1 + Member 2 models -> first fusion -> active learning
```


## Gender metadata: historical repair vs final architecture

### Historical development path

`cleaned unresolved gender`
→ `rebuild thread_id+author+posted_at key`
→ `recover profile_url from combined_all.csv`
→ `selective profile re-scrape`
→ `cache results`
→ `apply explicit male/female results`
→ `documented female assumption for remaining unresolved profiles`

### Final reproducible path

- Notebook 01 owns network access and robust gender parsing.
- Notebook 02 owns only cached data integration and provenance.
- Notebook 02 performs zero HTTP requests.
- `gender_source` distinguishes original, recovered, and imputed values.


## Category provenance: historical repair vs final architecture

Historical development:
`source CSV missing category fields` → `manually identify source section` →
`match thread_id+author+posted_at` → `99,442 rows restored` → `0 missing`.

Final reproduction:
`category_recovery.csv` is a frozen cache of that completed mapping; Cleaning only joins
it and records `category_source`.

Fresh acquisition:
Scraping assigns category/sub-category from `forum_sections.csv` before checkpointing,
so downstream repair is unnecessary.
