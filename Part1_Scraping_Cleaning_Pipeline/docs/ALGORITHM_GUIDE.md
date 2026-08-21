# Algorithm Guide — Early Data Pipeline

This is a system-level study guide. It focuses on **algorithms, assumptions, data flow, and failure modes**, not Python syntax.

## End-to-end flow

Ninisite forum
→ manifest-driven resumable scraping
→ durable post/profile checkpoints
→ recovery of unresolved work
→ cleaning and metadata normalization
→ conflict-aware identity and deduplication
→ cleaned post-level data contract
→ seed lexical features
→ weak initial stress proxy
→ differential Persian vocabulary induction
→ manually filtered broad lexicons
→ refined stress proxy
→ stress-enriched annotation sample

## Notebook 01: Resumable scraping

1. Discover expected thread IDs in a manifest.
2. Request thread pages.
3. Parse posts.
4. Persist success immediately.
5. Retry temporary failures with increasing delays.
6. On later runs, compute unresolved = expected − recovered.
7. Retry only unresolved work.

**Key idea:** checkpointing turns a fragile long-running scrape into a recoverable batch process.

## Notebook 02: Cleaning

1. Combine raw checkpoints.
2. Remove structurally unusable rows.
3. Normalize Persian representation.
4. Standardize age/education/profile fields.
5. Construct robust project identity.
6. Audit exact duplicates versus ID conflicts.
7. Preserve conflicting records safely.
8. Extract separate signature metadata features.
9. Assert the final data contract.

**Key idea:** cleaning protects identity and semantics; it is not just deleting nulls.

## Notebook 03: Proxy bootstrapping

1. Count small manually designed seed signals.
2. Build a transparent initial heuristic.
3. Use the heuristic only to form contrasting unlabeled reference groups.
4. Normalize/tokenize/stem Persian text.
5. Count stems in high- and low-proxy groups.
6. Select disproportionately represented stems using frequency ratios.
7. Remove neutral/noisy candidates manually.
8. Recompute broad lexical counts.
9. Combine precise seed features and broad learned features in a refined proxy.
10. Use the proxy for enriched annotation sampling.

**Key idea:** the proxy is a sampling mechanism, not a clinical label.

## Design decisions worth explaining orally

- `is_starter` is audited but excluded from proxy formulas to avoid a structural shortcut.
- Replies are numerous and often contain little self-disclosure, which motivates enriched sampling, but reply/starter status is not treated as stress itself.
- `post_neg_count_temp` and `post_pos_count_temp` are preserved because narrow seed lexicons and broad learned lexicons encode different kinds of evidence.
- Ratio-based differential vocabulary selection is preferable to raw count subtraction because it measures enrichment, not only absolute volume.
- Punctuation counts are clipped to avoid a few extreme posts dominating the proxy.
- Purposeful high-stress enrichment improves label efficiency but means the labeled set cannot estimate natural stress prevalence on the full forum.
- Checkpoints and assertions are part of the algorithm: they make failures observable instead of silent.

## Implementation details versus core concepts

Exact BeautifulSoup selectors, Pandas syntax, file-opening commands, plotting syntax, and individual regular expressions are implementation details rather than core methodological concepts.

Core concepts demonstrated by the pipeline:
- the input and output of each stage;
- why the stage exists;
- the rule/algorithm it implements;
- what can go wrong;
- what assumption or tradeoff is being made;
- how the next stage depends on its output.

## Real-data versus smoke-test figures

The project can run without the private dataset by creating a tiny synthetic corpus. This mode verifies code execution only.

Synthetic rows are deliberately repetitive, so their proxy distribution is not representative of Ninisite. Consequently:

- smoke-test figures are labeled `DEMO ONLY`;
- the canonical report figure `stress_proxy_distribution.png` is generated only when the real cleaned corpus is present;
- no scientific conclusion should be drawn from synthetic fallback outputs.


## Gender-recovery lesson

A data-cleaning stage should not need to repair a web-parser mistake by going back
to the website. That happened during development because the original gender
marker assumption was wrong and profile URLs had already been removed from an
intermediate table.

The final design moves the fix upstream:

1. retain `profile_url` in the raw acquisition contract;
2. parse multiple known marker families;
3. never infer gender from the absence of another marker;
4. preserve `unknown` as an explicit state;
5. retry unresolved profiles in the scraping stage;
6. in cleaning, replay only the frozen historical cache and record provenance.

This is an example of a general system-design principle: **repair the earliest
component that owns the information rather than adding repeated network work to
later stages.**
