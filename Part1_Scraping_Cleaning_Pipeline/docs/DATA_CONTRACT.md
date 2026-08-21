# Data contract

## Cleaned dataset required columns

`unique_post_id`, `content`, `thread_id`, `author`, `posted_at`, `is_starter`.

Additional columns are preserved where available: post/site IDs, likes, reply target, user post count, profile URL, gender, join date, age, education, category/sub-category, signature, and derived signature features.

## Proxy output additions

- `post_char_count`
- `post_punct_count`
- `post_question_count`
- `post_excl_count`
- `post_emoji_count`
- `post_pos_emoji`
- `post_neg_emoji`
- `post_word_count`
- `post_neg_count_temp`
- `post_pos_count_temp`
- `stress_proxy_initial`
- `post_neg_count`
- `post_pos_count`
- `stress_proxy`

`post_neg_count_temp` and `post_pos_count_temp` are intentionally retained because they encode the small, manually chosen high-precision lexicons and can remain useful downstream even after larger data-driven counts are added.

`is_starter` is never used in either stress-proxy formula.
