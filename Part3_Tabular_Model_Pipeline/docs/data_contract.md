# Data contract

The frozen handoff and manifest are authoritative. `unique_post_id` must be unique. Roles are `train`, `validation`, `test`, and `embargo`. Only train rows have `oof_fold` in 0–4. Only train rows may be fitted. Embargo rows are never transformed or predicted.

Forbidden model inputs include raw `content`, `thread_title`, `stress_proxy`, target labels, split/fold/group columns, and training weights. Content may only be used upstream to recreate the predefined numeric count/signal columns.
