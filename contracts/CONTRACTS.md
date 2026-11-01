# Contracts

## 1. Record schema
One row = one transaction. Required columns:
`record_id` (int, unique, event-time order), `step` (int, event time), `isFraud` (0/1 label),
plus the raw dataset columns. The pipeline adds `is_poisoned` (bool, ground truth for
evaluation only; learners and defenses must NEVER read it).

## 2. Attack signature (P2)
`attack(df, attack_name, budget, timing, seed, dataset) -> (df_poisoned, poison_mask)`
- entry point: `apply_attack` in `src/attacks/api.py`
- `attack_name`: `random_flip | targeted_flip | feature_shift | feature_scale | feature_noise | backdoor`
- `df`: TRAIN stream only (`step < test_from_step`), event-time order. The clean test set is never passed to an attack.
- `budget`: `random_flip` = fraction of all train rows tampered with; every other attack = fraction of fraud rows
- `timing`: `uniform | burst | low_and_slow`
- `dataset`: `paysim | ulb`
- `poison_mask`: numpy bool array, same length and order as `df_poisoned`; True = record was tampered with
- `df_poisoned` has no `is_poisoned` column (the pipeline adds it from `poison_mask`); keeps `record_id`, `step`; row order unchanged
- also provide `make_triggered_test(fraud_test_df, dataset) -> df` which stamps the backdoor trigger, used for `backdoor_hit_rate`

## 3. Defense signature (P3)
`defend(window_df, state: dict) -> (accepted_df, quarantined_df, state)`
- `window_df`: all records of one event-time window, no `is_poisoned` column
- `accepted_df` goes to the learner; `quarantined_df` is held back
- every row of `window_df` appears in exactly one of the two outputs
- `state` carries detector statistics between windows (start as `{}`)
- entry point: `src/defense/pipeline.py` function `defend`

## 4. Config
All settings come from `config/default.yaml`. No hard-coded constants in code.

## 5. Metrics CSV (`results/metrics.csv`, one row per window per run)
`run_id, dataset, attack, budget, timing, defense, seed, window, step_end, n_seen,
n_quarantined, pr_auc, recall, f1, det_precision, det_recall, det_fpr,
time_to_detect, backdoor_hit_rate`
- `pr_auc, recall, f1`: on the clean held-out test set after learning that window
- `det_precision, det_recall`: quarantine vs `is_poisoned` for that window (blank if defense off or no poison)
- `det_fpr`: clean records quarantined / clean records in that window (blank if defense off)
- `time_to_detect`: windows from the first window containing poison to the first window where a poisoned record is quarantined; one value per run, repeated on every row; blank if never detected or no poison
- `backdoor_hit_rate`: share of triggered fraud test records predicted legit; blank unless attack is `backdoor`

## 6. Other files
- `results/benchmark.csv`: throughput and latency (P1), separate from metrics
- ULB loader renames `Time` to `step` (hours) and `Class` to `isFraud`